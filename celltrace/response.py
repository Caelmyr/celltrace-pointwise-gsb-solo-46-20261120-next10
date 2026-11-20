from dataclasses import dataclass

import numpy as np

from .alignment import AlignmentMap, local_alignment
from .calibration import PanelImage
from .geometry import Cell
from .images import distribution, histogram_rows
from .model import AnalysisSettings, Capture, Issue


@dataclass(frozen=True)
class CurrentResponse:
    ratio: np.ndarray
    relative: np.ndarray
    eligible: np.ndarray
    common: np.ndarray
    weak: np.ndarray
    alignment: AlignmentMap
    records: list[dict]
    report: dict


def explain_ratio_requirements(high: Capture, low: Capture) -> list[str]:
    warnings = []
    if high.current_a is None or low.current_a is None:
        warnings.append("Drive currents are missing. Ratio is labeled by selected role only.")
    elif high.current_a <= low.current_a:
        warnings.append("The selected high-current image must have the larger drive current.")
    if not 0.02 <= low.exposure_ms / high.exposure_ms <= 50:
        warnings.append("Exposure times differ by more than 50×. Verify the metadata.")
    if high.bit_depth != low.bit_depth:
        warnings.append("Capture code depths differ. Ratios use the declared code ranges.")
    return warnings


def response_map(high: PanelImage, low: PanelImage, high_capture: Capture, low_capture: Capture,
                 cells: list[Cell], settings: AnalysisSettings,
                 plane: tuple[int, int]) -> CurrentResponse:
    if high.signal.shape != low.signal.shape:
        raise Issue("Both acquisitions need to use the same rectified panel plane.", 409)
    messages = explain_ratio_requirements(high_capture, low_capture)
    if (
        high_capture.current_a is not None
        and low_capture.current_a is not None
        and high_capture.current_a <= low_capture.current_a
    ):
        raise Issue(messages[0], 409)
    common = high.valid & low.valid
    alignment = local_alignment(high, low, cells, plane)
    comparable = alignment.compared
    stable = comparable & (high.signal > 0.02) & (low.signal > 0.002)
    exposure_scale = high_capture.exposure_ms / low_capture.exposure_ms
    current_ratio = np.full(high.signal.shape, np.nan, dtype=np.float32)
    np.divide(low.signal, high.signal, where=stable, out=current_ratio)
    current_ratio[stable] *= exposure_scale
    eligible = np.zeros(high.signal.shape, dtype=bool)
    rows = []
    references = []
    for cell in cells:
        view = cell.view
        aligned = alignment.cells[cell.id]
        aligned_view = alignment.compared[view]
        count_common = int(common[view].sum())
        count_aligned = int(aligned_view.sum())
        count_stable = int(stable[view].sum())
        qualified = count_stable / cell.pixels >= settings.usable_cell
        stats = distribution(current_ratio[view][stable[view]])
        median_value = stats["median"] if qualified else None
        if median_value is not None:
            references.append(median_value)
            eligible[view] = stable[view]
        rows.append({
            "cell": cell.id,
            "string": cell.string,
            "expected_pixels": cell.pixels,
            "common_pixels": count_common,
            "aligned_pixels": count_aligned,
            "stable_pixels": count_stable,
            "alignment_state": aligned["state"],
            "alignment_note": "; ".join(aligned["reasons"]) or "aligned locally",
            "aligned_fraction": aligned["aligned_fraction"],
            "support": round(count_stable / cell.pixels, 5),
            "ratio": median_value,
            "ratio_p10": stats["p10"] if qualified else None,
            "ratio_p90": stats["p90"] if qualified else None,
        })
    reference = float(np.median(references)) if references else None
    relative = np.full(current_ratio.shape, np.nan, dtype=np.float32)
    weak = np.zeros(current_ratio.shape, dtype=bool)
    if reference is not None and reference > 1e-8:
        relative[eligible] = current_ratio[eligible] / reference
        weak = eligible & (relative < settings.current_ratio)
    qualified_cells = 0
    weak_cells = 0
    for row, cell in zip(rows, cells):
        if row["ratio"] is None or reference is None or reference <= 1e-8:
            row.update(relative=None, weak_fraction=None, assessment="insufficient aligned overlap")
            continue
        qualified_cells += 1
        relative_value = float(row["ratio"] / reference)
        count_weak = int((weak & eligible)[cell.view].sum())
        is_weak = relative_value < settings.current_ratio
        if is_weak:
            weak_cells += 1
        row.update(
            relative=round(relative_value, 5),
            weak_fraction=round(count_weak / max(1, int(eligible[cell.view].sum())), 6),
            assessment="low relative response" if is_weak else "in range",
        )
    if qualified_cells == 0:
        messages.append("No cell reaches the usable aligned-overlap threshold.")
    alignment_messages(alignment, messages)
    report = {
        "low_capture": low_capture.id, "high_capture": high_capture.id,
        "high_current_a": high_capture.current_a, "low_current_a": low_capture.current_a,
        "exposure_scale": round(exposure_scale, 5),
        "reference_ratio": round(reference, 5) if reference is not None else None,
        "usable_cells": qualified_cells,
        "weak_cells": weak_cells,
        "shared_pixels": int(common.sum()),
        "comparable_pixels": int(comparable.sum()),
        "stable_pixels": int(stable.sum()),
        "weak_pixels": int(weak.sum()),
        "alignment": alignment.report,
        "relative_histogram": histogram_rows(relative[eligible], (0, 1.6), 20),
        "messages": messages,
    }
    return CurrentResponse(current_ratio, relative, eligible, common, weak, alignment, rows, report)


def alignment_messages(alignment: AlignmentMap, messages: list[str]):
    report = alignment.report
    if not report["field"]["supported"]:
        messages.append(
            "Local alignment could not be modelled across the plane. Only cells with their own "
            "confirmed alignment contribute to the ratio."
        )
    if report["cells_recapture"]:
        messages.append(
            f"{report['cells_recapture']} cell(s) need recapture for the ratio: their local alignment "
            "is offset, ambiguous, or not confirmable."
        )
    if report["cells_partial"]:
        messages.append(
            f"{report['cells_partial']} cell(s) only partly contribute: part of the cell is not "
            "registered or not inside both frames."
        )


def probe_response(analysis: CurrentResponse, x: int, y: int) -> dict:
    relative = analysis.relative[y, x]
    ratio = analysis.ratio[y, x]
    status = analysis.alignment.status_at(x, y)
    tile = analysis.alignment.tile_at(x, y) or {}
    shift = correspondence(analysis, x, y)
    return {
        "reported": bool(analysis.eligible[y, x]),
        "common": bool(analysis.common[y, x]),
        "alignment": status,
        "alignment_reason": tile.get("reason"),
        "local_shift": shift["shift"],
        "high_point": [x, y],
        "low_point": [x, y] if analysis.common[y, x] else None,
        "match_point": shift["match"],
        "ratio": round(float(ratio), 6) if np.isfinite(ratio) else None,
        "relative": round(float(relative), 6) if np.isfinite(relative) else None,
        "weak": bool(analysis.weak[y, x]),
    }


def correspondence(analysis: CurrentResponse, x: int, y: int) -> dict:
    tile = analysis.alignment.tile_at(x, y)
    if tile is None:
        return {"shift": None, "match": None}
    if tile.get("status") == "inferred" and tile.get("predicted_shift") is not None:
        shift = tile["predicted_shift"]
    elif tile.get("shift_x") is not None:
        shift = [tile["shift_x"], tile["shift_y"]]
    elif tile.get("predicted_shift") is not None:
        shift = tile["predicted_shift"]
    else:
        return {"shift": None, "match": None}
    height, width = analysis.ratio.shape
    match_x = int(np.clip(round(x + shift[0]), 0, width - 1))
    match_y = int(np.clip(round(y + shift[1]), 0, height - 1))
    return {"shift": [round(float(shift[0]), 3), round(float(shift[1]), 3)], "match": [match_x, match_y]}
