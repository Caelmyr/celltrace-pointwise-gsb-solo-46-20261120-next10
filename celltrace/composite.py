from dataclasses import dataclass, replace

import numpy as np
from scipy import ndimage

from .calibration import PanelImage
from .geometry import Cell
from .images import distribution, histogram_rows
from .model import Capture, ExposureMerge, Issue
from .quality import supported_blur


MERGE_VERSION = "paired-exposure-composite-v1"
REGIMES = {
    0: "no support",
    1: "mutually corroborated",
    2: "long exposure only",
    3: "short exposure only",
    4: "no conclusion",
}
MISSING, CORROBORATED, LONG_ONLY, SHORT_ONLY, UNDECIDED = range(5)
REGION_VERIFIED, REGION_ADJUSTED, REGION_UNVERIFIED = 1, 2, 3
LONG_TRUST_FLOOR = 0.02
SHORT_FLOOR_LEVEL = 0.0025
SHORT_FLOOR_SIGMA = 5.0
SHORT_FLOOR_LEVEL_PERCENTILE = 20.0
SEAM_RING = 4.0
SEAM_MIN_DEVIATION = 0.01


@dataclass(frozen=True)
class MergeResult:
    long_panel: PanelImage
    short_panel: PanelImage
    long_capture: Capture
    short_capture: Capture
    panel: PanelImage
    short_final: np.ndarray
    regime: np.ndarray
    region_status: np.ndarray
    local_ratio: np.ndarray
    reference_ratio: float
    short_floor: float
    exposure_scale: float
    settings: ExposureMerge
    records: list[dict]
    report: dict


def seam_ring(selection: np.ndarray, long_read: np.ndarray,
              short_read: np.ndarray) -> np.ndarray:
    """Cross-check samples banding the edge of a short-only region.

    A skin of overlap just outside the region is preferred because it is nearest and
    least affected by illumination drift. If the patch is enclosed by the clipped
    shadow of the long exposure, the band widens until readable overlap appears.
    """
    readable = long_read & short_read
    for width in (int(SEAM_RING), 2 * int(SEAM_RING), 4 * int(SEAM_RING), 8 * int(SEAM_RING)):
        band = ndimage.binary_dilation(selection, iterations=width) & ~selection & readable
        if int(band.sum()) >= max(12, int(selection.sum()) * 0.15) or width == 8 * int(SEAM_RING):
            return band
    return np.zeros(selection.shape, dtype=bool)


def read_floor(signal: np.ndarray, valid: np.ndarray) -> float:
    """A robust read floor in normalized sensor units for one acquisition."""
    values = signal[valid]
    if values.size < 128:
        return SHORT_FLOOR_LEVEL * 2
    faint = values[values <= np.percentile(values, SHORT_FLOOR_LEVEL_PERCENTILE)]
    if faint.size < 32:
        return SHORT_FLOOR_LEVEL
    level = float(np.median(faint))
    spread = 1.4826 * float(np.median(np.abs(faint - level)))
    return max(SHORT_FLOOR_LEVEL, level + SHORT_FLOOR_SIGMA * spread)


def pad_window(region, shape, padding: int):
    """Grow a bounding region so a ring can be read outside it."""
    y0 = max(0, region[0].start - padding)
    y1 = min(shape[0], region[0].stop + padding)
    x0 = max(0, region[1].start - padding)
    x1 = min(shape[1], region[1].stop + padding)
    return np.s_[y0:y1, x0:x1]


def exposure_scale(long_capture: Capture, short_capture: Capture) -> float:
    return float(long_capture.exposure_ms) / float(short_capture.exposure_ms)


def compose(
    long_panel: PanelImage,
    short_panel: PanelImage,
    long_capture: Capture,
    short_capture: Capture,
    settings: ExposureMerge,
    cells: list[Cell],
) -> MergeResult:
    if long_panel.signal.shape != short_panel.signal.shape:
        raise Issue("Both exposures need to use the same rectified panel plane.", 409)
    scale = exposure_scale(long_capture, short_capture)
    converted = (short_panel.signal * scale).astype(np.float32)
    short_floor = read_floor(short_panel.signal, short_panel.valid) * scale
    long_read = long_panel.valid & (long_panel.signal > LONG_TRUST_FLOOR)
    short_read = short_panel.valid & np.isfinite(converted) & (converted >= short_floor)
    overlap = long_read & short_read

    ratio = np.full(converted.shape, np.nan, dtype=np.float32)
    np.divide(converted, long_panel.signal, out=ratio, where=overlap)
    reference = float(np.median(ratio[overlap])) if int(overlap.sum()) >= 64 else 1.0
    reference = max(min(reference, 8.0), 0.125) if np.isfinite(reference) else 1.0

    local = np.full(converted.shape, np.nan, dtype=np.float32)
    if overlap.any():
        blurred, weights = supported_blur(np.where(overlap, ratio, 0.0), overlap, 3.0)
        measured = overlap & (weights >= 0.5) & np.isfinite(blurred)
        local[measured] = blurred[measured]
    disagreement = overlap & np.isfinite(local) & (
        np.abs(local / reference - 1.0) > settings.agreement
    )

    regime = np.full(converted.shape, MISSING, dtype=np.uint8)
    regime[long_panel.valid] = LONG_ONLY
    regime[short_read & ~long_panel.valid] = SHORT_ONLY
    regime[overlap & ~disagreement] = CORROBORATED
    regime[disagreement] = UNDECIDED

    short_final = converted.copy()
    region_status = np.zeros(converted.shape, dtype=np.uint8)
    labels, count = ndimage.label(regime == SHORT_ONLY, structure=np.ones((3, 3), dtype=bool))
    rejected = 0
    adjusted = 0
    padding = int(SEAM_RING) * 8
    for marker, region in enumerate(ndimage.find_objects(labels, count), 1):
        if region is None:
            continue
        window = pad_window(region, converted.shape, padding)
        selection = labels[window] == marker
        area = int(selection.sum())
        ring = seam_ring(selection, long_read[window], short_read[window])
        required = min(settings.minimum_agreement_pixels, max(4, area // 2))
        if int(ring.sum()) < required:
            regime[window][selection] = UNDECIDED
            region_status[window][selection] = REGION_UNVERIFIED
            rejected += 1
            continue
        gain = float(np.median(long_panel.signal[window][ring] / converted[window][ring]))
        if not np.isfinite(gain) or abs(gain - 1.0) > settings.agreement:
            regime[window][selection] = UNDECIDED
            region_status[window][selection] = REGION_UNVERIFIED
            rejected += 1
            continue
        if abs(gain - 1.0) > SEAM_MIN_DEVIATION:
            short_final[window][selection] = (converted[window][selection] * gain).astype(np.float32)
            region_status[window][selection] = REGION_ADJUSTED
            adjusted += 1
        else:
            region_status[window][selection] = REGION_VERIFIED

    signal = np.full(long_panel.signal.shape, np.nan, dtype=np.float32)
    kept = (regime == CORROBORATED) | (regime == LONG_ONLY)
    signal[kept] = long_panel.signal[kept]
    borrowed = regime == SHORT_ONLY
    signal[borrowed] = short_final[borrowed]
    valid = np.isfinite(signal)
    clipped = ~valid & (long_panel.clipped | (short_panel.clipped & ~long_panel.any_support))
    calibration_gap = ~valid & (
        long_panel.calibration_gap | (short_panel.calibration_gap & ~long_panel.any_support)
    )
    floor_hit = (long_panel.floor_hit & kept) | (short_panel.floor_hit & borrowed)
    panel = PanelImage(
        signal,
        valid,
        long_panel.any_support | short_panel.any_support,
        clipped & ~valid,
        calibration_gap & ~valid,
        floor_hit,
        long_panel.matrix,
        {
            "method": MERGE_VERSION,
            "long": long_panel.details,
            "short": short_panel.details,
            "dark": long_panel.details.get("dark"),
            "flat": long_panel.details.get("flat"),
            "exposure_scale": round(scale, 6),
        },
    )
    result = MergeResult(
        long_panel, short_panel, long_capture, short_capture, panel, short_final,
        regime, region_status, local, reference, short_floor, scale, settings, [], {},
    )
    records = cell_records(result, cells)
    report = merge_report(result, records, int(count), adjusted, rejected)
    return replace(result, records=records, report=report)


def merge_report(result: MergeResult, records: list[dict], regions: int,
                 adjusted: int, rejected: int) -> dict:
    regime = result.regime
    total = int(regime.size)
    messages = []
    if abs(result.reference_ratio - 1.0) > result.settings.agreement * 0.5:
        messages.append(
            f"Exposure-normalized levels differ by {100 * (result.reference_ratio - 1):+.1f}% overall. "
            "Verify both exposures and their acquisition times."
        )
    if result.short_capture.exposure_ms >= result.long_capture.exposure_ms:
        messages.append("The selected short exposure is not shorter than the long exposure.")
    if (
        result.long_capture.current_a is not None and result.short_capture.current_a is not None
        and abs(result.long_capture.current_a - result.short_capture.current_a) > 0.001
    ):
        messages.append("The two exposures declare different drive currents. Composition assumes one drive setpoint.")
    if adjusted:
        messages.append(
            f"{adjusted} short-exposure region{'' if adjusted == 1 else 's'} "
            f"{'was' if adjusted == 1 else 'were'} aligned by a local gain so the join does not add a dark edge. "
            "The scale is listed per cell."
        )
    if rejected:
        messages.append(
            f"{rejected} short-exposure region{'' if rejected == 1 else 's'} could not be cross-checked "
            f"and {'remains' if rejected == 1 else 'remain'} undecided."
        )
    undecided = int((regime == UNDECIDED).sum())
    if undecided:
        messages.append(
            f"{undecided} overlapping pixels disagree beyond the {100 * result.settings.agreement:.0f}% "
            "agreement tolerance. They are left without a conclusion rather than reading either exposure alone."
        )
    cells = [row for row in records if row["support"] > 0]
    weak = [row for row in cells if row["flags"]]
    return {
        "long_capture": result.long_capture.id, "short_capture": result.short_capture.id,
        "long_exposure_ms": result.long_capture.exposure_ms,
        "short_exposure_ms": result.short_capture.exposure_ms,
        "exposure_scale": round(result.exposure_scale, 6),
        "reference_ratio": round(result.reference_ratio, 6),
        "agreement": result.settings.agreement,
        "short_floor": round(result.short_floor, 6),
        "short_calibration": result.settings.short_calibration.dark_id,
        "pixels": total,
        "corroborated_pixels": int((regime == CORROBORATED).sum()),
        "long_only_pixels": int((regime == LONG_ONLY).sum()),
        "short_only_pixels": int((regime == SHORT_ONLY).sum()),
        "undecided_pixels": int((regime == UNDECIDED).sum()),
        "missing_pixels": int((regime == MISSING).sum()),
        "merged_pixels": int(result.panel.valid.sum()),
        "coverage": round(float(result.panel.valid.sum() / total), 6),
        "short_regions": regions,
        "adjusted_regions": adjusted,
        "unverified_regions": rejected,
        "cells": len(records),
        "cells_with_flags": len(weak),
        "histogram": histogram(result.panel.signal[result.panel.valid]),
        "messages": messages,
    }


def histogram(values: np.ndarray) -> list[dict]:
    return histogram_rows(values, (0, 1.2), 26)


def cell_records(result: MergeResult, cells: list[Cell]) -> list[dict]:
    records = []
    for cell in cells:
        view = cell.view
        regime = result.regime[view]
        total = cell.pixels
        support = int(result.panel.valid[view].sum())
        corroborated = int((regime == CORROBORATED).sum())
        long_only = int((regime == LONG_ONLY).sum())
        short_only = int((regime == SHORT_ONLY).sum())
        undecided = int((regime == UNDECIDED).sum())
        stats = distribution(result.panel.signal[view][result.panel.valid[view]])
        gains = result.short_final[view][regime == SHORT_ONLY] / result.short_panel.signal[view][regime == SHORT_ONLY]
        flags = []
        if support / total < 0.6:
            flags.append("insufficient merged support")
        if undecided > total * 0.02 and undecided >= 8:
            flags.append("unresolved overlap")
        if short_only > total * 0.02:
            flags.append("short exposure carries cell")
        if corroborated < 0.5 * max(1, support) and support:
            flags.append("limited cross-check")
        relative_gain = (
            round(float(np.nanmedian(gains)) / result.exposure_scale, 6)
            if short_only and np.isfinite(gains).any() else None
        )
        records.append({
            "cell": cell.id,
            "string": cell.string,
            "expected_pixels": total,
            "support": round(support / total, 5),
            "corroborated": round(corroborated / total, 5),
            "long_only": round(long_only / total, 5),
            "short_only": round(short_only / total, 5),
            "no_conclusion": round(undecided / total, 5),
            "short_gain": relative_gain,
            "median": stats["median"],
            "flags": flags,
        })
    return records


def cell_record(result: MergeResult, cell_id: str) -> dict | None:
    return next((row for row in result.records if row["cell"] == cell_id), None)


def status_reasons(result: MergeResult, x: int, y: int) -> list[str]:
    long_panel, short_panel = result.long_panel, result.short_panel
    reasons = []
    if not (long_panel.any_support[y, x] or short_panel.any_support[y, x]):
        return ["outside both source frames"]
    if not long_panel.any_support[y, x]:
        reasons.append("long exposure outside frame")
    elif not long_panel.valid[y, x]:
        if long_panel.clipped[y, x]:
            reasons.append("long exposure clipped")
        elif long_panel.calibration_gap[y, x]:
            reasons.append("long exposure calibration gap")
        else:
            reasons.append("long exposure unusable")
    if not short_panel.any_support[y, x]:
        reasons.append("short exposure outside frame")
    elif not short_panel.valid[y, x]:
        if short_panel.clipped[y, x]:
            reasons.append("short exposure clipped")
        elif short_panel.calibration_gap[y, x]:
            reasons.append("short exposure calibration gap")
        else:
            reasons.append("short exposure unusable")
    elif not np.isfinite(result.short_final[y, x]) or result.short_final[y, x] < result.short_floor:
        reasons.append("short exposure below read floor")
    regime = result.regime[y, x]
    if regime == UNDECIDED:
        status = int(result.region_status[y, x])
        if status == REGION_UNVERIFIED:
            reasons.append("short-exposure patch has no cross-check at its rim")
        elif np.isfinite(result.local_ratio[y, x]):
            reasons.append("exposures disagree beyond the agreement tolerance")
        else:
            reasons.append("no overlapping reading to cross-check")
    return reasons


def region_merge_reading(result: MergeResult, mask: np.ndarray) -> dict:
    """How the composed plane was assembled over a selected region."""
    regime = result.regime[mask]
    total = int(regime.size)
    if total == 0:
        return {"pixels": 0}
    counts = {
        key: int((regime == value).sum())
        for key, value in (
            ("corroborated", CORROBORATED), ("long_only", LONG_ONLY),
            ("short_only", SHORT_ONLY), ("no_conclusion", UNDECIDED), ("missing", MISSING),
        )
    }
    short = regime == SHORT_ONLY
    gains = result.short_final[mask][short] / (
        result.short_panel.signal[mask][short] * result.exposure_scale
    )
    return {
        "pixels": total,
        **{key: round(value / total, 5) for key, value in counts.items()},
        "short_scale": round(float(np.nanmedian(gains)), 6) if short.any() else None,
        "exposure_scale": round(result.exposure_scale, 6),
        "agreement": result.settings.agreement,
        "message": "Share of the region read from both exposures, the long exposure, the short exposure, "
                   "or left without a conclusion.",
    }


def merge_probe(result: MergeResult | None, x: int, y: int, settings: ExposureMerge | None = None) -> dict:
    if result is None:
        return {"available": False}
    settings = settings or result.settings
    regime = int(result.regime[y, x])
    mode = settings.readings
    long_value = result.long_panel.signal[y, x]
    short_value = result.short_panel.signal[y, x]
    converted = result.short_final[y, x]
    if mode == "long":
        used = "long" if result.long_panel.valid[y, x] and long_value > LONG_TRUST_FLOOR else None
    elif mode == "short":
        used = "short" if result.short_panel.valid[y, x] and np.isfinite(converted) else None
    else:
        used = {CORROBORATED: "both", LONG_ONLY: "long", SHORT_ONLY: "short"}.get(regime)
    record = {
        "available": True,
        "mode": mode,
        "regime": REGIMES[regime],
        "used": used,
        "used_capture": None,
        "used_name": None,
        "long": {
            "capture": result.long_capture.id, "name": result.long_capture.name,
            "exposure_ms": result.long_capture.exposure_ms,
            "value": round(float(long_value), 6) if np.isfinite(long_value) else None,
            "readable": bool(result.long_panel.valid[y, x] and long_value > LONG_TRUST_FLOOR),
        },
        "short": {
            "capture": result.short_capture.id, "name": result.short_capture.name,
            "exposure_ms": result.short_capture.exposure_ms,
            "value": round(float(short_value), 6) if np.isfinite(short_value) else None,
            "converted": round(float(converted), 6) if np.isfinite(converted) else None,
            "gain": round(
                float(result.short_final[y, x] / (short_value * result.exposure_scale)), 6
            ) if np.isfinite(short_value) and short_value > 0 and np.isfinite(converted) else None,
        },
        "exposure_scale": round(result.exposure_scale, 6),
        "reference_ratio": round(result.reference_ratio, 6),
        "local_ratio": round(float(result.local_ratio[y, x]), 6)
        if np.isfinite(result.local_ratio[y, x]) else None,
        "agreement": result.settings.agreement,
        "short_region": {REGION_VERIFIED: "verified", REGION_ADJUSTED: "gain adjusted",
                         REGION_UNVERIFIED: "unverified"}.get(int(result.region_status[y, x])),
        "composed_regime": REGIMES[regime],
        "reasons": status_reasons(result, x, y),
        "warnings": status_reasons(result, x, y) if used is None else [],
    }
    if used == "long":
        record["used_capture"] = result.long_capture.id
        record["used_name"] = result.long_capture.name
    elif used == "short":
        record["used_capture"] = result.short_capture.id
        record["used_name"] = result.short_capture.name
    elif used == "both":
        record["used_capture"] = result.long_capture.id
        record["used_name"] = result.long_capture.name
        record["corroborated_by"] = result.short_capture.name
    return record
