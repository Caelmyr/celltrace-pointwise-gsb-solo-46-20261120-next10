from dataclasses import dataclass

import numpy as np
from scipy import fft

from .calibration import PanelImage
from .geometry import Cell
from .quality import supported_blur


ALIGNMENT_VERSION = "local-texture-v1"
MAX_SHIFT = 8.0
MIN_TILE = 12
MIN_TILE_PIXELS = 40
CONFIRM_SCORE = 0.80
CONFIRM_EXCESS = 0.72
COMPARABLE_RIVAL = 0.93
COMPETING_MARGIN = 0.055
COMPETING_SEPARATION = 2.0
SHIFT_TOLERANCE = 0.45
PREDICT_MARGIN = 0.15
MIN_FIELD_TILES = 6
RECAPTURE_SHARE = 0.34
STATUSES = ("confirmed", "inferred", "ambiguous", "offset", "no-texture")
RECAPTURE_STATUSES = ("ambiguous", "offset", "no-texture")


@dataclass(frozen=True)
class AlignmentMap:
    masks: dict[str, np.ndarray]
    covered: np.ndarray
    cells: dict[str, dict]
    tiles: list[dict]
    field: dict
    report: dict

    @property
    def compared(self) -> np.ndarray:
        return self.masks["confirmed"] | self.masks["inferred"]

    @property
    def recapture(self) -> np.ndarray:
        return self.covered & ~self.compared

    def status_at(self, x: int, y: int) -> str:
        if not self.covered[y, x]:
            return "outside low-current frame"
        for name in STATUSES:
            if self.masks[name][y, x]:
                return name
        return "outside low-current frame"

    def tile_at(self, x: int, y: int) -> dict | None:
        found = None
        for tile in self.tiles:
            if tile["x0"] <= x < tile["x1"] and tile["y0"] <= y < tile["y1"]:
                found = tile
        return found


def texture(panel: PanelImage, fine: float = 1.1, coarse: float = 6.5) -> np.ndarray:
    values = np.nan_to_num(panel.signal, nan=0.0).astype(np.float32)
    detail, _ = supported_blur(values, panel.valid, fine)
    base, _ = supported_blur(values, panel.valid, coarse)
    return np.where(panel.valid, detail - base, 0.0).astype(np.float32)


def tile_bounds(x0: int, y0: int, x1: int, y1: int, across: int = 3) -> list[tuple[int, int, int, int]]:
    columns = max(1, min(across, (x1 - x0) // MIN_TILE))
    rows = max(1, min(across, (y1 - y0) // MIN_TILE))
    xs = np.linspace(x0, x1, columns + 1).round().astype(int)
    ys = np.linspace(y0, y1, rows + 1).round().astype(int)
    return [
        (int(ax), int(ay), int(bx), int(by))
        for ay, by in zip(ys, ys[1:]) for ax, bx in zip(xs, xs[1:])
    ]


def _peak_offset(surface: np.ndarray, limit: int):
    center = surface.shape[0] // 2
    radius = min(limit, center - 1)
    window = surface[center - radius:center + radius + 1, center - radius:center + radius + 1]
    row, column = np.unravel_index(int(np.argmax(window)), window.shape)
    peak = float(window[row, column])
    competing = window.copy()
    competing[max(0, row - 2):row + 3, max(0, column - 2):column + 3] = -np.inf
    second = float(competing.max()) if np.isfinite(competing).any() else -1.0
    second_row, second_column = (int(value) for value in np.unravel_index(int(np.argmax(competing)), competing.shape))

    def refined(previous, current, following):
        curvature = previous - 2 * current + following
        if curvature > -1e-9:
            return 0.0
        return float(np.clip(0.5 * (previous - following) / curvature, -0.5, 0.5))

    on_edge = row in (0, window.shape[0] - 1) or column in (0, window.shape[1] - 1)
    dx = column - radius + (0.0 if on_edge else refined(
        window[row, max(0, column - 1)], peak, window[row, min(window.shape[1] - 1, column + 1)]
    ))
    dy = row - radius + (0.0 if on_edge else refined(
        window[max(0, row - 1), column], peak, window[min(window.shape[0] - 1, row + 1), column]
    ))
    return peak, dx, dy, second, float(second_column - radius), float(second_row - radius)


def _background(surface: np.ndarray, limit: int) -> float:
    center = surface.shape[0] // 2
    radius = min(limit, center - 1)
    reach = center - 1
    far = surface[center - reach:center + reach + 1, center - reach:center + reach + 1].copy()
    local_center = reach
    far[local_center - radius - 2:local_center + radius + 3,
        local_center - radius - 2:local_center + radius + 3] = np.nan
    values = far[np.isfinite(far)]
    if values.size < 16:
        return 0.0
    return float(np.percentile(values, 99))


def tile_shift(first: np.ndarray, second: np.ndarray, common: np.ndarray) -> dict:
    height, width = first.shape
    result = {
        "status": "no-texture", "shift_x": None, "shift_y": None, "rival": None,
        "score": None, "competing": None, "pixels": int(common.sum()),
    }
    if int(common.sum()) < MIN_TILE_PIXELS or min(height, width) < 8:
        return result
    window = np.outer(np.hanning(height), np.hanning(width)).astype(np.float32)
    weights = common.astype(np.float32)
    count = max(1, int(weights.sum()))
    first_centered = (first - float((first * weights).sum() / count)) * window * weights
    second_centered = (second - float((second * weights).sum() / count)) * window * weights
    first_energy = float(np.square(first_centered, dtype=np.float32).sum())
    second_energy = float(np.square(second_centered, dtype=np.float32).sum())
    if first_energy <= 1e-12 or second_energy <= 1e-12:
        return result
    size = fft.next_fast_len(max(height, width) * 2)
    surface = fft.fftshift(fft.irfft2(
        fft.rfft2(first_centered, s=(size, size)) * np.conj(fft.rfft2(second_centered, s=(size, size))),
        s=(size, size),
    )) / np.sqrt(first_energy * second_energy)
    limit = int(min(MAX_SHIFT, min(height, width) // 3))
    peak, raw_x, raw_y, rival, rival_x, rival_y = _peak_offset(surface, limit)
    background = _background(surface, limit)
    excess = (peak - background) / max(1e-9, 1.0 - background)
    # The correlation peak locates the second tile inside the first. Report the motion of
    # low-current content instead, so that low at (x + dx, y + dy) matches high at (x, y).
    measured = {
        "shift_x": round(-raw_x, 3), "shift_y": round(-raw_y, 3),
        "score": round(peak, 4), "excess": round(float(excess), 4),
        "competing": round(rival / peak, 4) if peak > 1e-9 else None,
    }
    if peak < CONFIRM_SCORE or excess < CONFIRM_EXCESS:
        # Below these levels the peak is not distinct enough to quote a shift.
        return {**result, "score": measured["score"], "excess": measured["excess"],
                "competing": measured["competing"]}
    rivals = (-rival_x, -rival_y)
    if (
        rival >= CONFIRM_SCORE
        and rival / peak >= COMPARABLE_RIVAL
        and peak - rival <= COMPETING_MARGIN
        and max(abs(rivals[0] - measured["shift_x"]), abs(rivals[1] - measured["shift_y"])) >= COMPETING_SEPARATION
    ):
        measured["rival"] = [round(rivals[0], 3), round(rivals[1], 3)]
    return {**result, **measured, "status": "measured"}


def fit_field(tiles: list[dict], plane: tuple[int, int]) -> dict:
    usable = [
        tile for tile in tiles
        if tile["status"] == "measured" and tile["rival"] is None and tile["shift_x"] is not None
    ]
    empty = {"supported": False, "tiles": len(usable), "coefficients": None, "residual": None}
    if len(usable) < MIN_FIELD_TILES:
        return empty
    width, height = plane
    x = np.array([(tile["x0"] + tile["x1"]) / 2 / width for tile in usable])
    y = np.array([(tile["y0"] + tile["y1"]) / 2 / height for tile in usable])
    design = np.column_stack((np.ones_like(x), x, y))
    shifts = np.array([[tile["shift_x"], tile["shift_y"]] for tile in usable], dtype=np.float64)
    keep = np.ones(len(usable), dtype=bool)
    coefficients = np.zeros((3, 2))
    for _ in range(4):
        if int(keep.sum()) < MIN_FIELD_TILES:
            return empty
        coefficients, *_ = np.linalg.lstsq(design[keep], shifts[keep], rcond=None)
        residual = np.linalg.norm(shifts - design @ coefficients, axis=1)
        limit = max(SHIFT_TOLERANCE * 0.75, float(np.median(residual[keep])) * 2.5)
        updated = residual <= limit
        if np.array_equal(updated, keep):
            break
        keep = updated
    residual = np.linalg.norm(shifts - design @ coefficients, axis=1)
    return {
        "supported": True,
        "tiles": int(keep.sum()),
        "coefficients": np.round(coefficients, 5).tolist(),
        "residual": round(float(np.median(residual[keep])), 4),
    }


def predict(field: dict, x: float, y: float, plane: tuple[int, int]) -> tuple[float, float] | None:
    """Predicted motion of low-current content at a plane point."""
    if not field["supported"]:
        return None
    width, height = plane
    value = np.array([1.0, x / width, y / height]) @ np.asarray(field["coefficients"])
    return float(value[0]), float(value[1])


def _resolve(tile: dict, field: dict, plane: tuple[int, int]) -> dict:
    predicted = predict(field, (tile["x0"] + tile["x1"]) / 2, (tile["y0"] + tile["y1"]) / 2, plane)
    if predicted is not None:
        tile["predicted_shift"] = [round(predicted[0], 3), round(predicted[1], 3)]
    if tile["status"] == "measured":
        primary = (tile["shift_x"], tile["shift_y"])
        if tile["rival"] is not None:
            rival = tuple(tile["rival"])
            if predicted is None:
                tile["status"] = "ambiguous"
                tile["reason"] = "two comparable local offsets"
                return tile
            primary_error = max(abs(primary[0] - predicted[0]), abs(primary[1] - predicted[1]))
            rival_error = max(abs(rival[0] - predicted[0]), abs(rival[1] - predicted[1]))
            if min(primary_error, rival_error) >= SHIFT_TOLERANCE:
                tile["status"] = "ambiguous"
                tile["reason"] = "two comparable local offsets"
                return tile
            if rival_error < primary_error:
                tile["shift_x"], tile["shift_y"] = tile["rival"]
        magnitude = max(abs(tile["shift_x"]), abs(tile["shift_y"]))
        if magnitude >= SHIFT_TOLERANCE:
            tile["status"] = "offset"
            tile["reason"] = (
                f"low-current plane displaced by {magnitude:.1f} px"
                if predicted is not None and max(abs(predicted[0]), abs(predicted[1])) >= SHIFT_TOLERANCE
                else f"local misregistration of {magnitude:.1f} px"
            )
        else:
            tile["status"] = "confirmed"
            tile["reason"] = None
        return tile
    tile["reason"] = "alignment not confirmable inside the cell"
    if predicted is not None:
        magnitude = max(abs(predicted[0]), abs(predicted[1]))
        if magnitude >= SHIFT_TOLERANCE:
            tile["status"] = "offset"
            tile["reason"] = f"low-current plane displaced by {magnitude:.1f} px"
        elif magnitude < SHIFT_TOLERANCE - PREDICT_MARGIN:
            tile["status"] = "inferred"
            tile["reason"] = "alignment inferred from nearby tiles"
    return tile


def cell_alignment(cell: Cell, first: np.ndarray, second: np.ndarray, common: np.ndarray,
                   plane: tuple[int, int], field: dict, across: int, resolve: bool) -> dict:
    expected = cell.pixels
    covered = common[cell.view]
    masks = {name: np.zeros(covered.shape, dtype=bool) for name in STATUSES}
    tiles = []
    for x0, y0, x1, y1 in tile_bounds(*cell.interior, across):
        view = np.s_[y0:y1, x0:x1]
        tile = tile_shift(first[view], second[view], common[view])
        tile.update(x0=int(x0), y0=int(y0), x1=int(x1), y1=int(y1), cell=cell.id)
        if resolve:
            tile = _resolve(tile, field, plane)
        tiles.append(tile)
        if resolve:
            masks[tile["status"]][y0 - cell.interior[1]:y1 - cell.interior[1],
                                  x0 - cell.interior[0]:x1 - cell.interior[0]] = common[view]
    share = {name: float(masks[name].sum() / expected) for name in STATUSES}
    uncovered = 1.0 - covered.sum() / expected
    reasons = []
    if uncovered > 0.01:
        reasons.append("cell is not fully inside the low-current frame")
    seen = []
    for name in RECAPTURE_STATUSES:
        if share[name] <= 0:
            continue
        sample = next(tile for tile in tiles if tile["status"] == name)
        if sample["reason"] and sample["reason"] not in seen:
            seen.append(sample["reason"])
    reasons.extend(reason for reason in seen if reason not in reasons)
    blocked = sum(share[name] for name in RECAPTURE_STATUSES) + uncovered
    if share["confirmed"] + share["inferred"] == 0:
        state = "recapture"
    elif blocked > RECAPTURE_SHARE:
        state = "recapture"
    elif any(share[name] > 0 for name in RECAPTURE_STATUSES) or uncovered > 0.01:
        state = "partial"
    elif share["inferred"] > 0:
        state = "limited"
        reasons.append("low-signal area included from nearby alignment")
    else:
        state = "measured"
    if state in ("recapture", "partial"):
        # A cell that is not fully confirmed must not put any pixel into the shared
        # reference ratio. Only its own locally confirmed tiles are counted.
        masks["no-texture"] |= masks["inferred"]
        masks["inferred"] = np.zeros_like(masks["inferred"])
    aligned = float((masks["confirmed"] | masks["inferred"]).sum() / expected)
    blocked = 1.0 - aligned
    share = {name: float(masks[name].sum() / expected) for name in STATUSES}
    shifts = [
        (tile["shift_x"], tile["shift_y"]) for tile in tiles
        if tile["status"] in ("confirmed", "offset", "ambiguous") and tile["shift_x"] is not None
    ]
    return {
        "cell": cell.id, "state": state, "reasons": reasons, "tiles": tiles,
        "aligned_fraction": round(aligned, 5), "recapture_fraction": round(min(1.0, blocked), 5),
        "covered_fraction": round(float(covered.sum() / expected), 5),
        **{f"{name}_fraction": round(share[name], 5) for name in STATUSES},
        "median_shift": (
            [round(float(np.median([value[0] for value in shifts])), 3),
             round(float(np.median([value[1] for value in shifts])), 3)] if shifts else None
        ),
        "masks": masks,
    }


def local_alignment(high: PanelImage, low: PanelImage, cells: list[Cell],
                    plane: tuple[int, int]) -> AlignmentMap:
    first, second = texture(high), texture(low)
    common = high.valid & low.valid
    across = 3 if sum(cell.pixels for cell in cells) <= 120_000 else 2
    survey = [cell_alignment(cell, first, second, common, plane, {"supported": False}, across, False) for cell in cells]
    field = fit_field([tile for result in survey for tile in result["tiles"]], plane)
    results = [
        cell_alignment(cell, first, second, common, plane, field, across, True) for cell in cells
    ]
    shape = high.signal.shape
    maps = {name: np.zeros(shape, dtype=bool) for name in STATUSES}
    records, tiles = {}, []
    for cell, result in zip(cells, results):
        for name, values in result.pop("masks").items():
            maps[name][cell.view] = values
        records[result["cell"]] = result
        tiles.extend(result["tiles"])
    compared = maps["confirmed"] | maps["inferred"]
    offsets = [
        max(abs(row["median_shift"][0]), abs(row["median_shift"][1]))
        for row in records.values() if row["median_shift"]
    ]
    report = {
        "version": ALIGNMENT_VERSION,
        **{f"{name}_pixels": int(maps[name].sum()) for name in STATUSES},
        "covered_pixels": int(common.sum()),
        "compared_pixels": int(compared.sum()),
        "recapture_pixels": int((common & ~compared).sum()),
        "plane_compared_fraction": round(float(compared.sum() / max(1, common.sum())), 5),
        **{
            f"cells_{state}": sum(row["state"] == state for row in records.values())
            for state in ("measured", "limited", "partial", "recapture")
        },
        "median_abs_shift": round(float(np.median(offsets)), 3) if offsets else None,
        "maximum_abs_shift": round(float(np.max(offsets)), 3) if offsets else None,
        "field": field,
        "settings": {
            "tile_grid": across, "minimum_tile": MIN_TILE, "confirm_score": CONFIRM_SCORE,
            "confirm_excess": CONFIRM_EXCESS, "competing_margin": COMPETING_MARGIN,
            "shift_tolerance": SHIFT_TOLERANCE, "recapture_share": RECAPTURE_SHARE,
        },
    }
    return AlignmentMap(
        maps, common, records,
        sorted(tiles, key=lambda tile: (tile["cell"], tile["y0"], tile["x0"])), field, report,
    )
