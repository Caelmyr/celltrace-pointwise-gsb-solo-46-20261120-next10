from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .calibration import PanelImage
from .geometry import Cell, millimeter_scale
from .model import AnalysisSettings, Plane
from .placement_check import refined_minimum


MIN_CONTRAST = 0.12
AMBIGUITY_DEPTH = 0.78
MIN_SEPARATION = 3
COVERAGE_FLOOR = 0.55
SEGMENT_TARGET = 22
MAX_SEGMENTS = 12

STATUS_COLORS = {
    "tracked": (139, 231, 201),
    "review": (240, 201, 115),
    "obscured": (233, 138, 163),
    "lost": (154, 163, 178),
}
NOMINAL_COLOR = (98, 158, 197)


@dataclass(frozen=True)
class BusbarMap:
    records: list[dict]
    report: dict


def busbar_check(panel: PanelImage, plane: Plane, cells: list[Cell],
                 settings: AnalysisSettings) -> BusbarMap:
    records = []
    messages = []
    skipped = 0
    scale = millimeter_scale(plane)
    if not settings.busbars:
        messages.append(
            "No nominal busbar positions are configured. Set them under Thresholds to compare lanes."
        )
    for cell in cells:
        left, top, right, bottom = cell.interior
        height = bottom - top
        if height < 24 or right - left < 12:
            skipped += len(settings.busbars)
            continue
        count = max(3, min(MAX_SEGMENTS, int(round(height / SEGMENT_TARGET))))
        edges = np.rint(np.linspace(top, bottom, count + 1)).astype(int)
        for lane, fraction in enumerate(settings.busbars, start=1):
            nominal = cell.bounds[0] + fraction * (cell.bounds[2] - cell.bounds[0])
            center = int(np.floor(nominal))
            search = settings.busbar_radius + 4
            x_lo = max(left, center - search)
            x_hi = min(right, center + search + 1)
            if x_hi - x_lo < 5:
                skipped += 1
                continue
            minimum_width = max(2.0, settings.busbar_radius * 0.8)
            for segment, (y0, y1) in enumerate(zip(edges, edges[1:]), start=1):
                records.append(measure_segment(
                    panel, cell, lane, fraction, segment, int(y0), int(y1),
                    x_lo, x_hi, nominal, minimum_width, scale,
                ))
    if skipped:
        messages.append(f"{skipped} busbar lane(s) sit too close to a cell edge to measure.")
    return BusbarMap(records, lane_report(records, settings, scale, messages))


def measure_segment(panel: PanelImage, cell: Cell, lane: int, fraction: float, segment: int,
                    y0: int, y1: int, x_lo: int, x_hi: int, nominal: float,
                    minimum_width: float, scale) -> dict:
    signal = panel.signal[y0:y1, x_lo:x_hi]
    valid = panel.valid[y0:y1, x_lo:x_hi]
    coverage = float(valid.mean()) if valid.size else 0.0
    row = {
        "id": f"{cell.id}-B{lane}-S{segment:02d}",
        "cell": cell.id,
        "busbar": lane,
        "position": fraction,
        "segment": segment,
        "bounds": [x_lo, y0, x_hi, y1],
        "nominal_x": round(nominal, 2),
        "detected_x": None,
        "offset_px": None,
        "offset_mm": None,
        "contrast": None,
        "coverage": round(coverage, 4),
        "status": "obscured",
        "obstruction": [],
        "candidates": [],
    }
    obstruction = obstruction_reasons(panel, y0, y1, x_lo, x_hi)
    if coverage < COVERAGE_FLOOR:
        row["obstruction"] = obstruction or ["unsupported"]
        return row
    profile = column_profile(signal, valid)
    finite = np.isfinite(profile)
    if finite.sum() < 4:
        row["obstruction"] = obstruction or ["unsupported"]
        return row
    filled = np.interp(np.arange(profile.size), np.flatnonzero(finite), profile[finite])
    smooth = ndimage.gaussian_filter1d(filled, 0.85, mode="nearest")
    brightest = np.sort(smooth)[-max(3, smooth.size // 4):]
    shoulder = float(np.median(brightest))
    if shoulder < 0.04:
        row["obstruction"] = obstruction + ["low signal"]
        return row
    candidates = valley_candidates(smooth, filled, shoulder, x_lo)
    row["obstruction"] = obstruction
    if not candidates:
        row["status"] = "lost"
        return row
    best = max(candidates, key=lambda candidate: candidate["depth"])
    nearest = min(candidates, key=lambda candidate: abs(candidate["x"] - nominal))
    rivals = [
        candidate for candidate in candidates if candidate is not best
        and candidate["depth"] >= best["depth"] * AMBIGUITY_DEPTH
        and abs(candidate["x"] - best["x"]) >= MIN_SEPARATION
    ]
    row["contrast"] = round(best["depth"], 4)
    if rivals or best is not nearest or best["width"] < minimum_width:
        row["status"] = "review"
        row["candidates"] = [
            [candidate["x"], round(candidate["depth"], 4)]
            for candidate in sorted(candidates, key=lambda candidate: candidate["x"])
        ]
        return row
    row["status"] = "tracked"
    row["detected_x"] = round(best["x"], 2)
    offset = best["x"] - nominal
    row["offset_px"] = round(offset, 2)
    if scale is not None:
        row["offset_mm"] = round(offset * scale[0], 3)
    return row


def column_profile(signal: np.ndarray, valid: np.ndarray) -> np.ndarray:
    profile = np.full(signal.shape[1], np.nan, dtype=np.float64)
    for x in range(signal.shape[1]):
        rows = valid[:, x]
        if rows.sum() >= 3:
            profile[x] = float(np.median(signal[rows, x]))
    return profile


def valley_candidates(smooth: np.ndarray, filled: np.ndarray, shoulder: float, x_lo: int) -> list[dict]:
    candidates = []
    for index in range(1, smooth.size - 1):
        center = float(smooth[index])
        if not (center < smooth[index - 1] and center <= smooth[index + 1]):
            continue
        depth = (shoulder - center) / shoulder
        if depth < MIN_CONTRAST:
            continue
        level = shoulder * (1 - depth * 0.5)
        lo = index
        while lo > 0 and filled[lo - 1] <= level:
            lo -= 1
        hi = index
        while hi < filled.size - 1 and filled[hi + 1] <= level:
            hi += 1
        candidates.append({
            "x": x_lo + refined_minimum(smooth, index),
            "depth": float(depth),
            "width": float(hi - lo + 1),
        })
    return candidates


def obstruction_reasons(panel: PanelImage, y0: int, y1: int, x_lo: int, x_hi: int) -> list[str]:
    reasons = []
    if float((~panel.any_support[y0:y1, x_lo:x_hi]).mean()) > 0.25:
        reasons.append("outside source")
    if float(panel.clipped[y0:y1, x_lo:x_hi].mean()) > 0.2:
        reasons.append("clipped capture")
    if float(panel.calibration_gap[y0:y1, x_lo:x_hi].mean()) > 0.2:
        reasons.append("calibration gap")
    return reasons


def lane_report(records: list[dict], settings: AnalysisSettings, scale, messages: list[str]) -> dict:
    offsets = [abs(row["offset_px"]) for row in records if row["offset_px"] is not None]
    statuses = {}
    for row in records:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    tracked = statuses.get("tracked", 0)
    return {
        "lanes": len({(row["cell"], row["busbar"]) for row in records}),
        "segments": len(records),
        "tracked": tracked,
        "review": statuses.get("review", 0),
        "obscured": statuses.get("obscured", 0),
        "lost": statuses.get("lost", 0),
        "tracked_fraction": round(tracked / len(records), 4) if records else None,
        "median_abs_offset_px": round(float(np.median(offsets)), 3) if offsets else None,
        "max_abs_offset_px": round(float(max(offsets)), 3) if offsets else None,
        "max_abs_offset_mm": round(max(offsets) * scale[0], 3) if offsets and scale else None,
        "lanes_with_unclear": len({
            (row["cell"], row["busbar"]) for row in records if row["status"] != "tracked"
        }),
        "cells_with_unclear": len({row["cell"] for row in records if row["status"] != "tracked"}),
        "busbar_positions": list(settings.busbars),
        "busbar_half_width": settings.busbar_radius,
        "messages": messages,
        "measurement": "Configured busbar lanes checked segment by segment against column valleys inside cell interiors",
    }


def paint_busbar_check(image, records: list[dict]):
    from PIL import ImageDraw
    from .render import draw_tag, font

    painter = ImageDraw.Draw(image, "RGBA")
    face = font(10)
    for row in records:
        x0, y0, x1, y1 = row["bounds"]
        nominal = row["nominal_x"]
        painter.line((nominal, y0, nominal, y1), fill=(*NOMINAL_COLOR, 110), width=1)
        if row["segment"] == 1:
            draw_tag(painter, (nominal + 3, y1 - 15), f"B{row['busbar']}", NOMINAL_COLOR, face)
        status = row["status"]
        color = STATUS_COLORS[status]
        if status == "tracked":
            detected = row["detected_x"]
            painter.line((detected, y0, detected, y1), fill=(*color, 215), width=2)
            offset = row["offset_px"]
            if abs(offset) >= 2.5:
                draw_tag(painter, (detected + 4, (y0 + y1) / 2 - 6), f"{offset:+.1f}", color, face)
            continue
        painter.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(*color, 200), width=1)
        if status == "review":
            for candidate_x, _ in row["candidates"]:
                painter.line((candidate_x, y0 + 2, candidate_x, y1 - 3), fill=(*color, 170), width=1)
        if status == "obscured":
            painter.line((x0 + 2, y0 + 2, x1 - 3, y1 - 3), fill=(*color, 190))
            painter.line((x0 + 2, y1 - 3, x1 - 3, y0 + 2), fill=(*color, 190))
        draw_tag(painter, (x0 + 2, y1 - 13), f"B{row['busbar']}-S{row['segment']:02d}", color, face)
    return image
