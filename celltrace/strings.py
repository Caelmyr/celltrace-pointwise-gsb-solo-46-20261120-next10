from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .geometry import Cell
from .intensity import IntensityMap
from .model import Plane

COVERAGE_FLOOR = 0.7
DIM_FLOOR = 0.8
LOW_CELL_FLOOR = 0.72
STEP_LIMIT = 0.35
IMBALANCE_LIMIT = 0.2
GAP_SPAN = 3


@dataclass(frozen=True)
class StringMap:
    records: list[dict]
    heat: np.ndarray
    support: np.ndarray
    report: dict


def physical_runs(cells: list[Cell], plane: Plane) -> dict[int, list[Cell]]:
    groups = defaultdict(list)
    for cell in cells:
        groups[cell.string].append(cell)
    if plane.string_axis == "half_columns":
        half_rows = max(plane.rows // 2, 1)
        key = lambda cell: (
            cell.col,
            (cell.row % half_rows) if cell.col % 2 == 0 else -(cell.row % half_rows),
        )
    elif plane.string_axis == "columns":
        key = lambda cell: (cell.col, cell.row if cell.col % 2 == 0 else -cell.row)
    else:
        key = lambda cell: (cell.row, cell.col if cell.row % 2 == 0 else -cell.col)
    return {number: sorted(groups[number], key=key) for number in sorted(groups)}


def readings_along_run(run: list[Cell], map_data: IntensityMap) -> list[dict]:
    readings = []
    for index, cell in enumerate(run):
        median = map_data.medians[cell.id]
        relative = None
        if median is not None and map_data.panel_median is not None and map_data.panel_median > 1e-6:
            relative = median / map_data.panel_median
        readings.append({
            "cell": cell.id,
            "index": index + 1,
            "signal": median,
            "relative": round(relative, 6) if relative is not None else None,
        })
    return readings


def largest_step(readings: list[dict]) -> dict | None:
    largest = None
    for first, second in zip(readings, readings[1:]):
        a, b = first["relative"], second["relative"]
        if a is None or b is None:
            continue
        change = b - a
        if largest is None or abs(change) > abs(largest["delta"]):
            largest = {"from": first["cell"], "to": second["cell"], "delta": round(change, 6)}
    return largest


def low_runs(readings: list[dict], floor: float) -> list[list[str]]:
    clusters = []
    active = []
    for row in readings:
        value = row["relative"]
        if value is not None and value < floor:
            active.append(row["cell"])
        elif active:
            clusters.append(active)
            active = []
    if active:
        clusters.append(active)
    return clusters


def longest_gap(readings: list[dict]) -> int:
    longest = current = 0
    for row in readings:
        current = current + 1 if row["relative"] is None else 0
        longest = max(longest, current)
    return longest


def reliable(row: dict) -> bool:
    return row["coverage"] >= COVERAGE_FLOOR and row["median_relative"] is not None


def pair_branches(results: list[dict]) -> list[dict]:
    by_section = defaultdict(list)
    for row in results:
        number = row["string"]
        row["section"] = (number - 1) // 2 + 1
        row["half"] = "upper" if number % 2 == 1 else "lower"
        row["sibling"] = number + 1 if number % 2 == 1 else number - 1
        row["paired_relative"] = None
        row["section_note"] = None
        by_section[row["section"]].append(row)
    sections = []
    for section in sorted(by_section):
        branches = by_section[section]
        note = None
        if len(branches) == 2:
            upper, lower = branches
            if reliable(upper) and reliable(lower):
                upper_med, lower_med = upper["median_relative"], lower["median_relative"]
                if lower_med > 1e-6:
                    upper["paired_relative"] = round(upper_med / lower_med, 5)
                if upper_med > 1e-6:
                    lower["paired_relative"] = round(lower_med / upper_med, 5)
                if upper_med < DIM_FLOOR and lower_med < DIM_FLOOR:
                    assessment = "common dimming"
                    note = "Both branches are dim together, which points to section-level dimming rather than a few cells."
                elif abs(upper_med - lower_med) >= IMBALANCE_LIMIT:
                    assessment = "unbalanced"
                    note = f"Parallel branches differ by {abs(upper_med - lower_med):.2f}; one branch runs dimmer than its sibling."
                else:
                    assessment = "balanced"
            elif not reliable(upper) and not reliable(lower):
                assessment = "insufficient data"
                note = "Both branches lack valid readings, so common dimming cannot be told apart from missing capture."
            else:
                assessment = "single branch"
                note = "One branch is largely unmeasured; the measured branch is assessed on its own, not against the missing sibling."
        else:
            assessment = "single branch"
            note = "This section has no measured parallel sibling to compare against."
        for row in branches:
            row["section_note"] = note
        sections.append({
            "section": section,
            "branches": [row["string"] for row in branches],
            "assessment": assessment,
            "note": note,
        })
    return sections


def inspect_strings(cells: list[Cell], plane: Plane, map_data: IntensityMap) -> StringMap:
    heat = np.full((plane.height, plane.width), np.nan, dtype=np.float32)
    support = map_data.comparable.copy()
    half_cut = plane.string_axis == "half_columns"
    results = []
    for number, run in physical_runs(cells, plane).items():
        readings = readings_along_run(run, map_data)
        available = [row["relative"] for row in readings if row["relative"] is not None]
        fraction = len(available) / len(run)
        robust = float(np.median(available)) if available else None
        variation = float(np.percentile(available, 90) - np.percentile(available, 10)) if available else None
        low = low_runs(readings, LOW_CELL_FLOOR)
        step = largest_step(readings)
        low_cluster = any(len(cluster) >= 2 for cluster in low)
        low_share = sum(1 for value in available if value < LOW_CELL_FLOOR)
        review = []
        if fraction < COVERAGE_FLOOR:
            review.append("incomplete run")
        if robust is not None and fraction >= COVERAGE_FLOOR and robust < DIM_FLOOR:
            review.append("dim branch" if half_cut else "dim string")
        if low_cluster and (not half_cut or low_share < len(available)):
            review.append("adjacent dim cells")
        if step and abs(step["delta"]) > STEP_LIMIT:
            review.append("large neighboring step")
        if half_cut and longest_gap(readings) >= GAP_SPAN:
            review.append("contiguous capture gap")
        results.append({
            "string": number, "cells": [cell.id for cell in run], "readings": readings,
            "coverage": round(fraction, 5),
            "median_relative": round(robust, 5) if robust is not None and fraction >= COVERAGE_FLOOR else None,
            "p90_p10_span": round(variation, 5) if variation is not None else None,
            "largest_step": step,
            "low_runs": low,
            "flags": review,
        })
        for cell, row in zip(run, readings):
            if row["relative"] is not None:
                heat[cell.view][support[cell.view]] = row["relative"]
    report = {
        "groups": len(results), "groups_with_flags": sum(bool(row["flags"]) for row in results),
        "axis": plane.string_axis, "string_width": plane.string_size,
        "weighted": "One median per supported cell in the declared serpentine order",
    }
    if half_cut:
        sections = pair_branches(results)
        report["weighted"] = (
            "One median per supported cell along each half-cut branch; "
            "every bypass section pairs an upper and a lower parallel branch"
        )
        report["sections"] = len(sections)
        report["section_assessments"] = sections
        report["sections_to_review"] = sum(section["assessment"] != "balanced" for section in sections)
        report["messages"] = [
            f"Section {section['section']}: {section['note']}" for section in sections if section["note"]
        ]
    return StringMap(results, heat, np.isfinite(heat), report)
