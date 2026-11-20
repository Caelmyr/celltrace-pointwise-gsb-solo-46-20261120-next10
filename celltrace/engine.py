from dataclasses import asdict, dataclass
from threading import BoundedSemaphore

import numpy as np

from .cache import MemoryCache
from .calibration import (
    CAL_VERSION, PanelImage, ReferenceImage, calibrate_sensor,
    panel_calibration_report, project_panel,
)
from .composite import MERGE_VERSION, MergeResult, compose, merge_probe
from .cracks import CrackMap, crack_evidence
from .findings import get_finding, measure_polygon, pixel_evidence
from .geometry import Cell, SamplingMap, cells_for, default_placement
from .images import decode_source, verify_capture
from .intensity import DarkMap, IntensityMap, cell_intensity, dark_area_map
from .model import Calibration, Capture, Inspection, Issue, SENSOR_ROLES, choice, integer, parse_cell, xy_list
from .quality import QualityMap, capture_quality, source_quality
from .response import CurrentResponse, probe_response, response_map
from .store import Store, data_digest
from .strings import StringMap, inspect_strings
from .transects import Transect, analyze_transect


MEASUREMENT_PLANES = frozenset(("quality", "calibration", "intensity", "dark", "cracks", "response", "strings", "merge"))
EL_LAYERS = frozenset(("panel", "support", "grid")) | MEASUREMENT_PLANES
MODEL_REVISION = "celltrace-plane-v1"


@dataclass(frozen=True)
class PanelContext:
    capture: Capture
    panel: PanelImage
    cells: list[Cell]
    pixel_key: tuple
    merge: MergeResult | None = None


@dataclass(frozen=True)
class PlaneAnalysis:
    context: PanelContext
    data: QualityMap | DarkMap | IntensityMap | CrackMap | CurrentResponse | StringMap | dict


def leg_calibration(inspection: Inspection, capture: Capture) -> Calibration:
    return inspection.merge.leg_calibration(inspection, capture)


def panel_identity(inspection: Inspection, capture: Capture) -> tuple:
    placement = inspection.placements.get(capture.id)
    calibration = leg_calibration(inspection, capture)
    bindings = []
    for reference_id in (calibration.dark_id, calibration.flat_id):
        reference = inspection.captures.get(reference_id)
        bindings.append(data_digest(asdict(reference)) if reference else None)
    return (
        MODEL_REVISION, CAL_VERSION, data_digest(asdict(capture)),
        data_digest(asdict(placement)) if placement else None,
        inspection.plane.width, inspection.plane.height,
        calibration.flat_floor, *bindings,
    )


def merge_identity(inspection: Inspection, long: Capture, short: Capture) -> tuple:
    settings = inspection.merge
    return (
        MERGE_VERSION,
        panel_identity(inspection, long), panel_identity(inspection, short),
        settings.short_id, settings.agreement, settings.minimum_agreement_pixels,
        data_digest(asdict(inspection.plane)),
    )


class Engine:
    def __init__(self, store: Store):
        self.store = store
        self.originals = MemoryCache(66 * 1024 * 1024)
        self.panels = MemoryCache(105 * 1024 * 1024)
        self.measurements = MemoryCache(75 * 1024 * 1024)
        self.materials = MemoryCache(30 * 1024 * 1024)
        self.slots = BoundedSemaphore(2)

    def selected_reference(self, inspection: Inspection, role: str) -> ReferenceImage | None:
        key = getattr(inspection.calibration, f"{role}_id")
        if key is None:
            return None
        capture = inspection.capture(key)
        if capture.role != role:
            raise Issue(f"The calibration source is not a {role} image.", 409)
        return ReferenceImage(capture, self.read_source(capture))

    def leg_reference(self, inspection: Inspection, capture: Capture, role: str) -> ReferenceImage | None:
        key = getattr(leg_calibration(inspection, capture), f"{role}_id")
        if key is None:
            return None
        reference = inspection.capture(key)
        if reference.role != role:
            raise Issue(f"The calibration reference for {capture.name} is not a {role} image.", 409)
        return ReferenceImage(reference, self.read_source(reference))

    def read_source(self, capture: Capture):
        source = self.originals.get(capture.digest, lambda: decode_source(self.store.blobs.read(capture.digest)))
        verify_capture(source, capture)
        return source

    def panel_context(self, inspection: Inspection, role: str) -> PanelContext:
        choice(role, "EL capture role", SENSOR_ROLES)
        capture = inspection.selected(role)
        merge = None
        if role == "high":
            merge = self.merge(inspection)
            if merge is not None:
                if inspection.merge.readings == "long":
                    return PanelContext(merge.long_capture, merge.long_panel, cells_for(inspection.plane),
                                        panel_identity(inspection, merge.long_capture), merge)
                if inspection.merge.readings == "short":
                    return PanelContext(merge.short_capture, merge.short_panel, cells_for(inspection.plane),
                                        panel_identity(inspection, merge.short_capture), merge)
                return PanelContext(merge.long_capture, merge.panel, cells_for(inspection.plane),
                                    merge_identity(inspection, merge.long_capture, merge.short_capture), merge)
        key = panel_identity(inspection, capture)
        panel = self.panels.get(key, lambda: self.calculate_panel(inspection, capture))
        return PanelContext(capture, panel, cells_for(inspection.plane), key)

    def merge(self, inspection: Inspection) -> MergeResult | None:
        settings = inspection.merge
        if not settings.enabled or settings.short_id is None:
            return None
        long_capture = inspection.selected("high")
        if long_capture.id == settings.short_id:
            raise Issue("The long and short exposure must be two different high-current images.", 409)
        short_capture = inspection.capture(settings.short_id)
        if short_capture.role != "high":
            raise Issue("The second exposure is not a high-current image.", 409)
        if short_capture.exposure_ms >= long_capture.exposure_ms:
            raise Issue("The paired exposure is not shorter than the selected long exposure.", 409)
        key = merge_identity(inspection, long_capture, short_capture)
        return self.panels.get(key, lambda: self.calculate_merge(
            inspection, long_capture, short_capture,
        ))

    def calculate_merge(self, inspection: Inspection, long_capture: Capture,
                        short_capture: Capture) -> MergeResult:
        long_panel = self.calculate_panel(inspection, long_capture)
        short_panel = self.calculate_panel(inspection, short_capture)
        if long_panel.signal.shape != short_panel.signal.shape:
            raise Issue("Both high-current exposures must share the rectified panel plane.", 409)
        return compose(
            long_panel, short_panel, long_capture, short_capture,
            inspection.merge, cells_for(inspection.plane),
        )

    def calculate_panel(self, inspection: Inspection, capture: Capture) -> PanelImage:
        source = self.read_source(capture)
        calibration = leg_calibration(inspection, capture)
        dark = self.leg_reference(inspection, capture, "dark")
        flat = self.leg_reference(inspection, capture, "flat")
        native = calibrate_sensor(capture, source, dark, flat, calibration.flat_floor)
        placement = inspection.placements.get(capture.id) or default_placement(capture.width, capture.height)
        sampling = SamplingMap.for_plane(inspection.plane, placement, capture.width, capture.height)
        return project_panel(native, sampling)

    def capture_stats(self, inspection: Inspection, identifier: str) -> dict:
        capture = inspection.capture(identifier)
        key = ("native-quality", capture.digest, capture.white_level)
        return self.measurements.get(
            key, lambda: source_quality(self.read_source(capture).pixels, capture.white_level),
        )

    def identity(self, inspection: Inspection, kind: str, context: PanelContext, low: PanelContext | None = None):
        return (
            kind, context.pixel_key, low.pixel_key if low else None,
            data_digest(asdict(inspection.plane)), data_digest(asdict(inspection.analysis)),
        )

    def intensity(self, inspection: Inspection, context: PanelContext):
        identity = self.identity(inspection, "intensity", context)
        return self.measurements.get(identity, lambda: cell_intensity(context.panel, context.cells, inspection.analysis))

    def dark(self, inspection: Inspection, context: PanelContext):
        identity = self.identity(inspection, "dark", context)
        return self.measurements.get(
            identity,
            lambda: dark_area_map(
                context.panel, inspection.plane, context.cells,
                inspection.analysis, self.intensity(inspection, context),
            ),
        )

    def compare(self, inspection: Inspection):
        high = self.panel_context(inspection, "high")
        low = self.panel_context(inspection, "low")
        key = self.identity(inspection, "response", high, low)
        result = self.measurements.get(
            key,
            lambda: response_map(high.panel, low.panel, high.capture, low.capture, high.cells, inspection.analysis),
        )
        return PlaneAnalysis(high, result)

    def measure(self, inspection: Inspection, role: str, kind: str) -> PlaneAnalysis:
        choice(kind, "Measurement", MEASUREMENT_PLANES)
        if kind == "response":
            return self.compare(inspection)
        context = self.panel_context(inspection, role)
        if kind == "merge":
            merge = context.merge
            if merge is None:
                raise Issue("Enable the paired high-current exposures to read the merge partition.", 409)
            return PlaneAnalysis(context, merge)
        if kind == "intensity":
            return PlaneAnalysis(context, self.intensity(inspection, context))
        if kind == "dark":
            return PlaneAnalysis(context, self.dark(inspection, context))
        identity = self.identity(inspection, kind, context)
        result = self.measurements.get(identity, lambda: self.calculate(inspection, kind, context))
        return PlaneAnalysis(context, result)

    def calculate(self, inspection: Inspection, kind: str, context: PanelContext):
        if kind == "calibration":
            report = panel_calibration_report(context.panel)
            if context.merge is not None and not isinstance(report.get("long"), dict):
                report["composed"] = False
                report["pair"] = {
                    "long": context.merge.long_panel.details,
                    "short": context.merge.short_panel.details,
                }
            return report
        if kind == "quality":
            return capture_quality(context.panel, context.capture, context.cells, inspection.analysis)
        if kind == "cracks":
            return crack_evidence(
                context.panel, inspection.plane, context.cells,
                inspection.analysis, self.dark(inspection, context).mask,
            )
        if kind == "merge":
            if context.merge is None:
                raise Issue("Enable the paired high-current exposures first.", 409)
            return context.merge.report
        if kind == "strings":
            return inspect_strings(context.cells, inspection.plane, self.intensity(inspection, context))
        raise Issue("This measurement is unavailable.", 404)

    def table(self, inspection: Inspection, role: str, kind: str) -> dict:
        analysis = self.measure(inspection, role, kind)
        context, result = analysis.context, analysis.data
        report, records = report_and_rows(kind, result)
        if kind == "merge":
            report = {**report, "readings": inspection.merge.readings}
        return {
            "kind": kind,
            "role": context.capture.role,
            "capture": context.capture.id,
            "plane_size": [inspection.plane.width, inspection.plane.height],
            "revision": inspection.revision,
            "summary": report,
            "rows": records,
        }

    def pixel(self, inspection: Inspection, role: str, x_input, y_input, compare: bool) -> dict:
        x = integer(x_input, "Plane x", 0, inspection.plane.width - 1)
        y = integer(y_input, "Plane y", 0, inspection.plane.height - 1)
        context = self.panel_context(inspection, role)
        result = pixel_evidence(inspection, context.panel, x, y)
        reading = reading_summary(inspection, context, x, y)
        if reading is not None:
            result["reading"] = reading
        if compare:
            comparison = self.compare(inspection)
            result["response"] = probe_response(comparison.data, x, y)
        return {
            "capture": context.capture.id, "role": context.capture.role,
            "revision": inspection.revision, **result,
        }

    def merge_report(self, inspection: Inspection) -> dict:
        merge = self.merge(inspection)
        if merge is None:
            return {
                "enabled": False,
                "readings": inspection.merge.readings,
                "available": False,
                "short_id": inspection.merge.short_id,
                "message": "Exposure merge is off. High-current readings use the single selected image.",
            }
        context = self.panel_context(inspection, "high")
        messages = list(merge.report.get("messages", []))
        for name, details in (("Long", merge.long_panel.details), ("Short", merge.short_panel.details)):
            for note in details.get("messages", []):
                messages.append(f"{name} exposure: {note}")
        summary = {**merge.report, "messages": messages}
        return {
            "enabled": True,
            "available": True,
            "readings": inspection.merge.readings,
            "short_id": inspection.merge.short_id,
            "selected_leg": context.capture.id,
            "long": {"id": merge.long_capture.id, "name": merge.long_capture.name,
                     "exposure_ms": merge.long_capture.exposure_ms},
            "short": {"id": merge.short_capture.id, "name": merge.short_capture.name,
                      "exposure_ms": merge.short_capture.exposure_ms},
            "short_calibration": asdict(inspection.merge.short_calibration),
            "summary": summary,
            "rows": merge.records,
        }

    def cell_details(self, inspection: Inspection, role: str, identifier: str):
        row, col = parse_cell(identifier, inspection.plane.rows, inspection.plane.cols)
        context = self.panel_context(inspection, role)
        cell = next(item for item in context.cells if (item.row, item.col) == (row, col))
        intensity = self.intensity(inspection, context).records[row * inspection.plane.cols + col]
        dark = self.dark(inspection, context)
        cracks = self.measure(inspection, role, "cracks").data
        merge = context.merge
        merge_row = (
            next((entry for entry in merge.records if entry["cell"] == cell.id), None)
            if merge is not None else None
        )
        short = reading_switch(inspection)
        result = {
            "capture": context.capture.id,
            "revision": inspection.revision,
            "cell": cell.id,
            "bounds": cell.bounds,
            "interior": cell.interior,
            "string": cell.string,
            "intensity": intensity,
            "dark": [item for item in dark.records if item["cell"] == cell.id],
            "cracks": [item for item in cracks.records if item["cell"] == cell.id],
            "findings": [item.id for item in inspection.findings if item.cell == cell.id],
            "merge": merge_row,
            "reading": short,
        }
        if short is not None:
            try:
                result["single"] = self.single_cell(inspection, cell.id)
            except Issue as error:
                result["single"] = {"error": str(error)}
        return result

    def single_cell(self, inspection: Inspection, cell_id: str) -> dict:
        """The historical single-image reading, ignoring any composed high-current mode."""
        row, col = parse_cell(cell_id, inspection.plane.rows, inspection.plane.cols)
        merge = self.merge(inspection)
        if merge is None:
            raise Issue("The high-current image is already read as a single exposure.", 409)
        context = PanelContext(
            merge.long_capture, merge.long_panel, cells_for(inspection.plane),
            panel_identity(inspection, merge.long_capture),
        )
        intensity = self.intensity(inspection, context)
        return {
            "capture": merge.long_capture.id,
            "name": merge.long_capture.name,
            "exposure_ms": merge.long_capture.exposure_ms,
            "cell": cell_id,
            "intensity": intensity.records[row * inspection.plane.cols + col],
        }

    def polygon(self, inspection: Inspection, role: str, points, finding_id: str | None = None):
        if finding_id:
            finding = get_finding(inspection, finding_id)
            from .findings import finding_status
            if finding_status(inspection, finding) != "current":
                raise Issue("The source has changed since this finding. Open its original review or redraw the polygon.", 409)
            points = finding.points
            role = finding.source_role
        else:
            points = xy_list(points, "Region polygon", 3, 48, (0, 1), (0, 1))
        context = self.panel_context(inspection, role)
        report, mask = measure_polygon(inspection, context.panel, points)
        values = {
            "capture": context.capture.id, "source_role": role,
            "revision": inspection.revision, "points": points, **report,
        }
        return values, mask, context

    def transect(self, inspection: Inspection, role: str, raw: dict):
        transect = Transect.from_dict(raw)
        context = self.panel_context(inspection, role)
        report = analyze_transect(context.panel, inspection.plane, context.cells, transect)
        return {"capture": context.capture.id, "role": role, "revision": inspection.revision, **report}

    def view_key(self, inspection: Inspection, role: str, kind: str, grid: bool, labels: bool):
        source = inspection.selected("high" if kind == "response" else role)
        low_source = inspection.selected("low") if kind == "response" else None
        return (
            "render", kind, panel_identity(inspection, source),
            panel_identity(inspection, low_source) if low_source else None,
            data_digest(asdict(inspection.plane)), data_digest(asdict(inspection.analysis)),
            data_digest(asdict(inspection.display)), grid, labels,
            "merge", data_digest(asdict(inspection.merge)),
            inspection.merge.enabled and inspection.merge.short_id or None,
        )


def reading_summary(inspection: Inspection, context: PanelContext, x: int, y: int) -> dict | None:
    if context.merge is None:
        return None
    return merge_probe(context.merge, x, y, inspection.merge)


def reading_switch(inspection: Inspection) -> dict | None:
    merge = inspection.merge
    if not merge.enabled or merge.short_id is None:
        return None
    return {
        "mode": merge.readings,
        "short_id": merge.short_id,
        "agreement": merge.agreement,
        "options": ["composite", "long", "short"],
    }


def report_and_rows(kind: str, result) -> tuple[dict, list[dict]]:
    if kind == "calibration":
        return result, []
    if kind in ("merge", "response", "dark", "cracks", "strings"):
        return result.report, result.records
    if kind == "intensity":
        return {
            "eligible_cells": sum(value is not None for value in result.medians.values()),
            "median": result.panel_median,
            "cells": len(result.records),
        }, result.records
    return result.report, result.records
