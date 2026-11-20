from dataclasses import asdict
from hashlib import sha256
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw, TiffImagePlugin

from .engine import Engine
from .findings import finding_status
from .images import distribution, png_bytes
from .model import Inspection, Issue, parse_cell
from .packets import PacketWriter
from .render import RenderOptions, cell_crop, draw_tag, font, panel_preview, render_layer
from .store import data_digest, serial_json
from .tables import csv_data


EXPORT_SCHEMA = "celltrace-raster-v1"
SUPPORT_CODING = {
    "outside_source": 0,
    "usable": 1,
    "clipped": 2,
    "invalid_reference": 4,
    "signal_floor": 8,
}
EXPOSURE_REGIME = {
    0: "no support",
    1: "mutually corroborated",
    2: "long exposure only",
    3: "short exposure only",
    4: "no conclusion",
}


def raw_float_tiff(values: np.ndarray, source_description: str) -> bytes:
    target = BytesIO()
    image = Image.fromarray(values.astype(np.float32, copy=False))
    info = TiffImagePlugin.ImageFileDirectory_v2()
    info[270] = source_description[:300]
    info[305] = "CellTrace EL"
    image.save(target, format="TIFF", compression="tiff_deflate", tiffinfo=info)
    return target.getvalue()


def bit_field(panel) -> np.ndarray:
    bitmap = np.zeros(panel.signal.shape, dtype=np.uint8)
    bitmap[panel.valid] |= SUPPORT_CODING["usable"]
    bitmap[panel.clipped] |= SUPPORT_CODING["clipped"]
    bitmap[panel.calibration_gap] |= SUPPORT_CODING["invalid_reference"]
    bitmap[panel.floor_hit & panel.valid] |= SUPPORT_CODING["signal_floor"]
    return bitmap


def merge_field(result) -> np.ndarray:
    """Codes for how the composed high-current plane assembled each pixel."""
    return result.regime.astype(np.uint8)


def observation_record(inspection: Inspection, context, kind: str, bounds=None):
    capture = context.capture
    placement = inspection.placements.get(capture.id)
    calibration = inspection.merge.leg_calibration(inspection, capture)
    pair = None
    if context.merge is not None:
        merge = context.merge
        pair = {
            "settings": asdict(inspection.merge),
            "long": {"capture": merge.long_capture.id, "exposure_ms": merge.long_capture.exposure_ms},
            "short": {"capture": merge.short_capture.id, "exposure_ms": merge.short_capture.exposure_ms},
            "exposure_scale": merge.exposure_scale,
            "reference_ratio": round(merge.reference_ratio, 6),
            "short_floor": round(merge.short_floor, 6),
            "summary": merge.report,
        }
    return {
        "inspection": inspection.id,
        "revision": inspection.revision,
        "module": inspection.module,
        "lot": inspection.lot,
        "kind": kind,
        "role": capture.role,
        "original_capture": capture.id,
        "original_sha256": capture.digest,
        "capture_metadata": asdict(capture),
        "plane": asdict(inspection.plane),
        "bounds": bounds or [0, 0, inspection.plane.width, inspection.plane.height],
        "configured_placement": asdict(placement) if placement else None,
        "plane_to_sensor": context.panel.matrix.tolist(),
        "calibration": context.panel.details,
        "calibration_binding": asdict(calibration),
        "exposure_pair": pair,
        "measurement_settings": asdict(inspection.analysis),
        "display_settings": asdict(inspection.display),
        "support_bits": SUPPORT_CODING,
        "exposure_regime_bits": EXPOSURE_REGIME if pair else None,
        "values": "32-bit floating normalized and calibrated sensor codes. NaN samples are unavailable.",
        "coordinates": "Plane origin is the upper left edge. Pixel centers use half-integer coordinates.",
    }


def float_plane(engine: Engine, inspection: Inspection, role: str) -> bytes:
    context = engine.panel_context(inspection, role)
    panel = context.panel
    record = observation_record(inspection, context, "calibrated plane")
    preview = render_layer(inspection, engine, role, "panel", RenderOptions(True, False))
    tiff = raw_float_tiff(panel.signal, f"Normalized EL signal in {context.capture.role} capture")
    bitmap = png_bytes(Image.fromarray(bit_field(panel), mode="L"))
    report = {
        "coverage": int(panel.any_support.sum()),
        "distribution": distribution(panel.signal[panel.valid]),
        "pixels": int(panel.signal.size),
    }
    with PacketWriter() as packet:
        packet.add("plane/signal.tiff", tiff, "image/tiff")
        packet.add("plane/support.png", bitmap, "image/png")
        packet.add("plane/preview.png", preview, "image/png")
        report_json = report
        if context.merge is not None:
            packet.add("plane/exposure-regime.png",
                       png_bytes(Image.fromarray(merge_field(context.merge), mode="L")), "image/png")
            report_json = {**report, "exposure_regime": context.merge.report}
        packet.add("plane/measurement.json", serial_json(report_json), "application/json")
        return packet.finish({
            "schema": EXPORT_SCHEMA, "geometry": record, "plane_sha256": sha256(tiff).hexdigest(),
        })


def chosen_cells(context, inspection: Inspection, requested) -> list:
    if not requested:
        return context.cells
    selected = set()
    for identifier in requested:
        selected.add(parse_cell(identifier, inspection.plane.rows, inspection.plane.cols))
    return [cell for cell in context.cells if (cell.row, cell.col) in selected]


def write_cell_bundle(engine: Engine, inspection: Inspection, role: str, requested: list[str]):
    context = engine.panel_context(inspection, role)
    picked = chosen_cells(context, inspection, requested)
    if len(picked) > 18:
        raise Issue("Choose at most 18 cell evidence crops per package.")
    payload = observation_record(inspection, context, "individual cell views")
    observation = {}
    intensity = engine.table(inspection, role, "intensity")
    dark = engine.table(inspection, role, "dark")
    cracks = engine.table(inspection, role, "cracks")
    grouped = {
        key: {cell.id: [] for cell in picked}
        for key in ("dark", "cracks")
    }
    for kind, layer in (("dark", dark), ("cracks", cracks)):
        for record in layer["rows"]:
            if record["cell"] in grouped[kind]:
                grouped[kind][record["cell"]].append(record)
    intensity_rows = {row["cell"]: row for row in intensity["rows"]}
    relevant = []
    with PacketWriter() as packet:
        for cell in picked:
            view = png_bytes(cell_crop(inspection, engine, role, cell, "panel"))
            path = f"cells/{cell.id}.png"
            packet.add(path, view, "image/png")
            observation[cell.id] = {
                "view": path, "bounds": cell.bounds, "measurement_inset": cell.interior,
                "string": cell.string,
                "measurements": intensity_rows[cell.id],
                "exposure": (
                    next((row for row in context.merge.records if row["cell"] == cell.id), None)
                    if context.merge is not None else None
                ),
                "dark": grouped["dark"][cell.id],
                "cracks": grouped["cracks"][cell.id],
                "findings": [
                    row.id for row in inspection.findings
                    if row.cell == cell.id and row.source_role == role
                    and finding_status(inspection, row) == "current"
                ],
            }
            relevant.append(intensity_rows[cell.id])
        packet.add("cells/intensity.csv", csv_data("intensity", relevant), "text/csv")
        packet.add("cells/measurements.json", serial_json(observation), "application/json")
        return packet.finish({"schema": EXPORT_SCHEMA, "geometry": payload, "cells": list(observation)})


def contact_layout(number: int, columns: int, tile: tuple[int, int]) -> tuple[int, int]:
    width = columns * tile[0] + 24
    height = ((number + columns - 1) // columns) * tile[1] + 64
    return width, height


def grid_contact_sheet(engine: Engine, inspection: Inspection, role: str, requested: list[str]) -> bytes:
    context = engine.panel_context(inspection, role)
    chosen = chosen_cells(context, inspection, requested)
    if len(chosen) > 70:
        raise Issue("Choose at most 70 cells in a contact sheet.")
    canvas_shape = contact_layout(len(chosen), min(6, len(chosen)), (165, 146))
    image = Image.new("RGB", canvas_shape, (19, 30, 36))
    painter = ImageDraw.Draw(image, "RGBA")
    header = f"{inspection.module[:50]} / {role} current cells"
    painter.text((16, 14), header, font=font(19), fill=(218, 236, 229))
    measurements = {
        row["cell"]: row for row in engine.table(inspection, role, "intensity")["rows"]
    }
    context_preview = panel_preview(context.panel, inspection.display)
    for index, cell in enumerate(chosen):
        tile = cell_tile(context_preview, cell, measurements[cell.id])
        xpos = 12 + index % min(6, len(chosen)) * 165
        ypos = 46 + index // min(6, len(chosen)) * 146
        image.paste(tile, (xpos, ypos))
    return png_bytes(image)


def cell_tile(base: Image.Image, cell, stats: dict) -> Image.Image:
    tile = Image.new("RGB", (158, 138), (27, 44, 48))
    cropped = base.crop(cell.bounds)
    cropped.thumbnail((145, 106), Image.Resampling.LANCZOS)
    left = (tile.width - cropped.width) // 2
    top = 5 + (107 - cropped.height) // 2
    tile.paste(cropped, (left, top))
    painter = ImageDraw.Draw(tile, "RGBA")
    draw_tag(painter, (8, 6), cell.id, (180, 233, 208), font(10))
    support = stats["support"]
    median = "unavailable" if stats["median"] is None else f'{stats["median"]:.3f}'
    painter.text((7, 116), f"{support:.0%} support / {median}", fill=(186, 210, 207), font=font(11))
    return tile
