from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from .calibration import PanelImage
from .engine import EL_LAYERS, MEASUREMENT_PLANES, Engine
from .findings import finding_status
from .geometry import Cell, plane_polygon
from .images import (
    compact_thumbnail, continuous_palette, grayscale_preview, mixed_overlay,
    png_bytes, set_invalid_pattern,
)
from .model import Capture, Inspection, ViewSettings, choice


DIVERGING = ((161, 56, 123), (214, 99, 79), (246, 202, 120), (59, 175, 174), (32, 87, 158))
PLASMA = ((27, 28, 57), (70, 65, 151), (24, 132, 167), (52, 192, 167), (228, 217, 120))
QUALITY_COLORS = {
    "clipped": (240, 89, 141),
    "invalid calibration": (220, 155, 73),
    "outside source": (94, 75, 130),
    "near floor": (93, 171, 213),
}
FINDING_COLORS = {
    "dark": (250, 158, 95), "crack": (243, 98, 138), "suspect": (240, 201, 115),
    "capture": (135, 202, 255), "general": (123, 223, 188),
}
SUPPORT_COLORS = {
    "usable": (33, 108, 102),
    "outside": (38, 37, 64),
    "reference_gap": (191, 129, 58),
    "clipped": (184, 59, 109),
    "floor": (56, 100, 157),
}


@dataclass(frozen=True)
class RenderOptions:
    grid: bool = False
    labels: bool = False
    findings: bool = False
    highlight: str | None = None


def font(size=12):
    system_face = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    try:
        return ImageFont.truetype(str(system_face), size)
    except OSError:
        return ImageFont.load_default(size=size)


def panel_preview(panel: PanelImage, display: ViewSettings) -> Image.Image:
    return grayscale_preview(panel.signal, panel.valid, display.lower, display.upper, display.gamma)


def source_preview(engine: Engine, capture: Capture, display: ViewSettings | None = None,
                   *, long_edge: int | None = None, highlight: bool = False) -> Image.Image:
    source = engine.read_source(capture)
    signal = source.pixels.astype(np.float32) / capture.white_level
    if display is None:
        lower, upper = 0.0, float(np.percentile(signal, 99))
        display = ViewSettings(lower, max(upper, 0.03), 1.0)
    valid = np.ones(signal.shape, dtype=bool)
    rendered = grayscale_preview(signal, valid, display.lower, display.upper, display.gamma)
    if highlight:
        rendered = mixed_overlay(rendered, source.pixels >= capture.white_level, QUALITY_COLORS["clipped"], 0.75)
    return compact_thumbnail(rendered, long_edge) if long_edge else rendered


def support_rgb(panel: PanelImage) -> Image.Image:
    raster = np.empty((*panel.signal.shape, 3), dtype=np.uint8)
    raster[:] = SUPPORT_COLORS["outside"]
    raster[panel.valid] = SUPPORT_COLORS["usable"]
    raster[panel.floor_hit & panel.valid] = SUPPORT_COLORS["floor"]
    raster[panel.calibration_gap] = SUPPORT_COLORS["reference_gap"]
    raster[panel.clipped] = SUPPORT_COLORS["clipped"]
    return Image.fromarray(raster, "RGB")


def calibration_rgb(panel: PanelImage, display: ViewSettings) -> Image.Image:
    base = panel_preview(panel, display)
    base = mixed_overlay(base, panel.calibration_gap, QUALITY_COLORS["invalid calibration"], 0.72)
    return mixed_overlay(base, panel.floor_hit & panel.valid, SUPPORT_COLORS["floor"], 0.68)


def quality_rgb(panel: PanelImage, display: ViewSettings, masks: dict) -> Image.Image:
    base = panel_preview(panel, display)
    for label, mask in masks.items():
        base = mixed_overlay(base, mask, QUALITY_COLORS[label], 0.78)
    return base


def scalar_rgb(values: np.ndarray, eligible: np.ndarray, limits, palette) -> Image.Image:
    raster = continuous_palette(np.nan_to_num(values, nan=limits[0]), limits, palette)
    set_invalid_pattern(raster, eligible)
    return Image.fromarray(raster, "RGB")


def color_edges(image: Image.Image, mask: np.ndarray, color: tuple, alpha=0.82) -> Image.Image:
    edge = ndimage.binary_dilation(mask) ^ ndimage.binary_erosion(mask)
    return mixed_overlay(image, edge, color, alpha)


def render_base(inspection: Inspection, engine: Engine, role: str, kind: str):
    context = engine.panel_context(inspection, role) if kind != "response" else None
    if kind in MEASUREMENT_PLANES:
        result = engine.measure(inspection, role, kind)
        context, data = result.context, result.data
    display = inspection.display
    panel = context.panel
    if kind in ("panel", "grid"):
        image = panel_preview(panel, display)
    elif kind == "support":
        image = support_rgb(panel)
    elif kind == "calibration":
        image = calibration_rgb(panel, display)
    elif kind == "quality":
        image = quality_rgb(panel, display, data.layers)
    elif kind == "intensity":
        image = scalar_rgb(data.heat, data.comparable, (0.12, 0.65), PLASMA)
    elif kind == "dark":
        image = mixed_overlay(panel_preview(panel, display), data.mask, (251, 150, 75), 0.56)
        image = color_edges(image, data.mask, (252, 193, 117), 0.76)
        draw_measurements(image, data.records, (252, 193, 117), 24)
    elif kind == "cracks":
        image = mixed_overlay(panel_preview(panel, display), data.busbars, (98, 158, 197), 0.22)
        image = mixed_overlay(image, data.evidence, (239, 103, 133), 0.52)
        image = mixed_overlay(image, ndimage.binary_dilation(data.skeleton) & ~data.busbars, (255, 122, 151), 0.92)
        draw_endpoints(image, data.records)
    elif kind == "response":
        image = scalar_rgb(data.relative, data.eligible, (0.34, 1.32), DIVERGING)
        image = mixed_overlay(image, data.alignment.recapture, SUPPORT_COLORS["outside"], 0.55)
        image = color_edges(image, data.weak, (247, 204, 106), 0.63)
        draw_recapture_cells(image, data, context.cells)
    else:
        image = scalar_rgb(data.heat, data.support, (0.5, 1.3), DIVERGING)
    return image, context


def render_layer(inspection: Inspection, engine: Engine, role: str, kind: str, options: RenderOptions) -> bytes:
    choice(kind, "Image layer", EL_LAYERS)
    key = engine.view_key(inspection, role, kind, options.grid, options.labels)
    if options.findings or options.highlight:
        from .store import data_digest
        key += (
            "annotations", inspection.id, inspection.revision, data_digest(inspection.to_dict()),
            options.findings, options.highlight,
        )
    return engine.materials.get(key, lambda: make_layer(inspection, engine, role, kind, options))


def make_layer(inspection: Inspection, engine: Engine, role: str, kind: str, options: RenderOptions) -> bytes:
    image, context = render_base(inspection, engine, role, kind)
    if options.grid or kind == "grid" or options.highlight:
        decorate_grid(image, context.cells, labels=options.labels or kind == "grid", selected=options.highlight)
    if options.findings:
        decorate_findings(image, inspection, context.capture.role)
    return png_bytes(image)


def decorate_grid(image: Image.Image, cells: list[Cell], *, labels: bool, selected: str | None = None):
    painter = ImageDraw.Draw(image, "RGBA")
    face = font(11)
    for cell in cells:
        x0, y0, x1, y1 = cell.bounds
        painter.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(108, 173, 171, 119))
        if cell.id == selected:
            ix0, iy0, ix1, iy1 = cell.interior
            painter.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(253, 207, 107, 245), width=2)
            painter.rectangle((ix0, iy0, ix1 - 1, iy1 - 1), outline=(253, 207, 107, 130), width=1)
        if labels and x1 - x0 >= 38 and y1 - y0 >= 20:
            draw_tag(painter, (x0 + 4, y0 + 4), cell.id, (148, 233, 214), face)


def draw_tag(painter, pos, title, color, face=None):
    face = face or font(12)
    x, y = pos
    x1, y1, x2, y2 = painter.textbbox((x, y), title, font=face)
    painter.rounded_rectangle((x1 - 3, y1 - 2, x2 + 3, y2 + 2), radius=2, fill=(17, 24, 36, 213))
    painter.text((x, y), title, fill=(*color, 255), font=face)


def draw_measurements(image: Image.Image, records: list[dict], color: tuple, limit: int):
    painter = ImageDraw.Draw(image, "RGBA")
    face = font(10)
    for row in records[:limit]:
        x0, y0, x1, y1 = row["bounds"]
        radius = 2
        painter.rounded_rectangle((x0, y0, x1 - 1, y1 - 1), radius=radius,
                                  outline=(*color, 167), width=1)
        if x1 - x0 >= 25 and y1 - y0 >= 20:
            draw_tag(painter, (x0 + 3, y0 + 2), row["id"].split("-")[-1], color, face)


def draw_recapture_cells(image: Image.Image, data, cells: list[Cell]):
    painter = ImageDraw.Draw(image, "RGBA")
    face = font(9)
    for cell in cells:
        row = data.alignment.cells[cell.id]
        if row["state"] not in ("recapture", "partial"):
            continue
        x0, y0, x1, y1 = cell.bounds
        painter.rectangle((x0 + 1, y0 + 1, x1 - 2, y1 - 2), outline=(238, 162, 121, 214), width=1)
        if x1 - x0 >= 54 and y1 - y0 >= 18:
            draw_tag(painter, (x0 + 4, y1 - 15), "recapture" if row["state"] == "recapture" else "part",
                     (238, 162, 121), face)


def draw_endpoints(image: Image.Image, records: list[dict]):
    painter = ImageDraw.Draw(image, "RGBA")
    for row in records[:60]:
        for x, y in row["endpoints"]:
            painter.ellipse((x - 2, y - 2, x + 2, y + 2), outline=(255, 229, 161, 228))
        for x, y in row["junctions"]:
            painter.rectangle((x - 2, y - 2, x + 2, y + 2), outline=(150, 234, 211, 232))


def decorate_findings(image: Image.Image, inspection: Inspection, role: str):
    painter = ImageDraw.Draw(image, "RGBA")
    face = font(11)
    for index, finding in enumerate(inspection.findings, start=1):
        if finding.source_role != role:
            continue
        if finding_status(inspection, finding) != "current":
            continue
        points = [tuple(point) for point in plane_polygon(finding.points, inspection.plane)]
        color = FINDING_COLORS[finding.category]
        painter.polygon(points, fill=(*color, 43))
        painter.line(points + [points[0]], fill=(*color, 215), width=2, joint="curve")
        draw_tag(painter, (max(2, points[0][0]), max(0, points[0][1] - 14)), str(index), color, face)


def polygon_preview(context, mask: np.ndarray, display: ViewSettings) -> Image.Image:
    image = panel_preview(context.panel, display)
    image = mixed_overlay(image, mask, (115, 219, 180), 0.35)
    return color_edges(image, mask, (140, 249, 203), 0.9)


def cell_crop(inspection: Inspection, engine: Engine, role: str, cell: Cell, layer: str) -> Image.Image:
    image, _ = render_base(inspection, engine, role, layer)
    x0, y0, x1, y1 = cell.bounds
    padding = max(3, min(28, int(0.13 * min(x1 - x0, y1 - y0))))
    crop = (
        max(0, x0 - padding), max(0, y0 - padding),
        min(inspection.plane.width, x1 + padding), min(inspection.plane.height, y1 + padding),
    )
    image = image.crop(crop)
    painter = ImageDraw.Draw(image, "RGBA")
    a, b, c, d = cell.interior
    painter.rectangle((a - crop[0], b - crop[1], c - crop[0] - 1, d - crop[1] - 1),
                      outline=(225, 198, 133, 225), width=1)
    draw_tag(painter, (6, 6), cell.id, (225, 198, 133))
    return image


def placement_preview(engine: Engine, inspection: Inspection, capture: Capture) -> Image.Image:
    base = source_preview(engine, capture, long_edge=1000, highlight=True)
    placement = inspection.placements.get(capture.id)
    if placement is None:
        return base
    scale = base.width / capture.width, base.height / capture.height
    points = [(x * scale[0], y * scale[1]) for x, y in placement.corners]
    painter = ImageDraw.Draw(base, "RGBA")
    painter.line(points + [points[0]], fill=(119, 247, 209, 222), width=2, joint="curve")
    for index, (x, y) in enumerate(points, 1):
        painter.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(119, 247, 209, 235))
        draw_tag(painter, (x + 8, y + 2), str(index), (119, 247, 209))
    return base
