from dataclasses import dataclass
from math import ceil, floor

import numpy as np
from PIL import Image, ImageDraw

from .engine import Engine
from .geometry import Cell, cells_for
from .images import grayscale_preview, png_bytes
from .model import Inspection, Issue, real, xy_list
from .render import draw_tag, font
from .store import data_digest


PAIR_VIEW_SCHEMA = "celltrace-paired-views-v1"


@dataclass(frozen=True)
class PairWindow:
    bounds: tuple[int, int, int, int]
    center: tuple[float, float]
    size_x: int
    size_y: int


def region_from_body(inspection: Inspection, body: dict) -> PairWindow:
    width, height = inspection.plane.width, inspection.plane.height
    a, b = xy_list(body.get("corners"), "Paired review bounds", 2, 2, (0, 1), (0, 1))
    x0 = max(0, floor(min(a[0], b[0]) * width))
    y0 = max(0, floor(min(a[1], b[1]) * height))
    x1 = min(width, ceil(max(a[0], b[0]) * width))
    y1 = min(height, ceil(max(a[1], b[1]) * height))
    if min(x1 - x0, y1 - y0) < 12:
        raise Issue("The paired review window needs a minimum size of 12 pixels per side.")
    return PairWindow((x0, y0, x1, y1), ((x0 + x1) / 2, (y0 + y1) / 2), x1 - x0, y1 - y0)


def full_window(inspection: Inspection):
    return region_from_body(inspection, {"corners": ((0, 0), (1, 1))})


def window_for_cell(inspection: Inspection, cell: Cell):
    x0, y0, x1, y1 = cell.bounds
    return PairWindow(cell.bounds, ((x0 + x1) / 2, (y0 + y1) / 2), x1 - x0, y1 - y0)


def common_contrast(request, high, low, high_valid, low_valid):
    if request == "shared":
        use_high = high[high_valid]
        use_low = low[low_valid]
        samples = use_high[::max(1, len(use_high) // 45000)], use_low[::max(1, len(use_low) // 45000)]
        if sum(sample.size for sample in samples) == 0:
            return (0, 1), (0, 1)
        joint = np.concatenate(samples)
        p10, p99 = np.percentile(joint, (2, 99))
        levels = float(p10), max(float(p10) + 0.008, float(p99))
        return levels, levels
    return histogram_levels(high, high_valid), histogram_levels(low, low_valid)


def histogram_levels(signal: np.ndarray, valid: np.ndarray):
    usable = signal[valid]
    if usable.size < 20:
        return (0.0, 1.0)
    first, last = np.percentile(usable, (2, 99))
    return float(first), max(float(first) + 0.008, float(last))


def views(engine: Engine, inspection: Inspection, window: PairWindow, contrast: str, gamma: float):
    high = engine.panel_context(inspection, "high")
    low = engine.panel_context(inspection, "low")
    xs, ys, xe, ye = window.bounds
    region = np.s_[ys:ye, xs:xe]
    h_signal, l_signal = high.panel.signal[region], low.panel.signal[region]
    h_valid, l_valid = high.panel.valid[region], low.panel.valid[region]
    levels = common_contrast(contrast, h_signal, l_signal, h_valid, l_valid)
    first = grayscale_preview(h_signal, h_valid, *levels[0], gamma)
    second = grayscale_preview(l_signal, l_valid, *levels[1], gamma)
    meta = {
        "schema": PAIR_VIEW_SCHEMA,
        "plane": [inspection.plane.width, inspection.plane.height],
        "bounds": window.bounds,
        "revision": inspection.revision,
        "inspection": inspection.id,
        "high": describe(high, levels[0], h_valid),
        "low": describe(low, levels[1], l_valid),
        "paired_pixels": int((h_valid & l_valid).sum()),
        "high_only": int((h_valid & ~l_valid).sum()),
        "low_only": int((l_valid & ~h_valid).sum()),
        "missing_both": int((~h_valid & ~l_valid).sum()),
        "contrast": contrast,
        "gamma": gamma,
        "window": (window.size_x, window.size_y),
        "signature": data_digest({"low": low.pixel_key, "high": high.pixel_key, "bounds": window.bounds}),
    }
    return first, second, meta


def describe(context, levels, valid):
    capture = context.capture
    return {
        "id": capture.id,
        "original": capture.name, "digest": capture.digest,
        "drive_a": capture.current_a, "ms": capture.exposure_ms,
        "display_limits": [round(float(value), 5) for value in levels],
        "usable": int(valid.sum()),
    }


def pair_result(engine: Engine, inspection: Inspection, window: PairWindow, fmt: str, mode: str,
                contrast: str, gamma_raw, wipe_raw, tile_raw, grid: bool):
    gamma = real(gamma_raw, "Paired display gamma", 0.2, 3.0)
    wipe = real(wipe_raw, "Divider", 0, 1)
    fraction = real(tile_raw, "Checker width", 0.045, 0.5)
    key = (
        "pair-review-view", engine.view_key(inspection, "high", "response", grid, False),
        inspection.id, inspection.revision, window.bounds, fmt, mode, contrast, gamma, wipe, fraction, grid,
    )
    return engine.materials.get(key, lambda: draw_result(
        engine, inspection, window, fmt, mode, contrast, gamma, wipe, fraction, grid
    ))


def draw_result(engine, inspection, window, fmt, mode, contrast, gamma, wipe, fraction, grid):
    high, low, metadata = views(engine, inspection, window, contrast, gamma)
    if fmt == "data":
        return metadata
    visible_cells = list(crossed_cells(cells_for(inspection.plane), window)) if grid else []
    if mode == "side":
        output = side_by_side(high, low, metadata, visible_cells, window)
    elif mode == "checker":
        output = toggled_view(high, low, metadata, visible_cells, window, fraction)
    else:
        output = wipe_view(high, low, metadata, visible_cells, window, wipe)
    return png_bytes(output)


def crossed_cells(cells: list[Cell], window: PairWindow):
    x0, y0, x1, y1 = window.bounds
    return (
        cell for cell in cells
        if cell.bounds[0] < x1 and cell.bounds[2] > x0 and cell.bounds[1] < y1 and cell.bounds[3] > y0
    )


def fit_display(image: Image.Image, limit: tuple[int, int]):
    texture = image.copy()
    texture.thumbnail(limit, Image.Resampling.LANCZOS)
    return texture


def overlay_grid(image, window, cells):
    if not cells:
        return
    ix0, iy0, _, _ = window.bounds
    sx = image.width / window.size_x
    sy = image.height / window.size_y
    drawer = ImageDraw.Draw(image, "RGBA")
    for cell in cells:
        x0, y0, x1, y1 = cell.bounds
        bounds = ((x0 - ix0) * sx, (y0 - iy0) * sy, (x1 - ix0) * sx, (y1 - iy0) * sy)
        drawer.rectangle(bounds, outline=(115, 216, 193, 131))
        if bounds[2] - bounds[0] >= 52 and bounds[3] - bounds[1] >= 33 and bounds[0] >= 0 and bounds[1] >= 0:
            draw_tag(drawer, (bounds[0] + 4, bounds[1] + 4), cell.id, (167, 239, 207), font(10))


def wrap_pair(image: Image.Image, metadata: dict, mode: str) -> Image.Image:
    canvas = Image.new("RGB", (max(274, image.width + 28), image.height + 74), (20, 33, 40))
    drawer = ImageDraw.Draw(canvas, "RGBA")
    canvas.paste(image, ((canvas.width - image.width) // 2, 45))
    drawer.text((14, 13), mode, font=font(18), fill=(224, 237, 230))
    data = f"{metadata['paired_pixels']:,} paired px / {metadata['contrast']} display scale"
    drawer.text((14, canvas.height - 23), data, font=font(11), fill=(171, 200, 195))
    return canvas


def side_by_side(high, low, metadata, cells, window):
    first, second = fit_display(high, (702, 710)), fit_display(low, (702, 710))
    gap = 6
    frame = Image.new("RGB", (first.width + second.width + gap, first.height), (49, 64, 70))
    overlay_grid(first, window, cells)
    overlay_grid(second, window, cells)
    frame.paste(first, (0, 0))
    frame.paste(second, (first.width + gap, 0))
    drawer = ImageDraw.Draw(frame, "RGBA")
    draw_tag(drawer, (8, 5), role_tag("High", metadata["high"]), (162, 225, 203), font(11))
    draw_tag(drawer, (first.width + gap + 8, 5), role_tag("Low", metadata["low"]), (179, 210, 242), font(11))
    return wrap_pair(frame, metadata, "Paired current views")


def role_tag(title: str, source: dict):
    if source["drive_a"] is None:
        return f"{title} / {source['ms']:g} ms"
    return f"{title} / {source['drive_a']:g} A / {source['ms']:g} ms"


def wipe_view(high, low, metadata, cells, window, wipe):
    blend = fit_display(high, (1240, 740))
    second = fit_display(low, (1240, 740))
    dividing = int(round(blend.width * wipe))
    if dividing < blend.width:
        blend.paste(second.crop((dividing, 0, blend.width, blend.height)), (dividing, 0))
    overlay_grid(blend, window, cells)
    painter = ImageDraw.Draw(blend, "RGBA")
    if 0 < dividing < blend.width:
        painter.line((dividing, 0, dividing, blend.height), fill=(220, 224, 160, 255), width=2)
    if dividing >= 70:
        draw_tag(painter, (8, 6), "High side", (140, 230, 199), font(11))
    if blend.width - dividing >= 70:
        draw_tag(painter, (dividing + 8, 6), "Low side", (167, 204, 243), font(11))
    return wrap_pair(blend, metadata, "Paired current divider")


def toggled_view(high, low, metadata, cells, window, fraction):
    width, height = high.size
    pitch = max(4, int(min(width, height) * fraction))
    y, x = np.ogrid[:height, :width]
    alternating = ((x // pitch + y // pitch) % 2).astype(bool)
    merged = np.asarray(high).copy()
    merged[alternating] = np.asarray(low)[alternating]
    output = fit_display(Image.fromarray(merged, "RGB"), (1240, 740))
    overlay_grid(output, window, cells)
    painter = ImageDraw.Draw(output, "RGBA")
    draw_tag(painter, (8, 6), "Top left / high current", (150, 236, 205), font(11))
    return wrap_pair(output, metadata, "Paired current checkerboard")


CORRESPONDENCE_SIZE = 168


def point_window(inspection: Inspection, x: int, y: int, size: int = CORRESPONDENCE_SIZE) -> PairWindow:
    width, height = inspection.plane.width, inspection.plane.height
    half = max(24, size // 2)
    x0 = max(0, min(width - 2 * half, x - half))
    y0 = max(0, min(height - 2 * half, y - half))
    x1 = min(width, x0 + 2 * half)
    y1 = min(height, y0 + 2 * half)
    return PairWindow((x0, y0, x1, y1), ((x0 + x1) / 2, (y0 + y1) / 2), x1 - x0, y1 - y0)


def correspondence_result(engine: Engine, inspection: Inspection, x_input, y_input, contrast: str, gamma_raw):
    from .model import integer
    from .response import probe_response
    gamma = real(gamma_raw, "Paired display gamma", 0.2, 3.0)
    x = integer(x_input, "Plane x", 0, inspection.plane.width - 1)
    y = integer(y_input, "Plane y", 0, inspection.plane.height - 1)
    analysis = engine.compare(inspection).data
    point = probe_response(analysis, x, y)
    window = point_window(inspection, x, y)
    key = (
        "correspondence", engine.view_key(inspection, "high", "response", False, False),
        inspection.id, inspection.revision, (x, y), contrast, gamma,
    )
    return engine.materials.get(key, lambda: draw_correspondence(
        engine, inspection, analysis, point, window, x, y, contrast, gamma
    ))


def draw_correspondence(engine, inspection, analysis, point, window, x, y, contrast, gamma):
    high, low, metadata = views(engine, inspection, window, contrast, gamma)
    high = fit_display(high, (CORRESPONDENCE_SIZE, CORRESPONDENCE_SIZE))
    low = fit_display(low, (CORRESPONDENCE_SIZE, CORRESPONDENCE_SIZE))
    x0, y0, x1, y1 = window.bounds
    scale_x = high.width / (x1 - x0)
    scale_y = high.height / (y1 - y0)
    status = analysis.alignment.status_at(x, y)
    tile = analysis.alignment.tile_at(x, y) or {}
    measured = analysis.eligible[y, x]
    color = (248, 208, 113) if measured else (235, 160, 128)
    canvas = Image.new("RGB", (high.width + low.width + 34, max(high.height, low.height) + 66), (20, 33, 40))
    painter = ImageDraw.Draw(canvas, "RGBA")
    canvas.paste(high, (12, 38))
    canvas.paste(low, (high.width + 22, 38))
    for image, offset_x in ((high, 12), (low, high.width + 22)):
        draw_tag(painter, (offset_x + 6, 40), "high" if offset_x == 12 else "low", (176, 220, 205), font(10))
    draw_tag(painter, (12, 8), "circle: compared pixel / ring: low-current match", (176, 220, 205), font(9))
    high_point = ((x - x0 + 0.5) * scale_x + 12, (y - y0 + 0.5) * scale_y + 38)
    circle(painter, high_point, color)
    if point["low_point"] is not None:
        low_point = (
            (point["low_point"][0] - x0 + 0.5) * scale_x + high.width + 22,
            (point["low_point"][1] - y0 + 0.5) * scale_y + 38,
        )
        circle(painter, low_point, color)
    if point["match_point"] is not None:
        match_point = (
            (point["match_point"][0] - x0 + 0.5) * scale_x + high.width + 22,
            (point["match_point"][1] - y0 + 0.5) * scale_y + 38,
        )
        ring(painter, match_point, (144, 196, 238))
    shift = point["local_shift"]
    caption = (
        f"Local shift {shift[0]:+.2f}, {shift[1]:+.2f} px" if shift is not None
        else "No local shift available"
    )
    painter.text((12, canvas.height - 22), f"{status} / {caption}", fill=(173, 203, 197), font=font(10))
    if tile.get("reason"):
        painter.text((12, canvas.height - 35), str(tile["reason"])[:64], fill=(230, 178, 136), font=font(9))
    return png_bytes(canvas)


def ring(painter, point, color):
    x, y = point
    painter.ellipse((x - 5, y - 5, x + 5, y + 5), outline=(*color, 245), width=2)


def circle(painter, point, color):
    x, y = point
    painter.ellipse((x - 9, y - 9, x + 9, y + 9), outline=(*color, 235), width=2)
    painter.line((x - 14, y, x - 4, y), fill=(*color, 200), width=1)
    painter.line((x + 4, y, x + 14, y), fill=(*color, 200), width=1)
