from dataclasses import asdict, replace
from hashlib import sha256
from math import ceil, floor

import numpy as np
from PIL import ImageDraw

from .engine import Engine
from .findings import get_finding, measure_polygon, region_source_link, validate_source_anchor
from .geometry import plane_polygon, project, transform_for
from .images import compact_thumbnail, png_bytes
from .model import Calibration, ExposureMerge, Inspection, Issue, Placement, Plane
from .packets import PacketWriter
from .render import FINDING_COLORS, draw_tag, font, panel_preview, polygon_preview, source_preview
from .store import Store, data_digest, serial_json


EVIDENCE_SCHEMA = "celltrace-local-image-evidence-v1"


def historical_inspection(inspection: Inspection, finding) -> Inspection:
    validate_source_anchor(inspection, finding)
    anchor = finding.anchor
    source = inspection.capture(anchor["image"])
    plane = Plane.from_dict(anchor["plane"])
    placement = anchor.get("placement")
    placements = {}
    if placement:
        checked = Placement.from_dict(placement, source.width, source.height)
        transform_for(plane, checked)
        placements[source.id] = checked
    pair = anchor.get("exposure_pair")
    if pair:
        merge = ExposureMerge(
            enabled=True, short_id=pair["short"], short_calibration=pair["short_calibration"],
            agreement=pair["agreement"],
            minimum_agreement_pixels=pair["minimum_agreement_pixels"],
            readings=pair.get("readings", "composite"),
        )
    else:
        merge = ExposureMerge(enabled=False)
    return replace(
        inspection, plane=plane, merge=merge,
        active={finding.source_role: source.id},
        calibration=Calibration.from_dict(anchor["calibration"]), placements=placements,
    )


def clip_box(points: np.ndarray, width: int, height: int, relative_padding=0.15):
    xlo, ylo = np.min(points, axis=0)
    xhi, yhi = np.max(points, axis=0)
    dx = max(6, ceil((xhi - xlo) * relative_padding))
    dy = max(6, ceil((yhi - ylo) * relative_padding))
    x0 = max(0, floor(xlo) - dx)
    y0 = max(0, floor(ylo) - dy)
    x1 = min(width, ceil(xhi) + dx)
    y1 = min(height, ceil(yhi) + dy)
    if x1 <= x0 or y1 <= y0:
        raise Issue("The evidence polygon falls outside its source image.")
    return x0, y0, x1, y1


def finding_evidence(inspection: Inspection, engine: Engine, finding_id: str) -> dict:
    finding = get_finding(inspection, finding_id)
    snapshot = historical_inspection(inspection, finding)
    context = engine.panel_context(snapshot, finding.source_role)
    polygon = plane_polygon(finding.points, snapshot.plane)
    ix, iy = project(context.panel.matrix, polygon[:, 0], polygon[:, 1])
    source_points = np.column_stack((ix, iy))
    report, mask = measure_polygon(snapshot, context.panel, finding.points)
    panel_bounds = clip_box(polygon, snapshot.plane.width, snapshot.plane.height)
    image = polygon_preview(context, mask, snapshot.display)
    original_crop, original_box = native_crop(
        engine, context.capture, source_points, FINDING_COLORS[finding.category], finding.title
    )
    return {
        "finding": asdict(finding),
        "lineage": region_source_link(inspection, finding),
        "plane": asdict(snapshot.plane),
        "capture": {
            "id": context.capture.id, "digest": context.capture.digest,
            "name": context.capture.name, "size": [context.capture.width, context.capture.height],
            "white_level": context.capture.white_level, "exposure_ms": context.capture.exposure_ms,
            "bit_depth": context.capture.bit_depth, "synthetic": context.capture.synthetic,
        },
        "image_points": np.round(source_points, 4).tolist(),
        "source_box": original_box,
        "plane_box": panel_bounds,
        "settings_fingerprint": data_digest(finding.anchor),
        "measurement": report,
        "rendered": image.crop(panel_bounds),
        "native": original_crop,
    }


def native_crop(engine, capture, source_points, color, title):
    try:
        box = clip_box(source_points, capture.width, capture.height)
    except Issue:
        return None, None
    preview = source_preview(engine, capture, highlight=True)
    preview = preview.crop(box)
    rendered = compact_thumbnail(preview, 1024)
    mx = rendered.width / preview.width
    my = rendered.height / preview.height
    painted = [((x - box[0]) * mx, (y - box[1]) * my) for x, y in source_points]
    canvas = ImageDraw.Draw(rendered, "RGBA")
    canvas.line(painted + [painted[0]], fill=(*color, 255), width=2, joint="curve")
    if rendered.width >= 180 and rendered.height >= 24:
        draw_tag(canvas, (6, 4), title[:24], color, font(10))
    return rendered, box


def evidence_response(store: Store, engine: Engine, inspection: Inspection, finding_id: str, view: str):
    evidence = finding_evidence(inspection, engine, finding_id)
    if view == "plane":
        return png_bytes(evidence["rendered"]), "image/png"
    if view == "source":
        if evidence["native"] is None:
            raise Issue("The evidence polygon falls outside its source image.", 404)
        return png_bytes(evidence["native"]), "image/png"
    if view == "packet":
        return evidence_packet(store, inspection, evidence), "application/zip"
    return {
        key: value for key, value in evidence.items() if key not in ("rendered", "native")
    }


def evidence_packet(store: Store, inspection: Inspection, entry: dict) -> bytes:
    capture = inspection.capture(entry["capture"]["id"])
    source = store.blobs.read(capture.digest)
    if sha256(source).hexdigest() != capture.digest:
        raise Issue("The source image does not match the evidence digest.", 409)
    data = {key: value for key, value in entry.items() if key not in ("rendered", "native")}
    with PacketWriter() as packet:
        packet.add("evidence.json", serial_json(data), "application/json")
        packet.add("views/plane.png", png_bytes(entry["rendered"]), "image/png")
        if entry["native"] is not None:
            packet.add("views/native.png", png_bytes(entry["native"]), "image/png")
        extension = "tiff" if capture.format == "TIFF" else "png"
        packet.add(f"original/source.{extension}", source, f"image/{extension}")
        return packet.finish({
            "schema": EVIDENCE_SCHEMA,
            "inspection": inspection.id,
            "current_revision": inspection.revision,
            "finding": entry["finding"]["id"],
        })
