from dataclasses import asdict, replace

import numpy as np
from PIL import ImageDraw
from scipy import ndimage

from .geometry import millimeter_scale
from .images import mixed_overlay, png_bytes
from .model import (
    MAX_LINK_REGIONS, MAX_REVIEW_LINKS, MAX_REVIEWS, REGION_RE, SENSOR_ROLES,
    Inspection, Issue, ReviewLink, StepReview, file_id, mapping, sequence, text, utcnow,
)
from .render import decorate_grid, draw_tag, font, panel_preview
from .store import data_digest, new_id


GROWN_RATIO = 1.15
SHRUNK_RATIO = 0.87
NEIGHBOR_PX = 2
VERDICTS = ("new", "grown", "shrunk", "steady", "merged", "split", "regrouped", "cleared")
GROUP_COLORS = (
    (242, 166, 90), (125, 214, 194), (229, 127, 161), (159, 184, 242),
    (217, 224, 107), (199, 155, 242), (242, 136, 123), (111, 199, 232),
    (168, 217, 124), (232, 195, 106), (143, 159, 240), (240, 149, 201),
)


def color_hex(index: int) -> str:
    red, green, blue = GROUP_COLORS[index % len(GROUP_COLORS)]
    return f"#{red:02x}{green:02x}{blue:02x}"


def capture_pin(inspection: Inspection, capture) -> dict:
    placement = inspection.placements.get(capture.id)
    return {
        "capture": capture.id,
        "digest": capture.digest,
        "placement": asdict(placement) if placement is not None else None,
    }


def review_anchor(inspection: Inspection, before_id: str, after_id: str) -> dict:
    before = inspection.capture(before_id)
    after = inspection.capture(after_id)
    if before.id == after.id:
        raise Issue("Choose two different captures for the step review.")
    for capture in (before, after):
        if capture.role not in SENSOR_ROLES:
            raise Issue("Step reviews compare high- or low-current EL captures.", 409)
    dark = inspection.captures.get(inspection.calibration.dark_id)
    flat = inspection.captures.get(inspection.calibration.flat_id)
    values = {
        "plane": asdict(inspection.plane),
        "analysis": asdict(inspection.analysis),
        "calibration": asdict(inspection.calibration),
        "reference_digests": {"dark": dark.digest if dark else None, "flat": flat.digest if flat else None},
        "before": capture_pin(inspection, before),
        "after": capture_pin(inspection, after),
    }
    return {"signature": data_digest(values), **values}


def review_status(inspection: Inspection, review: StepReview) -> str:
    try:
        anchor = review_anchor(inspection, review.before_capture, review.after_capture)
    except Issue:
        return "source unavailable"
    return "current" if anchor["signature"] == review.anchor.get("signature") else "source changed"


def review_meta(review: StepReview) -> dict:
    return {
        "id": review.id,
        "title": review.title,
        "before_step": review.before_step,
        "after_step": review.after_step,
        "links": len(review.links),
        "created_at": review.created_at,
        "updated_at": review.updated_at,
    }


def capture_brief(inspection: Inspection, capture_id: str, step: str) -> dict:
    capture = inspection.captures.get(capture_id)
    if capture is None:
        return {"id": capture_id, "step": step, "missing": True}
    placement = inspection.placements.get(capture_id)
    return {
        "id": capture.id,
        "step": step,
        "missing": False,
        "name": capture.name,
        "role": capture.role,
        "digest": capture.digest,
        "current_a": capture.current_a,
        "exposure_ms": capture.exposure_ms,
        "width": capture.width,
        "height": capture.height,
        "placement_explicit": placement is not None,
        "orientation": placement.orientation if placement is not None else "identity",
    }


def review_provenance(inspection: Inspection, review: StepReview) -> dict:
    return {
        "before": capture_brief(inspection, review.before_capture, review.before_step),
        "after": capture_brief(inspection, review.after_capture, review.after_step),
    }


def list_reviews(inspection: Inspection) -> list[dict]:
    output = []
    for review in inspection.reviews:
        before = inspection.captures.get(review.before_capture)
        after = inspection.captures.get(review.after_capture)
        output.append({
            **review_meta(review),
            "before_capture": review.before_capture,
            "after_capture": review.after_capture,
            "before_name": before.name if before else None,
            "after_name": after.name if after else None,
            "status": review_status(inspection, review),
        })
    return output


def make_review(inspection: Inspection, raw: dict) -> StepReview:
    if len(inspection.reviews) >= MAX_REVIEWS:
        raise Issue(f"An inspection can hold up to {MAX_REVIEWS} step reviews.", 409)
    title = text(raw.get("title"), "Review title", 120)
    before_step = text(raw.get("before_step"), "Earlier process step", 80)
    after_step = text(raw.get("after_step"), "Later process step", 80)
    before_id = file_id(raw.get("before_capture"), "Earlier capture ID")
    after_id = file_id(raw.get("after_capture"), "Later capture ID")
    anchor = review_anchor(inspection, before_id, after_id)
    before = inspection.capture(before_id)
    after = inspection.capture(after_id)
    now = utcnow()
    return StepReview(
        new_id(), title, before_step, after_step,
        before_id, after_id, before.digest, after.digest,
        (), anchor, now, now,
    )


def get_review(inspection: Inspection, review_id: str) -> StepReview:
    file_id(review_id, "Review ID")
    for review in inspection.reviews:
        if review.id == review_id:
            return review
    raise Issue("This step review does not exist.", 404)


def apply_review_edit(review: StepReview, raw: dict, links: list[ReviewLink] | None) -> StepReview:
    title = text(raw.get("title", review.title), "Review title", 120)
    return replace(
        review,
        title=title,
        links=review.links if links is None else tuple(links),
        updated_at=utcnow(),
    )


def descriptor(record: dict) -> dict:
    return {
        "id": record["id"],
        "cell": record["cell"],
        "center": [float(record["center"][0]), float(record["center"][1])],
        "bounds": [float(value) for value in record["bounds"]],
        "pixels": int(record["pixels"]),
        "area_mm2": record["area_mm2"],
    }


def resolve_links(computed: dict, submitted) -> list[ReviewLink]:
    submitted = sequence(submitted, "Review links", 0, MAX_REVIEW_LINKS)
    index = {
        side: {record["id"]: record for record in computed["regions"][side]}
        for side in ("before", "after")
    }
    used = {"before": set(), "after": set()}
    links = []
    for entry in submitted:
        entry = mapping(entry, "Review link")
        resolved = {}
        for side in ("before", "after"):
            identifiers = sequence(entry.get(side, []), f"Link {side} regions", 0, MAX_LINK_REGIONS)
            picked = []
            for region_id in identifiers:
                if not isinstance(region_id, str) or not REGION_RE.fullmatch(region_id):
                    raise Issue("A link refers to an unknown dark region identifier.")
                if region_id in used[side]:
                    raise Issue(f"Dark region {region_id} is assigned to two links.")
                record = index[side].get(region_id)
                if record is None:
                    raise Issue(f"Dark region {region_id} is not part of the current measurement.", 409)
                used[side].add(region_id)
                picked.append(descriptor(record))
            resolved[side] = tuple(picked)
        if not resolved["before"] and not resolved["after"]:
            raise Issue("A review link needs at least one dark region.")
        links.append(ReviewLink(
            new_id(), resolved["before"], resolved["after"],
            text(entry.get("note", ""), "Link note", 400, allow_blank=True),
        ))
    return links


def region_labels(mask: np.ndarray, records: list[dict]) -> np.ndarray:
    markers, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    if count != len(records):
        raise Issue("The dark region index could not be rebuilt for comparison.", 500)
    labels = np.zeros(mask.shape, dtype=np.int32)
    if not count:
        return labels
    centroids = ndimage.center_of_mass(mask, markers, range(1, count + 1))
    free = set(range(count))
    order = []
    for record in records:
        cx, cy = record["center"]
        best = min(free, key=lambda item: (centroids[item][1] - cx) ** 2 + (centroids[item][0] - cy) ** 2)
        free.discard(best)
        order.append(best)
    remap = np.zeros(count + 1, dtype=np.int32)
    for index, marker in enumerate(order):
        remap[marker + 1] = index + 1
    return remap[markers]


def label_edges(first: np.ndarray, second: np.ndarray) -> set[tuple[int, int]]:
    edges = set()
    size = 2 * NEIGHBOR_PX + 1
    for source, target, flip in ((first, second, False), (second, first, True)):
        grown = ndimage.grey_dilation(source, size=size)
        take = target > 0
        src = grown[take]
        tgt = target[take]
        keep = src > 0
        if not keep.any():
            continue
        pairs = np.unique(np.stack((src[keep], tgt[keep]), axis=1), axis=0)
        for a_value, b_value in pairs:
            pair = (int(a_value), int(b_value))
            edges.add(pair if not flip else (pair[1], pair[0]))
    return edges


def cluster_regions(free_before: set[int], free_after: set[int],
                    edges: set[tuple[int, int]]) -> list[tuple[list[int], list[int]]]:
    parent = {}

    def find(node):
        parent.setdefault(node, node)
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    for index in free_before:
        find(("b", index))
    for index in free_after:
        find(("a", index))
    for before, after in edges:
        if before in free_before and after in free_after:
            first, second = find(("b", before)), find(("a", after))
            if first != second:
                parent[max(first, second)] = min(first, second)
    clusters = {}
    for node in parent:
        clusters.setdefault(find(node), [[], []])
    for node in parent:
        side, index = node
        clusters[find(node)][0 if side == "b" else 1].append(index)
    return [(sorted(before), sorted(after)) for before, after in clusters.values()]


def classify(before_count: int, after_count: int, before_px: int, after_px: int) -> str:
    if before_count == 0:
        return "new"
    if after_count == 0:
        return "cleared"
    if before_count > 1 and after_count == 1:
        return "merged"
    if before_count == 1 and after_count > 1:
        return "split"
    if before_count > 1 and after_count > 1:
        return "regrouped"
    ratio = after_px / before_px if before_px else float("inf")
    if ratio >= GROWN_RATIO:
        return "grown"
    if ratio <= SHRUNK_RATIO:
        return "shrunk"
    return "steady"


def side_data(engine, inspection: Inspection, capture_id: str) -> dict:
    context = engine.panel_context_for(inspection, capture_id)
    dark = engine.dark(inspection, context)
    intensity = engine.intensity(inspection, context)
    unreliable = {row["cell"] for row in intensity.records if not row["reliable"]}
    regions = []
    for record in dark.records:
        regions.append({
            "id": record["id"],
            "cell": record["cell"],
            "bounds": record["bounds"],
            "center": record["center"],
            "pixels": record["pixels"],
            "area_mm2": record["area_mm2"],
            "depth": record["depth"],
            "open_boundary": record["open_boundary"],
            "limited": record["cell"] in unreliable,
        })
    return {
        "context": context,
        "dark": dark,
        "intensity": intensity,
        "regions": regions,
        "labels": region_labels(dark.mask, dark.records),
        "unreliable": unreliable,
    }


def group_payload(number: int, link, before_regions: list[dict], after_regions: list[dict],
                  scale, limited_cells: set[str]) -> dict:
    before_px = sum(record["pixels"] for record in before_regions)
    after_px = sum(record["pixels"] for record in after_regions)
    caution = set()
    for record in before_regions + after_regions:
        if record["open_boundary"]:
            caution.add("region touches an unsupported edge")
        if record["cell"] in limited_cells:
            caution.add("cell support is limited")
    area = lambda pixels: round(pixels * scale[0] * scale[1], 3) if scale else None
    return {
        "key": f"G{number}",
        "link_id": link.id if link is not None else None,
        "manual": link is not None,
        "note": link.note if link is not None else "",
        "before": [record["id"] for record in before_regions],
        "after": [record["id"] for record in after_regions],
        "cells": sorted({record["cell"] for record in before_regions + after_regions}),
        "verdict": classify(len(before_regions), len(after_regions), before_px, after_px),
        "before_pixels": before_px,
        "after_pixels": after_px,
        "delta_pixels": after_px - before_px,
        "before_mm2": area(before_px),
        "after_mm2": area(after_px),
        "delta_mm2": area(after_px - before_px),
        "area_ratio": round(after_px / before_px, 4) if before_px else None,
        "caution": sorted(caution),
        "color": color_hex(number - 1),
    }


def inconclusive_cells(before: dict, after: dict, review: StepReview) -> list[dict]:
    rows = []
    for earlier, later in zip(before["intensity"].records, after["intensity"].records):
        if earlier["reliable"] and later["reliable"]:
            continue
        if not earlier["reliable"] and not later["reliable"]:
            reason = "Both captures lack usable support here."
        elif not later["reliable"]:
            reason = f"Blocked or outside the frame in the {review.after_step} capture."
        else:
            reason = f"Blocked or outside the frame in the {review.before_step} capture."
        rows.append({
            "cell": earlier["cell"],
            "before_support": earlier["support"],
            "after_support": later["support"],
            "reason": reason,
        })
    return rows


def summarize(groups: list[dict], before: dict, after: dict, scale, inconclusive: list[dict]) -> dict:
    counts = {verdict: 0 for verdict in VERDICTS}
    for group in groups:
        counts[group["verdict"]] += 1
    before_px = sum(record["pixels"] for record in before["regions"])
    after_px = sum(record["pixels"] for record in after["regions"])
    area = lambda pixels: round(pixels * scale[0] * scale[1], 3) if scale else None
    return {
        "groups": len(groups),
        "manual_links": sum(1 for group in groups if group["manual"]),
        **counts,
        "before_dark_pixels": before_px,
        "after_dark_pixels": after_px,
        "before_dark_mm2": area(before_px),
        "after_dark_mm2": area(after_px),
        "delta_mm2": area(after_px - before_px),
        "inconclusive_cells": len(inconclusive),
    }


def compute_review(engine, inspection: Inspection, review: StepReview) -> dict:
    before = side_data(engine, inspection, review.before_capture)
    after = side_data(engine, inspection, review.after_capture)
    scale = millimeter_scale(inspection.plane)
    edges = {(b - 1, a - 1) for b, a in label_edges(before["labels"], after["labels"])}
    before_index = {record["id"]: pos for pos, record in enumerate(before["regions"])}
    after_index = {record["id"]: pos for pos, record in enumerate(after["regions"])}
    used_before, used_after = set(), set()
    grouped = []
    for link in review.links:
        b_idx = sorted(before_index[region["id"]] for region in link.before)
        a_idx = sorted(after_index[region["id"]] for region in link.after)
        used_before.update(b_idx)
        used_after.update(a_idx)
        grouped.append((link, b_idx, a_idx))
    free_before = set(range(len(before["regions"]))) - used_before
    free_after = set(range(len(after["regions"]))) - used_after
    auto = cluster_regions(free_before, free_after, edges)
    auto.sort(key=lambda pair: group_sort_key(before["regions"], after["regions"], pair))
    grouped.extend((None, b_idx, a_idx) for b_idx, a_idx in auto)
    limited = before["unreliable"] | after["unreliable"]
    groups = [
        group_payload(
            number, link,
            [before["regions"][pos] for pos in b_idx],
            [after["regions"][pos] for pos in a_idx],
            scale, limited,
        )
        for number, (link, b_idx, a_idx) in enumerate(grouped, start=1)
    ]
    inconclusive = inconclusive_cells(before, after, review)
    return {
        "status": "current",
        "review": review_meta(review),
        "captures": review_provenance(inspection, review),
        "plane": {
            "width": inspection.plane.width, "height": inspection.plane.height,
            "rows": inspection.plane.rows, "cols": inspection.plane.cols,
        },
        "physical_scale": [round(value, 6) for value in scale] if scale else None,
        "regions": {"before": before["regions"], "after": after["regions"]},
        "groups": groups,
        "inconclusive_cells": inconclusive,
        "summary": summarize(groups, before, after, scale, inconclusive),
    }


def group_sort_key(before_regions: list[dict], after_regions: list[dict], pair) -> tuple:
    b_idx, a_idx = pair
    centers = [before_regions[pos]["center"] for pos in b_idx]
    centers += [after_regions[pos]["center"] for pos in a_idx]
    y = min(point[1] for point in centers)
    x = min(point[0] for point in centers)
    return (y, x)


def stale_payload(inspection: Inspection, review: StepReview, status: str) -> dict:
    groups = []
    for number, link in enumerate(review.links, start=1):
        before_px = sum(region["pixels"] for region in link.before)
        after_px = sum(region["pixels"] for region in link.after)
        area = lambda regions: (
            round(sum(region["area_mm2"] for region in regions), 3)
            if all(region["area_mm2"] is not None for region in regions) else None
        )
        before_mm2 = area(link.before)
        after_mm2 = area(link.after)
        groups.append({
            "key": f"G{number}",
            "link_id": link.id,
            "manual": True,
            "note": link.note,
            "before": [region["id"] for region in link.before],
            "after": [region["id"] for region in link.after],
            "cells": sorted({region["cell"] for region in (*link.before, *link.after)}),
            "verdict": classify(len(link.before), len(link.after), before_px, after_px),
            "before_pixels": before_px,
            "after_pixels": after_px,
            "delta_pixels": after_px - before_px,
            "before_mm2": before_mm2,
            "after_mm2": after_mm2,
            "delta_mm2": round(after_mm2 - before_mm2, 3) if before_mm2 is not None and after_mm2 is not None else None,
            "area_ratio": round(after_px / before_px, 4) if before_px else None,
            "caution": [],
            "color": color_hex(number - 1),
        })
    counts = {verdict: 0 for verdict in VERDICTS}
    for group in groups:
        counts[group["verdict"]] += 1
    before_px = sum(group["before_pixels"] for group in groups)
    after_px = sum(group["after_pixels"] for group in groups)
    return {
        "status": status,
        "review": review_meta(review),
        "captures": review_provenance(inspection, review),
        "plane": None,
        "physical_scale": None,
        "regions": {"before": [], "after": []},
        "groups": groups,
        "inconclusive_cells": [],
        "summary": {
            "groups": len(groups),
            "manual_links": len(groups),
            **counts,
            "before_dark_pixels": before_px,
            "after_dark_pixels": after_px,
            "before_dark_mm2": None,
            "after_dark_mm2": None,
            "delta_mm2": None,
            "inconclusive_cells": 0,
        },
    }


def review_computed(engine, inspection: Inspection, review: StepReview) -> dict:
    key = (
        "step-review", review.anchor.get("signature"),
        data_digest([asdict(link) for link in review.links]),
    )
    return engine.measurements.get(key, lambda: compute_review(engine, inspection, review))


def review_image(engine, inspection: Inspection, review: StepReview, computed: dict, side: str) -> bytes:
    capture_id = review.before_capture if side == "before" else review.after_capture
    context = engine.panel_context_for(inspection, capture_id)
    dark = engine.dark(inspection, context)
    image = panel_preview(context.panel, inspection.display)
    image = mixed_overlay(image, dark.mask, (247, 166, 90), 0.28)
    painter = ImageDraw.Draw(image, "RGBA")
    cells = {cell.id: cell for cell in context.cells}
    for entry in computed["inconclusive_cells"]:
        cell = cells.get(entry["cell"])
        if cell is None:
            continue
        x0, y0, x1, y1 = cell.bounds
        painter.rectangle((x0, y0, x1 - 1, y1 - 1), fill=(151, 117, 205, 24), outline=(174, 143, 224, 140))
    records = {record["id"]: record for record in computed["regions"][side]}
    face = font(10)
    for group in computed["groups"]:
        rgb = tuple(int(group["color"][pos:pos + 2], 16) for pos in (1, 3, 5))
        for region_id in group[side]:
            record = records.get(region_id)
            if record is None:
                continue
            x0, y0, x1, y1 = record["bounds"]
            painter.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(*rgb, 235), width=2)
            draw_tag(painter, (x0 + 2, max(0, y0 - 13)), group["key"], rgb, face)
    decorate_grid(image, context.cells, labels=True)
    return png_bytes(image)
