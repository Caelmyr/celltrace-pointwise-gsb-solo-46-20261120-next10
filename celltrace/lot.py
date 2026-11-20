from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw

from .engine import Engine
from .images import compact_thumbnail, distribution, png_bytes
from .model import Inspection, Issue, text
from .render import RenderOptions, draw_tag, font, render_layer
from .store import Store
from .tables import csv_data


MAX_LOT_INSPECTIONS = 12


@dataclass(frozen=True)
class LotRecord:
    inspection: Inspection
    source: dict
    measurements: list[dict]
    summary: dict


def comparability_key(item: LotRecord) -> tuple:
    inspection, src = item.inspection, item.source
    calibration = src["references"]
    return (
        inspection.plane.rows, inspection.plane.cols, src["bit_depth"], src["white_level"],
        None if src["current_a"] is None else round(src["current_a"], 3),
        round(src["exposure_ms"], 3), calibration.get("dark") is not None, calibration.get("flat") is not None,
        src["exposure_pair"] is not None,
    )


def source(entry: Inspection, calibration: dict) -> dict:
    high = entry.selected("high")
    merge = entry.merge
    pair = None
    if merge.enabled and merge.short_id in entry.captures:
        short = entry.captures[merge.short_id]
        pair = {
            "short": short.id, "short_exposure_ms": short.exposure_ms,
            "readings": merge.readings, "agreement": merge.agreement,
        }
    return {
        "capture": high.id, "digest": high.digest, "exposure_ms": high.exposure_ms, "current_a": high.current_a,
        "white_level": high.white_level, "bit_depth": high.bit_depth,
        "references": {"dark": calibration.get("dark"), "flat": calibration.get("flat")},
        "exposure_pair": pair,
        "synthetic": high.synthetic,
    }


class LotReview:
    def __init__(self, store: Store, engine: Engine):
        self.store = store
        self.engine = engine

    def inventory(self, lot: str):
        name = text(lot, "Production lot", 80)
        rows = self.store.list(lot=name, limit=MAX_LOT_INSPECTIONS + 1)
        if not rows:
            raise Issue("This lot has no inspections.", 404)
        return name, rows[:MAX_LOT_INSPECTIONS], len(rows) > MAX_LOT_INSPECTIONS

    def open_record(self, inspection: Inspection) -> LotRecord:
        intensity = self.engine.table(inspection, "high", "intensity")
        quality = self.engine.table(inspection, "high", "calibration")
        cracks = self.engine.table(inspection, "high", "cracks")
        dark = self.engine.table(inspection, "high", "dark")
        details = {
            "measurement": "current working plane",
            "median": intensity["summary"]["median"],
            "usable_cells": intensity["summary"]["eligible_cells"],
            "total_cells": intensity["summary"]["cells"],
            "crack_count": cracks["summary"]["count"],
            "dark_count": dark["summary"]["count"],
        }
        return LotRecord(inspection, source(inspection, quality["summary"]), intensity["rows"], details)

    def issued_record(self, inspection: Inspection) -> LotRecord:
        import json
        attachment = inspection.issue.get("lot_snapshot")
        if not attachment:
            live = self.open_record(inspection)
            live.summary["measurement"] = "issued settings recomputed"
            return live
        raw = self.store.blobs.read(attachment)
        if sha256(raw).hexdigest() != attachment:
            raise Issue("The issued lot measurements failed verification.", 409)
        values = json.loads(raw)
        return LotRecord(inspection, values["source"], values["measurements"], values["summary"])

    def read_records(self, lot: str):
        label, originals, limited = self.inventory(lot)
        records = []
        issues = []
        for row in originals:
            inspection = self.store.get(row["id"])
            if "high" not in inspection.active:
                issues.append({"module": inspection.module, "reason": "No high-current original"})
                continue
            try:
                read = self.issued_record(inspection) if inspection.state == "issued" else self.open_record(inspection)
            except Issue as error:
                issues.append({"module": inspection.module, "reason": str(error)})
                continue
            records.append(read)
        return label, originals, records, issues, limited

    def overview(self, lot: str) -> dict:
        label, inventory, records, issues, limited = self.read_records(lot)
        pools = group_records(records)
        rows = [
            {
                "id": row.inspection.id, "module": row.inspection.module, "lot": row.inspection.lot,
                "state": row.inspection.state, **row.summary, "source": row.source,
                "pool": pools[comparability_key(row)]["id"],
            }
            for row in records
        ]
        return {
            "lot": label, "shown": len(inventory), "limited": limited,
            "ready": len(records), "pools": list(pools.values()),
            "unavailable": issues, "rows": rows,
        }

    def sheets(self, lot: str, kind: str) -> bytes:
        label, _, records, _, _ = self.read_records(lot)
        pools = group_records(records)
        if kind == "lot":
            rows = [
                {"module": row.inspection.module, "lot": label, "state": row.inspection.state, **row.summary}
                for row in records
            ]
        elif kind == "lot-cells":
            rows = detailed_lot_rows(records, pools)
        else:
            raise Issue("This lot export is unavailable.", 404)
        return csv_data(kind, rows)

    def contact_sheet(self, lot: str) -> bytes:
        label, inventory, _ = self.inventory(lot)
        tile_width, tile_height = 330, 255
        cols = min(3, len(inventory))
        rows = (len(inventory) + cols - 1) // cols
        margin = 20
        width = tile_width * cols + margin * 2
        height = tile_height * rows + 78
        sheet = Image.new("RGB", (width, height), (18, 28, 34))
        painter = ImageDraw.Draw(sheet, "RGBA")
        painter.text((margin + 5, 18), f"CellTrace EL / {label}", fill=(225, 236, 229), font=font(24))
        for index, row in enumerate(inventory):
            x = margin + (index % cols) * tile_width
            y = 59 + (index // cols) * tile_height
            inspection = self.store.get(row["id"])
            thumb = self.lot_thumbnail(inspection)
            draw_card(sheet, painter, inspection, thumb, x, y, tile_width, tile_height)
        return png_bytes(sheet)

    def lot_thumbnail(self, inspection: Inspection):
        try:
            if inspection.issue and inspection.issue.get("cover"):
                content = self.store.blobs.read(inspection.issue["cover"])
            else:
                content = render_layer(inspection, self.engine, "high", "panel", RenderOptions(grid=True))
        except Issue:
            return None
        with Image.open(BytesIO(content)) as source_file:
            return compact_thumbnail(source_file.convert("RGB"), 300)


def group_records(records: list[LotRecord]) -> dict:
    pools = {}
    for row in records:
        key = comparability_key(row)
        pools.setdefault(key, []).append(row)
    output = {}
    for index, (key, rows) in enumerate(sorted(pools.items(), key=lambda item: str(item[0])), 1):
        qualified = [entry.summary["median"] for entry in rows if entry.summary["median"] is not None]
        output[key] = {
            "id": f"P{index:02}",
            "count": len(rows),
            "modules": [entry.inspection.module for entry in rows],
            "synthetic": any(entry.source["synthetic"] for entry in rows),
            "grid": list(key[0:2]),
            "bit_depth": key[2],
            "white_level": key[3],
            "current_a": key[4],
            "exposure_ms": key[5],
            "references": {"dark": key[6], "flat": key[7]},
            "paired_exposure": key[8],
            "supported_module_medians": distribution(np.asarray(qualified)),
            "captured_settings_match": key[4] is not None,
        }
    return output


def detailed_lot_rows(records: list[LotRecord], pools: dict) -> list:
    output = []
    for row in records:
        inspection = row.inspection
        pool = pools[comparability_key(row)]
        for cell in row.measurements:
            output.append({
                "module": inspection.module, "lot": inspection.lot, "cell": cell["cell"],
                "pool": pool["id"], "median": cell["median"], "support": cell["support"],
                "median_relative": cell["relative_panel"], "capture": row.source["capture"],
                "digest": row.source["digest"], "issue_date": inspection.issue["at"] if inspection.issue else None,
            })
    return output


def draw_card(sheet, painter, inspection, thumb, x, y, width, height):
    painter.rounded_rectangle((x + 3, y + 3, x + width - 6, y + height - 7), radius=7,
                              fill=(24, 41, 47, 255), outline=(71, 99, 99, 255))
    painter.text((x + 14, y + 13), inspection.module[:38], fill=(227, 238, 232), font=font(15))
    image_top = y + 40
    image_height = height - 66
    if thumb is None:
        painter.text((x + 16, image_top + 32), "High-current view unavailable", fill=(159, 181, 180), font=font(14))
    else:
        thumb.thumbnail((width - 24, image_height))
        sheet.paste(thumb, (x + (width - thumb.width) // 2, image_top + (image_height - thumb.height) // 2))
    label = "Issued evidence" if inspection.issue else "Working evidence"
    draw_tag(painter, (x + 15, y + height - 25), label, (113, 219, 182), font(11))
