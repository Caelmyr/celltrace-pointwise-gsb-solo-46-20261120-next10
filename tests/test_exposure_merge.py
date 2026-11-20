"""Regression checks for the paired high-current exposure reading.

Run from the project root with ``python3 -m unittest discover -s tests``.
These use the synthetic fixtures only; they do not touch a data directory.
"""

import io
import tempfile
import unittest
import zipfile

import numpy as np

from celltrace.composite import (
    CORROBORATED, LONG_ONLY, SHORT_ONLY, UNDECIDED, merge_probe, read_floor,
)
from celltrace.engine import Engine
from celltrace.inspection import Inspections
from celltrace.store import Store
from celltrace.web import build_app


class MergeBench(unittest.TestCase):
    def setUp(self):
        self.data = tempfile.mkdtemp()
        self.store = Store(self.data)
        self.service = Inspections(self.store)
        self.engine = Engine(self.store)

    def set_mode(self, item, **values):
        body = {
            "enabled": True, "short_id": item.merge.short_id,
            "short_calibration": item.merge.short_calibration,
            "readings": item.merge.readings,
        }
        body.update(values)
        return self.service.change(item.id, item.revision, "merge", body)

    def test_composition_keeps_the_clipped_patch_readable(self):
        item = self.service.demo("standard")
        merge = self.engine.panel_context(item, "high").merge
        self.assertIsNotNone(merge)
        # The long exposure's clipped hotspot is borrowed from the short exposure.
        self.assertGreater(int((merge.regime == SHORT_ONLY).sum()), 0)
        self.assertTrue(merge.panel.valid[merge.regime == SHORT_ONLY].all())
        # The composed plane reaches where the long exposure alone has no usable reading.
        self.assertTrue(merge.panel.valid[~merge.long_panel.valid & merge.short_panel.valid].any())
        # No pixel is counted in more than one partition.
        self.assertEqual(sum(merge.report[key] for key in (
            "corroborated_pixels", "long_only_pixels", "short_only_pixels",
            "undecided_pixels", "missing_pixels",
        )), merge.report["pixels"])

    def test_seam_gain_avoids_a_dark_edge(self):
        item = self.service.demo("stress")
        merge = self.engine.panel_context(item, "high").merge
        self.assertEqual(merge.report["adjusted_regions"], 1)
        borrowed = merge.regime == SHORT_ONLY
        transformed = merge.short_final[borrowed]
        plain = merge.short_panel.signal[borrowed] * merge.exposure_scale
        surrounding = float(np.median(merge.long_panel.signal[merge.long_panel.valid]))
        # Correcting the local gain moves the borrowed patch closer to its surroundings.
        before = abs(float(np.median(plain)) - surrounding)
        after = abs(float(np.median(transformed)) - surrounding)
        self.assertLess(after, before)

    def test_read_switch_returns_the_single_exposure(self):
        item = self.service.demo("standard")
        y, x = np.nonzero(self.engine.panel_context(item, "high").merge.regime == SHORT_ONLY)
        row, col = int(np.median(y)), int(np.median(x))
        merged = self.engine.pixel(item, "high", col, row, False)
        self.assertEqual(merged["reading"]["used"], "short")
        item = self.set_mode(item, readings="long")
        single = self.engine.pixel(item, "high", col, row, False)
        self.assertIsNone(single["signal"])
        self.assertEqual(single["reading"]["used"], None)
        self.assertIn("long exposure clipped", single["reading"]["reasons"])
        item = self.set_mode(item, enabled=False)
        self.assertIsNone(self.engine.panel_context(item, "high").merge)

    def test_off_on_memory_restores_the_pair(self):
        item = self.service.demo("standard")
        short = item.merge.short_id
        item = self.set_mode(item, enabled=False)
        self.assertFalse(item.merge.enabled)
        self.assertEqual(item.merge.short_id, short)
        item = self.service.change(item.id, item.revision, "merge", {
            "enabled": True, "short_id": item.merge.short_id, "readings": "composite",
            "short_calibration": item.merge.short_calibration,
        })
        self.assertTrue(item.merge.enabled)
        self.assertIsNotNone(self.engine.panel_context(item, "high").merge)

    def test_paired_exposure_must_be_shorter(self):
        item = self.service.demo("standard")
        with self.assertRaises(Exception):
            self.service.change(item.id, item.revision, "merge", {
                "enabled": True, "short_id": item.active["high"], "readings": "composite",
            })

    def test_pair_guards_keep_the_long_and_short_roles(self):
        item = self.service.demo("standard")
        short = item.merge.short_id
        with self.assertRaises(Exception):
            self.service.change(item.id, item.revision, "selection", {"capture": short})
        with self.assertRaises(Exception):
            self.service.change(item.id, item.revision, "capture-metadata", {
                "capture": short, "exposure_ms": 500, "white_level": 65535,
            })
        detached = self.service.remove_capture(item.id, item.revision, short)
        self.assertFalse(detached.merge.enabled)
        self.assertIsNone(detached.merge.short_id)
        self.assertIsNone(self.engine.panel_context(detached, "high").merge)

    def test_detaching_a_short_reference_clears_the_link(self):
        item = self.service.demo("standard")
        dark = item.merge.short_calibration.dark_id
        item = self.set_mode(item, enabled=False)
        item = self.service.remove_capture(item.id, item.revision, dark)
        self.assertIsNone(item.merge.short_calibration.dark_id)

    def test_http_round_trip_preserves_the_reading(self):
        app = build_app(self.data)
        client = app.test_client()
        item = client.post("/api/demo/stress").get_json()
        index = item["id"]
        item = client.patch(f"/api/inspections/{index}/merge", json={
            "enabled": True, "short_id": item["merge"]["short_id"], "readings": "long",
            "short_calibration": item["merge"]["short_calibration"], "revision": item["revision"],
        }).get_json()
        package = client.get(f"/api/inspections/{index}/export/exchange").data
        restored = client.post(
            "/api/import/restore",
            data={"file": (io.BytesIO(package), "restored.zip")},
            content_type="multipart/form-data",
        )
        self.assertEqual(restored.status_code, 200)
        self.assertEqual(restored.get_json()["merge"]["readings"], "long")

    def test_plane_export_records_the_partition(self):
        app = build_app(self.data)
        client = app.test_client()
        item = client.post("/api/demo/standard").get_json()
        archive = zipfile.ZipFile(io.BytesIO(
            client.get(f"/api/inspections/{item['id']}/export/plane").data
        ))
        with archive:
            self.assertIn("plane/exposure-regime.png", archive.namelist())
            self.assertIn("plane/measurement.json", archive.namelist())


if __name__ == "__main__":
    unittest.main()
