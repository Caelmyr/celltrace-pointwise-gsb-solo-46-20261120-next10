import sys
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from celltrace.alignment import STATUSES, local_alignment  # noqa: E402
from celltrace.engine import Engine  # noqa: E402
from celltrace.inspection import Inspections  # noqa: E402
from celltrace.store import Store  # noqa: E402


class AlignmentCase(unittest.TestCase):
    variants = ("standard", "stress")

    @classmethod
    def setUpClass(cls):
        cls.directory = TemporaryDirectory()
        cls.store = Store(cls.directory.name)
        cls.engine = Engine(cls.store)
        cls.service = Inspections(cls.store)
        cls.items = {variant: cls.service.demo(variant) for variant in cls.variants}
        cls.panels = {
            variant: (
                cls.engine.panel_context(item, "high").panel,
                cls.engine.panel_context(item, "low").panel,
                cls.engine.panel_context(item, "high").cells,
            )
            for variant, item in cls.items.items()
        }

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def align(self, variant, low=None):
        high, native, cells = self.panels[variant]
        item = self.items[variant]
        return local_alignment(high, low if low is not None else native, cells, (item.plane.width, item.plane.height))

    def warp(self, panel, dx, dy, box):
        """Move panel content by (dx, dy) plane pixels inside box."""
        x0, y0, x1, y1 = box
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        signal = panel.signal.copy()
        valid = panel.valid.copy()
        signal[y0:y1, x0:x1] = ndimage.map_coordinates(
            np.nan_to_num(panel.signal), (yy - dy, xx - dx), order=1, mode="nearest",
        )
        valid[y0:y1, x0:x1] = ndimage.map_coordinates(
            panel.valid.astype(np.float32), (yy - dy, xx - dx), order=1, mode="nearest",
        ) > 0.999
        signal[~valid] = np.nan
        return replace(panel, signal=signal.astype(np.float32), valid=valid)

    def test_well_registered_demos_keep_their_weak_cells(self):
        expected = {
            "standard": {"R02C04", "R04C03", "R04C05"},
            "stress": {"R01C01", "R02C04", "R04C03", "R04C05"},
        }
        for variant in self.variants:
            alignment = self.align(variant)
            self.assertEqual(alignment.report["cells_recapture"], 0, variant)
            self.assertTrue(alignment.field["supported"])
            self.assertFalse(alignment.masks["offset"].any(), variant)
            weak = {
                record["cell"] for record in alignment.cells.values()
                if record["state"] in ("measured", "limited")
            }
            self.assertTrue(expected[variant] <= weak, variant)
            self.assertGreaterEqual(alignment.report["plane_compared_fraction"], 0.7, variant)

    def test_missed_separator_columns_are_sent_to_recapture(self):
        box = (0, 270, 280, 540)
        for variant in self.variants:
            _, low, _ = self.panels[variant]
            native = self.align(variant)
            shifted = self.align(variant, self.warp(low, 1.5, -1.5, box))
            flagged = {
                cell for cell, record in shifted.cells.items()
                if record["state"] != "measured" and native.cells[cell]["state"] == "measured"
            }
            self.assertGreaterEqual(len(flagged), 3, variant)
            self.assertGreaterEqual(shifted.report["cells_recapture"], 3, variant)
            for cell in flagged:
                self.assertTrue(
                    shifted.cells[cell]["offset_fraction"] > 0
                    or shifted.cells[cell]["ambiguous_fraction"] > 0
                    or shifted.cells[cell]["no-texture_fraction"] > 0,
                    f"{variant} {cell}",
                )

    def test_reported_shift_is_the_motion_of_low_current_content(self):
        _, low, _ = self.panels["standard"]
        box = (0, 270, 280, 540)
        for dx, dy in ((3.0, 0.0), (0.0, -2.0)):
            shifted = self.warp(low, dx, dy, box)
            record = self.align("standard", shifted).cells["R03C01"]
            self.assertAlmostEqual(record["median_shift"][0], dx, delta=0.45)
            self.assertAlmostEqual(record["median_shift"][1], dy, delta=0.45)
            self.assertEqual(record["state"], "recapture")

    def test_half_pixel_offset_is_detected(self):
        _, low, _ = self.panels["standard"]
        shifted = self.align("standard", self.warp(low, 0.5, 0.0, (0, 270, 280, 540)))
        self.assertGreaterEqual(shifted.report["maximum_abs_shift"], 0.4)

    def test_unaligned_cells_do_not_feed_the_ratio(self):
        item = self.items["stress"]
        table = self.engine.table(item, "high", "response")
        summary = table["summary"]
        self.assertLessEqual(summary["comparable_pixels"], summary["shared_pixels"])
        self.assertAlmostEqual(
            summary["comparable_pixels"] + summary["alignment"]["recapture_pixels"],
            summary["alignment"]["covered_pixels"],
            delta=1,
        )
        for row in table["rows"]:
            self.assertLessEqual(row["aligned_pixels"], row["common_pixels"], row["cell"])
            self.assertLessEqual(row["stable_pixels"], row["aligned_pixels"], row["cell"])
            if row["alignment_state"] == "recapture":
                self.assertEqual(row["aligned_pixels"], 0, row["cell"])
                self.assertIsNone(row["ratio"], row["cell"])

    def test_recapture_pixels_are_outside_the_counted_reference(self):
        item = self.items["stress"]
        analysis = self.engine.compare(item).data
        self.assertFalse((analysis.eligible & analysis.alignment.recapture).any())
        self.assertTrue(set(analysis.alignment.masks) == set(STATUSES))

    def test_correspondence_reports_the_same_compare_point(self):
        item = self.items["stress"]
        comparison = self.engine.compare(item).data
        for x, y in ((210, 300), (620, 500), (100, 100)):
            probe = self.engine.pixel(item, "high", x, y, True)["response"]
            self.assertEqual(probe["high_point"], [x, y])
            if probe["common"]:
                self.assertEqual(probe["low_point"], [x, y])
            self.assertEqual(probe["ratio"], (
                None if not np.isfinite(comparison.ratio[y, x]) else round(float(comparison.ratio[y, x]), 6)
            ))


class EmptyTextureCase(unittest.TestCase):
    def test_flat_pair_is_not_reported_as_aligned(self):
        from celltrace.calibration import PanelImage
        from celltrace.geometry import cells_for
        from celltrace.model import AnalysisSettings, Capture, Issue, Plane
        from celltrace.response import probe_response, response_map

        plane = Plane(width=120, height=80, rows=2, cols=3, inset=0.1)
        shape = (plane.height, plane.width)

        def flat(value):
            signal = np.full(shape, value, dtype=np.float32)
            valid = np.ones(shape, dtype=bool)
            return PanelImage(signal, valid, valid.copy(), np.zeros(shape, bool),
                              np.zeros(shape, bool), np.zeros(shape, bool), np.eye(3), {})

        def capture(identifier, role, current):
            return Capture(identifier, role, "n.png", "PNG", "a" * 64, 100,
                           plane.width, plane.height, 16, 65535, 100, current, "t", True)

        result = response_map(flat(0.4), flat(0.2), capture("a", "high", 8.0), capture("b", "low", 1.0),
                              cells_for(plane), AnalysisSettings(), (plane.width, plane.height))
        self.assertEqual(result.report["usable_cells"], 0)
        self.assertFalse(result.report["alignment"]["field"]["supported"])
        self.assertTrue(all(row["alignment_state"] == "recapture" for row in result.records))
        self.assertFalse(result.eligible.any())
        probe = probe_response(result, 10, 10)
        self.assertIsNone(probe["local_shift"])
        self.assertIsNone(probe["match_point"])
        self.assertEqual(probe["alignment"], "no-texture")

    def test_low_current_ordering_is_rejected(self):
        from celltrace.calibration import PanelImage
        from celltrace.geometry import cells_for
        from celltrace.model import AnalysisSettings, Capture, Issue, Plane
        from celltrace.response import response_map

        plane = Plane(width=120, height=80, rows=2, cols=3, inset=0.1)
        shape = (plane.height, plane.width)
        signal = np.full(shape, 0.3, dtype=np.float32)
        valid = np.ones(shape, dtype=bool)
        panel = PanelImage(signal, valid, valid.copy(), np.zeros(shape, bool),
                           np.zeros(shape, bool), np.zeros(shape, bool), np.eye(3), {})

        def capture(identifier, role, current):
            return Capture(identifier, role, "n.png", "PNG", "a" * 64, 100,
                           plane.width, plane.height, 16, 65535, 100, current, "t", True)

        with self.assertRaises(Issue):
            response_map(panel, panel, capture("a", "high", 1.0), capture("b", "low", 8.0),
                         cells_for(plane), AnalysisSettings(), (plane.width, plane.height))


if __name__ == "__main__":
    unittest.main()
