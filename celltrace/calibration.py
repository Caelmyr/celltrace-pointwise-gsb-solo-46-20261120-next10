from dataclasses import dataclass

import numpy as np

from .geometry import SamplingMap
from .images import SensorImage, distribution
from .model import Capture, Issue


CAL_VERSION = "single-dark-and-flat-v1"


@dataclass(frozen=True)
class ReferenceImage:
    capture: Capture
    source: SensorImage


@dataclass(frozen=True)
class CalibratedImage:
    signal: np.ndarray
    valid: np.ndarray
    saturated: np.ndarray
    calibration_gap: np.ndarray
    dark_lost: np.ndarray
    details: dict


@dataclass(frozen=True)
class PanelImage:
    signal: np.ndarray
    valid: np.ndarray
    any_support: np.ndarray
    clipped: np.ndarray
    calibration_gap: np.ndarray
    floor_hit: np.ndarray
    matrix: np.ndarray
    details: dict


def reference_matches(capture: Capture, reference: Capture) -> str | None:
    if (capture.width, capture.height) != (reference.width, reference.height):
        return "source and calibration dimensions differ"
    if capture.bit_depth != reference.bit_depth or capture.white_level != reference.white_level:
        return "source and calibration code ranges differ"
    return None


def dark_matches(capture: Capture, reference: Capture) -> str | None:
    issue = reference_matches(capture, reference)
    if issue is not None:
        return issue
    tolerance = max(0.025, 0.005 * capture.exposure_ms)
    if abs(capture.exposure_ms - reference.exposure_ms) > tolerance:
        return "dark exposure and signal exposure differ"
    return None


def corrected_codes(source: SensorImage, white_level: int, dark: ReferenceImage | None):
    values = source.pixels.astype(np.float32)
    if dark is None:
        dark_clipped = np.zeros(values.shape, dtype=bool)
    else:
        dark_values = dark.source.pixels
        dark_clipped = dark_values >= dark.capture.white_level
        values -= dark_values
    lost = values <= 0
    np.maximum(values, 0, out=values)
    values /= white_level
    return values, dark_clipped, lost


def calibrate_sensor(
    capture: Capture, source: SensorImage, dark: ReferenceImage | None,
    flat: ReferenceImage | None, flat_floor: float
) -> CalibratedImage:
    details = {
        "method": CAL_VERSION, "dark": None, "flat": None,
        "flat_normalization": None, "messages": [],
    }
    selected_dark = dark
    if dark is not None:
        issue = dark_matches(capture, dark.capture)
        if issue:
            details["messages"].append(f"Dark reference skipped because {issue}.")
            selected_dark = None
    values, dark_clipped, floor_hit = corrected_codes(source, capture.white_level, selected_dark)
    saturated = source.pixels >= capture.white_level
    calibration_gap = dark_clipped.copy()
    if selected_dark is not None:
        details["dark"] = selected_dark.capture.id
    if flat is not None:
        mismatch = reference_matches(capture, flat.capture)
        if mismatch:
            details["messages"].append(f"Flat reference skipped because {mismatch}.")
        else:
            flat_dark = dark
            if flat_dark is not None and dark_matches(flat.capture, flat_dark.capture):
                flat_dark = None
                details["messages"].append("Dark subtraction is unavailable for this flat exposure.")
            flat_values, flat_dark_clipped, _ = corrected_codes(
                flat.source, flat.capture.white_level, flat_dark
            )
            reference_support = (
                (flat.source.pixels < flat.capture.white_level)
                & ~flat_dark_clipped & (flat_values > 0)
            )
            if reference_support.sum() < 0.30 * reference_support.size:
                raise Issue("The selected flat image contains too little usable reference area.", 409)
            median = float(np.median(flat_values[reference_support]))
            reference_support &= flat_values >= median * flat_floor
            calibration_gap |= ~reference_support
            gains = np.ones(values.shape, dtype=np.float32)
            np.divide(median, flat_values, out=gains, where=reference_support)
            values *= gains
            details["flat"] = flat.capture.id
            details["flat_normalization"] = round(median, 6)
            details["flat_floor"] = flat_floor
    valid = ~saturated & ~calibration_gap
    details["sensor_pixels"] = int(valid.size)
    details["usable_pixels"] = int(valid.sum())
    details["clipped_pixels"] = int(saturated.sum())
    details["invalid_calibration_pixels"] = int(calibration_gap.sum())
    return CalibratedImage(values, valid, saturated, calibration_gap, floor_hit, details)


def project_panel(corrected: CalibratedImage, sampling: SamplingMap) -> PanelImage:
    valid = sampling.usable(corrected.valid)
    signal = sampling.samples(corrected.signal)
    signal[~valid] = np.nan
    return PanelImage(
        signal,
        valid,
        sampling.in_bounds,
        sampling.present(corrected.saturated),
        sampling.present(corrected.calibration_gap),
        sampling.present(corrected.dark_lost),
        sampling.matrix,
        corrected.details,
    )


def panel_calibration_report(panel: PanelImage) -> dict:
    count = panel.signal.size
    report = {
        **panel.details,
        "plane_pixels": count,
        "coverage": round(float(panel.any_support.sum() / count), 6),
        "usable": round(float(panel.valid.sum() / count), 6),
        "clipped": round(float(panel.clipped.sum() / count), 6),
        "invalid_calibration": round(float(panel.calibration_gap.sum() / count), 6),
        "floor_signal": round(float((panel.floor_hit & panel.valid).sum() / count), 6),
        "signal": distribution(panel.signal[panel.valid]),
    }
    if isinstance(panel.details.get("long"), dict):
        report["composed"] = True
        report["legs"] = {
            leg: {key: value for key, value in panel.details[leg].items() if key != "messages"}
            for leg in ("long", "short")
            if isinstance(panel.details.get(leg), dict)
        }
    return report


def support_reason(panel: PanelImage, x: int, y: int) -> list[str]:
    if not panel.any_support[y, x]:
        return ["outside source"]
    reasons = []
    if panel.clipped[y, x]:
        reasons.append("clipped capture")
    if panel.calibration_gap[y, x]:
        reasons.append("calibration gap")
    if panel.floor_hit[y, x] and panel.valid[y, x]:
        reasons.append("signal at floor")
    return reasons
