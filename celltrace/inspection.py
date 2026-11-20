from dataclasses import asdict, replace

from .calibration import dark_matches, reference_matches
from .demo import DEMO_NOTICE, make_demo
from .findings import get_finding, list_findings, make_finding
from .geometry import default_placement, transform_for
from .images import SensorImage, decode_source, safe_name
from .model import (
    AnalysisSettings, CAPTURE_ROLES, Calibration, Capture, ExposureMerge, Inspection, Issue,
    MERGE_READINGS, Placement, Plane, SENSOR_ROLES, ViewSettings, boolean, choice, file_id,
    integer, mapping, real, text, utcnow,
)
from .store import Store, new_id, summary


MAX_INSPECTION_IMAGES = 18


def new_inspection(module: str, lot: str, operator: str, notes: str) -> Inspection:
    created = utcnow()
    return Inspection(new_id(), module, lot, operator, notes, created, created)


def edit_metadata(raw: dict) -> dict:
    return {
        "module": text(raw.get("module"), "Module serial", 80),
        "lot": text(raw.get("lot", ""), "Production lot", 80, allow_blank=True),
        "operator": text(raw.get("operator", ""), "Operator", 80, allow_blank=True),
        "notes": text(raw.get("notes", ""), "Inspection notes", 2000, allow_blank=True),
    }


def capture_metadata(role: str, name: str, source: SensorImage, raw: dict, *, synthetic=False) -> Capture:
    choice(role, "Capture role", CAPTURE_ROLES)
    current = raw.get("current_a")
    if current in (None, ""):
        current = None
    elif role in SENSOR_ROLES:
        current = real(current, "Drive current (A)", 0.001, 100)
    else:
        raise Issue("The drive current is only used for low- and high-current captures.")
    fallback = "image.tif" if source.format == "TIFF" else "image.png"
    return Capture(
        new_id(), role, safe_name(name, fallback),
        source.format, source.digest, source.byte_size,
        source.width, source.height, source.bit_depth,
        integer(raw.get("white_level") or source.maximum, "Sensor white level", 16, source.maximum),
        real(raw.get("exposure_ms", 100), "Exposure (ms)", 0.01, 600000),
        current,
        utcnow(),
        synthetic,
    )


def inspection_payload(inspection: Inspection) -> dict:
    output = inspection.to_dict()
    output["findings"] = list_findings(inspection)
    output["summary"] = summary(inspection)
    output["placements"] = {}
    for capture in inspection.captures.values():
        if capture.role in SENSOR_ROLES:
            configured = inspection.placements.get(capture.id)
            value = configured or default_placement(capture.width, capture.height)
            output["placements"][capture.id] = {**asdict(value), "explicit": configured is not None}
    return output


class Inspections:
    def __init__(self, store: Store):
        self.store = store

    def create(self, raw: dict) -> Inspection:
        item = new_inspection(**edit_metadata(raw))
        return self.store.add(item, "created", {"module": item.module})

    def demo(self, variant: str) -> Inspection:
        choice(variant, "Synthetic sample", frozenset(("standard", "stress")))
        plane, images = make_demo(variant)
        module = "EL-24017" if variant == "standard" else "EL-24029"
        item = new_inspection(module, "DEMO-EL04", "Training", DEMO_NOTICE)
        item.plane = plane
        item.analysis = replace(item.analysis, busbars=(0.28, 0.72))
        high_images = []
        dark_images = []
        flat_images = []
        for frame in images:
            source = decode_source(frame.content)
            capture = capture_metadata(frame.role, frame.name, source, {
                "exposure_ms": frame.exposure_ms, "current_a": frame.current_a,
            }, synthetic=True)
            self.store.blobs.put(frame.content)
            item.captures[capture.id] = capture
            item.active[capture.role] = capture.id
            if frame.role == "high":
                high_images.append(capture)
            elif frame.role == "dark":
                dark_images.append(capture)
            elif frame.role == "flat":
                flat_images.append(capture)
            if frame.corners is not None:
                item.placements[capture.id] = Placement(frame.corners, frame.orientation)
        long_leg = max(high_images, key=lambda frame: frame.exposure_ms)
        short_leg = min(high_images, key=lambda frame: frame.exposure_ms)
        item.active["high"] = long_leg.id
        if short_leg.id != long_leg.id:
            long_dark = max(dark_images, key=lambda frame: frame.exposure_ms)
            short_dark = min(dark_images, key=lambda frame: frame.exposure_ms)
            long_flat = max(flat_images, key=lambda frame: frame.exposure_ms)
            short_flat = min(flat_images, key=lambda frame: frame.exposure_ms)
            item.active["dark"] = long_dark.id
            item.active["flat"] = long_flat.id
            item.calibration = Calibration(long_dark.id, long_flat.id)
            item.merge = ExposureMerge(
                enabled=True, short_id=short_leg.id,
                short_calibration=Calibration(short_dark.id, short_flat.id),
            )
        else:
            item.calibration = Calibration(item.active["dark"], item.active["flat"])
        return self.store.add(item, "synthetic fixture loaded", {"sample": variant, "images": len(images)})

    def checked(self, identifier: str, expected, *, editable=True) -> Inspection:
        expected = integer(expected, "Revision", 0, 10**9)
        item = self.store.get(identifier)
        if item.revision != expected:
            raise Issue("The inspection changed. Reload before applying more changes.", 409)
        if editable and item.state == "issued":
            raise Issue("This inspection is issued. Make a working copy to continue review.", 409)
        return item

    def change(self, identifier: str, expected, operation: str, raw: dict) -> Inspection:
        item = self.checked(identifier, expected)
        if operation == "metadata":
            body = edit_metadata(raw)
            item = replace(item, **body)
            message, details = "module details updated", {"module": item.module}
        elif operation == "plane":
            item.plane = Plane.from_dict(raw)
            for placement in item.placements.values():
                transform_for(item.plane, placement)
            message, details = "panel plane updated", asdict(item.plane)
        elif operation == "placement":
            capture = item.capture(raw.get("capture"))
            if capture.role not in SENSOR_ROLES:
                raise Issue("Only an EL frame can receive a panel placement.")
            placement = Placement.from_dict(raw, capture.width, capture.height)
            transform_for(item.plane, placement)
            item.placements[capture.id] = placement
            message, details = "image placement updated", {"capture": capture.id, **asdict(placement)}
        elif operation == "calibration":
            item.calibration = Calibration.from_dict(raw)
            details = check_calibration(item)
            message = "references selected"
        elif operation == "merge":
            item.merge = merge_settings(item, raw)
            details = merge_details(item)
            message = "high-current exposure reading updated"
        elif operation == "display":
            item.display = ViewSettings.from_dict(raw)
            message, details = "display levels updated", asdict(item.display)
        elif operation == "analysis":
            item.analysis = AnalysisSettings.from_dict(raw)
            message, details = "measurement thresholds updated", asdict(item.analysis)
        elif operation == "selection":
            capture = item.capture(raw.get("capture"))
            if (
                capture.role == "high" and item.merge.enabled
                and capture.id == item.merge.short_id
            ):
                raise Issue(
                    "This image is the paired short exposure. Change or close the exposure pair "
                    "before making it the long exposure.", 409,
                )
            item.active[capture.role] = capture.id
            message, details = "capture selected", {"role": capture.role, "capture": capture.id}
        elif operation == "capture-metadata":
            capture = item.capture(raw.get("capture"))
            if capture_applies_to_findings(item, capture.id):
                raise Issue("Review evidence uses this capture. Upload a new capture entry with corrected metadata.", 409)
            current = raw.get("current_a")
            current = real(current, "Drive current", 0.001, 100) if current not in ("", None) else None
            if capture.role not in SENSOR_ROLES and current is not None:
                raise Issue("Only EL captures can carry a drive current.")
            exposure = real(raw.get("exposure_ms"), "Exposure (ms)", 0.01, 600000)
            _check_pair_exposure(item, capture.id, exposure)
            item.captures[capture.id] = replace(
                capture,
                exposure_ms=exposure,
                current_a=current,
                white_level=integer(raw.get("white_level"), "Sensor white level", 16, (1 << capture.bit_depth) - 1),
            )
            message, details = "capture metadata corrected", {"capture": capture.id}
        else:
            raise Issue("This inspection operation is not available.", 404)
        return self.store.save(item, item.revision, message, details)

    def upload(self, identifier: str, expected, role: str, name: str, content: bytes, raw: dict) -> Inspection:
        item = self.checked(identifier, expected)
        if len(item.captures) >= MAX_INSPECTION_IMAGES:
            raise Issue(f"An inspection can hold up to {MAX_INSPECTION_IMAGES} originals.", 409)
        source = decode_source(content)
        capture = capture_metadata(role, name, source, raw)
        if any(
            existing.role == capture.role and existing.digest == capture.digest
            and existing.exposure_ms == capture.exposure_ms and existing.current_a == capture.current_a
            and existing.white_level == capture.white_level
            for existing in item.captures.values()
        ):
            raise Issue("That image and its acquisition settings are already present in this role.", 409)
        self.store.blobs.put(content)
        item.captures[capture.id] = capture
        if raw.get("select", "true") == "true" or role not in item.active:
            item.active[role] = capture.id
        return self.store.save(item, item.revision, "original imported", {
            "capture": capture.id, "role": capture.role, "digest": capture.digest,
            "size": [capture.width, capture.height], "format": capture.format,
        })

    def remove_capture(self, identifier: str, expected, capture_id: str) -> Inspection:
        item = self.checked(identifier, expected)
        capture = item.capture(capture_id)
        if capture_applies_to_findings(item, capture.id):
            raise Issue("Review evidence uses this capture. Keep its original for traceability.", 409)
        merge = item.merge
        if capture.id == merge.short_id:
            merge = replace(merge, enabled=False, short_id=None)
        short_calibration = replace(
            merge.short_calibration,
            dark_id=None if capture.id == merge.short_calibration.dark_id else merge.short_calibration.dark_id,
            flat_id=None if capture.id == merge.short_calibration.flat_id else merge.short_calibration.flat_id,
        )
        item.merge = replace(merge, short_calibration=short_calibration)
        if capture.id in (item.calibration.dark_id, item.calibration.flat_id):
            raise Issue("Clear the selected calibration reference before removing this capture.", 409)
        del item.captures[capture.id]
        item.placements.pop(capture.id, None)
        if item.active.get(capture.role) == capture.id:
            candidates = [entry for entry in item.captures.values() if entry.role == capture.role]
            if candidates:
                replacement = max(candidates, key=lambda entry: (entry.captured_at, entry.id))
                item.active[capture.role] = replacement.id
            else:
                del item.active[capture.role]
        return self.store.save(item, item.revision, "original detached", {"capture": capture.id})

    def clone(self, source_id: str, raw: dict):
        source = self.store.get(source_id)
        preview = f"{source.module} review copy"[:80]
        requested_module = text(raw.get("module", preview), "Module serial", 80)
        cloned = replace(
            source,
            id=new_id(), state="open", issue=None, module=requested_module,
            revision=0, created_at=utcnow(), updated_at=utcnow(),
        )
        return self.store.add(cloned, "working copy created", {
            "source": source.id, "source_revision": source.revision, "source_state": source.state,
        })

    def finding(self, identifier: str, expected, raw: dict, finding_id=None) -> Inspection:
        item = self.checked(identifier, expected)
        entry = make_finding(item, raw, finding_id)
        if finding_id is None:
            item.findings.append(entry)
        else:
            item.findings = [entry if value.id == finding_id else value for value in item.findings]
        return self.store.save(item, item.revision, "finding updated" if finding_id else "finding recorded", {
            "finding": entry.id, "status": entry.status, "category": entry.category, "cell": entry.cell,
        })

    def remove_finding(self, identifier: str, expected, finding_id) -> Inspection:
        item = self.checked(identifier, expected)
        removed = get_finding(item, finding_id)
        item.findings = [entry for entry in item.findings if entry.id != removed.id]
        return self.store.save(item, item.revision, "finding withdrawn", {"finding": removed.id})


def _check_pair_exposure(item: Inspection, capture_id: str, exposure: float):
    """Keep an exposure edit from silently breaking the pair ordering."""
    merge = item.merge
    if not merge.enabled or merge.short_id is None:
        return
    long_id = item.active.get("high")
    if capture_id == merge.short_id and long_id in item.captures:
        if exposure >= item.captures[long_id].exposure_ms:
            raise Issue("The paired short exposure must stay shorter than the long exposure.", 409)
    elif capture_id == long_id and merge.short_id in item.captures:
        if exposure <= item.captures[merge.short_id].exposure_ms:
            raise Issue("The long exposure must stay longer than the paired short exposure.", 409)


def merge_settings(item: Inspection, raw: dict) -> ExposureMerge:
    """Validate the paired-exposure settings against the current inspection."""
    enabled = boolean(raw.get("enabled", item.merge.enabled), "Exposure merge switch")
    readings = choice(raw.get("readings", item.merge.readings), "High-current read mode",
                      MERGE_READINGS)
    short_raw = raw.get("short_id", item.merge.short_id)
    if short_raw in ("", None):
        short_id = None
    else:
        short_id = file_id(short_raw, "Short exposure image ID")
    calibration_raw = raw.get("short_calibration")
    if calibration_raw is None:
        calibration = item.merge.short_calibration
    elif isinstance(calibration_raw, Calibration):
        calibration = calibration_raw
    else:
        calibration = Calibration.from_dict(calibration_raw)
    for role in ("dark", "flat"):
        reference_id = getattr(calibration, f"{role}_id")
        if reference_id is None:
            continue
        reference = item.captures.get(file_id(reference_id, f"{role.title()} reference image ID"))
        if reference is None or reference.role != role:
            raise Issue(f"The short-exposure {role} reference must be an original in the {role} role.")
    if enabled:
        if "high" not in item.active:
            raise Issue("Select a long high-current image before pairing exposures.")
        long_capture = item.selected("high")
        if short_id is None:
            raise Issue("Choose the short high-current exposure to pair.")
        short = item.captures.get(short_id)
        if short is None or short.role != "high":
            raise Issue("The second exposure must be a high-current image in this inspection.")
        if short.id == long_capture.id:
            raise Issue("Choose a second high-current image, not the selected one.")
        if short.exposure_ms >= long_capture.exposure_ms:
            raise Issue("The paired exposure must be shorter than the selected long exposure.")
    if not enabled:
        return ExposureMerge(enabled=False, short_id=short_id, short_calibration=calibration,
                             agreement=item.merge.agreement,
                             minimum_agreement_pixels=item.merge.minimum_agreement_pixels,
                             readings="composite")
    return ExposureMerge(
        True, short_id, calibration,
        real(raw.get("agreement", item.merge.agreement), "Merge agreement tolerance", 0.01, 0.6),
        integer(raw.get("minimum_agreement_pixels", item.merge.minimum_agreement_pixels),
                "Merge agreement sample size", 4, 4000),
        readings,
    )


def merge_details(item: Inspection) -> dict:
    leg = item.merge.leg_calibration(item, item.captures[item.merge.short_id]) if item.merge.short_id else None
    return {
        "enabled": item.merge.enabled,
        "short": item.merge.short_id,
        "readings": item.merge.readings,
        "short_calibration": asdict(leg) if leg else None,
    }


def validate_merge(inspection: Inspection):
    """Check a restored exposure pair against the images it references."""
    merge = inspection.merge
    if not merge.enabled or merge.short_id is None:
        return
    if "high" not in inspection.active:
        raise Issue("A paired exposure needs a selected long high-current image.")
    long_capture = inspection.selected("high")
    short = inspection.captures.get(merge.short_id)
    if short is None or short.role != "high":
        raise Issue("A paired exposure does not refer to a high-current image.")
    if short.id == long_capture.id or short.exposure_ms >= long_capture.exposure_ms:
        raise Issue("A paired exposure must be shorter and distinct from the selected long exposure.")
    for role in ("dark", "flat"):
        reference_id = getattr(merge.short_calibration, f"{role}_id")
        if reference_id is None:
            continue
        reference = inspection.captures.get(reference_id)
        if reference is None or reference.role != role:
            raise Issue(f"A paired exposure refers to an invalid {role} reference.")


def capture_applies_to_findings(inspection: Inspection, identifier: str) -> bool:
    for finding in inspection.findings:
        if finding.anchor.get("image") == identifier:
            return True
        reference = finding.anchor.get("calibration", {})
        if identifier in (reference.get("dark_id"), reference.get("flat_id")):
            return True
    return False


def check_calibration(inspection: Inspection) -> dict:
    notes = []
    pairs = [("long", inspection.calibration, inspection.active.get("high"))]
    if inspection.merge.short_id is not None:
        pairs.append(("short", inspection.merge.short_calibration, inspection.merge.short_id))
    for label, calibration, high_id in pairs:
        for role in ("dark", "flat"):
            key = getattr(calibration, f"{role}_id")
            if key is None:
                continue
            capture = inspection.capture(key)
            if capture.role != role:
                raise Issue(f"Select an original in the {role} reference role.")
            targets = [high_id] if label == "short" else [inspection.active.get(role_name) for role_name in ("high", "low")]
            for target_id in targets:
                active = inspection.captures.get(target_id)
                if active is None:
                    continue
                mismatch = dark_matches if role == "dark" else reference_matches
                problem = mismatch(active, capture)
                if problem:
                    notes.append({"exposure": label, "role": active.role, "reference": role, "reason": problem})
    return {
        "references": asdict(inspection.calibration),
        "short_references": asdict(inspection.merge.short_calibration),
        "compatibility": notes,
    }
