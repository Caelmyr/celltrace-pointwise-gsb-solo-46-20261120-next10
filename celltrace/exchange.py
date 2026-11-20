from dataclasses import replace
from hashlib import sha256

from .findings import validate_source_anchor
from .geometry import transform_for
from .images import decode_source, verify_capture
from .inspection import MAX_INSPECTION_IMAGES, check_calibration, validate_merge
from .model import Inspection, Issue, mapping, sequence, text, utcnow
from .packets import PacketReader, PacketWriter
from .store import Store, new_id


EXCHANGE_SCHEMA = "celltrace-inspection-v1"
SOURCE_MIME = {"PNG": "image/png", "TIFF": "image/tiff"}


def original_name(digest: str, format: str) -> str:
    return f"originals/{digest}.{'png' if format == 'PNG' else 'tiff'}"


def create_exchange(store: Store, inspection_id: str) -> bytes:
    inspection, events = store.bundle(inspection_id)
    with PacketWriter() as packet:
        for capture in sorted(inspection.captures.values(), key=lambda frame: frame.id):
            role_path = original_name(capture.digest, capture.format)
            if role_path in packet.members:
                continue
            content = store.blobs.read(capture.digest)
            if capture.digest != sha256(content).hexdigest():
                raise Issue("An original image failed its storage digest check.", 409)
            packet.add(role_path, content, SOURCE_MIME[capture.format])
        return packet.finish({
            "schema": EXCHANGE_SCHEMA,
            "inspection": exchange_state(inspection),
            "source": {"inspection": inspection.id, "revision": inspection.revision, "state": inspection.state},
            "events": events,
            "purpose": "Working inspection transfer with original sensor bytes and selected settings",
        })


def exchange_state(inspection: Inspection) -> dict:
    state = inspection.to_dict()
    state["state"] = "open"
    state["issue"] = None
    return state


def preview_exchange(content: bytes) -> dict:
    with PacketReader(content) as package:
        inspection, original_members = parse_exchange(package)
        return {
            "module": inspection.module,
            "production_lot": inspection.lot,
            "source_revision": inspection.revision,
            "operator": inspection.operator,
            "capture_count": len(inspection.captures),
            "unique_originals": len(original_members),
            "original_bytes": sum(package.members[member].size for member in original_members),
            "findings": len(inspection.findings),
            "source": package.manifest.get("source"),
            "verification": "Manifest and references read. Original image digests and formats will be checked on import.",
        }


def parse_exchange(package: PacketReader):
    if package.manifest.get("schema") != EXCHANGE_SCHEMA:
        raise Issue("This file is not a CellTrace inspection exchange package.")
    state = mapping(package.manifest.get("inspection"), "Inspection state")
    if len(mapping(state.get("captures", {}), "Image inventory")) > MAX_INSPECTION_IMAGES:
        raise Issue("The archive contains too many captured originals.")
    inspection = Inspection.from_dict(state)
    if inspection.state != "open":
        raise Issue("Exchange packages contain editable working inspections.")
    originals = {}
    for capture in inspection.captures.values():
        name = original_name(capture.digest, capture.format)
        if name in originals:
            prior = originals[name]
            if (prior.width, prior.height, prior.bit_depth, prior.byte_size) != (
                capture.width, capture.height, capture.bit_depth, capture.byte_size
            ):
                raise Issue("Copies of one image disagree about its native format.")
        originals[name] = capture
    if set(originals) != set(package.members):
        raise Issue("The package original images do not match the inspection.")
    for name, capture in originals.items():
        member = package.members[name]
        if member.size != capture.byte_size or member.digest != capture.digest:
            raise Issue("A source image declaration disagrees with the file inventory.")
        if member.content_type != SOURCE_MIME[capture.format]:
            raise Issue("A source image declaration has an inconsistent format.")
    for finding in inspection.findings:
        validate_source_anchor(inspection, finding)
    for placement in inspection.placements.values():
        transform_for(inspection.plane, placement)
    check_calibration(inspection)
    validate_merge(inspection)
    return inspection, originals


def import_exchange(store: Store, content: bytes, label: str | None) -> Inspection:
    with PacketReader(content) as package:
        origin, originals = parse_exchange(package)
        for name, capture in originals.items():
            raw = package.read(name)
            verify_capture(decode_source(raw), capture)
        for name in originals:
            store.blobs.put(package.read(name))
        events = sequence(package.manifest.get("events", []), "Export event history", 0, 80)
        provenance = mapping(package.manifest.get("source", {}), "Source declaration")
    label = text(label, "Imported module serial", 80) if label else origin.module
    current = utcnow()
    restored = replace(origin, id=new_id(), module=label, revision=0, created_at=current, updated_at=current)
    return store.add(restored, "inspection restored", {
        "source": origin.id,
        "source_revision": origin.revision,
        "source_state": provenance.get("state", "open"),
        "source_events": len(events),
        "original_count": len(originals),
        "package_sha256": sha256(content).hexdigest(),
    })
