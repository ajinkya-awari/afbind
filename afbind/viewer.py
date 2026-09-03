"""Dependency-light boundary for sanitized cached benchmark artifacts."""

from copy import deepcopy
from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Any, Optional


class ViewerContractError(ValueError):
    """Raised when a record or request violates the cached-viewer contract."""


@dataclass(frozen=True)
class ViewerSelection:
    status: str
    reason: str
    record: Optional[Mapping[str, Any]] = None


_REQUIRED_FIELDS = ("schema_version", "run_id", "created_at", "provenance", "n", "missing_count", "reviewed")
_FORBIDDEN_PATH_FIELDS = {
    "structure_path",
    "raw_structure_path",
    "pdb_path",
    "raw_pdb_path",
    "af2_structure_path",
    "structure_file",
    "pdb_file",
}


def _validate_record(record: Any, expected_schema_version: str) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise ViewerContractError("cached record must be a mapping")
    missing = [field for field in _REQUIRED_FIELDS if field not in record]
    if missing:
        raise ViewerContractError("missing " + ", ".join(missing))
    if record["schema_version"] != expected_schema_version:
        raise ViewerContractError("schema_version does not match cached viewer contract")
    if record["reviewed"] is not True:
        raise ViewerContractError("reviewed must be exactly True for cached artifacts")
    if not isinstance(record["provenance"], Mapping) or not record["provenance"]:
        raise ViewerContractError("provenance must be a non-empty mapping")
    for field in ("n", "missing_count"):
        value = record[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ViewerContractError(f"{field} must be a non-negative integer count")
    for key in record:
        lowered = str(key).lower()
        if key in _FORBIDDEN_PATH_FIELDS or ("path" in lowered and ("structure" in lowered or "pdb" in lowered or "af2" in lowered)):
            raise ViewerContractError("raw structure paths are not allowed in cached records")
    status = record.get("status", "available")
    if status not in ("available", "unavailable"):
        raise ViewerContractError("status must be available or unavailable")
    if status == "unavailable" and not str(record.get("reason", "")).strip():
        raise ViewerContractError("unavailable records require a reason")
    if not str(record.get("record_id", "")).strip():
        raise ViewerContractError("cached record requires record_id")
    return deepcopy(dict(record))


class CachedBenchmarkViewer:
    """Select verified cached records; never predicts from ligand or UniProt requests."""

    def __init__(self, records: Sequence[Mapping[str, Any]], *, schema_version: str = "benchmark/v1"):
        if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
            raise ViewerContractError("cached records must be a sequence")
        validated = [_validate_record(record, schema_version) for record in records]
        ids = [record["record_id"] for record in validated]
        if len(ids) != len(set(ids)):
            raise ViewerContractError("cached record_id values must be unique")
        self._records = {record["record_id"]: record for record in validated}

    @property
    def records(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(deepcopy(record) for record in self._records.values())

    def select(self, *, record_id: Optional[str] = None, **request: Any) -> ViewerSelection:
        if request:
            raise ViewerContractError("viewer accepts cached record ids only; arbitrary prediction requests are not supported")
        if not isinstance(record_id, str) or not record_id:
            raise ViewerContractError("record_id is required for cached selection")
        record = self._records.get(record_id)
        if record is None:
            return ViewerSelection("unavailable", "cached_record_not_found")
        status = record.get("status", "available")
        return ViewerSelection(status, str(record.get("reason", "cached record available")), deepcopy(record))


def load_cached_records(records: Sequence[Mapping[str, Any]], *, schema_version: str = "benchmark/v1") -> CachedBenchmarkViewer:
    """Load only sanitized benchmark artifact records with provenance and counts."""

    return CachedBenchmarkViewer(records, schema_version=schema_version)
