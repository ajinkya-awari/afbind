"""Offline handling for prediction records returned by an AF2 API."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PredictionRecord(Mapping[str, Any]):
    """A selected API record, including an explicit non-available outcome."""

    status: str
    pdb_url: str | None
    reason: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return {
            "status": self.status,
            "pdb_url": self.pdb_url,
            "pdbUrl": self.pdb_url,
            "reason": self.reason,
            "metadata": self.metadata,
        }[key]

    def __iter__(self) -> Iterator[str]:
        return iter(("status", "pdb_url", "reason", "metadata"))

    def __len__(self) -> int:
        return 4


def _result(status: str, reason: str, metadata: Mapping[str, Any] | None = None, pdb_url: str | None = None) -> PredictionRecord:
    return PredictionRecord(status, pdb_url, reason, dict(metadata or {}))


def _records_from_payload(payload: Any) -> tuple[list[Mapping[str, Any]] | None, Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        response_metadata = dict(payload)
        status = payload.get("status")
        status_code = payload.get("status_code")
        if status == 404 or status_code == 404 or str(status).lower() in {"not_found", "404"}:
            return [], response_metadata
        for key in ("predictions", "results", "data"):
            value = payload.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                if any(not isinstance(item, Mapping) for item in value):
                    return None, response_metadata
                return list(value), response_metadata
        if "pdbUrl" in payload:
            return [payload], response_metadata
        return None, response_metadata
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        if any(not isinstance(item, Mapping) for item in payload):
            return None, {}
        return list(payload), {}
    return None, {}


def select_prediction_record(api_payload: Any) -> PredictionRecord:
    """Select the first API record that supplies a usable, returned ``pdbUrl``.

    The function never derives a URL from an accession, model label, or version.
    """

    records, response_metadata = _records_from_payload(api_payload)
    if records is None:
        return _result("invalid", "invalid_api_payload", response_metadata)
    if not records:
        if response_metadata.get("status") == 404 or str(response_metadata.get("status")).lower() in {"not_found", "404"}:
            return _result("not_found", "api_not_found", response_metadata)
        return _result("not_found", "no_prediction_records", response_metadata)
    for record in records:
        pdb_url = record.get("pdbUrl")
        if isinstance(pdb_url, str) and pdb_url.strip():
            metadata = dict(response_metadata)
            metadata.update(record)
            return _result("available", "selected_api_record", metadata, pdb_url)
    metadata = dict(response_metadata)
    metadata["records"] = tuple(dict(record) for record in records)
    return _result("unavailable", "pdb_url_unavailable", metadata)
