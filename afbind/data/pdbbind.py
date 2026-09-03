"""Offline parsing contracts for affinity rows.

The parser deliberately accepts a small, documented synthetic row shape while also
handling mappings and common delimiter choices.  It never reaches out to a data
source; callers provide the row or local path.
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


SUPPORTED_UNITS = {
    "M": 1.0,
    "mM": 1e-3,
    "uM": 1e-6,
    "µM": 1e-6,
    "μM": 1e-6,
    "nM": 1e-9,
    "pM": 1e-12,
}
SUPPORTED_MEASUREMENTS = {"ki": "Ki", "kd": "Kd"}
EXCLUDED_MEASUREMENTS = {"ic50", "ec50", "kact", "ka"}
_CENSOR_RE = re.compile(r"^\s*(?:<|>|≤|≥)")
_MISSING = object()


@dataclass(frozen=True, slots=True)
class AffinityRecord:
    """A validated Ki/Kd observation with conversion provenance."""

    record_id: str
    target_id: str
    ligand_id: str
    ligand_scaffold: str
    measurement_type: str
    raw_value: str
    raw_unit: str
    molar_concentration: float
    pki: float

    @property
    def pdb_id(self) -> str:
        return self.record_id

    @property
    def raw_measurement(self) -> str:
        return f"{self.raw_value} {self.raw_unit}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "target_id": self.target_id,
            "ligand_id": self.ligand_id,
            "ligand_scaffold": self.ligand_scaffold,
            "measurement_type": self.measurement_type,
            "raw_value": self.raw_value,
            "raw_unit": self.raw_unit,
            "molar_concentration": self.molar_concentration,
            "pki": self.pki,
        }


@dataclass(frozen=True, slots=True)
class ExcludedRow:
    """A source row excluded without losing its evidence or reason."""

    raw_line: str
    reason: str
    record_id: Optional[str] = None

    def to_dict(self) -> dict[str, Optional[str]]:
        return {
            "record_id": self.record_id,
            "raw_line": self.raw_line,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class ParseOutcome:
    record: Optional[AffinityRecord]
    exclusion: Optional[ExcludedRow]


@dataclass(frozen=True, slots=True)
class AffinityTable:
    """Stable in-memory table containing accepted records and exclusions."""

    records: tuple[AffinityRecord, ...]
    excluded: tuple[ExcludedRow, ...] = ()

    @classmethod
    def from_lines(cls, lines: Iterable[str]) -> "AffinityTable":
        records: list[AffinityRecord] = []
        excluded: list[ExcludedRow] = []
        for line in lines:
            outcome = parse_index_row(line)
            if outcome.record is not None:
                records.append(outcome.record)
            elif outcome.exclusion is not None:
                excluded.append(outcome.exclusion)
        return cls(tuple(records), tuple(excluded))

    @classmethod
    def from_path(cls, path: str | Path) -> "AffinityTable":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_lines(handle)

    @property
    def exclusions(self) -> tuple[ExcludedRow, ...]:
        return self.excluded

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "records": [record.to_dict() for record in self.records],
            "excluded": [item.to_dict() for item in self.excluded],
        }


def _value(mapping: Mapping[str, Any], *names: str, default: Any = _MISSING) -> Any:
    lowered = {str(key).strip().lower(): value for key, value in mapping.items()}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return default


def _mapping_row(row: Mapping[str, Any]) -> tuple[list[str], Optional[str]]:
    record_id = _value(row, "record_id", "pdb_id", "pdb", "id", "code")
    target_id = _value(row, "target_id", "target", "protein_id", "protein")
    ligand_id = _value(row, "ligand_id", "ligand", "compound_id", "compound")
    scaffold = _value(row, "ligand_scaffold", "scaffold", "scaffold_id")
    measurement = _value(row, "measurement_type", "affinity_type", "measurement", "measure", "type")
    raw_value = _value(row, "raw_value", "value", "affinity", "concentration")
    raw_unit = _value(row, "raw_unit", "unit", "units")
    values = [record_id, target_id, ligand_id, scaffold, measurement, raw_value, raw_unit]
    if scaffold is _MISSING and ligand_id is not _MISSING:
        values[3] = ligand_id
    if any(value is _MISSING or value is None or not str(value).strip() for value in values):
        return [], "malformed_row"
    return [str(value).strip() for value in values], None


def _split_line(line: str) -> tuple[list[str], Optional[str]]:
    stripped = line.strip()
    if not stripped:
        return [], "blank_row"
    if stripped.startswith("#"):
        return [], "comment_row"
    delimiter = "|" if "|" in stripped else "\t" if "\t" in stripped else "," if "," in stripped else None
    if delimiter is not None:
        fields = next(csv.reader([stripped], delimiter=delimiter))
    else:
        fields = stripped.split()
    fields = [field.strip() for field in fields]
    if len(fields) == 6:
        fields.insert(3, fields[2])
    if len(fields) != 7 or any(not field for field in fields):
        return [], "malformed_row"
    return fields, None


def _normalize_unit(unit: str) -> Optional[str]:
    candidate = unit.strip()
    for supported in SUPPORTED_UNITS:
        if candidate == supported or candidate.lower() == supported.lower():
            return supported
    return None


def parse_index_row(row: str | Mapping[str, Any] | Sequence[Any]) -> ParseOutcome:
    """Parse one row and retain an explicit exclusion when validation fails."""

    raw_line = row if isinstance(row, str) else repr(row)
    if isinstance(row, Mapping):
        fields, reason = _mapping_row(row)
    elif isinstance(row, str):
        fields, reason = _split_line(row)
    else:
        fields = [str(value).strip() for value in row]
        if len(fields) == 6:
            fields.insert(3, fields[2])
        reason = "malformed_row" if len(fields) != 7 or any(not field for field in fields) else None
    if reason is not None:
        return ParseOutcome(None, ExcludedRow(raw_line, reason))

    record_id, target_id, ligand_id, scaffold, measurement, raw_value, raw_unit = fields
    normalized_measurement = SUPPORTED_MEASUREMENTS.get(measurement.lower())
    if normalized_measurement is None:
        return ParseOutcome(None, ExcludedRow(raw_line, "unsupported_measurement_type", record_id))
    if _CENSOR_RE.match(raw_value):
        return ParseOutcome(None, ExcludedRow(raw_line, "censored_value", record_id))
    unit = _normalize_unit(raw_unit)
    if unit is None:
        return ParseOutcome(None, ExcludedRow(raw_line, "unsupported_unit", record_id))
    try:
        numeric_value = float(raw_value)
    except (TypeError, ValueError):
        return ParseOutcome(None, ExcludedRow(raw_line, "invalid_numeric_value", record_id))
    if not math.isfinite(numeric_value) or numeric_value <= 0:
        return ParseOutcome(None, ExcludedRow(raw_line, "invalid_numeric_value", record_id))
    molar = numeric_value * SUPPORTED_UNITS[unit]
    return ParseOutcome(
        AffinityRecord(
            record_id=record_id,
            target_id=target_id,
            ligand_id=ligand_id,
            ligand_scaffold=scaffold,
            measurement_type=normalized_measurement,
            raw_value=raw_value,
            raw_unit=raw_unit,
            molar_concentration=molar,
            pki=-math.log10(molar),
        ),
        None,
    )


def parse_index_line(line: str | Mapping[str, Any] | Sequence[Any]) -> Optional[AffinityRecord]:
    """Return a validated record, or ``None`` for an excluded row."""

    return parse_index_row(line).record


def load_index(path: str | Path) -> AffinityTable:
    """Load a local index file without accessing external data."""

    return AffinityTable.from_path(path)
