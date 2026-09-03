"""Evidence-first scans for retained HETATM records."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class HeteroReport(Mapping[str, Any]):
    num_complexes_scanned: int
    num_with_water: int
    num_with_metal: int
    metal_counts: Mapping[str, int]
    water_counts: Mapping[str, int]
    selenium_records_excluded: int
    source_file_manifest: tuple[str, ...]

    def __getitem__(self, key: str) -> Any:
        return {
            "num_complexes_scanned": self.num_complexes_scanned,
            "num_with_water": self.num_with_water,
            "num_with_metal": self.num_with_metal,
            "metal_counts": self.metal_counts,
            "water_counts": self.water_counts,
            "selenium_records_excluded": self.selenium_records_excluded,
            "source_file_manifest": self.source_file_manifest,
        }[key]

    def __iter__(self) -> Iterator[str]:
        return iter(("num_complexes_scanned", "num_with_water", "num_with_metal", "metal_counts", "water_counts", "selenium_records_excluded", "source_file_manifest"))

    def __len__(self) -> int:
        return 7


_WATER_NAMES = {"HOH", "WAT", "H2O", "DOD"}
_METAL_NAMES = {
    "AL", "BA", "CA", "CD", "CO", "CR", "CU", "FE", "HG", "K", "LI", "MG", "MN", "NA", "NI", "PB", "SR", "V", "ZN"
}


def _normalized(value: Any) -> str:
    return str(value or "").strip().upper()


def scan_hetero(structure: Any) -> HeteroReport:
    """Count retained waters/metals and expose raw evidence before classification."""

    if isinstance(structure, Mapping):
        records = structure.get("records", structure.get("atoms", ()))
        source = structure.get("source_file")
        source_manifest = structure.get("source_file_manifest")
        num_complexes = structure.get("num_complexes_scanned", 1)
    else:
        records = structure if isinstance(structure, (list, tuple)) else ()
        source = None
        source_manifest = None
        num_complexes = 1
    if not isinstance(records, (list, tuple)):
        records = ()
    try:
        num_complexes = int(num_complexes)
    except (TypeError, ValueError):
        num_complexes = 1

    water_counts: Counter[str] = Counter()
    metal_counts: Counter[str] = Counter()
    selenium_records_excluded = 0
    has_water = False
    has_metal = False
    for record in records:
        if not isinstance(record, Mapping) or _normalized(record.get("record_name", record.get("record"))) != "HETATM":
            continue
        residue_name = _normalized(record.get("residue_name", record.get("resname")))
        element = _normalized(record.get("element"))
        if residue_name in _WATER_NAMES:
            water_counts[residue_name] += 1
            has_water = True
        if residue_name == "SE" or element == "SE":
            selenium_records_excluded += 1
            continue
        metal_name = element if element in _METAL_NAMES else residue_name if residue_name in _METAL_NAMES else ""
        if metal_name:
            metal_counts[metal_name] += 1
            has_metal = True

    if source_manifest is not None and isinstance(source_manifest, (list, tuple, set, frozenset)):
        manifest = tuple(str(path) for path in source_manifest)
    elif source is not None:
        manifest = (str(source),)
    else:
        manifest = ()
    return HeteroReport(num_complexes, int(has_water), int(has_metal), dict(sorted(metal_counts.items())), dict(sorted(water_counts.items())), selenium_records_excluded, manifest)


def classify_water(report: HeteroReport) -> bool:
    """Classify only after the evidence report has been produced."""

    return report.num_with_water > 0


def classify_metal(report: HeteroReport) -> bool:
    """Classify only after the evidence report has been produced."""

    return report.num_with_metal > 0
