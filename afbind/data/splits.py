"""Deterministic target/scaffold grouped split manifests."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


_GROUP_ALIASES = {
    "target": "target_id",
    "protein": "target_id",
    "scaffold": "ligand_scaffold",
    "ligand_scaffold": "ligand_scaffold",
}
_SPLITS = ("train", "validation", "test")


@dataclass(frozen=True, slots=True)
class SplitManifest:
    schema_version: str
    seed: int
    group_keys: tuple[str, ...]
    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    counts: dict[str, int]
    excluded_ids: tuple[str, ...] = ()
    exclusion_reasons: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if self.exclusion_reasons is None:
            object.__setattr__(self, "exclusion_reasons", {})

    @property
    def split_ids(self) -> dict[str, tuple[str, ...]]:
        return {
            "train": self.train_ids,
            "validation": self.validation_ids,
            "test": self.test_ids,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "seed": self.seed,
            "group_keys": list(self.group_keys),
            "train_ids": list(self.train_ids),
            "validation_ids": list(self.validation_ids),
            "test_ids": list(self.test_ids),
            "counts": dict(self.counts),
            "excluded_ids": list(self.excluded_ids),
            "exclusion_reasons": dict(self.exclusion_reasons or {}),
        }


def _records_and_exclusions(records: Any, excluded: Any) -> tuple[list[Any], dict[str, str]]:
    if hasattr(records, "records"):
        source_records = list(records.records)
        reasons = {
            str(item.record_id): str(item.reason)
            for item in getattr(records, "excluded", getattr(records, "exclusions", ()))
            if item.record_id
        }
    else:
        source_records = list(records)
        reasons = {}
    if excluded is not None:
        if isinstance(excluded, Mapping):
            reasons.update({str(key): str(value) for key, value in excluded.items()})
        else:
            for item in excluded:
                if hasattr(item, "record_id") and item.record_id:
                    reasons[str(item.record_id)] = str(item.reason)
    return source_records, reasons


def _keys_for(group_by: str | Sequence[str] | None) -> tuple[str, ...]:
    if group_by is None:
        return ("target_id", "ligand_scaffold")
    if isinstance(group_by, str):
        group_by = (group_by,)
    return tuple(_GROUP_ALIASES.get(key, key) for key in group_by)


def _group_value(record: Any, key: str) -> str | None:
    value = getattr(record, key, None)
    if value is None and isinstance(record, Mapping):
        value = record.get(key)
    value = "" if value is None else str(value).strip()
    return value or None


def _record_id(record: Any) -> str:
    if hasattr(record, "record_id"):
        return str(record.record_id)
    return str(record.get("record_id"))


def _component_groups(records: list[Any], keys: tuple[str, ...]) -> dict[str, list[Any]]:
    """Make connected components when both target and scaffold are required."""

    if len(keys) == 1:
        groups: dict[str, list[Any]] = {}
        for record in records:
            groups.setdefault(_group_value(record, keys[0]), []).append(record)
        return groups

    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    record_groups: list[tuple[Any, tuple[str, ...]]] = []
    for record in records:
        values = tuple(_group_value(record, key) for key in keys)
        record_groups.append((record, values))
        for value in values:
            if value is not None:
                find(f"{keys[0]}={value}")
        present = [f"{key}={value}" for key, value in zip(keys, values) if value is not None]
        for left, right in zip(present, present[1:]):
            union(left, right)
    groups: dict[str, list[Any]] = {}
    for record, values in record_groups:
        present = [f"{key}={value}" for key, value in zip(keys, values) if value is not None]
        root = find(present[0]) if present else "missing"
        groups.setdefault(root, []).append(record)
    return groups


def _assign_groups(groups: dict[str, list[Any]], seed: int, ratios: Sequence[float]) -> dict[str, list[Any]]:
    order = sorted(groups)
    random.Random(seed).shuffle(order)
    targets = [sum(len(groups[key]) for key in order) * ratio / sum(ratios) for ratio in ratios]
    assigned: list[list[Any]] = [[], [], []]
    for group_key in order:
        remaining = [targets[index] - len(assigned[index]) for index in range(3)]
        index = max(range(3), key=lambda candidate: (remaining[candidate], -candidate))
        assigned[index].extend(groups[group_key])
    return {split: assigned[index] for index, split in enumerate(_SPLITS)}


def make_grouped_splits(
    records: Iterable[Any],
    seed: int,
    ratios: Sequence[float] = (0.8, 0.1, 0.1),
    group_by: str | Sequence[str] | None = None,
    excluded: Mapping[str, str] | Iterable[Any] | None = None,
) -> SplitManifest:
    """Create a deterministic leakage-safe manifest for one grouping policy.

    With no ``group_by`` argument, target and scaffold are both applicable and are
    assigned as connected components, preventing either group from crossing splits.
    """

    if len(ratios) != 3 or any(ratio < 0 for ratio in ratios) or sum(ratios) <= 0:
        raise ValueError("ratios must contain three non-negative values with a positive total")
    source_records, exclusion_reasons = _records_and_exclusions(records, excluded)
    ids = [_record_id(record) for record in source_records]
    if len(ids) != len(set(ids)):
        raise ValueError("record_id values must be unique")
    keys = _keys_for(group_by)
    applicable: list[Any] = []
    for record, record_id in zip(source_records, ids):
        if all(_group_value(record, key) is not None for key in keys):
            applicable.append(record)
        else:
            exclusion_reasons.setdefault(record_id, "missing_group_key")
    groups = _component_groups(applicable, keys)
    assignments = _assign_groups(groups, seed, ratios)
    ids_by_split = {
        split: tuple(_record_id(record) for record in assignments[split])
        for split in _SPLITS
    }
    excluded_ids = tuple(sorted(exclusion_reasons))
    counts = {
        "total": len(source_records),
        "train": len(ids_by_split["train"]),
        "validation": len(ids_by_split["validation"]),
        "test": len(ids_by_split["test"]),
        "excluded": len(excluded_ids),
    }
    return SplitManifest(
        schema_version="1.0",
        seed=int(seed),
        group_keys=keys,
        train_ids=ids_by_split["train"],
        validation_ids=ids_by_split["validation"],
        test_ids=ids_by_split["test"],
        counts=counts,
        excluded_ids=excluded_ids,
        exclusion_reasons={key: exclusion_reasons[key] for key in excluded_ids},
    )


def make_grouped_split_manifests(
    records: Iterable[Any],
    seed: int,
    ratios: Sequence[float] = (0.8, 0.1, 0.1),
    excluded: Mapping[str, str] | Iterable[Any] | None = None,
) -> dict[str, SplitManifest]:
    """Return independent target-grouped and scaffold-grouped manifests."""

    materialized = records if hasattr(records, "records") else list(records)
    return {
        "target": make_grouped_splits(materialized, seed, ratios, "target", excluded),
        "ligand_scaffold": make_grouped_splits(materialized, seed, ratios, "ligand_scaffold", excluded),
    }


make_target_scaffold_splits = make_grouped_split_manifests


def make_target_grouped_splits(
    records: Iterable[Any],
    seed: int,
    ratios: Sequence[float] = (0.8, 0.1, 0.1),
    excluded: Mapping[str, str] | Iterable[Any] | None = None,
) -> SplitManifest:
    return make_grouped_splits(records, seed, ratios, "target", excluded)


def make_scaffold_grouped_splits(
    records: Iterable[Any],
    seed: int,
    ratios: Sequence[float] = (0.8, 0.1, 0.1),
    excluded: Mapping[str, str] | Iterable[Any] | None = None,
) -> SplitManifest:
    return make_grouped_splits(records, seed, ratios, "ligand_scaffold", excluded)
