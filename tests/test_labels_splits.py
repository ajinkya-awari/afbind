import json
import math

import pytest

from afbind.data.pdbbind import (
    AffinityRecord,
    AffinityTable,
    parse_index_line,
)
from afbind.data.splits import make_grouped_splits


def test_supported_concentration_units_preserve_provenance_and_convert_to_pki():
    expected = {
        "M": 0.0,
        "mM": 3.0,
        "uM": 6.0,
        "µM": 6.0,
        "nM": 9.0,
        "pM": 12.0,
    }

    for unit, expected_pki in expected.items():
        raw_value = "1.0"
        record = parse_index_line(f"pdb-{unit}|target-1|ligand-1|scaf-1|Ki|{raw_value}|{unit}")

        assert record is not None
        assert record.raw_value == raw_value
        assert record.raw_unit == unit
        assert record.measurement_type == "Ki"
        assert record.pki == pytest.approx(expected_pki)
        assert record.molar_concentration == pytest.approx(10 ** (-expected_pki))


def test_parser_accepts_kd_and_table_records_explicit_exclusions(tmp_path):
    path = tmp_path / "synthetic-index.txt"
    path.write_text(
        "\n".join(
            [
                "valid-ki|target-1|ligand-1|scaf-1|Ki|2|nM",
                "valid-kd|target-2|ligand-2|scaf-2|Kd|3|uM",
                "ic50|target-3|ligand-3|scaf-3|IC50|2|nM",
                "ec50|target-4|ligand-4|scaf-4|EC50|2|nM",
                "kact|target-5|ligand-5|scaf-5|Kact|2|nM",
                "ka|target-6|ligand-6|scaf-6|Ka|2|nM",
                "censored|target-7|ligand-7|scaf-7|Ki|>2|nM",
                "bad-unit|target-8|ligand-8|scaf-8|Kd|2|fM",
                "bad-number|target-9|ligand-9|scaf-9|Kd|not-a-number|nM",
                "short|target-10|Kd",
                "",
                "",
            ]
        ),
        encoding="utf-8",
    )

    table = AffinityTable.from_path(path)

    assert [record.record_id for record in table.records] == ["valid-ki", "valid-kd"]
    assert {item.reason for item in table.excluded} == {
        "unsupported_measurement_type",
        "censored_value",
        "unsupported_unit",
        "invalid_numeric_value",
        "malformed_row",
        "blank_row",
    }
    assert all(item.raw_line is not None for item in table.excluded)


def _record(record_id, target_id, scaffold):
    return AffinityRecord(
        record_id=record_id,
        target_id=target_id,
        ligand_id=f"ligand-{record_id}",
        ligand_scaffold=scaffold,
        measurement_type="Ki",
        raw_value="1",
        raw_unit="nM",
        molar_concentration=1e-9,
        pki=9.0,
    )


def _groups(records, ids, field):
    return {getattr(record, field) for record in records if record.record_id in ids}


def test_target_grouped_manifest_is_deterministic_and_isolates_targets():
    records = [
        _record("a", "target-a", "scaf-a"),
        _record("b", "target-a", "scaf-b"),
        _record("c", "target-b", "scaf-c"),
        _record("d", "target-c", "scaf-d"),
        _record("e", "target-d", "scaf-e"),
        _record("f", "target-e", "scaf-f"),
    ]

    first = make_grouped_splits(records, seed=19, ratios=(0.5, 0.25, 0.25), group_by="target")
    second = make_grouped_splits(records, seed=19, ratios=(0.5, 0.25, 0.25), group_by="target")

    assert first.to_dict() == second.to_dict()
    assert first.schema_version
    assert first.group_keys == ("target_id",)
    assert set(first.train_ids) | set(first.validation_ids) | set(first.test_ids) == {
        record.record_id for record in records
    }
    partitions = [first.train_ids, first.validation_ids, first.test_ids]
    for left_index, left in enumerate(partitions):
        for right in partitions[left_index + 1 :]:
            assert _groups(records, left, "target_id").isdisjoint(_groups(records, right, "target_id"))
    assert first.counts["total"] == len(records)


def test_scaffold_grouped_manifest_isolates_shared_scaffolds_and_serializes():
    records = [
        _record("a", "target-a", "shared"),
        _record("b", "target-b", "shared"),
        _record("c", "target-c", "unique-c"),
        _record("d", "target-d", "unique-d"),
        _record("e", "target-e", "unique-e"),
        _record("f", "target-f", "unique-f"),
    ]

    manifest = make_grouped_splits(records, seed=7, ratios=(0.5, 0.25, 0.25), group_by="ligand_scaffold")
    serialized = manifest.to_dict()

    json.dumps(serialized, sort_keys=True)
    assert manifest.group_keys == ("ligand_scaffold",)
    partitions = [manifest.train_ids, manifest.validation_ids, manifest.test_ids]
    for left_index, left in enumerate(partitions):
        for right in partitions[left_index + 1 :]:
            assert _groups(records, left, "ligand_scaffold").isdisjoint(
                _groups(records, right, "ligand_scaffold")
            )


def test_manifest_carries_excluded_ids_and_reasons():
    records = [_record("a", "target-a", "scaf-a"), _record("b", "target-b", "scaf-b")]
    manifest = make_grouped_splits(
        records,
        seed=3,
        ratios=(0.5, 0.25, 0.25),
        group_by="target",
        excluded={"bad-row": "unsupported_unit"},
    )

    assert manifest.excluded_ids == ("bad-row",)
    assert manifest.exclusion_reasons == {"bad-row": "unsupported_unit"}
    assert set(manifest.to_dict()) >= {
        "schema_version",
        "seed",
        "group_keys",
        "train_ids",
        "validation_ids",
        "test_ids",
        "counts",
        "excluded_ids",
        "exclusion_reasons",
    }
