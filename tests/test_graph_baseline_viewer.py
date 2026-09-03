import pytest

from afbind.data.ligand_graph import (
    extract_ecfp6,
    smiles_to_graph,
    validate_ligand_graph,
)
from afbind.models.fingerprint import fit_baseline, predict
from afbind.viewer import ViewerContractError, load_cached_records


class FakeRDKitAdapter:
    def parse_smiles(self, smiles):
        return object() if smiles == "CCO" else None

    def mol_to_graph(self, mol):
        return {"node_features": [[6], [6], [8]], "edge_index": [[0, 1], [1, 2]]}

    def ecfp6(self, mol, *, radius, n_bits):
        assert radius == 3
        assert n_bits == 2048
        return [1] + [0] * (n_bits - 1)


def _manifest(train_ids, validation_ids=(), group_keys=("target_id", "scaffold_id")):
    return {
        "schema_version": "split/v1",
        "seed": 7,
        "group_keys": list(group_keys),
        "train_ids": list(train_ids),
        "validation_ids": list(validation_ids),
        "test_ids": [],
    }


def _train_input():
    rows = [
        {"id": "a", "fingerprint": [1.0, 0.0], "label": 1.0, "target_id": "t1", "scaffold_id": "s1", "split": "train"},
        {"id": "b", "fingerprint": [0.0, 1.0], "label": 3.0, "target_id": "t2", "scaffold_id": "s2", "split": "train"},
    ]
    return {"split_manifest": _manifest(["a", "b"], ["c"]), "rows": rows}


def _artifact(**overrides):
    record = {
        "record_id": "case-1",
        "schema_version": "benchmark/v1",
        "run_id": "synthetic-run",
        "created_at": "2026-08-19T00:00:00Z",
        "provenance": {"source": "synthetic", "artifact_id": "fixture-1"},
        "split_manifest_id": "split-1",
        "label_policy": "Ki/Kd only",
        "condition": "experimental",
        "model": "ecfp6-ridge",
        "n": 1,
        "missing_count": 0,
        "reviewed": True,
        "status": "available",
    }
    record.update(overrides)
    return record


def test_blank_smiles_is_deterministically_invalid_without_rdkit():
    result = smiles_to_graph("   ", adapter=False)

    assert result.status == "invalid"
    assert result.reason == "blank_smiles"
    assert result.graph is None


def test_obviously_malformed_smiles_is_invalid_without_rdkit():
    result = smiles_to_graph("C1CC[", adapter=False)

    assert result.status == "invalid"
    assert result.reason == "malformed_smiles"


def test_nonempty_well_formed_smiles_reports_rdkit_unavailable():
    result = smiles_to_graph("CCO", adapter=False)

    assert result.status == "dependency_unavailable"
    assert "RDKit" in result.reason
    assert result.graph is None


def test_rdkit_adapter_produces_a_documented_graph_result_and_ecfp6():
    adapter = FakeRDKitAdapter()

    graph_result = smiles_to_graph("CCO", adapter=adapter)
    fingerprint_result = extract_ecfp6("CCO", adapter=adapter)

    assert graph_result.status == "valid"
    assert graph_result.feature_width == 1
    assert graph_result.graph["node_features"]
    assert fingerprint_result.status == "valid"
    assert fingerprint_result.radius == 3
    assert fingerprint_result.n_bits == 2048
    assert len(fingerprint_result.bits) == 2048
    assert fingerprint_result.bits[0] == 1


def test_graph_validation_rejects_empty_and_inconsistent_graphs():
    empty = validate_ligand_graph({"node_features": [], "edge_index": [[], []]})
    inconsistent = validate_ligand_graph({"node_features": [[1]], "edge_index": [[0, 1], [0]]})

    assert empty.status == "invalid"
    assert empty.reason == "empty_graph"
    assert inconsistent.status == "invalid"
    assert inconsistent.reason == "edge_index_shape"


def test_fit_requires_a_split_aware_training_input():
    with pytest.raises(ValueError, match="split_manifest"):
        fit_baseline({"rows": _train_input()["rows"]})


def test_fit_and_predict_use_explicit_fingerprint_matrices_deterministically():
    training = _train_input()
    model = fit_baseline(training, alpha=0.5)
    evaluation = {
        "split_manifest": _manifest(["a", "b"], ["c"]),
        "rows": [{"id": "c", "fingerprint": [1.0, 1.0], "target_id": "t3", "scaffold_id": "s3", "split": "validation"}],
    }

    first = predict(model, evaluation)
    second = predict(model, evaluation)

    assert first == second
    assert len(first) == 1
    assert isinstance(first[0], float)


def test_predict_rejects_a_group_that_was_seen_in_training():
    model = fit_baseline(_train_input())
    evaluation = {
        "split_manifest": _manifest(["a", "b"], ["c"]),
        "rows": [{"id": "c", "fingerprint": [1.0, 1.0], "target_id": "t1", "scaffold_id": "s3", "split": "validation"}],
    }

    with pytest.raises(ValueError, match="group"):
        predict(model, evaluation)


def test_cached_loader_requires_provenance_schema_count_and_review_fields():
    for missing in ("provenance", "schema_version", "n", "missing_count", "reviewed"):
        record = _artifact()
        record.pop(missing)
        with pytest.raises(ViewerContractError, match=missing):
            load_cached_records([record])


def test_cached_loader_rejects_artifact_not_marked_reviewed():
    record = _artifact()
    record.pop("reviewed")

    with pytest.raises(ViewerContractError, match="reviewed"):
        load_cached_records([record])


def test_cached_loader_rejects_raw_structure_paths():
    with pytest.raises(ViewerContractError, match="raw structure"):
        load_cached_records([_artifact(structure_path="private/raw.pdb")])


def test_viewer_selects_only_cached_ids_and_rejects_prediction_requests():
    viewer = load_cached_records([_artifact()])

    selected = viewer.select(record_id="case-1")

    assert selected.status == "available"
    assert selected.record["record_id"] == "case-1"
    with pytest.raises(ViewerContractError, match="cached"):
        viewer.select(smiles="CCO", uniprot="P00001")


def test_viewer_exposes_unavailable_cached_records_without_inventing_a_prediction():
    unavailable = _artifact(
        record_id="case-missing",
        status="unavailable",
        n=0,
        missing_count=1,
        reason="no verified structure substitution",
    )
    viewer = load_cached_records([unavailable])

    selected = viewer.select(record_id="case-missing")

    assert selected.status == "unavailable"
    assert selected.record["record_id"] == "case-missing"
    assert selected.reason == "no verified structure substitution"


def test_cached_loader_rejects_invalid_status_value():
    with pytest.raises(ViewerContractError, match="status"):
        load_cached_records([_artifact(status="pending")])


def test_cached_loader_rejects_unavailable_record_without_reason():
    with pytest.raises(ViewerContractError, match="reason"):
        load_cached_records([_artifact(status="unavailable")])
