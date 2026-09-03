import math

import pytest

from afbind.analysis.failure_modes import scan_hetero
from afbind.analysis.metrics import match_ca_atoms, pocket_ca_rmsd
from afbind.data.af2_fetch import select_prediction_record
from afbind.data.mapping import resolve_candidates


def test_select_prediction_record_uses_returned_url_and_preserves_metadata():
    payload = [
        {"modelVersion": "old", "pdbUrl": None},
        {
            "modelVersion": "current",
            "uniprotAccession": "P12345",
            "pdbUrl": "https://api.example.test/predictions/P12345.pdb",
            "extra": {"confidence": 0.91},
        },
    ]

    result = select_prediction_record(payload)

    assert result.status == "available"
    assert result.pdb_url == payload[1]["pdbUrl"]
    assert result.metadata["modelVersion"] == "current"
    assert result.metadata["extra"] == {"confidence": 0.91}
    assert result.reason == "selected_api_record"


def test_select_prediction_record_reports_unavailable_not_found_and_invalid():
    unavailable = select_prediction_record([{"modelVersion": "current"}])
    not_found = select_prediction_record({"status": 404, "detail": "missing"})
    invalid = select_prediction_record({"unexpected": "shape"})

    assert unavailable.status == "unavailable"
    assert unavailable.reason == "pdb_url_unavailable"
    assert not_found.status == "not_found"
    assert not_found.reason == "api_not_found"
    assert invalid.status == "invalid"
    assert invalid.reason == "invalid_api_payload"


def test_resolve_candidates_selects_an_unambiguous_chain_and_keeps_candidates():
    result = resolve_candidates(
        "1abc",
        [
            {"chain_id": "B", "uniprot_id": "P2"},
            {"chain_id": "A", "uniprot_id": "P1"},
        ],
    )

    assert result.status == "resolved"
    assert result.candidate_uniprots == ("P1", "P2")
    assert result.chains[0].chain_id == "A"
    assert result.chains[0].selected_uniprot == "P1"
    assert result.selected_uniprot is None


def test_resolve_candidates_marks_a_chain_ambiguous_without_guessing():
    result = resolve_candidates(
        "1abc",
        [
            {"chain_id": "A", "uniprot_id": "P1", "metadata": {"source": "sifts"}},
            {"chain_id": "A", "uniprot_id": "P2", "metadata": {"source": "sifts"}},
        ],
    )

    assert result.status == "ambiguous"
    assert result.selected_uniprot is None
    assert result.candidate_uniprots == ("P1", "P2")
    assert result.chains[0].status == "ambiguous"
    assert result.chains[0].selected_uniprot is None
    assert result.reason == "multiple_candidates_for_chain"


def test_match_ca_atoms_uses_chain_residue_and_insertion_keys():
    experimental = [
        {"atom_name": "CA", "chain_id": "A", "residue_number": 1, "insertion_code": "", "coordinates": (0, 0, 0)},
        {"atom_name": "CA", "chain_id": "A", "residue_number": 2, "insertion_code": "A", "coordinates": (1, 0, 0)},
        {"atom_name": "CB", "chain_id": "A", "residue_number": 3, "insertion_code": "", "coordinates": (9, 9, 9)},
        {"atom_name": "CA", "chain_id": "B", "residue_number": 1, "insertion_code": "", "coordinates": (2, 1, 0)},
    ]
    predicted = [
        {"atom_name": "CA", "chain_id": "A", "residue_number": 2, "insertion_code": "A", "coordinates": (2, 0, 0)},
        {"atom_name": "CA", "chain_id": "A", "residue_number": 1, "insertion_code": "", "coordinates": (1, 0, 0)},
        {"atom_name": "CA", "chain_id": "A", "residue_number": 4, "insertion_code": "", "coordinates": (8, 8, 8)},
        {"atom_name": "CA", "chain_id": "B", "residue_number": 1, "insertion_code": "", "coordinates": (3, 1, 0)},
    ]

    report = match_ca_atoms(experimental, predicted)

    assert report.status == "matched"
    assert report.keys == (("A", 1, ""), ("A", 2, "A"), ("B", 1, ""))
    assert len(report.experimental_coordinates) == len(report.predicted_coordinates) == 3
    assert report.experimental_coordinates[1] == (1.0, 0.0, 0.0)
    assert report.predicted_coordinates[1] == (2.0, 0.0, 0.0)
    assert math.isclose(pocket_ca_rmsd(experimental, predicted), 0.0, abs_tol=1e-9)


def test_pocket_ca_rmsd_is_unavailable_below_three_common_residues():
    atoms = [
        {"atom_name": "CA", "chain_id": "A", "residue_number": 1, "coordinates": (0, 0, 0)},
        {"atom_name": "CA", "chain_id": "A", "residue_number": 2, "coordinates": (1, 0, 0)},
    ]

    report = match_ca_atoms(atoms, atoms)

    assert report.status == "unavailable"
    assert report.reason == "insufficient_common_residues"
    assert pocket_ca_rmsd(atoms, atoms) is None


def test_pocket_ca_rmsd_superposes_translated_and_rotated_point_sets():
    experimental = [
        {"atom_name": "CA", "chain_id": "A", "residue_number": 1, "coordinates": (0, 0, 0)},
        {"atom_name": "CA", "chain_id": "A", "residue_number": 2, "coordinates": (1, 0, 0)},
        {"atom_name": "CA", "chain_id": "A", "residue_number": 3, "coordinates": (0, 2, 0)},
        {"atom_name": "CA", "chain_id": "A", "residue_number": 4, "coordinates": (0, 0, 3)},
    ]
    predicted = [
        {"atom_name": "CA", "chain_id": "A", "residue_number": 1, "coordinates": (5, -2, 7)},
        {"atom_name": "CA", "chain_id": "A", "residue_number": 2, "coordinates": (5, -1, 7)},
        {"atom_name": "CA", "chain_id": "A", "residue_number": 3, "coordinates": (3, -2, 7)},
        {"atom_name": "CA", "chain_id": "A", "residue_number": 4, "coordinates": (5, -2, 10)},
    ]

    assert pocket_ca_rmsd(experimental, predicted) == pytest.approx(0.0, abs=1e-9)


def test_pocket_ca_rmsd_returns_none_for_degenerate_collinear_points():
    atoms = [
        {"atom_name": "CA", "chain_id": "A", "residue_number": 1, "coordinates": (0, 0, 0)},
        {"atom_name": "CA", "chain_id": "A", "residue_number": 2, "coordinates": (1, 0, 0)},
        {"atom_name": "CA", "chain_id": "A", "residue_number": 3, "coordinates": (2, 0, 0)},
    ]

    assert pocket_ca_rmsd(atoms, atoms) is None


def test_scan_hetero_exposes_retained_water_metal_and_selenium_evidence():
    structure = {
        "source_file": "synthetic.cif",
        "records": [
            {"record_name": "HETATM", "residue_name": " hoh ", "element": "O"},
            {"record_name": "HETATM", "residue_name": "ZN", "element": "ZN"},
            {"record_name": "HETATM", "residue_name": "se", "element": "SE"},
            {"record_name": "HETATM", "residue_name": "LIG", "element": "C"},
            {"record_name": "ATOM", "residue_name": "HOH", "element": "O"},
        ],
    }

    report = scan_hetero(structure)

    assert report.num_complexes_scanned == 1
    assert report.num_with_water == 1
    assert report.num_with_metal == 1
    assert report.water_counts == {"HOH": 1}
    assert report.metal_counts == {"ZN": 1}
    assert report.selenium_records_excluded == 1
    assert report.source_file_manifest == ("synthetic.cif",)


def _four_noncollinear_atoms():
    return [
        {"atom_name": "CA", "chain_id": "A", "residue_number": i, "coordinates": c}
        for i, c in enumerate([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)], start=1)
    ]


def test_pocket_ca_rmsd_respects_minimum_common_residues_argument():
    atoms = _four_noncollinear_atoms()

    assert pocket_ca_rmsd(atoms, atoms, minimum_common_residues=4) == pytest.approx(0.0, abs=1e-9)
    assert pocket_ca_rmsd(atoms, atoms[:3], minimum_common_residues=4) is None


def test_pocket_ca_rmsd_minimum_common_residues_clamped_to_three():
    atoms = _four_noncollinear_atoms()[:3]

    assert pocket_ca_rmsd(atoms, atoms, minimum_common_residues=1) == pytest.approx(0.0, abs=1e-9)
