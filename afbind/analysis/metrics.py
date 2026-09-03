"""Dependency-light CA matching and matched-coordinate RMSD."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MatchReport:
    keys: tuple[tuple[str, int, str], ...]
    experimental_coordinates: tuple[tuple[float, float, float], ...]
    predicted_coordinates: tuple[tuple[float, float, float], ...]
    status: str
    reason: str
    common_residue_count: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return {
            "keys": self.keys,
            "experimental_coordinates": self.experimental_coordinates,
            "predicted_coordinates": self.predicted_coordinates,
            "status": self.status,
            "reason": self.reason,
            "common_residue_count": self.common_residue_count,
            "metadata": self.metadata,
        }[key]

    def __iter__(self) -> Iterator[str]:
        return iter(("keys", "experimental_coordinates", "predicted_coordinates", "status", "reason", "common_residue_count", "metadata"))

    def __len__(self) -> int:
        return 7


def _records(structure: Any) -> list[Any]:
    if isinstance(structure, Mapping):
        for key in ("atoms", "records"):
            value = structure.get(key)
            if isinstance(value, (list, tuple)):
                return list(value)
        return []
    if isinstance(structure, (list, tuple)):
        return list(structure)
    return []


def _atom_key(atom: Mapping[str, Any]) -> tuple[str, int, str] | None:
    try:
        chain_id = str(atom.get("chain_id", atom.get("chain", ""))).strip()
        residue_number = int(atom.get("residue_number", atom.get("res_seq")))
        insertion_code = str(atom.get("insertion_code", atom.get("icode", "")) or "").strip()
    except (TypeError, ValueError):
        return None
    return chain_id, residue_number, insertion_code


def _coordinates(atom: Mapping[str, Any]) -> tuple[float, float, float] | None:
    value = atom.get("coordinates", atom.get("coord"))
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        return tuple(float(component) for component in value)  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None


def _ca_index(structure: Any) -> dict[tuple[str, int, str], tuple[float, float, float]]:
    indexed: dict[tuple[str, int, str], tuple[float, float, float]] = {}
    for atom in _records(structure):
        if not isinstance(atom, Mapping) or str(atom.get("atom_name", atom.get("name", ""))).strip().upper() != "CA":
            continue
        key = _atom_key(atom)
        coordinates = _coordinates(atom)
        if key is not None and coordinates is not None and key not in indexed:
            indexed[key] = coordinates
    return indexed


def match_ca_atoms(experimental: Any, predicted: Any) -> MatchReport:
    """Return equal-length CA coordinate lists keyed by chain/residue/insertion code."""

    experimental_index = _ca_index(experimental)
    predicted_index = _ca_index(predicted)
    keys = tuple(sorted(experimental_index.keys() & predicted_index.keys(), key=lambda key: (key[0], key[1], key[2])))
    experimental_coordinates = tuple(experimental_index[key] for key in keys)
    predicted_coordinates = tuple(predicted_index[key] for key in keys)
    count = len(keys)
    if count < 3:
        return MatchReport(keys, experimental_coordinates, predicted_coordinates, "unavailable", "insufficient_common_residues", count)
    return MatchReport(keys, experimental_coordinates, predicted_coordinates, "matched", "common_residues_matched", count)


def _dot(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _norm(vector: tuple[float, float, float]) -> float:
    return math.sqrt(_dot(vector, vector))


def _cross(left: tuple[float, float, float], right: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _normalize(vector: tuple[float, float, float], tolerance: float) -> tuple[float, float, float] | None:
    length = _norm(vector)
    if length <= tolerance:
        return None
    return tuple(component / length for component in vector)  # type: ignore[return-value]


def _mat_vec(matrix: tuple[tuple[float, float, float], ...], vector: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(sum(matrix[row][column] * vector[column] for column in range(3)) for row in range(3))  # type: ignore[return-value]


def _mat_mul(left: tuple[tuple[float, float, float], ...], right: tuple[tuple[float, float, float], ...]) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        tuple(sum(left[row][inner] * right[inner][column] for inner in range(3)) for column in range(3))
        for row in range(3)
    )


def _transpose(matrix: tuple[tuple[float, float, float], ...]) -> tuple[tuple[float, float, float], ...]:
    return tuple(tuple(matrix[row][column] for row in range(3)) for column in range(3))


def _determinant(matrix: tuple[tuple[float, float, float], ...]) -> float:
    return (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )


def _symmetric_eigenvectors(matrix: tuple[tuple[float, float, float], ...]) -> tuple[tuple[float, ...], tuple[tuple[float, ...], ...]]:
    """Jacobi eigensolver for the symmetric 3x3 matrix used by Kabsch."""

    values = [list(row) for row in matrix]
    vectors = [[1.0 if row == column else 0.0 for column in range(3)] for row in range(3)]
    for _ in range(32):
        row, column = max(((0, 1), (0, 2), (1, 2)), key=lambda pair: abs(values[pair[0]][pair[1]]))
        if abs(values[row][column]) <= 1e-14:
            break
        angle = 0.5 * math.atan2(2.0 * values[row][column], values[column][column] - values[row][row])
        cosine, sine = math.cos(angle), math.sin(angle)
        diagonal_row, diagonal_column, off_diagonal = values[row][row], values[column][column], values[row][column]
        values[row][row] = cosine * cosine * diagonal_row - 2.0 * sine * cosine * off_diagonal + sine * sine * diagonal_column
        values[column][column] = sine * sine * diagonal_row + 2.0 * sine * cosine * off_diagonal + cosine * cosine * diagonal_column
        values[row][column] = values[column][row] = 0.0
        for other in range(3):
            if other in (row, column):
                continue
            other_row, other_column = values[other][row], values[other][column]
            values[other][row] = values[row][other] = cosine * other_row - sine * other_column
            values[other][column] = values[column][other] = sine * other_row + cosine * other_column
        for other in range(3):
            vector_row, vector_column = vectors[other][row], vectors[other][column]
            vectors[other][row] = cosine * vector_row - sine * vector_column
            vectors[other][column] = sine * vector_row + cosine * vector_column
    order = sorted(range(3), key=lambda index: values[index][index], reverse=True)
    eigenvalues = tuple(values[index][index] for index in order)
    eigenvectors = tuple(tuple(vectors[row][index] for row in range(3)) for index in order)
    return eigenvalues, eigenvectors


def _kabsch_rmsd(source: tuple[tuple[float, float, float], ...], target: tuple[tuple[float, float, float], ...]) -> float | None:
    """Return aligned RMSD; ``None`` means insufficient rank for a unique rigid fit."""

    if len(source) != len(target) or len(source) < 3:
        return None
    source_center = tuple(sum(point[index] for point in source) / len(source) for index in range(3))
    target_center = tuple(sum(point[index] for point in target) / len(target) for index in range(3))
    centered_source = tuple(tuple(point[index] - source_center[index] for index in range(3)) for point in source)
    centered_target = tuple(tuple(point[index] - target_center[index] for index in range(3)) for point in target)
    covariance = tuple(
        tuple(sum(point[row] * target_point[column] for point, target_point in zip(centered_source, centered_target)) for column in range(3))
        for row in range(3)
    )
    covariance_transpose = _transpose(covariance)
    normal_matrix = _mat_mul(covariance_transpose, covariance)
    eigenvalues, right_vectors = _symmetric_eigenvectors(normal_matrix)
    scale = max(1.0, max(abs(value) for row in normal_matrix for value in row))
    tolerance = scale * 1e-12
    rank = sum(value > tolerance for value in eigenvalues)
    if rank < 2:
        return None

    left_vectors: list[tuple[float, float, float]] = []
    for singular_value_squared, right_vector in zip(eigenvalues, right_vectors):
        if singular_value_squared <= tolerance:
            left_vectors.append((0.0, 0.0, 0.0))
            continue
        singular_value = math.sqrt(singular_value_squared)
        left_vector = _normalize(_mat_vec(covariance, right_vector), tolerance)
        if left_vector is None:
            return None
        left_vectors.append(left_vector)
    if rank == 2:
        missing_index = next(index for index, vector in enumerate(left_vectors) if vector == (0.0, 0.0, 0.0))
        first, second = (index for index in range(3) if index != missing_index)
        completion = _normalize(_cross(left_vectors[first], left_vectors[second]), tolerance)
        if completion is None:
            return None
        left_vectors[missing_index] = completion

    left_matrix = tuple(tuple(left_vectors[column][row] for column in range(3)) for row in range(3))
    right_matrix = tuple(tuple(right_vectors[column][row] for column in range(3)) for row in range(3))
    rotation = _mat_mul(left_matrix, _transpose(right_matrix))
    if _determinant(rotation) < 0.0:
        smallest = len(left_vectors) - 1
        corrected_left = [list(vector) for vector in left_vectors]
        corrected_left[smallest] = [-component for component in corrected_left[smallest]]
        left_matrix = tuple(tuple(corrected_left[column][row] for column in range(3)) for row in range(3))
        rotation = _mat_mul(left_matrix, _transpose(right_matrix))

    squared_error = 0.0
    for source_point, target_point in zip(centered_source, centered_target):
        rotated = _mat_vec(_transpose(rotation), source_point)
        squared_error += sum((rotated[index] - target_point[index]) ** 2 for index in range(3))
    return math.sqrt(squared_error / len(source))


def pocket_ca_rmsd(experimental: Any, predicted: Any, minimum_common_residues: int = 3) -> float | None:
    """Superpose matched CA atoms with Kabsch, or return ``None`` if unavailable/degenerate."""

    report = match_ca_atoms(experimental, predicted)
    required_residues = max(3, int(minimum_common_residues))
    if report.common_residue_count < required_residues:
        return None
    return _kabsch_rmsd(report.predicted_coordinates, report.experimental_coordinates)
