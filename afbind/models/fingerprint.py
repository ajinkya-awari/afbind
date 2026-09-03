"""Deterministic, split-aware ECFP6 ridge-style baseline."""

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RidgeBaseline:
    """Fitted closed-form Ridge model and the groups used to fit it."""

    weights: np.ndarray
    intercept: float
    alpha: float
    feature_width: int
    group_keys: tuple[str, ...]
    training_groups: frozenset[tuple[str, Any]]
    training_ids: tuple[str, ...]


def _split_parts(dataset: Any) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    if not isinstance(dataset, Mapping) or "split_manifest" not in dataset:
        raise ValueError("split_manifest is required for split-aware fitting and prediction")
    manifest = dataset["split_manifest"]
    rows = dataset.get("rows")
    if not isinstance(manifest, Mapping):
        raise ValueError("split_manifest must be a mapping")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("rows must be a sequence")
    required = ("group_keys", "train_ids", "validation_ids", "test_ids")
    missing = [key for key in required if key not in manifest]
    if missing:
        raise ValueError("split_manifest missing " + ", ".join(missing))
    group_keys = manifest["group_keys"]
    if not isinstance(group_keys, Sequence) or isinstance(group_keys, (str, bytes)) or not group_keys:
        raise ValueError("split_manifest group_keys must be non-empty")
    return manifest, list(rows)


def _row_features(row: Mapping[str, Any]) -> Any:
    features = row.get("fingerprint", row.get("features"))
    if features is None:
        raise ValueError("each row requires an explicit fingerprint matrix row")
    return features


def _matrix(rows: list[Mapping[str, Any]], *, labels: bool = False) -> tuple[np.ndarray, np.ndarray | None]:
    if not rows:
        raise ValueError("at least one row is required")
    try:
        matrix = np.asarray([_row_features(row) for row in rows], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("fingerprint matrix must be rectangular and numeric") from exc
    if matrix.ndim != 2 or matrix.shape[1] == 0 or not np.isfinite(matrix).all():
        raise ValueError("fingerprint matrix must be a finite, non-empty 2-D matrix")
    values = None
    if labels:
        try:
            values = np.asarray([row["label"] for row in rows], dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("each training row requires a numeric label") from exc
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("labels must be finite numeric values")
    return matrix, values


def _validate_row_ids(rows: list[Mapping[str, Any]], manifest: Mapping[str, Any], expected_split: str) -> None:
    ids = [row.get("id") for row in rows]
    if any(not isinstance(identifier, str) or not identifier for identifier in ids):
        raise ValueError("every split row requires a non-empty id")
    if len(set(ids)) != len(ids):
        raise ValueError("split rows contain duplicate ids")
    declared = set(manifest[f"{expected_split}_ids"])
    if set(ids) - declared:
        raise ValueError(f"rows contain ids outside the declared {expected_split} split")
    for row in rows:
        if row.get("split") != expected_split:
            raise ValueError(f"row {row['id']} is not explicitly marked {expected_split}")


def _groups(rows: list[Mapping[str, Any]], group_keys: Sequence[str]) -> frozenset[tuple[str, Any]]:
    result = []
    for row in rows:
        for key in group_keys:
            if key not in row:
                raise ValueError(f"row missing split group key: {key}")
            result.append((key, row[key]))
    return frozenset(result)


def fit_baseline(train: Any, *, alpha: float = 1.0) -> RidgeBaseline:
    """Fit deterministic closed-form Ridge on explicit training fingerprints only."""

    manifest, rows = _split_parts(train)
    _validate_row_ids(rows, manifest, "train")
    try:
        alpha = float(alpha)
    except (TypeError, ValueError) as exc:
        raise ValueError("alpha must be a positive finite number") from exc
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be a positive finite number")
    matrix, labels = _matrix(rows, labels=True)
    centered_x = matrix - matrix.mean(axis=0)
    centered_y = labels - labels.mean()
    system = centered_x.T @ centered_x + alpha * np.eye(matrix.shape[1])
    weights = np.linalg.solve(system, centered_x.T @ centered_y)
    intercept = float(labels.mean() - matrix.mean(axis=0) @ weights)
    group_keys = tuple(manifest["group_keys"])
    return RidgeBaseline(
        weights=weights,
        intercept=intercept,
        alpha=alpha,
        feature_width=matrix.shape[1],
        group_keys=group_keys,
        training_groups=_groups(rows, group_keys),
        training_ids=tuple(row["id"] for row in rows),
    )


def predict(model: RidgeBaseline, rows: Any) -> list[float]:
    """Predict only explicitly declared validation/test rows with unseen groups."""

    manifest, evaluation_rows = _split_parts(rows)
    if not isinstance(model, RidgeBaseline):
        raise ValueError("model must be a RidgeBaseline")
    if list(manifest["group_keys"]) != list(model.group_keys):
        raise ValueError("prediction split group_keys do not match the training split")
    declared_eval = set(manifest["validation_ids"]) | set(manifest["test_ids"])
    if not evaluation_rows:
        raise ValueError("at least one evaluation row is required")
    if not all(row.get("split") in ("validation", "test") for row in evaluation_rows):
        raise ValueError("prediction rows must be explicitly marked validation or test")
    if len({row.get("id") for row in evaluation_rows}) != len(evaluation_rows):
        raise ValueError("prediction rows contain duplicate ids")
    for split in ("validation", "test"):
        split_rows = [row for row in evaluation_rows if row.get("split") == split]
        if split_rows:
            _validate_row_ids(split_rows, manifest, split)
    if any(row["id"] not in declared_eval for row in evaluation_rows):
        raise ValueError("prediction rows must belong to the declared evaluation split")
    evaluation_groups = _groups(evaluation_rows, model.group_keys)
    if model.training_groups & evaluation_groups:
        raise ValueError("prediction groups overlap training groups")
    matrix, _ = _matrix(evaluation_rows)
    if matrix.shape[1] != model.feature_width:
        raise ValueError("prediction fingerprint width does not match training")
    return [float(value) for value in matrix @ model.weights + model.intercept]


fit_ridge_baseline = fit_baseline
