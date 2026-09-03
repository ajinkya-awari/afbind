"""Kaggle-oriented GraphGPS regression boundary with offline CPU smoke support."""

from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import torch
    from torch import Tensor, nn
    from torch.nn import functional as F
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import GINEConv, GPSConv, global_mean_pool
except ImportError as exc:  # pragma: no cover - exercised only in dependency-light environments.
    raise ImportError("GraphGPS training requires torch and torch_geometric") from exc


@dataclass(frozen=True)
class GraphGPSConfig:
    node_dim: int
    edge_dim: int
    hidden_dim: int = 64
    layers: int = 3
    heads: int = 4
    dropout: float = 0.1
    learning_rate: float = 1e-3
    epochs: int = 10
    batch_size: int = 32
    seed: int = 0
    device: str = "cuda"

    def __post_init__(self) -> None:
        integer_fields = {
            "node_dim": self.node_dim,
            "edge_dim": self.edge_dim,
            "hidden_dim": self.hidden_dim,
            "layers": self.layers,
            "heads": self.heads,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
        }
        if any(not isinstance(value, int) or value <= 0 for value in integer_fields.values()):
            raise ValueError("GraphGPS dimensions, layers, epochs, and batch_size must be positive integers")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.hidden_dim % self.heads:
            raise ValueError("hidden_dim must be divisible by heads")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GraphGPSTrainingReport:
    seed: int
    device: str
    config: Mapping[str, Any]
    package_versions: Mapping[str, str]
    train_losses: tuple[float, ...]
    validation_losses: tuple[float, ...]
    checkpoint_sha256: str | None


def _mlp(hidden_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
    )


class GraphGPSRegressor(nn.Module):
    """Ligand/pocket graph regressor using GINE local message passing plus GPSConv."""

    def __init__(self, config: GraphGPSConfig):
        super().__init__()
        self.config = config
        self.node_encoder = nn.Linear(config.node_dim, config.hidden_dim)
        self.layers = nn.ModuleList(
            [
                GPSConv(
                    channels=config.hidden_dim,
                    conv=GINEConv(_mlp(config.hidden_dim), edge_dim=config.edge_dim),
                    heads=config.heads,
                    dropout=config.dropout,
                    norm="batch_norm",
                )
                for _ in range(config.layers)
            ]
        )
        self.regressor = nn.Linear(config.hidden_dim, 1)

    def forward(self, data: Data) -> Tensor:
        x = data.x.float()
        edge_attr = data.edge_attr.float()
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = self.node_encoder(x)
        for layer in self.layers:
            x = layer(x, data.edge_index, batch=batch, edge_attr=edge_attr)
        return self.regressor(global_mean_pool(x, batch)).squeeze(-1)


def _validate_rows(rows: Sequence[Mapping[str, Any]], expected_split: str) -> list[Mapping[str, Any]]:
    if not rows:
        raise ValueError(f"{expected_split} rows must not be empty")
    validated = list(rows)
    for row in validated:
        if row.get("split") != expected_split:
            raise ValueError(f"all rows must be explicitly marked {expected_split}")
        if not isinstance(row.get("graph"), Data):
            raise ValueError("each row requires a torch_geometric Data graph")
        if "label" not in row:
            raise ValueError("each row requires a numeric label")
        for key in ("target_id", "ligand_scaffold"):
            if not str(row.get(key, "")).strip():
                raise ValueError(f"each row requires {key}")
    return validated


def _groups(rows: Sequence[Mapping[str, Any]]) -> set[tuple[str, str]]:
    return {
        (key, str(row[key]))
        for row in rows
        for key in ("target_id", "ligand_scaffold")
    }


def _prepare_graph(row: Mapping[str, Any]) -> Data:
    graph = row["graph"].clone()
    graph.y = torch.tensor(float(row["label"]), dtype=torch.float32)
    return graph


def _checkpoint_hash(path: str | Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_graphgps(
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    config: GraphGPSConfig,
    *,
    checkpoint_path: str | Path | None = None,
) -> GraphGPSTrainingReport:
    """Train on explicit train rows and evaluate on disjoint validation rows."""

    train = _validate_rows(train_rows, "train")
    validation = _validate_rows(validation_rows, "validation")
    overlap = _groups(train) & _groups(validation)
    if overlap:
        raise ValueError(f"train/validation group overlap: {sorted(overlap)}")

    requested_device = torch.device(config.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable; run this configuration on Kaggle GPU")
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if requested_device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    model = GraphGPSRegressor(config).to(requested_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    train_loader = DataLoader(
        [_prepare_graph(row) for row in train],
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    validation_loader = DataLoader(
        [_prepare_graph(row) for row in validation],
        batch_size=config.batch_size,
        shuffle=False,
    )
    train_losses: list[float] = []
    validation_losses: list[float] = []
    for _ in range(config.epochs):
        model.train()
        train_total = 0.0
        train_count = 0
        for batch in train_loader:
            batch = batch.to(requested_device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(batch), batch.y.view(-1).float())
            loss.backward()
            optimizer.step()
            train_total += float(loss.detach()) * batch.num_graphs
            train_count += batch.num_graphs
        train_losses.append(train_total / train_count)

        model.eval()
        validation_total = 0.0
        validation_count = 0
        with torch.no_grad():
            for batch in validation_loader:
                batch = batch.to(requested_device)
                loss = F.mse_loss(model(batch), batch.y.view(-1).float())
                validation_total += float(loss) * batch.num_graphs
                validation_count += batch.num_graphs
        validation_losses.append(validation_total / validation_count)

    if checkpoint_path is not None:
        torch.save(model.state_dict(), checkpoint_path)
    return GraphGPSTrainingReport(
        seed=config.seed,
        device=str(requested_device),
        config=config.to_dict(),
        package_versions={"torch": torch.__version__, "torch_geometric": _pyg_version()},
        train_losses=tuple(train_losses),
        validation_losses=tuple(validation_losses),
        checkpoint_sha256=_checkpoint_hash(checkpoint_path),
    )


def _pyg_version() -> str:
    import torch_geometric

    return torch_geometric.__version__
