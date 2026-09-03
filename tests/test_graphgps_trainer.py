import pytest

torch = pytest.importorskip("torch")
from torch_geometric.data import Data

from afbind.models.graphgps import (
    GraphGPSConfig,
    GraphGPSRegressor,
    train_graphgps,
)


def _graph(label: float) -> Data:
    return Data(
        x=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
        edge_attr=torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        y=torch.tensor([label], dtype=torch.float32),
    )


def _rows(split: str, target: str, scaffold: str, label: float):
    return {
        "graph": _graph(label),
        "label": label,
        "split": split,
        "target_id": target,
        "ligand_scaffold": scaffold,
    }


def test_graphgps_forward_uses_declared_edge_width_and_batch_pooling():
    from torch_geometric.data import Batch

    config = GraphGPSConfig(node_dim=3, edge_dim=2, hidden_dim=8, layers=1, heads=2)
    model = GraphGPSRegressor(config)
    batch = Batch.from_data_list([_graph(1.0), _graph(2.0)])

    output = model(batch)

    assert output.shape == (2,)
    assert torch.isfinite(output).all()


def test_train_graphgps_records_seed_device_and_losses():
    config = GraphGPSConfig(
        node_dim=3,
        edge_dim=2,
        hidden_dim=8,
        layers=1,
        heads=2,
        epochs=1,
        batch_size=2,
        seed=11,
        device="cpu",
    )

    report = train_graphgps(
        [_rows("train", "target-a", "scaffold-a", 1.0), _rows("train", "target-b", "scaffold-b", 2.0)],
        [_rows("validation", "target-c", "scaffold-c", 1.5)],
        config,
    )

    assert report.seed == 11
    assert report.device == "cpu"
    assert len(report.train_losses) == 1
    assert len(report.validation_losses) == 1
    assert report.checkpoint_sha256 is None


def test_train_graphgps_rejects_target_or_scaffold_overlap():
    config = GraphGPSConfig(node_dim=3, edge_dim=2, hidden_dim=8, layers=1, epochs=1)

    with pytest.raises(ValueError, match="overlap"):
        train_graphgps(
            [_rows("train", "target-a", "scaffold-a", 1.0)],
            [_rows("validation", "target-a", "scaffold-b", 1.5)],
            config,
        )
