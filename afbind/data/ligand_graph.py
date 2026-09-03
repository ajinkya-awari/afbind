"""Offline ligand graph and optional-RDKit fingerprint contracts.

The module deliberately distinguishes deterministic input rejection from an unavailable
optional chemistry dependency.  A well-formed non-empty SMILES is not parsed by a
replacement implementation when RDKit is absent.
"""

from dataclasses import dataclass
import re
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class GraphValidationResult:
    """Structured result for ligand SMILES/graph validation."""

    status: str
    reason: str
    graph: Optional[Mapping[str, Any]] = None
    feature_width: Optional[int] = None
    input_kind: str = "unknown"

    @property
    def ok(self) -> bool:
        return self.status == "valid"


@dataclass(frozen=True)
class FingerprintResult:
    """Structured result for an optional ECFP6 (Morgan radius 3) extraction."""

    status: str
    reason: str
    bits: Optional[list[int]] = None
    radius: int = 3
    n_bits: int = 2048

    @property
    def ok(self) -> bool:
        return self.status == "valid"


def _smiles_shape_is_well_formed(smiles: str) -> bool:
    if any(character.isspace() for character in smiles):
        return False
    if re.search(r"[^A-Za-z0-9@+\-\[\]\(\)=#$%./\\:]", smiles):
        return False
    stack = []
    for character in smiles:
        if character in "([":
            stack.append(character)
        elif character == ")":
            if not stack or stack.pop() != "(":
                return False
        elif character == "]":
            if not stack or stack.pop() != "[":
                return False
    if stack:
        return False
    ring_tokens = re.findall(r"(?:%[0-9]{2}|[0-9])", smiles)
    return len(ring_tokens) % 2 == 0


class _RDKitAdapter:
    def __init__(self, chem: Any, all_chem: Any, descriptors: Any):
        self._chem = chem
        self._all_chem = all_chem
        self._descriptors = descriptors

    @classmethod
    def load(cls) -> Optional["_RDKitAdapter"]:
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem, rdMolDescriptors
        except ImportError:
            return None
        return cls(Chem, AllChem, rdMolDescriptors)

    def parse_smiles(self, smiles: str) -> Any:
        return self._chem.MolFromSmiles(smiles)

    def mol_to_graph(self, mol: Any) -> Mapping[str, Any]:
        node_features = []
        for atom in mol.GetAtoms():
            node_features.append([
                int(atom.GetAtomicNum()),
                int(atom.GetTotalDegree()),
                int(atom.GetFormalCharge()),
                int(atom.GetIsAromatic()),
            ])
        edges = [[], []]
        for bond in mol.GetBonds():
            start = int(bond.GetBeginAtomIdx())
            end = int(bond.GetEndAtomIdx())
            edges[0].extend((start, end))
            edges[1].extend((end, start))
        return {"node_features": node_features, "edge_index": edges}

    def ecfp6(self, mol: Any, *, radius: int, n_bits: int) -> list[int]:
        generator = getattr(self._descriptors, "GetMorganGenerator", None)
        if generator is not None:
            bit_vector = generator(radius=radius, fpSize=n_bits).GetFingerprint(mol)
        else:
            bit_vector = self._all_chem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return [int(bit) for bit in bit_vector.ToBitString()]


def _resolve_adapter(adapter: Any) -> Optional[Any]:
    if adapter is False:
        return None
    return _RDKitAdapter.load() if adapter is None else adapter


def validate_ligand_graph(graph: Any) -> GraphValidationResult:
    """Validate a dependency-free graph mapping with node and edge shape checks."""

    if not isinstance(graph, Mapping):
        return GraphValidationResult("invalid", "graph_type", input_kind="graph")
    nodes = graph.get("node_features", graph.get("x"))
    edges = graph.get("edge_index")
    if not isinstance(nodes, (list, tuple)) or not nodes:
        return GraphValidationResult("invalid", "empty_graph", input_kind="graph")
    if any(not isinstance(row, (list, tuple)) or not row for row in nodes):
        return GraphValidationResult("invalid", "node_feature_shape", input_kind="graph")
    width = len(nodes[0])
    if any(len(row) != width for row in nodes):
        return GraphValidationResult("invalid", "node_feature_shape", input_kind="graph")
    if not isinstance(edges, (list, tuple)) or len(edges) != 2:
        return GraphValidationResult("invalid", "edge_index_shape", input_kind="graph")
    if not isinstance(edges[0], (list, tuple)) or not isinstance(edges[1], (list, tuple)):
        return GraphValidationResult("invalid", "edge_index_shape", input_kind="graph")
    if len(edges[0]) != len(edges[1]):
        return GraphValidationResult("invalid", "edge_index_shape", input_kind="graph")
    node_count = len(nodes)
    if any(not isinstance(index, int) or index < 0 or index >= node_count for row in edges for index in row):
        return GraphValidationResult("invalid", "edge_index_bounds", input_kind="graph")
    return GraphValidationResult("valid", "validated_graph", graph=graph, feature_width=width, input_kind="graph")


def smiles_to_graph(smiles: Any, *, adapter: Any = None) -> GraphValidationResult:
    """Validate SMILES and optionally convert it with an RDKit-compatible adapter."""

    if not isinstance(smiles, str) or not smiles.strip():
        return GraphValidationResult("invalid", "blank_smiles", input_kind="smiles")
    smiles = smiles.strip()
    if not _smiles_shape_is_well_formed(smiles):
        return GraphValidationResult("invalid", "malformed_smiles", input_kind="smiles")
    resolved = _resolve_adapter(adapter)
    if resolved is None:
        return GraphValidationResult("dependency_unavailable", "RDKit unavailable", input_kind="smiles")
    try:
        molecule = resolved.parse_smiles(smiles)
    except Exception:
        return GraphValidationResult("invalid", "smiles_parser_error", input_kind="smiles")
    if molecule is None:
        return GraphValidationResult("invalid", "malformed_smiles", input_kind="smiles")
    try:
        graph = resolved.mol_to_graph(molecule)
    except Exception:
        return GraphValidationResult("invalid", "graph_conversion_error", input_kind="smiles")
    result = validate_ligand_graph(graph)
    return GraphValidationResult(result.status, result.reason, result.graph, result.feature_width, "smiles")


def extract_ecfp6(smiles: Any, *, adapter: Any = None) -> FingerprintResult:
    """Extract ECFP6 using Morgan radius 3 and 2048 bits when RDKit is available."""

    if not isinstance(smiles, str) or not smiles.strip():
        return FingerprintResult("invalid", "blank_smiles")
    smiles = smiles.strip()
    if not _smiles_shape_is_well_formed(smiles):
        return FingerprintResult("invalid", "malformed_smiles")
    resolved = _resolve_adapter(adapter)
    if resolved is None:
        return FingerprintResult("dependency_unavailable", "RDKit unavailable")
    try:
        molecule = resolved.parse_smiles(smiles)
    except Exception:
        return FingerprintResult("invalid", "smiles_parser_error")
    if molecule is None:
        return FingerprintResult("invalid", "malformed_smiles")
    try:
        bits = list(resolved.ecfp6(molecule, radius=3, n_bits=2048))
    except Exception:
        return FingerprintResult("invalid", "fingerprint_extraction_error")
    if len(bits) != 2048 or any(bit not in (0, 1) for bit in bits):
        return FingerprintResult("invalid", "fingerprint_shape")
    return FingerprintResult("valid", "ecfp6_extracted", bits=bits)


validate_smiles = smiles_to_graph
ecfp6 = extract_ecfp6
