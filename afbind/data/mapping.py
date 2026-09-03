"""Offline, deterministic chain-level UniProt mapping."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ChainMapping(Mapping[str, Any]):
    chain_id: str
    candidate_uniprots: tuple[str, ...]
    selected_uniprot: str | None
    status: str
    reason: str
    api_record_metadata: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)

    def __getitem__(self, key: str) -> Any:
        return {
            "chain_id": self.chain_id,
            "candidate_uniprots": self.candidate_uniprots,
            "selected_uniprot": self.selected_uniprot,
            "status": self.status,
            "reason": self.reason,
            "api_record_metadata": self.api_record_metadata,
        }[key]

    def __iter__(self) -> Iterator[str]:
        return iter(("chain_id", "candidate_uniprots", "selected_uniprot", "status", "reason", "api_record_metadata"))

    def __len__(self) -> int:
        return 6


@dataclass(frozen=True)
class MappingDecision(Mapping[str, Any]):
    pdb_id: str
    candidate_uniprots: tuple[str, ...]
    selected_uniprot: str | None
    status: str
    reason: str
    chains: tuple[ChainMapping, ...] = field(default_factory=tuple)
    api_record_metadata: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)

    def __getitem__(self, key: str) -> Any:
        return {
            "pdb_id": self.pdb_id,
            "candidate_uniprots": self.candidate_uniprots,
            "selected_uniprot": self.selected_uniprot,
            "status": self.status,
            "reason": self.reason,
            "chains": self.chains,
            "api_record_metadata": self.api_record_metadata,
        }[key]

    def __iter__(self) -> Iterator[str]:
        return iter(("pdb_id", "candidate_uniprots", "selected_uniprot", "status", "reason", "chains", "api_record_metadata"))

    def __len__(self) -> int:
        return 7


def _row_candidates(row: Mapping[str, Any]) -> tuple[str, ...]:
    values = row.get("uniprot_ids")
    if values is None:
        values = row.get("uniprot_id", row.get("uniprot", row.get("accession")))
    if isinstance(values, str):
        values = (values,)
    elif values is None:
        values = ()
    elif not isinstance(values, (tuple, list, set, frozenset)):
        values = (str(values),)
    return tuple(sorted({str(value).strip() for value in values if str(value).strip()}))


def resolve_candidates(pdb_id: str, rows: Any) -> MappingDecision:
    """Resolve UniProt candidates per chain without guessing across ambiguity."""

    if not isinstance(pdb_id, str) or not pdb_id.strip():
        return MappingDecision(str(pdb_id), (), None, "invalid", "invalid_pdb_id")
    if isinstance(rows, Mapping):
        rows = (rows,)
    if rows is None or isinstance(rows, (str, bytes)):
        return MappingDecision(pdb_id, (), None, "invalid", "invalid_mapping_rows")
    try:
        row_list = list(rows)
    except TypeError:
        return MappingDecision(pdb_id, (), None, "invalid", "invalid_mapping_rows")
    if not row_list:
        return MappingDecision(pdb_id, (), None, "not_found", "no_mapping_candidates")

    grouped: dict[str, list[tuple[str, Mapping[str, Any]]]] = defaultdict(list)
    chain_metadata: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    all_metadata: list[Mapping[str, Any]] = []
    for row in row_list:
        if not isinstance(row, Mapping):
            return MappingDecision(pdb_id, (), None, "invalid", "invalid_mapping_row")
        row_pdb = row.get("pdb_id", row.get("pdbId"))
        if row_pdb is not None and str(row_pdb).lower() != pdb_id.lower():
            continue
        chain_id = row.get("chain_id", row.get("chain", row.get("auth_chain_id")))
        candidates = _row_candidates(row)
        if chain_id is None or not str(chain_id).strip():
            return MappingDecision(pdb_id, (), None, "invalid", "missing_chain_id", api_record_metadata=(dict(row),))
        all_metadata.append(dict(row))
        normalized_chain_id = str(chain_id).strip()
        chain_metadata[normalized_chain_id].append(dict(row))
        for candidate in candidates:
            grouped[normalized_chain_id].append((candidate, dict(row)))

    if not all_metadata:
        return MappingDecision(pdb_id, (), None, "not_found", "no_mapping_candidates")

    chains: list[ChainMapping] = []
    for chain_id in sorted(chain_metadata):
        candidates = tuple(sorted({candidate for candidate, _ in grouped[chain_id]}))
        metadata = tuple(chain_metadata[chain_id])
        if not candidates:
            chains.append(ChainMapping(chain_id, (), None, "not_found", "unresolved_chain_mapping", metadata))
        elif len(candidates) == 1:
            chains.append(ChainMapping(chain_id, candidates, candidates[0], "resolved", "single_unambiguous_candidate", metadata))
        else:
            chains.append(ChainMapping(chain_id, candidates, None, "ambiguous", "multiple_candidates_for_chain", metadata))

    candidate_uniprots = tuple(sorted({candidate for chain in chains for candidate in chain.candidate_uniprots}))
    if not chains or any(chain.status == "not_found" for chain in chains):
        status, reason, selected = "not_found", "unresolved_chain_mapping", None
    elif any(chain.status == "ambiguous" for chain in chains):
        status, reason, selected = "ambiguous", "multiple_candidates_for_chain", None
    else:
        selected_values = {chain.selected_uniprot for chain in chains}
        selected = next(iter(selected_values)) if len(selected_values) == 1 else None
        status, reason = "resolved", "resolved_chain_level"
    return MappingDecision(pdb_id, candidate_uniprots, selected, status, reason, tuple(chains), tuple(all_metadata))


def resolve_chain_mappings(pdb_id: str, rows: Any) -> MappingDecision:
    """Descriptive alias for callers that emphasize chain-level provenance."""

    return resolve_candidates(pdb_id, rows)
