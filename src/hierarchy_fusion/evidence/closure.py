"""Detect unsupported cross-case pattern composition before residual fusion."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass

from hierarchy_fusion.patterns.extractor import (
    ALLOWED_ROLE_PATHS,
    PatternInstance,
    pattern_identifier,
    signature,
)


@dataclass(frozen=True, slots=True)
class PatternSupportRecord:
    pattern_id: str
    signature: str
    supporting_case_ids: tuple[str, ...]
    case_support_count: int
    case_support_ratio: float
    evidence_admissible: bool
    shared_pattern: bool

    def to_dict(self) -> dict:
        return {
            "pattern_id": self.pattern_id,
            "signature": self.signature,
            "supporting_case_ids": list(self.supporting_case_ids),
            "case_support_count": self.case_support_count,
            "case_support_ratio": self.case_support_ratio,
            "evidence_admissible": self.evidence_admissible,
            "shared_pattern": self.shared_pattern,
        }


@dataclass(frozen=True, slots=True)
class ClosureAssessment:
    case_ids: tuple[str, ...]
    potential_signatures: frozenset[str]
    admitted_signatures: frozenset[str]
    shared_signatures: frozenset[str]
    unsupported_signatures: frozenset[str]
    pattern_support: tuple[PatternSupportRecord, ...]
    distortion: float
    truncated: bool

    def to_dict(self, include_signatures: bool = False) -> dict:
        value = asdict(self)
        value.update(
            {
                "case_ids": list(self.case_ids),
                "potential_pattern_count": len(self.potential_signatures),
                "admitted_pattern_count": len(self.admitted_signatures),
                "shared_pattern_count": len(self.shared_signatures),
                "unsupported_pattern_count": len(self.unsupported_signatures),
                "pattern_support": [record.to_dict() for record in self.pattern_support],
                "cfpr": len(self.unsupported_signatures) / max(1, len(self.potential_signatures)),
            }
        )
        if include_signatures:
            value["potential_signatures"] = sorted(self.potential_signatures)
            value["admitted_signatures"] = sorted(self.admitted_signatures)
            value["shared_signatures"] = sorted(self.shared_signatures)
            value["unsupported_signatures"] = sorted(self.unsupported_signatures)
        else:
            value.pop("potential_signatures", None)
            value.pop("admitted_signatures", None)
            value.pop("shared_signatures", None)
            value.pop("unsupported_signatures", None)
        return value


class EvidenceClosureIndex:
    def __init__(self, patterns_by_case: dict[str, list[PatternInstance]]) -> None:
        self.case_signatures: dict[str, set[str]] = {}
        self.support_cases: dict[str, set[str]] = defaultdict(set)
        self.parts: dict[str, tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = {}
        for case_id, patterns in patterns_by_case.items():
            signatures: set[str] = set()
            # q ≼ G_i means structural containment in one real case. Therefore
            # a verified long path also supports every contiguous role-valid
            # subpath; support is not restricted to exact serialized records.
            for pattern in patterns:
                signatures.add(pattern.signature)
                for start in range(len(pattern.roles) - 1):
                    for end in range(start + 2, len(pattern.roles) + 1):
                        roles = pattern.roles[start:end]
                        if roles not in ALLOWED_ROLE_PATHS:
                            continue
                        sub_signature = signature(
                            roles,
                            pattern.concepts[start:end],
                            pattern.relations[start : end - 1],
                        )
                        signatures.add(sub_signature)
                        self.parts[sub_signature] = (
                            roles,
                            pattern.concepts[start:end],
                            pattern.relations[start : end - 1],
                        )
            self.case_signatures[case_id] = signatures
            for pattern_signature in signatures:
                self.support_cases[pattern_signature].add(case_id)
            for pattern in patterns:
                self.parts[pattern.signature] = (pattern.roles, pattern.concepts, pattern.relations)
        self._cache: dict[tuple, ClosureAssessment] = {}

    def support(self, pattern_signature: str, case_ids: set[str]) -> int:
        return len(self.support_cases.get(pattern_signature, set()) & case_ids)

    def support_records(
        self,
        pattern_signatures: set[str] | frozenset[str],
        case_ids: set[str],
        *,
        min_case_support: int = 1,
        shared_min_case_support: int = 2,
    ) -> tuple[PatternSupportRecord, ...]:
        denominator = max(1, len(case_ids))
        records: list[PatternSupportRecord] = []
        for pattern_signature in sorted(pattern_signatures):
            supporting = tuple(sorted(self.support_cases.get(pattern_signature, set()) & case_ids))
            count = len(supporting)
            records.append(
                PatternSupportRecord(
                    pattern_id=pattern_identifier(pattern_signature),
                    signature=pattern_signature,
                    supporting_case_ids=supporting,
                    case_support_count=count,
                    case_support_ratio=count / denominator,
                    evidence_admissible=count >= min_case_support,
                    shared_pattern=count >= shared_min_case_support,
                )
            )
        return tuple(records)

    def _exact(self, case_ids: set[str]) -> set[str]:
        return (
            set().union(*(self.case_signatures.get(case_id, set()) for case_id in case_ids))
            if case_ids
            else set()
        )

    def assess(
        self,
        case_ids: set[str],
        *,
        max_path_edges: int = 4,
        min_case_support: int = 1,
        shared_min_case_support: int = 2,
        max_potential_paths: int = 20000,
    ) -> ClosureAssessment:
        cache_key = (
            tuple(sorted(case_ids)),
            max_path_edges,
            min_case_support,
            shared_min_case_support,
            max_potential_paths,
        )
        if cache_key in self._cache:
            return self._cache[cache_key]
        exact = self._exact(case_ids)
        transitions: dict[tuple[str, str], set[tuple[str, str, str]]] = defaultdict(set)
        for pattern_signature in exact:
            roles, concepts, relations = self.parts[pattern_signature]
            for idx, relation in enumerate(relations):
                transitions[(roles[idx], roles[idx + 1])].add(
                    (concepts[idx], relation, concepts[idx + 1])
                )

        potential = set(exact)
        truncated = False
        for role_path in sorted(ALLOWED_ROLE_PATHS, key=lambda item: (len(item), item)):
            if len(role_path) - 1 > max_path_edges:
                continue
            options = [
                transitions.get((role_path[idx], role_path[idx + 1]), set())
                for idx in range(len(role_path) - 1)
            ]
            if not options or any(not option for option in options):
                continue
            # Dynamic join on the shared canonical concept. This avoids the
            # cartesian-product explosion of unrelated adjacent transitions.
            partials = [((edge[0], edge[2]), (edge[1],)) for edge in options[0]]
            for next_options in options[1:]:
                by_source: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
                for edge in next_options:
                    by_source[edge[0]].append(edge)
                extended: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
                for concepts_so_far, relations_so_far in partials:
                    for edge in by_source.get(concepts_so_far[-1], []):
                        extended.append(((*concepts_so_far, edge[2]), (*relations_so_far, edge[1])))
                        if len(extended) + len(potential) >= max_potential_paths:
                            truncated = True
                            break
                    if truncated:
                        break
                partials = extended
                if truncated or not partials:
                    break
            if not truncated:
                for concepts, relations in partials:
                    potential.add(signature(role_path, concepts, relations))
                    if len(potential) >= max_potential_paths:
                        truncated = True
                        break
            if truncated:
                break
        admitted = {
            value for value in potential if self.support(value, case_ids) >= min_case_support
        }
        shared = {
            value for value in admitted if self.support(value, case_ids) >= shared_min_case_support
        }
        unsupported = potential - admitted
        support_records = self.support_records(
            potential,
            case_ids,
            min_case_support=min_case_support,
            shared_min_case_support=shared_min_case_support,
        )
        result = ClosureAssessment(
            case_ids=tuple(sorted(case_ids)),
            potential_signatures=frozenset(potential),
            admitted_signatures=frozenset(admitted),
            shared_signatures=frozenset(shared),
            unsupported_signatures=frozenset(unsupported),
            pattern_support=support_records,
            # A truncated enumeration has unknown omitted risk and therefore
            # must never pass the distortion gate as if it were complete.
            distortion=1.0 if truncated else len(unsupported) / max(1, len(potential)),
            truncated=truncated,
        )
        self._cache[cache_key] = result
        return result
