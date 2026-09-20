from __future__ import annotations

import re
from typing import Any

SYSTEM_TERMS = {
    "滑油系统": ("滑油", "回油", "油气分离"),
    "燃油系统": ("燃油", "喷嘴", "燃油泵"),
    "液压系统": ("液压", "作动筒"),
    "气路系统": ("气密", "引气", "气路", "燃烧室", "压气机", "涡轮"),
    "电气系统": ("传感器", "电缆", "插头", "信号", "电压"),
}
COMPONENT_FAMILIES = {
    "轴承": ("轴承",),
    "管路": ("管", "软管", "接头"),
    "密封": ("O型圈", "密封圈", "封严"),
    "法兰机匣": ("法兰", "机匣", "端盖", "检查盖"),
    "转子叶片": ("转子", "涡轮", "压气机", "叶片", "导向器"),
    "齿轮传动": ("齿轮", "轴颈", "驱动轴"),
    "传感电气": ("传感器", "电缆", "插头"),
    "执行机构": ("作动筒", "活门", "泵"),
    "紧固件": ("螺栓", "螺母", "垫片", "支架"),
}


def _all_text(slot_record: dict[str, Any], *, hard_evidence_only: bool = False) -> str:
    return " ".join(
        mention["value"]
        for mentions in slot_record["slots"].values()
        for mention in mentions
        if not hard_evidence_only or mention.get("eligible_for_hard_conflict", False)
    )


def _categories(text: str, mapping: dict[str, tuple[str, ...]]) -> set[str]:
    return {
        category
        for category, terms in mapping.items()
        if any(term.casefold() in text.casefold() for term in terms)
    }


def detect_conflicts(
    left: dict[str, Any], right: dict[str, Any], soft_gate: float = 0.35
) -> dict[str, Any]:
    left_text = _all_text(left, hard_evidence_only=True)
    right_text = _all_text(right, hard_evidence_only=True)
    left_systems = _categories(left_text, SYSTEM_TERMS)
    right_systems = _categories(right_text, SYSTEM_TERMS)
    left_components = _categories(left_text, COMPONENT_FAMILIES)
    right_components = _categories(right_text, COMPONENT_FAMILIES)
    conflicts: list[dict[str, Any]] = []

    if left_systems and right_systems and left_systems.isdisjoint(right_systems):
        conflicts.append(
            {
                "conflict_type": "system_conflict",
                "severity": "hard",
                "evidence_a": "、".join(sorted(left_systems)),
                "evidence_b": "、".join(sorted(right_systems)),
            }
        )
    elif left_components and right_components and left_components.isdisjoint(right_components):
        severity = "hard" if len(left_components) == len(right_components) == 1 else "soft"
        conflicts.append(
            {
                "conflict_type": "component_conflict",
                "severity": severity,
                "evidence_a": "、".join(sorted(left_components)),
                "evidence_b": "、".join(sorted(right_components)),
            }
        )

    left_damage = {
        mention["value"]
        for mention in left["slots"]["damage_state"]
        if mention["modality"] == "confirmed"
    }
    right_damage = {
        mention["value"]
        for mention in right["slots"]["damage_state"]
        if mention["modality"] == "confirmed"
    }
    if left_damage and right_damage and left_damage.isdisjoint(right_damage):
        normalized_left = {re.sub(r"[的了]", "", value) for value in left_damage}
        normalized_right = {re.sub(r"[的了]", "", value) for value in right_damage}
        if normalized_left.isdisjoint(normalized_right):
            conflicts.append(
                {
                    "conflict_type": "cause_conflict",
                    "severity": "soft",
                    "evidence_a": "；".join(sorted(left_damage)),
                    "evidence_b": "；".join(sorted(right_damage)),
                }
            )

    hard = any(conflict["severity"] == "hard" for conflict in conflicts)
    soft = any(conflict["severity"] == "soft" for conflict in conflicts)
    return {
        "conflicts": conflicts,
        "has_hard_conflict": hard,
        "has_soft_conflict": soft,
        "conflict_score": 1.0 if hard else soft_gate if soft else 0.0,
    }
