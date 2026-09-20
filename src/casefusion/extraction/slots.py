from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = PROJECT_ROOT / "schemas" / "diagnostic_slots.schema.json"
SLOT_NAMES = (
    "component",
    "location",
    "stage",
    "phenomenon",
    "damage_state",
    "inspection_evidence",
    "cause",
    "maintenance_action",
    "reassembly_control",
    "verification",
)

COMPONENT_PATTERN = re.compile(
    r"(?:\d+号)?(?:轴承|转子|涡轮|压气机|叶片|导向器|机匣|法兰|O型圈|密封圈|封严环|"
    r"回油管|燃油管|滑油管|液压管|管路|接头|软管|齿轮|齿轮箱|传感器|电缆|插头|"
    r"作动筒|活门|泵|喷嘴|燃烧室|整流罩|支架|螺栓|螺母|垫片|衬套|轴颈|轴|端盖)"
)
LOCATION_PATTERN = re.compile(
    r"[\u4e00-\u9fffA-Za-z0-9]{0,14}(?:进气端|排气端|前端|后端|内侧|外侧|上部|下部|"
    r"根部|尖部|边缘|端面|结合面|密封面|安装孔|轴承腔|检查盖|接头处|法兰面|沟底|槽底|局部)"
)
STAGE_PATTERN = re.compile(
    r"(?:总装|装配|复装|分解|拆卸|试装|试车|运转|运行|起动|巡航|盘转|气密试验|"
    r"功能试验|性能试验|入库|运输|检修|返工|对接|紧固)(?:过程中|阶段|后|前|时)?"
)
PHENOMENON_PATTERN = re.compile(
    r"(?:漏油|渗油|泄漏|冒泡|异响|打鼓声|啸叫声|振动(?:偏高|超限)?|温度(?:偏高|异常)|"
    r"卡滞|不灵活|压力(?:下降|异常|为零)|假信号|信号异常|金属屑|堵塞|不同步|间隙异常|"
    r"转动阻力(?:大|异常)|无法转动|无响应)"
)
DAMAGE_PATTERN = re.compile(
    r"(?:挤压坑|压伤|划伤|划痕|裂纹|断裂|磨损|擦伤|凹陷|变形|翘曲|切伤|咬伤|烧蚀|"
    r"剥落|脱落|掉块|崩边|毛刺|翻边|腐蚀|过热|变色|疲劳|松动|错位|窜动|报废)"
)
INSPECTION_PATTERN = re.compile(
    r"(?:检查|测量|复测|检验|鉴定|分析|显示|发现|确认|观察|试验|探伤|吹扫)"
)
CAUSE_PATTERN = re.compile(
    r"(?:原因|由于|因|导致|造成|形成|未按|未能|不当|错误|偏差|不匹配|不足|过大|过小)"
)
ACTION_PATTERN = re.compile(
    r"(?:更换|换新|拆除|拆下|分解|清理|清洗|修整|修平|打磨|补焊|重加工|调整|涂敷|润滑|返修|报废)"
)
REASSEMBLY_PATTERN = re.compile(
    r"(?:复装|重新安装|重新装配|对接|自然贴合|分级紧固|对称紧固|星形紧固|控制力矩|控制间隙)"
)
VERIFICATION_PATTERN = re.compile(
    r"(?:复查|复测|验证|确认合格|重新检查|气密试验|功能测试|功能试验|盘转检查|试车|压力稳定|无气泡|无异响)"
)

PATTERNS = {
    "component": COMPONENT_PATTERN,
    "location": LOCATION_PATTERN,
    "stage": STAGE_PATTERN,
    "phenomenon": PHENOMENON_PATTERN,
    "damage_state": DAMAGE_PATTERN,
    "inspection_evidence": INSPECTION_PATTERN,
    "cause": CAUSE_PATTERN,
    "maintenance_action": ACTION_PATTERN,
    "reassembly_control": REASSEMBLY_PATTERN,
    "verification": VERIFICATION_PATTERN,
}


def sentence_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    start = 0
    for match in re.finditer(r"[。！？；\n]", text):
        end = match.end()
        sentence = text[start:end].strip()
        if sentence:
            leading = len(text[start:end]) - len(text[start:end].lstrip())
            spans.append((start + leading, end, sentence))
        start = end
    if start < len(text):
        sentence = text[start:].strip()
        if sentence:
            leading = len(text[start:]) - len(text[start:].lstrip())
            spans.append((start + leading, len(text), sentence))
    return spans


def modality(sentence: str) -> str:
    if re.search(r"(?:怀疑|疑似|初步判断|初步认为)", sentence):
        return "suspected"
    if re.search(r"(?:可能|可导致|风险|或为|推测)", sentence):
        return "possible"
    if re.search(r"(?:确认|发现|显示|证实|结果为|检查可见)", sentence):
        return "confirmed"
    return "unknown"


def _value_for_slot(slot: str, match: re.Match[str], sentence: str) -> tuple[str, int, int]:
    if slot in {"component", "location", "stage", "phenomenon", "damage_state"}:
        return match.group(0), match.start(), match.end()
    relative_start = max(0, match.start() - 18)
    relative_end = min(len(sentence), match.end() + 28)
    raw = sentence[relative_start:relative_end]
    stripped = raw.strip("，。；： ")
    leading = len(raw) - len(raw.lstrip("，。；： "))
    return stripped, relative_start + leading, relative_start + leading + len(stripped)


def extract_slots(document_id: str, text: str) -> dict[str, Any]:
    slots: dict[str, list[dict[str, Any]]] = {name: [] for name in SLOT_NAMES}
    seen: dict[str, set[tuple[int, int, str]]] = {name: set() for name in SLOT_NAMES}
    for sentence_start, _, sentence in sentence_spans(text):
        sentence_modality = modality(sentence)
        for slot, pattern in PATTERNS.items():
            for match in pattern.finditer(sentence):
                value, value_start, value_end = _value_for_slot(slot, match, sentence)
                start = sentence_start + value_start
                end = sentence_start + value_end
                signature = (start, end, value)
                if signature in seen[slot]:
                    continue
                seen[slot].add(signature)
                base_confidence = (
                    0.90 if slot in {"component", "phenomenon", "damage_state"} else 0.78
                )
                if sentence_modality == "confirmed":
                    base_confidence = min(0.98, base_confidence + 0.06)
                eligible_for_hard_conflict = base_confidence >= 0.90 and sentence_modality not in {
                    "possible",
                    "suspected",
                }
                slots[slot].append(
                    {
                        "value": value,
                        "evidence_span": text[start:end],
                        "evidence_sentence": sentence,
                        "start": start,
                        "end": end,
                        "confidence": base_confidence,
                        "modality": sentence_modality,
                        "evidence_origin": "rule",
                        "eligible_for_hard_conflict": eligible_for_hard_conflict,
                    }
                )
    record = {"document_id": document_id, "slots": slots}
    validate_slot_record(record, text)
    return record


def validate_slot_record(record: dict[str, Any], text: str | None = None) -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(record)
    if text is not None:
        for mentions in record["slots"].values():
            for mention in mentions:
                if not 0 <= mention["start"] < mention["end"] <= len(text):
                    raise ValueError(f"{record['document_id']}: invalid slot offsets")
                if mention["evidence_sentence"] not in text:
                    raise ValueError(
                        f"{record['document_id']}: evidence sentence is not in source text"
                    )
                if text[mention["start"] : mention["end"]] != mention["evidence_span"]:
                    raise ValueError(
                        f"{record['document_id']}: evidence span does not match offsets"
                    )


def extract_corpus_slots(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [extract_slots(document["document_id"], document["text"]) for document in documents]
