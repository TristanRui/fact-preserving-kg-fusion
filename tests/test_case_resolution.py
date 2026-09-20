import numpy as np

from casefusion.case_resolution import CaseResolutionConfig, resolve_cases


def test_case_resolution_uses_frozen_hybrid_and_average_linkage() -> None:
    records = [
        {"document_id": "A", "text": "滑油系统轴承磨损，检查轴承并更换。"},
        {"document_id": "B", "text": "滑油系统轴承磨损，复查后更换轴承。"},
        {"document_id": "C", "text": "电气系统传感器电压异常，更换插头。"},
    ]
    embeddings = {
        "A": np.array([1.0, 0.0]),
        "B": np.array([1.0, 0.0]),
        "C": np.array([0.0, 1.0]),
    }
    result = resolve_cases(
        records,
        embeddings,
        config=CaseResolutionConfig(candidate_top_k=2),
    )
    assert result.assignments["A"] == result.assignments["B"]
    assert result.assignments["A"] != result.assignments["C"]
    assert all(
        decision.same_case_score == 0.0
        for decision in result.candidate_decisions
        if decision.hard_conflict
    )
