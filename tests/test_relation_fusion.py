from casefusion.fusion.relation_states import (
    RelationFeatureRecord,
    decide_relation_state,
    final_equivalence_score,
)


def feature(**overrides: float) -> RelationFeatureRecord:
    values = {
        "direct_head_text": 0.9,
        "direct_tail_text": 0.9,
        "reverse_head_text": 0.1,
        "reverse_tail_text": 0.1,
        "evidence_similarity": 0.8,
        "relation_type_match": 1.0,
        "relation_role_compatible": 1.0,
        "modality_match": 1.0,
        "head_mapping_support": 0.9,
        "tail_mapping_support": 0.9,
        "reverse_head_mapping_support": 0.0,
        "reverse_tail_mapping_support": 0.0,
        "head_forbidden_risk": 0.0,
        "tail_forbidden_risk": 0.0,
        "numeral_conflict": 0.0,
        "negation_conflict": 0.0,
        "maintenance_state_conflict": 0.0,
        "reverse_direction_conflict": 0.0,
        "bge_direct_head_text": 0.9,
        "bge_direct_tail_text": 0.85,
        "bge_reverse_head_text": 0.1,
        "bge_reverse_tail_text": 0.1,
        "bge_evidence_similarity": 0.8,
        "relation_semantic_similarity": 0.9,
        "action_semantic_similarity": 0.0,
        "action_semantic_available": 0.0,
        "narrative_position_similarity": 0.9,
    }
    values.update(overrides)
    return RelationFeatureRecord(**values)


def test_equivalence_score_uses_the_weaker_endpoint() -> None:
    strong = final_equivalence_score(feature())
    weak_tail = final_equivalence_score(feature(bge_direct_tail_text=0.2))
    assert strong > 0.8
    assert weak_tail == 0.2


def test_relation_decision_applies_conflict_before_equivalence() -> None:
    decision = decide_relation_state(
        {},
        feature(maintenance_state_conflict=1.0),
        {},
        conflict_threshold=0.4,
        equivalent_threshold=0.7,
        complementary_probability=0.9,
        complementary_threshold=0.5,
    )
    assert decision.predicted_state == "conflicting"
    assert decision.graph_action == "mark_conflict"
