import numpy as np

from casefusion.fusion.event_states import (
    apply_event_state_rule,
    event_hard_conflict,
)


def test_event_rule_is_equivalence_then_forbidden_then_related() -> None:
    states = apply_event_state_rule(
        np.array([0.9, 0.9, 0.2]),
        np.array([0.1, 0.8, 0.7]),
        np.array([False, True, False]),
        equivalent_threshold=0.7,
        forbidden_threshold=0.6,
    )
    assert states.tolist() == ["equivalent", "forbidden", "forbidden"]


def test_hard_conflict_is_non_compensatory() -> None:
    row = {
        "event_text_a": "轴承磨损 0.2 mm",
        "event_text_b": "轴承磨损 0.8 mm",
        "event_type_a": "FAILURE",
        "event_type_b": "FAILURE",
    }
    assert event_hard_conflict(row)
