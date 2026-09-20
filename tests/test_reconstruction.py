from casefusion.reconstruction.case_graphs import build_case_graphs


def test_reconstruction_preserves_source_provenance() -> None:
    local_graphs = [
        {
            "document_id": "D1",
            "text": "轴承磨损。",
            "nodes": [
                {
                    "event_id": "E1",
                    "start": 0,
                    "end": 4,
                    "text": "轴承磨损",
                    "type": "FAILURE",
                    "score": 0.9,
                }
            ],
            "relations": [],
        }
    ]
    artifacts = build_case_graphs(
        local_graphs,
        {"D1": "CASE_1"},
        [],
        [],
    )
    assert artifacts["validation_report"]["passed"]
    graph = artifacts["case_graphs"][0]
    assert graph["events"][0]["source_event_ids"] == ["D1_E1"]
    assert graph["events"][0]["canonical_text"] == "轴承磨损"
