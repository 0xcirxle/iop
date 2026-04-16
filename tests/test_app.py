from motor_fault.app import format_monitor_result, format_prediction_summary


def test_format_prediction_summary_includes_votes_when_requested():
    summary = format_prediction_summary(
        {
            "binary": "Faulty",
            "severity": "1u",
            "phase": "Phase 2",
            "load": "Full Load",
            "fault_votes": 3,
            "healthy_votes": 1,
            "window_filled": 4,
        },
        include_votes=True,
    )

    assert summary == "Faulty 1u Phase 2 Full Load [F=3 H=1 W=4]"


def test_format_monitor_result_includes_current_and_rolling_predictions():
    payload = {
        "timestamp": 0.0,
        "currents": {
            "I1": 1.5,
            "I2": 1.4,
            "I3": 1.3,
        },
        "current_prediction": {
            "binary": "Faulty",
            "binary_confidence": 0.91,
            "severity": "1u",
            "phase": "Phase 1",
            "load": "Full Load",
        },
        "rolling_decision": {
            "binary": "Faulty",
            "severity": "1u",
            "phase": "Phase 1",
            "load": "Full Load",
            "fault_votes": 3,
            "healthy_votes": 0,
            "window_filled": 3,
        },
    }

    assert format_monitor_result(payload) == (
        "currents: I1=1.500 I2=1.400 I3=1.300 | "
        "current: Faulty(0.91) 1u Phase 1 Full Load | "
        "rolling: Faulty 1u Phase 1 Full Load [F=3 H=0 W=3]"
    )
