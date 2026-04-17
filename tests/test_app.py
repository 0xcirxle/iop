from motor_fault.app import format_monitor_result, format_prediction_summary


def test_format_prediction_summary_handles_warmup_state():
    summary = format_prediction_summary(
        {
            "ready": False,
            "reason": "warmup",
            "buffer_fill": 7,
            "buffer_size": 128,
        }
    )

    assert summary == "warmup [7/128]"


def test_format_prediction_summary_handles_ready_state():
    summary = format_prediction_summary(
        {
            "ready": True,
            "label": "Faulty",
            "proba_fault": 0.91,
        }
    )

    assert summary == "Faulty p_fault=0.91"


def test_format_monitor_result_includes_currents_and_prediction():
    payload = {
        "timestamp": 0.0,
        "currents": {
            "I1": 1.5,
            "I2": 1.4,
            "I3": 1.3,
        },
        "prediction": {
            "ready": True,
            "label": "Healthy",
            "proba_fault": 0.02,
        },
    }

    assert format_monitor_result(payload) == (
        "currents: I1=1.500 I2=1.400 I3=1.300 | "
        "prediction: Healthy p_fault=0.02"
    )
