from motor_fault.app import build_prediction_summary, format_monitor_result
from motor_fault.predictor import PredictionResult


def test_build_prediction_summary_returns_labels_in_expected_order():
    predictions = {
        "binary": PredictionResult("binary", 0, "Healthy", {0: 0.9}),
        "severity": PredictionResult("severity", 0, "Healthy", {0: 0.9}),
        "phase": PredictionResult("phase", 0, "Healthy", {0: 0.9}),
        "load": PredictionResult("load", 2, "Full Load", {2: 0.8}),
    }

    assert build_prediction_summary(predictions) == {
        "binary": "Healthy",
        "severity": "Healthy",
        "phase": "Healthy",
        "load": "Full Load",
    }


def test_format_monitor_result_includes_currents_and_prediction_labels():
    payload = {
        "timestamp": 0.0,
        "currents": {
            "I1": 0.0,
            "I2": 0.1,
            "I3": -0.2,
        },
        "prediction_summary": {
            "binary": "Healthy",
            "severity": "Healthy",
            "phase": "Healthy",
            "load": "No Load",
        },
    }

    assert format_monitor_result(payload) == (
        "currents: I1=0.000 I2=0.100 I3=-0.200 | "
        "predictions: binary=Healthy severity=Healthy phase=Healthy load=No Load"
    )
