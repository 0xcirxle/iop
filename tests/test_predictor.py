from pathlib import Path

from motor_fault.predictor import MotorFaultPredictor, RollingMotorDecision


MODEL_DIR = Path(__file__).resolve().parent.parent / "trained_models_v2"


def test_saved_models_can_run_inference():
    predictor = MotorFaultPredictor(MODEL_DIR)
    results = predictor.predict(1.456810, 1.428707, 1.404529)

    assert set(results) == {"binary", "severity", "phase", "load"}
    assert results["binary"].label in {"Healthy", "Faulty"}
    for result in results.values():
        assert isinstance(result.class_id, int)
        assert result.probabilities
        assert result.confidence is not None


def test_predict_live_hides_fault_details_for_healthy_binary_result():
    predictor = MotorFaultPredictor(MODEL_DIR)
    results = predictor.predict(0.0, 0.0, 0.0)
    summary = predictor.summarize_prediction(results, 0.0, 0.0, 0.0)

    assert summary["binary"] in {"Healthy", "Faulty", "Uncertain"}
    if summary["binary"] == "Healthy":
        assert summary["severity"] == "N/A"
        assert summary["phase"] == "N/A"


def test_rolling_decision_promotes_fault_after_enough_votes():
    rolling = RollingMotorDecision(window_size=5, fault_votes_required=3)
    faulty_prediction = {
        "safety_status": "OK",
        "binary": "Faulty",
        "binary_confidence": 0.9,
        "severity": "1u",
        "severity_confidence": 0.8,
        "phase": "Phase 1",
        "phase_confidence": 0.8,
        "load": "Full Load",
        "load_confidence": 0.9,
    }

    rolling.update(faulty_prediction)
    rolling.update(faulty_prediction)
    result = rolling.update(faulty_prediction)

    assert result["rolling_decision"]["binary"] == "Faulty"
    assert result["rolling_decision"]["severity"] == "1u"
    assert result["rolling_decision"]["phase"] == "Phase 1"
