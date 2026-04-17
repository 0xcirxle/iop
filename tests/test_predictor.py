from pathlib import Path

from motor_fault.predictor import MotorFaultPredictor


MODEL_PATH = Path(__file__).resolve().parent.parent / "motor_fault_model" / "model.joblib"


def test_predictor_reports_warmup_before_buffer_fills():
    predictor = MotorFaultPredictor(MODEL_PATH)

    result = predictor.update(1.47, 1.46, 1.48)

    assert result.ready is False
    assert result.reason == "warmup"
    assert result.buffer_fill == 1
    assert result.buffer_size == 128
    assert result.label is None
    assert result.proba_fault is None


def test_predictor_emits_ready_result_after_rolling_buffer_fills():
    predictor = MotorFaultPredictor(MODEL_PATH)

    result = None
    for _ in range(128):
        result = predictor.update(1.47, 1.46, 1.48)

    assert result is not None
    assert result.ready is True
    assert result.reason is None
    assert result.label in {"Healthy", "Faulty"}
    assert 0.0 <= float(result.proba_fault) <= 1.0


def test_motor_off_clears_the_warmup_buffer():
    predictor = MotorFaultPredictor(MODEL_PATH)

    for _ in range(8):
        predictor.update(1.47, 1.46, 1.48)

    motor_off = predictor.update(0.0, 0.0, 0.0)
    next_result = predictor.update(1.47, 1.46, 1.48)

    assert motor_off.ready is False
    assert motor_off.reason == "motor_off"
    assert motor_off.buffer_fill == 0
    assert next_result.ready is False
    assert next_result.reason == "warmup"
    assert next_result.buffer_fill == 1
