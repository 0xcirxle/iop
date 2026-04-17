import numpy as np

from motor_fault.features import RollingFeatureBuffer, extract_base_feature_row


def test_extract_base_feature_row_matches_expected_ratios():
    result = extract_base_feature_row((1.50, 1.20, 1.30))

    assert result.ready is True
    np.testing.assert_allclose(
        result.feature_row[:4],
        np.array([1.3333333333, 1.125, 0.9, 0.975], dtype=np.float64),
        rtol=1e-6,
        atol=1e-6,
    )
    assert result.reason is None


def test_extract_base_feature_row_flags_motor_off():
    result = extract_base_feature_row((0.0, 0.0, 0.0))

    assert result.ready is False
    assert result.feature_row is None
    assert result.reason == "motor_off"


def test_window_feature_vector_has_expected_shape_once_buffer_is_full():
    rolling = RollingFeatureBuffer(buffer_n=128)
    for _ in range(128):
        rolling.update((1.47, 1.46, 1.48))

    feature_vector = rolling.current_feature_vector()

    assert feature_vector.shape == (38,)
