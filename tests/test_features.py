import numpy as np

from motor_fault.features import build_feature_vector


def test_build_feature_vector_matches_training_order():
    vector = build_feature_vector(2.0, 0.5, 1.5)
    i_mean = np.mean([2.0, 0.5, 1.5])
    eps = 1e-6
    expected = np.array(
        [[
            2.0,
            0.5,
            1.5,
            4.0,
            i_mean,
            2.0,
            0.5,
            1.5,
            np.std([2.0, 0.5, 1.5]),
            1.5 / (i_mean + eps),
            2.0 / (i_mean + eps),
            0.5 / (i_mean + eps),
            1.5 / (i_mean + eps),
            1.5,
            -1.0,
            -0.5,
        ]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(vector, expected)
