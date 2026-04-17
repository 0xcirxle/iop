"""Compatibility exports for the integrated threshold-model feature pipeline."""

from motor_fault_model.features import (
    BASE_FEATURE_NAMES,
    BUFFER_N,
    FEATURE_NAMES,
    MOTOR_OFF_THRESHOLD_AMPS,
    RMS_STRIDE_SAMPLES,
    RMS_WINDOW_SAMPLES,
    RollingFeatureBuffer,
    build_feature_matrix_from_currents,
    build_window_feature_matrix,
    extract_base_feature_matrix,
    extract_base_feature_row,
    simulate_rms_stream,
    window_feature_vector,
)

__all__ = [
    "BASE_FEATURE_NAMES",
    "BUFFER_N",
    "FEATURE_NAMES",
    "MOTOR_OFF_THRESHOLD_AMPS",
    "RMS_STRIDE_SAMPLES",
    "RMS_WINDOW_SAMPLES",
    "RollingFeatureBuffer",
    "build_feature_matrix_from_currents",
    "build_window_feature_matrix",
    "extract_base_feature_matrix",
    "extract_base_feature_row",
    "simulate_rms_stream",
    "window_feature_vector",
]
