"""
ElevatorGuard-AI: Telemetry Feature Extraction Engine
Extracts temporal, statistical, and spectral vibration/tension features
from tri-axial accelerometer (MPU-6050) and multi-rope load cell arrays (HX711).
"""

from typing import Dict, Any, List, Union
import numpy as np
from scipy import stats, signal


class TelemetryFeatureExtractor:
    """
    Extracts time-frequency features from elevator telemetry signals.
    Enforces ISO 18738 ride quality standards and EN 81-20 tension balance standards.
    """

    FEATURE_NAMES = [
        "acc_rms_x",
        "acc_rms_y",
        "acc_rms_z",
        "acc_kurtosis_z",
        "acc_peak_z",
        "spectral_kurtosis_z",
        "tension_mean",
        "tension_std",
        "tension_imbalance_ratio"
    ]

    def __init__(
        self,
        sampling_rate_hz: int = 100,
        tension_imbalance_limit: float = 0.15,
        vibration_rms_limit: float = 0.45
    ):
        self.fs = sampling_rate_hz
        self.tension_imbalance_limit = tension_imbalance_limit
        self.vibration_rms_limit = vibration_rms_limit

    def extract_features(
        self,
        accel_x: np.ndarray,
        accel_y: np.ndarray,
        accel_z: np.ndarray,
        tensions: np.ndarray
    ) -> Dict[str, Any]:
        """
        Process a window of telemetry data and compute feature vector.

        Args:
            accel_x: 1D array of X-axis vibration (lateral) [m/s^2]
            accel_y: 1D array of Y-axis vibration (longitudinal) [m/s^2]
            accel_z: 1D array of Z-axis vibration (vertical) [m/s^2]
            tensions: 2D array of rope tensions [samples x num_ropes] or 1D array of 4 tensions.

        Returns:
            Dictionary containing computed features, normalized feature vector, and alarm flags.
        """
        accel_x = np.asarray(accel_x, dtype=np.float32)
        accel_y = np.asarray(accel_y, dtype=np.float32)
        accel_z = np.asarray(accel_z, dtype=np.float32)
        tensions = np.asarray(tensions, dtype=np.float32)

        # 1. Root Mean Square (RMS) Vibration
        rms_x = float(np.sqrt(np.mean(accel_x ** 2))) if len(accel_x) > 0 else 0.0
        rms_y = float(np.sqrt(np.mean(accel_y ** 2))) if len(accel_y) > 0 else 0.0
        rms_z = float(np.sqrt(np.mean(accel_z ** 2))) if len(accel_z) > 0 else 0.0

        # 2. Statistical Moments (Kurtosis & Peak Amplitude)
        if len(accel_z) > 4:
            kurtosis_z = float(stats.kurtosis(accel_z, fisher=False)) # Pearson kurtosis (normal=3.0)
            peak_z = float(np.max(np.abs(accel_z)))
        else:
            kurtosis_z = 3.0
            peak_z = 0.0

        # 3. Spectral Kurtosis & FFT Peak
        if len(accel_z) >= 16:
            # Detrend for FFT
            detrended = accel_z - np.mean(accel_z)
            fft_vals = np.abs(np.fft.rfft(detrended))
            spec_kurtosis = float(stats.kurtosis(fft_vals, fisher=False)) if len(fft_vals) > 4 else 3.0
            freqs = np.fft.rfftfreq(len(detrended), d=1.0 / self.fs)
            dominant_freq = float(freqs[np.argmax(fft_vals[1:]) + 1]) if len(fft_vals) > 1 else 0.0
        else:
            spec_kurtosis = 3.0
            dominant_freq = 0.0

        # 4. Rope Tension Imbalance Analysis
        if tensions.ndim == 2:
            # Average across the window for each rope
            mean_rope_tensions = np.mean(tensions, axis=0)
        elif tensions.ndim == 1:
            mean_rope_tensions = tensions
        else:
            mean_rope_tensions = np.array([2500.0, 2500.0, 2500.0, 2500.0], dtype=np.float32)

        avg_tension = float(np.mean(mean_rope_tensions)) if len(mean_rope_tensions) > 0 else 1.0
        std_tension = float(np.std(mean_rope_tensions)) if len(mean_rope_tensions) > 0 else 0.0

        if avg_tension > 1e-3 and len(mean_rope_tensions) > 1:
            imbalance_ratio = float((np.max(mean_rope_tensions) - np.min(mean_rope_tensions)) / avg_tension)
        else:
            imbalance_ratio = 0.0

        # 5. Fault Alarms Evaluation
        has_tension_imbalance_alarm = imbalance_ratio > self.tension_imbalance_limit
        has_vibration_alarm = rms_z > self.vibration_rms_limit

        feature_vector = np.array([
            rms_x,
            rms_y,
            rms_z,
            kurtosis_z,
            peak_z,
            spec_kurtosis,
            avg_tension,
            std_tension,
            imbalance_ratio
        ], dtype=np.float32)

        return {
            "acc_rms_x": rms_x,
            "acc_rms_y": rms_y,
            "acc_rms_z": rms_z,
            "acc_kurtosis_z": kurtosis_z,
            "acc_peak_z": peak_z,
            "spectral_kurtosis_z": spec_kurtosis,
            "dominant_frequency_hz": dominant_freq,
            "tension_mean": avg_tension,
            "tension_std": std_tension,
            "tension_imbalance_ratio": imbalance_ratio,
            "individual_tensions": [float(t) for t in mean_rope_tensions],
            "tension_imbalance_alarm": has_tension_imbalance_alarm,
            "vibration_alarm": has_vibration_alarm,
            "feature_vector": feature_vector
        }
