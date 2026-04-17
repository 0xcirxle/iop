"""Optional ThingSpeak uploader."""

from __future__ import annotations

import time
from typing import Callable, Dict

import requests

from .predictor import PredictionResult


class ThingSpeakUploader:
    """Uploads currents and predictions while respecting ThingSpeak rate limits."""

    def __init__(
        self,
        api_key: str,
        url: str,
        min_interval_seconds: float = 15.0,
        request_get: Callable = requests.get,
    ):
        self.api_key = api_key
        self.url = url
        self.min_interval_seconds = min_interval_seconds
        self.request_get = request_get
        self._last_upload = 0.0

    def upload(
        self,
        currents: Dict[str, float],
        prediction: PredictionResult,
    ) -> bool:
        if not self.api_key or self.api_key.startswith("REPLACE_WITH_"):
            return False

        now = time.time()
        elapsed = now - self._last_upload
        if elapsed < self.min_interval_seconds:
            time.sleep(self.min_interval_seconds - elapsed)

        payload = {
            "api_key": self.api_key,
            "field1": f"{currents['I1']:.3f}",
            "field2": f"{currents['I2']:.3f}",
            "field3": f"{currents['I3']:.3f}",
            "field4": "" if prediction.label_id is None else prediction.label_id,
            "field5": "" if prediction.proba_fault is None else f"{prediction.proba_fault:.3f}",
            "field6": prediction.buffer_fill,
            "field7": prediction.buffer_size,
        }
        response = self.request_get(self.url, params=payload, timeout=10)
        self._last_upload = time.time()
        return response.status_code == 200 and response.text != "0"
