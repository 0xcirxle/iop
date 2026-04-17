from motor_fault.cloud import ThingSpeakUploader
from motor_fault.predictor import PredictionResult


class DummyResponse:
    def __init__(self, status_code=200, text="1"):
        self.status_code = status_code
        self.text = text


def test_thingspeak_payload_shape():
    captured = {}

    def fake_get(url, params, timeout):
        captured["url"] = url
        captured["params"] = params
        captured["timeout"] = timeout
        return DummyResponse()

    uploader = ThingSpeakUploader(
        api_key="TEST_KEY",
        url="https://api.thingspeak.com/update",
        min_interval_seconds=0.0,
        request_get=fake_get,
    )
    prediction = PredictionResult(
        label_id=1,
        label="Faulty",
        proba_fault=0.9123,
        ready=True,
        reason=None,
        buffer_fill=128,
        buffer_size=128,
    )

    ok = uploader.upload({"I1": 1.0, "I2": 2.0, "I3": 3.0}, prediction)

    assert ok is True
    assert captured["params"]["field1"] == "1.000"
    assert captured["params"]["field5"] == "0.912"
    assert captured["params"]["field7"] == 128
