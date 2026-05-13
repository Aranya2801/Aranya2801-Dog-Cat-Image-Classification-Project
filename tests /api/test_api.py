"""
Tests for the FastAPI prediction endpoints.
Uses httpx AsyncClient for async test support.
"""

import io
import pytest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def create_test_image(color=(200, 100, 50), size=(100, 100)) -> bytes:
    """Create a synthetic RGB image as JPEG bytes."""
    img = Image.fromarray(np.full((*size, 3), color, dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def client():
    from src.api.app import app
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "uptime_seconds" in data


def test_model_info_endpoint(client):
    resp = client.get("/model/info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["architecture"] == "EfficientNet-B4"
    assert data["classes"] == ["cat", "dog"]
    assert data["input_size"] == [380, 380]


def test_predict_single_image(client):
    image_bytes = create_test_image()
    resp = client.post(
        "/predict",
        files={"file": ("test.jpg", image_bytes, "image/jpeg")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["prediction"] in ["cat", "dog"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert "probabilities" in data
    assert abs(data["probabilities"]["cat"] + data["probabilities"]["dog"] - 1.0) < 1e-4
    assert data["inference_time_ms"] > 0


def test_predict_returns_emoji(client):
    image_bytes = create_test_image()
    resp = client.post(
        "/predict",
        files={"file": ("test.jpg", image_bytes, "image/jpeg")},
    )
    assert resp.status_code == 200
    assert resp.json()["emoji"] in ["🐶", "🐱"]


def test_predict_batch(client):
    images = [create_test_image(color=(c, c, c)) for c in [50, 100, 150]]
    files = [("files", (f"img{i}.jpg", img, "image/jpeg")) for i, img in enumerate(images)]
    resp = client.post("/predict/batch", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 3
    assert len(data["results"]) == 3
    assert data["total_time_ms"] > 0


def test_predict_invalid_file(client):
    resp = client.post(
        "/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert resp.status_code == 400


def test_predict_base64(client):
    import base64
    image_bytes = create_test_image()
    b64 = base64.b64encode(image_bytes).decode()
    resp = client.post("/predict/base64", json={"image": b64, "format": "jpeg"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["prediction"] in ["cat", "dog"]


def test_predict_probabilities_sum_to_one(client):
    for _ in range(5):
        image_bytes = create_test_image(
            color=tuple(np.random.randint(0, 255, 3).tolist())
        )
        resp = client.post(
            "/predict",
            files={"file": ("test.jpg", image_bytes, "image/jpeg")},
        )
        assert resp.status_code == 200
        probs = resp.json()["probabilities"]
        total = probs["cat"] + probs["dog"]
        assert abs(total - 1.0) < 1e-3, f"Probabilities sum to {total}"
