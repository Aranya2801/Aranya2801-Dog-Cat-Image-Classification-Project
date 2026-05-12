# 🔌 API Reference — DogCat Vision

## Base URL

```
http://localhost:8000
```

## Authentication

No authentication required for local deployment.
For production, add Bearer token middleware.

---

## Endpoints

### `GET /health`
Returns server health status and model availability.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda:0",
  "uptime_seconds": 3672.4
}
```

---

### `GET /model/info`
Returns model architecture and performance metadata.

**Response:**
```json
{
  "architecture": "EfficientNet-B4",
  "parameters": 19341616,
  "accuracy": 0.992,
  "input_size": [380, 380],
  "classes": ["cat", "dog"],
  "training_dataset": "Microsoft Cats vs Dogs (25,000 images)",
  "device": "cuda"
}
```

---

### `POST /predict`
Classify a single image file.

**Request:** `multipart/form-data`
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | File | ✅ | JPEG/PNG/WebP image, max 10MB |

**Response:**
```json
{
  "prediction": "dog",
  "confidence": 0.9987,
  "probabilities": {
    "dog": 0.9987,
    "cat": 0.0013
  },
  "inference_time_ms": 8.3,
  "model_version": "efficientnet-b4-v2.1",
  "emoji": "🐶"
}
```

**Error Responses:**
| Code | Reason |
|------|--------|
| `400` | Non-image file submitted |
| `413` | Image exceeds 10MB limit |
| `503` | Model not loaded |

---

### `POST /predict/batch`
Classify multiple images in one request (max 20).

**Request:** `multipart/form-data`
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `files` | File[] | ✅ | List of image files |

**Response:**
```json
{
  "results": [
    {
      "prediction": "cat",
      "confidence": 0.9834,
      "probabilities": {"cat": 0.9834, "dog": 0.0166},
      "inference_time_ms": 7.1,
      "model_version": "efficientnet-b4-v2.1",
      "emoji": "🐱"
    }
  ],
  "total_time_ms": 24.7,
  "count": 1
}
```

---

### `POST /predict/base64`
Classify an image provided as a base64-encoded string.

**Request body (JSON):**
```json
{
  "image": "<base64_encoded_string>",
  "format": "jpeg"
}
```

**Response:** Same as `/predict`

**JavaScript example:**
```javascript
const canvas = document.getElementById("myCanvas");
const base64 = canvas.toDataURL("image/jpeg").split(",")[1];

const response = await fetch("/predict/base64", {
  method: "POST",
  headers: {"Content-Type": "application/json"},
  body: JSON.stringify({image: base64, format: "jpeg"})
});
const result = await response.json();
console.log(result.prediction, result.confidence);
```

---

### `WebSocket /ws/predict`
Real-time prediction stream — send raw image bytes, receive JSON.

**Connection:**
```javascript
const ws = new WebSocket("ws://localhost:8000/ws/predict");

// Send image as ArrayBuffer
ws.send(imageBlob);

// Receive prediction
ws.onmessage = (event) => {
  const result = JSON.parse(event.data);
  console.log(result.prediction, result.confidence);
};
```

**Message format:** Same as `/predict` response.

**Use case:** Webcam streaming — send frames at 10-30 FPS for real-time classification.

---

## Interactive Docs

Full Swagger UI available at:
```
http://localhost:8000/docs
```

ReDoc alternative at:
```
http://localhost:8000/redoc
```

---

## Rate Limits

| Endpoint | Limit |
|----------|-------|
| `/predict` | 100 req/min |
| `/predict/batch` | 20 req/min |
| `/ws/predict` | 30 frames/sec |

---

## SDK Examples

### Python

```python
import requests

# Single image
with open("my_dog.jpg", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/predict",
        files={"file": ("dog.jpg", f, "image/jpeg")}
    )
result = resp.json()
print(f"{result['emoji']} {result['prediction']} ({result['confidence']:.1%})")
```

### JavaScript / Node.js

```javascript
const FormData = require('form-data');
const fs = require('fs');
const fetch = require('node-fetch');

const form = new FormData();
form.append('file', fs.createReadStream('my_cat.jpg'));

const response = await fetch('http://localhost:8000/predict', {
  method: 'POST', body: form
});
const result = await response.json();
console.log(`${result.emoji} ${result.prediction} (${(result.confidence*100).toFixed(1)}%)`);
```

### cURL

```bash
curl -X POST "http://localhost:8000/predict" \
     -F "file=@my_image.jpg" | python3 -m json.tool
```
