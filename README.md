# AI-Generated Voice Detection API

FastAPI-based service that classifies incoming audio clips as AI-generated or human speech across Tamil, English, Hindi, Malayalam, and Telugu. Phase 1 focuses on inference-only usage of a pretrained wav2vec2 encoder plus a lightweight classifier head.

## Features

- **Single `/detect` endpoint** accepting Base64-encoded MP3 audio.
- **API-key authentication** via the `X-API-KEY` header (can be disabled locally).
- **Audio safeguards**: Base64 size limits, duration checks, resampling, normalization.
- **Model lifecycle**: wav2vec2 embeddings + classifier head loaded once at startup.
- **Fallback handling**: predictable response if inference fails or times out.
- **Health probe**: `/health` route for liveness monitoring.

## Project Structure

```
app/
  api/routes/detect.py      # HTTP handlers
  core/config.py            # Pydantic Settings (env-driven)
  core/security.py          # API-key middleware
  models/components.py      # Embedding encoder & classifier head wrappers
  models/registry.py        # ModelRegistry orchestrating inference
  schemas/detection.py      # Request/response contracts
  services/audio/ingestion.py # Base64 decode + preprocessing
  main.py                   # FastAPI app factory & startup hooks
requirements.txt
README.md
```

## Configuration

Create a `.env` file at the repo root with at least the following variables:

```
VOICE_DETECT_API_KEY=dev-key
VOICE_DETECT_SAMPLE_RATE=16000
VOICE_DETECT_MAX_DURATION=30.0
VOICE_DETECT_MAX_B64_BYTES=6291456
VOICE_DETECT_INFERENCE_TIMEOUT=8.0
VOICE_DETECT_MODEL_CACHE=.cache/models
VOICE_DETECT_EMBEDDING_BUNDLE=WAV2VEC2_ASR_BASE_960H
VOICE_DETECT_CLASSIFIER_CKPT=app/models/artifacts/classifier_head.pt
VOICE_DETECT_CLASSIFIER_INPUT_DIM=768
```

> Tip: For local development, keep the API key simple and share it as a header when calling `/detect`. In production, use a secret store.

## Setup & Local Run

1. **Create a virtual environment**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the API**
   ```bash
   uvicorn app.main:app --reload
   ```

## API Usage

### Health Check
```bash
curl http://localhost:8000/health
```
Response:
```json
{ "status": "ok" }
```

### Detect Endpoint
```bash
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: dev-key" \
  -d '{
        "audio_base64": "<Base64 MP3 string>",
        "language": "en"
      }'
```
Response schema:
```json
{
  "classification": "AI_GENERATED" | "HUMAN",
  "confidence": 0.0 - 1.0,
  "explanation": ["..."]
}
```

If inference exceeds `VOICE_DETECT_INFERENCE_TIMEOUT` or errors, the service emits the fallback response:
```json
{
  "classification": "HUMAN",
  "confidence": 0.05,
  "explanation": ["Model fallback due to inference error"]
}
```

## Architecture Overview

1. **Middleware**: `APIKeyAuthMiddleware` blocks unauthorized requests.
2. **Routing**: `/detect` validates payloads via Pydantic schemas, then hands audio off to preprocessing and model inference.
3. **Audio Pipeline**: Base64 decoding, mono conversion, resampling to `VOICE_DETECT_SAMPLE_RATE`, duration validation, normalization.
4. **Model Registry**: On startup, loads the specified torchaudio bundle plus classifier checkpoint and keeps them in memory.
5. **Inference Flow**: `ModelRegistry.predict()` runs in a worker thread, returns classification/confidence/explanations. `asyncio.wait_for` enforces the timeout guard.

## Extending Beyond Phase 1

- Replace the placeholder classifier checkpoint with a trained head tuned on multilingual AI vs. human datasets.
- Add structured logging/metrics (e.g., OpenTelemetry, Prometheus).
- Build CI/CD pipelines and containerization artifacts for cloud deployment.
- Implement integration tests and load tests before going public.

## Troubleshooting

- **`torchaudio` backend errors**: Ensure `soundfile` is installed (already in requirements) and libsndfile is available on your OS.
- **Large payload rejection**: Increase `VOICE_DETECT_MAX_B64_BYTES` or clip audio duration before encoding.
- **Authentication failures**: Confirm the `X-API-KEY` header matches `VOICE_DETECT_API_KEY`.

---
Feel free to open issues or iterate on the classifier head as you move to data collection and calibration phases.
