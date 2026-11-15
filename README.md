# Real-Time Liveness Verification & Presentation Attack Detection

A production-grade face verification system for high-value banking transactions: document/account-to-live face matching, spatial and temporal liveness detection, and presentation attack detection (PAD), served as a decoupled GPU inference backend with web, Android, and iOS clients.

> **Public version:** This repository is a sanitized copy of a system built for production banking use. All credentials, internal service addresses, and environment-specific configuration have been replaced with placeholders and environment variables. Model checkpoints are not included.

## Results (locked evaluation set)

| Metric | Result |
|---|---|
| False Acceptance Rate (replay attacks) | **0.00%** (0/33) |
| False Rejection Rate (genuine live captures) | **3.125%** (1/32) |
| End-to-end verification latency | **~800 ms** (down from >5 s in the initial single-service design) |

The evaluation set is small and pilot-stage by design: thresholds were calibrated against a locked replay set under a strict promotion rule (FAR must stay 0%, FRR ≤ 15%) before any model version was promoted. Per-session detail is in [`docs/pilot-metrics.md`](docs/pilot-metrics.md).

## How it works

A verification session runs as a fail-closed cascade:

1. **Reference resolution** — the account number is resolved to reference photos (document/NID photo and account photo) through core-banking and identity-verification services (env-configured).
2. **Live capture** — the client streams ~10 seconds of camera frames over a WebSocket (`/ws/live`), or uploads a photo + video.
3. **Liveness cascade** — three independent checks must all pass:
   - *Spatial PAD* — Swin-B backbone with a UAD (uncertainty-aware) head, on MTCNN face crops
   - *Temporal liveness* — 3D ResNet-18 over the frame sequence
   - *Screen/replay detection* — spectral and motion cues for displayed-media attacks
4. **Face matching** — ArcFace embeddings matched against the reference photos (pgvector store), with provider fallback. Matching is **fail-closed**: if every provider fails, the session is rejected, never approved.
5. **Decision** — the decision API combines liveness and match results under session TTL, nonce, and authentication controls.

Heavy PyTorch/CUDA inference (`gpu-server/`) is decoupled from the stateless asyncio decision gateway (`local-api/`), which is what brought end-to-end latency from over 5 seconds to roughly 800 ms.

## Repository layout

| Directory | Contents |
|---|---|
| `local-api/` | FastAPI decision engine — sessions, orchestration, WebSocket live stream, auth, pgvector faceprint store |
| `gpu-server/` | GPU inference service — MTCNN detection, Swin-B + UAD spatial PAD, 3D ResNet-18 temporal model, ArcFace matching |
| `mobile-web/` | Mobile browser client — live camera streaming over WebSocket (React + TypeScript + Vite) |
| `mobile-web-legacy/` | Legacy upload-based web UI |
| `android-sdk/` | Native Android liveness SDK (Kotlin) + sample app |
| `ios-sdk/` | Native iOS liveness SDK (Swift) + sample app |
| `scripts/` | Training (ArcFace / UAD / temporal), dataset preparation and merging, threshold calibration, pilot metrics, smoke tests |
| `deploy/` | nginx reverse-proxy configuration for HTTPS deployment |
| `docs/` | Pilot metrics and native SDK attestation notes |

## Quick start (local development)

Prerequisites: Python 3.11+, a CUDA GPU for the inference service, Node.js for the web client, Docker for Postgres (optional — the API falls back to an in-memory store).

```bash
# 1. Faceprint database (optional)
docker compose up -d postgres

# 2. GPU inference service
cd gpu-server
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 3. Decision API
cd local-api
cp .env.example .env        # fill in your own values
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

- Health check: `http://127.0.0.1:8001/health`
- Mobile live app: `http://127.0.0.1:8001/app/` (camera requires HTTPS or localhost)
- Web client dev mode: `cd mobile-web && cp .env.example .env && npm install && npm run dev`

### Development mode: liveness only

Set `SKIP_FACE_MATCH=true` in `local-api/.env` to develop the liveness cascade without the face-matching providers. Approval then depends only on the liveness checks. **Never enable this in production.**

## Training and calibration

Training runs on the GPU host; checkpoints are loaded from `gpu-server/weights/` (not committed — see `.gitignore`).

```bash
# Prepare merged datasets (academic FAS datasets require registration)
python scripts/merge_fas_dataset.py --sources /data/celeba_spoof /data/casia_fasd --output /data/fas_merged
python scripts/merge_temporal_dataset.py --sources /data/replay_attack --output /data/temporal_merged

# Train spatial (Swin-B + UAD) and temporal (3D ResNet-18) models
python scripts/train_uad.py --data-dir /data/fas_merged --output-dir gpu-server/weights --batch-size 32 --epochs 20 --finetune-epochs 10
python scripts/train_temporal.py --data-dir /data/temporal_merged --output-dir gpu-server/weights --batch-size 8 --epochs 15

# Calibrate decision thresholds against a target FAR
python scripts/calibrate_thresholds.py --fas-dir /data/fas_merged --temporal-dir /data/temporal_merged --gpu-url http://localhost:8000 --target-far 0.01
```

## API summary

| Endpoint | Purpose |
|---|---|
| `GET /app/` | Mobile live web app |
| `WS /ws/live` | Real-time frame stream + decision (preferred) |
| `POST /verify` | Photo + video upload (disable with `ALLOW_VERIFY_UPLOAD=false`) |
| `POST /session/create` | Create session (API key / JWT) |
| `POST /session/{id}/stream` | Upload recorded video |
| `POST /auth/token/mint` | Mint short-lived JWT (admin key) |
| `GET /auth/status` | Whether authentication is required |

## Security model

- Face matching is **fail-closed** across all providers
- Short-lived sessions (`SESSION_TTL_SECONDS=120`), request nonces, API-key/JWT authentication
- Web capture is treated as soft trust; high-assurance deployments use the native SDKs with device attestation — see [`docs/native-sdk-attestation.md`](docs/native-sdk-attestation.md)
- Production exposure requires, at minimum: `AUTH_REQUIRED=true`, `SKIP_FACE_MATCH=false`, `REQUIRE_SPATIAL=true`, `REQUIRE_TEMPORAL=true`, `FACE_MATCH_FAIL_CLOSED=true`, `REQUIRE_NONCE=true`, and HTTPS (see [`deploy/nginx-liveness.conf`](deploy/nginx-liveness.conf))

## Limitations

- Evaluation so far is pilot-scale (see the locked-set sizes above); larger, more diverse attack corpora are needed before general claims can be made.
- Model checkpoints, training data, and the bank-side integration services are not part of this public copy; the training and calibration pipeline is included so the approach can be reproduced on public FAS datasets.
- The Android/iOS SDKs expect a reachable decision-API deployment; the sample apps point at a placeholder host by default.

## Author

**MD. Rafi Imam** — Software Development Officer (AI & Automation)
[Portfolio](https://rafiimam.github.io/Personal-website/) · [GitHub](https://github.com/rafiimam)
