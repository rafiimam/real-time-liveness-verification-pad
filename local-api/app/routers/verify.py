"""One-shot verification: reference photo + 10s video -> decision."""

import base64
import logging
import os
import tempfile

import cv2
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse

from app.db.pgvector import store_faceprint
from app.models.schemas import SessionStatus, VerificationResult
from app.routers.session import consume_session, create_session_record, update_session
from app.security.auth import require_api_auth
from app.services import gpu_client
from app.services.orchestrator import SKIP_FACE_MATCH, run_verification_pipeline

logger = logging.getLogger(__name__)

ALLOW_VERIFY_UPLOAD = os.getenv("ALLOW_VERIFY_UPLOAD", "true").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
HIGH_VALUE_REQUIRE_LIVE_WS = os.getenv("HIGH_VALUE_REQUIRE_LIVE_WS", "true").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
HIGH_VALUE_AMOUNT = float(os.getenv("HIGH_VALUE_AMOUNT", "100000"))

logger = logging.getLogger(__name__)
router = APIRouter()

UPLOAD_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
  <title>Liveness Verification</title>
  <style>
    :root {
      --bg: #0f1419; --card: #1a2332; --text: #e8eef6; --muted: #8b9bb4;
      --danger: #f87171; --ok: #4ade80; --border: #2a3a4f; --accent: #2dd4bf;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; min-height: 100vh;
      font-family: "Segoe UI", system-ui, sans-serif;
      background: radial-gradient(1200px 600px at 10% -10%, #1b3a3a 0%, var(--bg) 55%);
      color: var(--text);
      display: flex; align-items: flex-start; justify-content: center;
      padding: 18px;
    }
    .wrap {
      width: min(560px, 100%);
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 22px;
      box-shadow: 0 20px 60px rgba(0,0,0,.35);
      margin: 12px 0 40px;
    }
    h1 { margin: 0 0 6px; font-size: 1.35rem; }
    p.sub { margin: 0 0 14px; color: var(--muted); font-size: .92rem; line-height: 1.4; }
    .tip {
      background: #0f1722; border: 1px solid var(--border); border-radius: 10px;
      padding: 10px 12px; margin-bottom: 16px; font-size: .82rem; color: var(--muted); line-height: 1.45;
    }
    .tip code { color: var(--accent); word-break: break-all; }
    label { display: block; margin: 14px 0 6px; font-size: .85rem; color: var(--muted); }
    input[type=file], input[type=text] {
      width: 100%; background: #0f1722; border: 1px solid var(--border);
      color: var(--text); border-radius: 10px; padding: 12px;
    }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 8px; }
    .secondary, .danger, .primary {
      border: 0; border-radius: 10px; padding: 12px 14px; font-weight: 700; font-size: .95rem; cursor: pointer;
    }
    .primary {
      margin-top: 18px; width: 100%;
      background: linear-gradient(135deg, #14b8a6, #0ea5e9); color: #042f2e;
    }
    .secondary { background: #243247; color: var(--text); width: 100%; }
    .danger { background: #7f1d1d; color: #fecaca; width: 100%; }
    button:disabled { opacity: .55; cursor: wait; }
    video#preview {
      width: 100%; max-height: 320px; background: #000; border-radius: 12px;
      margin-top: 10px; object-fit: cover; display: none;
    }
    #timer {
      text-align: center; font-size: 1.4rem; font-weight: 700; color: var(--accent);
      margin-top: 8px; min-height: 1.5em;
    }
    #status { margin-top: 14px; color: var(--muted); font-size: .9rem; min-height: 1.2em; }
    #result {
      margin-top: 14px; display: none; background: #0f1722; border: 1px solid var(--border);
      border-radius: 12px; padding: 14px; white-space: pre-wrap;
      font-family: ui-monospace, Consolas, monospace; font-size: .8rem; line-height: 1.45;
    }
    .badge { display: inline-block; padding: 2px 8px; border-radius: 999px; font-weight: 700; }
    .ok { background: rgba(74,222,128,.15); color: var(--ok); }
    .bad { background: rgba(248,113,113,.15); color: var(--danger); }
    .hint { font-size: .78rem; color: var(--muted); margin-top: 4px; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Liveness Verification</h1>
    <p class="sub">Use your phone camera on the same Wi‑Fi. Capture a fresh photo + 10s live video — no USB needed.</p>
    <div class="tip">
      On phone browser open:<br/>
      <code id="lanUrl">http://&lt;this-pc-ip&gt;:8001/</code>
    </div>

    <form id="form">
      <label>Account ID (optional)</label>
      <input type="text" id="account_id" placeholder="ACC_001" />

      <label>1) Reference photo (face)</label>
      <input type="file" id="photo" accept="image/*" capture="user" />
      <div class="hint">On mobile, this opens the front camera for a fresh selfie.</div>

      <label>2) Live 10-second video</label>
      <input type="file" id="video" accept="video/*" capture="user" />
      <div class="hint">Or record in-page below (if browser allows camera).</div>

      <video id="preview" playsinline muted></video>
      <div id="timer"></div>
      <div class="row">
        <button type="button" class="secondary" id="btnStartCam">Open camera</button>
        <button type="button" class="danger" id="btnRecord" disabled>Record 10s</button>
      </div>

      <button id="btn" class="primary" type="submit">Verify</button>
    </form>
    <div id="status"></div>
    <div id="result"></div>
  </div>
  <script>
    const form = document.getElementById('form');
    const statusEl = document.getElementById('status');
    const resultEl = document.getElementById('result');
    const btn = document.getElementById('btn');
    const photoInput = document.getElementById('photo');
    const videoInput = document.getElementById('video');
    const preview = document.getElementById('preview');
    const timerEl = document.getElementById('timer');
    const btnStartCam = document.getElementById('btnStartCam');
    const btnRecord = document.getElementById('btnRecord');

    let stream = null;
    let recordedBlob = null;
    const RECORD_SECONDS = 10;

    // Show reachable URL for phone
    document.getElementById('lanUrl').textContent = location.origin + '/';

    btnStartCam.addEventListener('click', async () => {
      statusEl.textContent = '';
      try {
        if (!window.isSecureContext && location.hostname !== 'localhost') {
          statusEl.innerHTML = 'In-page camera may be blocked on HTTP. Use the file pickers above (they open the phone camera).';
        }
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'user', width: { ideal: 720 }, height: { ideal: 1280 } },
          audio: false
        });
        preview.srcObject = stream;
        preview.style.display = 'block';
        await preview.play();
        btnRecord.disabled = false;
        statusEl.textContent = 'Camera ready. Tap Record 10s.';
      } catch (err) {
        statusEl.innerHTML = '<span class="badge bad">CAMERA</span> ' + err.message +
          ' — use the video file picker above (capture from camera).';
      }
    });

    btnRecord.addEventListener('click', async () => {
      if (!stream) return;
      recordedBlob = null;
      btnRecord.disabled = true;
      btnStartCam.disabled = true;

      const mime = MediaRecorder.isTypeSupported('video/webm;codecs=vp9')
        ? 'video/webm;codecs=vp9'
        : (MediaRecorder.isTypeSupported('video/webm') ? 'video/webm' : 'video/mp4');
      const chunks = [];
      const recorder = new MediaRecorder(stream, { mimeType: mime });
      recorder.ondataavailable = (e) => { if (e.data && e.data.size) chunks.push(e.data); };

      const done = new Promise((resolve) => { recorder.onstop = resolve; });
      recorder.start(200);

      let left = RECORD_SECONDS;
      timerEl.textContent = left + 's';
      const tick = setInterval(() => {
        left -= 1;
        timerEl.textContent = left > 0 ? (left + 's') : 'Processing clip...';
        if (left <= 0) clearInterval(tick);
      }, 1000);

      await new Promise((r) => setTimeout(r, RECORD_SECONDS * 1000));
      if (recorder.state !== 'inactive') recorder.stop();
      await done;
      clearInterval(tick);

      recordedBlob = new Blob(chunks, { type: mime });
      // Mirror into the file input via DataTransfer so form submit works
      const file = new File([recordedBlob], 'live_10s.webm', { type: mime });
      const dt = new DataTransfer();
      dt.items.add(file);
      videoInput.files = dt.files;

      timerEl.textContent = 'Recorded ' + RECORD_SECONDS + 's ✓';
      statusEl.textContent = 'Live clip ready. Tap Verify.';
      btnRecord.disabled = false;
      btnStartCam.disabled = false;
    });

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      resultEl.style.display = 'none';
      resultEl.textContent = '';

      if (!photoInput.files[0]) {
        statusEl.innerHTML = '<span class="badge bad">ERROR</span> Reference photo required';
        return;
      }
      if (!videoInput.files[0] && !recordedBlob) {
        statusEl.innerHTML = '<span class="badge bad">ERROR</span> 10s video required';
        return;
      }

      btn.disabled = true;
      statusEl.textContent = 'Uploading & verifying... up to ~1 minute.';

      const fd = new FormData();
      const account = document.getElementById('account_id').value.trim();
      if (account) fd.append('account_id', account);
      fd.append('photo', photoInput.files[0]);
      fd.append('video', videoInput.files[0] || new File([recordedBlob], 'live_10s.webm', { type: recordedBlob.type }));

      try {
        const res = await fetch('/verify', { method: 'POST', body: fd });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || JSON.stringify(data));
        const badge = data.is_approved
          ? '<span class="badge ok">APPROVED</span>'
          : '<span class="badge bad">REJECTED</span>';
        statusEl.innerHTML = 'Done. ' + badge;
        resultEl.style.display = 'block';
        resultEl.textContent = JSON.stringify(data, null, 2);
      } catch (err) {
        statusEl.innerHTML = '<span class="badge bad">ERROR</span> ' + err.message;
      } finally {
        btn.disabled = false;
      }
    });
  </script>
</body>
</html>
"""


@router.get("/", response_class=HTMLResponse)
async def root_ui():
    """Friendly mobile flow lives at /app/ — send users there by default."""
    return RedirectResponse(url="/app/", status_code=302)


@router.get("/upload", response_class=HTMLResponse)
async def upload_ui():
    """Legacy photo + video file upload UI (testing / fallback)."""
    return UPLOAD_PAGE


@router.post("/verify", response_model=VerificationResult)
async def verify_photo_and_video(
    photo: UploadFile = File(..., description="Account holder reference photo"),
    video: UploadFile = File(..., description="10-second live video"),
    account_id: str = Form(default="REAL_USER"),
    account_number: str | None = Form(default=None),
    transaction_amount: float | None = Form(default=None),
    _auth: dict = Depends(require_api_auth),
):
    """One request: enroll photo + verify against uploaded video.

    Lower capture-trust than live WebSocket. Disable via ALLOW_VERIFY_UPLOAD=false
    for production high-assurance deployments.
    """
    # Prefer account_number when provided (SDK naming); fall back to account_id.
    if account_number and account_number.strip():
        account_id = account_number.strip()
    if not ALLOW_VERIFY_UPLOAD:
        raise HTTPException(
            status_code=403,
            detail="File upload verify disabled. Use live WebSocket /ws/live (higher capture trust).",
        )
    if (
        HIGH_VALUE_REQUIRE_LIVE_WS
        and transaction_amount is not None
        and float(transaction_amount) >= HIGH_VALUE_AMOUNT
    ):
        raise HTTPException(
            status_code=403,
            detail=(
                f"High-value amounts (>= {HIGH_VALUE_AMOUNT}) require live WebSocket capture, "
                "not file upload."
            ),
        )

    photo_bytes = await photo.read()
    video_bytes = await video.read()

    if not photo_bytes:
        raise HTTPException(status_code=400, detail="Empty photo file")
    if not video_bytes:
        raise HTTPException(status_code=400, detail="Empty video file")

    # Decode photo
    nparr = __import__("numpy").frombuffer(photo_bytes, __import__("numpy").uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid photo format")

    photo_b64 = base64.b64encode(photo_bytes).decode()

    # Embedding for faceprint store (optional when SKIP_FACE_MATCH)
    embedding = await gpu_client.get_embedding(image)
    if embedding is None:
        if SKIP_FACE_MATCH:
            embedding = [0.0] * 512
            logger.warning("No face in reference photo — continuing because SKIP_FACE_MATCH=true")
        else:
            raise HTTPException(status_code=400, detail="No face detected in reference photo")

    await store_faceprint(
        account_id=account_id,
        embedding=embedding,
        reference_image_b64=photo_b64,
    )

    record = create_session_record(
        account_id=account_id,
        transaction_amount=transaction_amount,
        status=SessionStatus.PROCESSING,
    )
    session_id = record["session_id"]

    # Extract frames from video (preserve real container — phone often sends webm)
    suffix = ".mp4"
    fname = (video.filename or "").lower()
    ctype = (video.content_type or "").lower()
    if fname.endswith(".webm") or "webm" in ctype:
        suffix = ".webm"
    elif fname.endswith(".mov") or "quicktime" in ctype:
        suffix = ".mov"
    elif fname.endswith(".avi"):
        suffix = ".avi"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(video_bytes)
        tmp_path = f.name

    cap = cv2.VideoCapture(tmp_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame is None or frame.size == 0:
            continue
        frames.append(frame)
    cap.release()

    logger.info(
        "Verify decoded %d frames from %s (%s, %d bytes)",
        len(frames),
        fname or suffix,
        ctype or "unknown",
        len(video_bytes),
    )
    print(
        f"[VERIFY] photo={photo.filename} video={video.filename} "
        f"decoded_frames={len(frames)} bytes={len(video_bytes)} account={account_id}",
        flush=True,
    )

    if not frames:
        update_session(session_id, status=SessionStatus.FAILED)
        consume_session(session_id)
        raise HTTPException(
            status_code=400,
            detail=f"No frames extracted from video ({fname or suffix}). Try MP4 or use /app/ live stream.",
        )

    try:
        result = await run_verification_pipeline(
            session_id=session_id,
            account_id=account_id,
            frames=frames,
            video_bytes=video_bytes,
            reference_image=image,
            source="verify_upload",
        )
    except Exception as e:
        logger.exception("Verify failed")
        update_session(session_id, status=SessionStatus.FAILED)
        consume_session(session_id)
        raise HTTPException(status_code=500, detail=f"Verification failed: {type(e).__name__}: {e}") from e

    update_session(session_id, status=SessionStatus.COMPLETED, result=result)
    consume_session(session_id)
    return result
