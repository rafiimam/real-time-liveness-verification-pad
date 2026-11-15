(() => {
  const $ = (id) => document.getElementById(id);

  const apiUrlInput = $("apiUrl");
  const accountIdInput = $("accountId");
  const preview = $("preview");
  const snap = $("snap");
  const photoFlash = $("photoFlash");
  const statusEl = $("status");
  const stepLabel = $("stepLabel");
  const guide = $("guide");
  const countdownEl = $("countdown");
  const progressBar = $("progressBar");
  const progressFill = $("progressFill");
  const resultCard = $("resultCard");
  const resultTitle = $("resultTitle");
  const resultDetail = $("resultDetail");
  const btnPrimary = $("btnPrimary");
  const btnRetry = $("btnRetry");
  const stage = document.querySelector(".stage");

  const TARGET_FPS = 8;
  const MAX_CLIENT_SECONDS = 8;

  /** @type {'photo'|'ready'|'streaming'|'done'} */
  let step = "photo";
  let mediaStream = null;
  let photoBase64 = null;
  let streaming = false;
  let activeWs = null;
  let frameTimer = null;

  const params = new URLSearchParams(location.search);
  const saved = localStorage.getItem("liveness_api_url") || "";
  apiUrlInput.value = params.get("api") || saved || location.origin;

  function apiBase() {
    return (apiUrlInput.value || "").trim().replace(/\/$/, "");
  }

  function wsBase() {
    const base = apiBase();
    if (base.startsWith("https://")) return "wss://" + base.slice(8);
    if (base.startsWith("http://")) return "ws://" + base.slice(7);
    return base;
  }

  function setStatus(text) {
    statusEl.textContent = text;
  }

  function setStepUI() {
    stage.classList.remove("checking", "done-ok", "done-bad");
    progressBar.classList.add("hidden");
    resultCard.classList.add("hidden");
    resultCard.classList.remove("ok", "bad");
    btnRetry.classList.add("hidden");
    countdownEl.textContent = "";

    if (step === "photo") {
      stepLabel.textContent = "Step 1 of 2 · Photo";
      guide.textContent = "Center your face, then take a photo";
      btnPrimary.textContent = "Start Test";
      btnPrimary.disabled = !mediaStream;
      photoFlash.classList.add("hidden");
      preview.classList.remove("hidden");
    } else if (step === "ready") {
      stepLabel.textContent = "Step 2 of 2 · Live check";
      guide.textContent = "Keep your face in the oval";
      btnPrimary.textContent = "Start verification";
      btnPrimary.disabled = !photoBase64;
      photoFlash.classList.remove("hidden");
      // keep preview live behind for next stream — show flash briefly then live
      preview.classList.remove("hidden");
    } else if (step === "streaming") {
      stepLabel.textContent = "Verifying…";
      guide.textContent = "Hold still — finishing as soon as it’s ready";
      btnPrimary.textContent = "Checking…";
      btnPrimary.disabled = true;
      progressBar.classList.remove("hidden");
      stage.classList.add("checking");
      photoFlash.classList.add("hidden");
    } else if (step === "done") {
      stepLabel.textContent = "Done";
      btnPrimary.classList.add("hidden");
      btnRetry.classList.remove("hidden");
      progressBar.classList.add("hidden");
    }
  }

  async function openCamera() {
    try {
      if (!window.isSecureContext && location.hostname !== "localhost") {
        setStatus("Camera needs HTTPS (or open via your LAN IP carefully).");
      }
      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: { ideal: "user" },
          width: { ideal: 720 },
          height: { ideal: 1280 },
        },
      });
      preview.srcObject = mediaStream;
      await preview.play();
      setStatus("Camera ready");
      if (step === "photo") btnPrimary.disabled = false;
    } catch (err) {
      setStatus("Camera error: " + err.message);
      btnPrimary.disabled = true;
    }
  }

  function capturePhoto() {
    if (!mediaStream) return;
    const w = preview.videoWidth || 720;
    const h = preview.videoHeight || 1280;
    snap.width = w;
    snap.height = h;
    const ctx = snap.getContext("2d");
    ctx.drawImage(preview, 0, 0, w, h);
    photoBase64 = snap.toDataURL("image/jpeg", 0.92);
    photoFlash.src = photoBase64;
    localStorage.setItem("liveness_api_url", apiBase());

    step = "ready";
    btnPrimary.classList.remove("hidden");
    setStepUI();
    setStatus("Photo saved. Tap to start the live check.");
    // Auto-start verification shortly — one less tap if user prefers
    window.setTimeout(() => {
      if (step === "ready") startLiveStream();
    }, 650);
  }

  function frameJpegBase64(quality = 0.68) {
    const w = preview.videoWidth || 640;
    const h = preview.videoHeight || 480;
    const maxW = 480;
    const scale = Math.min(1, maxW / w);
    const tw = Math.round(w * scale);
    const th = Math.round(h * scale);
    snap.width = tw;
    snap.height = th;
    const ctx = snap.getContext("2d");
    ctx.drawImage(preview, 0, 0, tw, th);
    return snap.toDataURL("image/jpeg", quality);
  }

  function showResult(data) {
    step = "done";
    setStepUI();
    const approved = !!data.is_approved;
    stage.classList.add(approved ? "done-ok" : "done-bad");
    guide.textContent = approved ? "You’re verified" : "Could not verify";
    resultCard.classList.remove("hidden");
    resultCard.classList.add(approved ? "ok" : "bad");
    resultTitle.textContent = approved ? "APPROVED" : "REJECTED";
    const bits = [];
    if (data.face_match != null) bits.push(`Match: ${data.face_match ? "yes" : "no"}`);
    if (data.liveness != null) bits.push(`Liveness: ${data.liveness ? "yes" : "no"}`);
    if (data.frames_received != null) bits.push(`${data.frames_received} frames`);
    resultDetail.textContent = bits.join(" · ") || "Verification finished";
    setStatus(approved ? "All set." : "Try again with better lighting.");
    countdownEl.textContent = "";
  }

  function stopStreamingTimers() {
    if (frameTimer) {
      clearInterval(frameTimer);
      frameTimer = null;
    }
    streaming = false;
  }

  function cleanupWs() {
    stopStreamingTimers();
    if (activeWs) {
      try { activeWs.close(); } catch {}
      activeWs = null;
    }
  }

  async function startLiveStream() {
    if (streaming || !photoBase64) return;
    if (!apiBase()) {
      setStatus("Set API URL in Settings.");
      return;
    }
    localStorage.setItem("liveness_api_url", apiBase());

    streaming = true;
    step = "streaming";
    setStepUI();
    setStatus("Connecting…");
    progressFill.style.width = "5%";

    const ws = new WebSocket(`${wsBase()}/ws/live`);
    activeWs = ws;
    let maxTimer = null;
    let finished = false;

    const finishClient = () => {
      if (finished) return;
      finished = true;
      stopStreamingTimers();
      if (maxTimer) clearTimeout(maxTimer);
    };

    ws.onopen = () => {
      ws.send(JSON.stringify({
        type: "start",
        account_id: accountIdInput.value.trim() || "MOBILE_USER",
        photo_base64: photoBase64,
      }));
    };

    ws.onmessage = (ev) => {
      let msg;
      try { msg = JSON.parse(ev.data); } catch { return; }

      if (msg.type === "ready") {
        setStatus("Recording live…");
        const early = Number(msg.early_seconds) || 3.5;
        const maxSec = Number(msg.max_seconds) || MAX_CLIENT_SECONDS;
        let started = Date.now();

        const interval = Math.round(1000 / TARGET_FPS);
        frameTimer = setInterval(() => {
          if (ws.readyState !== WebSocket.OPEN || finished) return;
          ws.send(JSON.stringify({
            type: "frame",
            image_base64: frameJpegBase64(0.68),
            ts: Date.now(),
          }));
          const elapsed = (Date.now() - started) / 1000;
          const pct = Math.min(95, (elapsed / early) * 90);
          progressFill.style.width = `${pct}%`;
          if (elapsed < early) {
            countdownEl.textContent = `${Math.max(0, Math.ceil(early - elapsed))}s`;
          } else {
            countdownEl.textContent = "Almost…";
          }
        }, interval);

        // Safety: ask server to end if auto-finish somehow missed
        maxTimer = setTimeout(() => {
          if (finished || ws.readyState !== WebSocket.OPEN) return;
          setStatus("Wrapping up…");
          ws.send(JSON.stringify({ type: "end" }));
        }, maxSec * 1000);
      }

      if (msg.type === "progress") {
        const frames = msg.frames || 0;
        const elapsed = (msg.elapsed_ms || 0) / 1000;
        setStatus(msg.can_finish ? "Almost done…" : `Live · ${frames} frames`);
        if (msg.can_finish) progressFill.style.width = "92%";
        else if (elapsed > 0) {
          progressFill.style.width = `${Math.min(88, 10 + elapsed * 18)}%`;
        }
      }

      if (msg.type === "status") {
        setStatus(msg.message || "Checking…");
        progressFill.style.width = "96%";
        countdownEl.textContent = "";
      }

      if (msg.type === "result") {
        finishClient();
        progressFill.style.width = "100%";
        showResult(msg);
        try { ws.close(); } catch {}
        activeWs = null;
      }

      if (msg.type === "error") {
        finishClient();
        step = "done";
        setStepUI();
        stage.classList.add("done-bad");
        guide.textContent = "Something went wrong";
        resultCard.classList.remove("hidden");
        resultCard.classList.add("bad");
        resultTitle.textContent = "FAILED";
        resultDetail.textContent = msg.message || "Unknown error";
        setStatus(msg.message || "Error");
        btnPrimary.classList.add("hidden");
        btnRetry.classList.remove("hidden");
        try { ws.close(); } catch {}
        activeWs = null;
      }
    };

    ws.onerror = () => {
      if (finished) return;
      finishClient();
      setStatus("Connection error. Check Settings → API URL.");
      step = "ready";
      btnPrimary.classList.remove("hidden");
      setStepUI();
    };

    ws.onclose = () => {
      if (!finished && streaming) {
        finishClient();
        setStatus("Connection closed.");
        step = "ready";
        btnPrimary.classList.remove("hidden");
        setStepUI();
      }
    };
  }

  function resetAll() {
    cleanupWs();
    photoBase64 = null;
    photoFlash.removeAttribute("src");
    photoFlash.classList.add("hidden");
    btnPrimary.classList.remove("hidden");
    step = "photo";
    setStepUI();
    setStatus(mediaStream ? "Camera ready" : "Starting camera…");
    if (!mediaStream) openCamera();
  }

  btnPrimary.addEventListener("click", () => {
    if (step === "photo") capturePhoto();
    else if (step === "ready") startLiveStream();
  });
  btnRetry.addEventListener("click", resetAll);

  setStepUI();
  openCamera();
})();
