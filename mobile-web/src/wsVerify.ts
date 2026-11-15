import type { VerificationResult } from "./types";
import { apiBaseFrom, withAccessToken, wsBaseFrom } from "./types";
import type { VerifyHandlers } from "./httpVerify";

const TARGET_FPS = 8;
const MAX_CLIENT_SECONDS = 8;

export function startWsVerification(opts: {
  apiUrl: string;
  accountId: string;
  photoBase64: string;
  mediaStream: MediaStream;
  handlers: VerifyHandlers;
  accessToken?: string;
  onOpen?: () => void;
  getFrame?: () => string;
  /** True on the second challenge round — disables another step-up. */
  stepUpCompleted?: boolean;
}): () => void {
  const { handlers } = opts;
  const base = apiBaseFrom(opts.apiUrl);
  localStorage.setItem("liveness_api_url", base);
  const token = opts.accessToken || localStorage.getItem("liveness_api_key") || "";

  const canvas = document.createElement("canvas");
  const videoEl = document.createElement("video");
  videoEl.playsInline = true;
  videoEl.muted = true;
  videoEl.srcObject = opts.mediaStream;
  void videoEl.play().catch(() => undefined);

  const grabFrame = () => {
    if (opts.getFrame) return opts.getFrame();
    const w = videoEl.videoWidth || 640;
    const h = videoEl.videoHeight || 480;
    const maxW = 480;
    const scale = Math.min(1, maxW / w);
    canvas.width = Math.round(w * scale);
    canvas.height = Math.round(h * scale);
    const ctx = canvas.getContext("2d");
    if (!ctx) return opts.photoBase64;
    ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.68);
  };

  const wsUrl = withAccessToken(`${wsBaseFrom(base)}/ws/live`, token);
  const ws = new WebSocket(wsUrl);
  let frameTimer: number | null = null;
  let maxTimer: number | null = null;
  let finished = false;
  let sessionNonce: string | null = null;

  const cleanup = () => {
    finished = true;
    if (frameTimer != null) window.clearInterval(frameTimer);
    if (maxTimer != null) window.clearTimeout(maxTimer);
    frameTimer = null;
    maxTimer = null;
    videoEl.srcObject = null;
    try {
      ws.close();
    } catch {
      /* ignore */
    }
  };

  ws.onopen = () => {
    opts.onOpen?.();
    handlers.onStatus("Connecting…");
    const accountNumber = opts.accountId.trim();
    ws.send(
      JSON.stringify({
        type: "start",
        account_number: accountNumber,
        account_id: accountNumber || "MOBILE_USER",
        photo_base64: opts.photoBase64,
        access_token: token || undefined,
        step_up_completed: !!opts.stepUpCompleted,
        challenge_round: !!opts.stepUpCompleted,
      }),
    );
  };

  ws.onmessage = (ev) => {
    let msg: Record<string, unknown>;
    try {
      msg = JSON.parse(String(ev.data)) as Record<string, unknown>;
    } catch {
      return;
    }
    const type = String(msg.type || "");

    if (type === "ready") {
      sessionNonce = String(msg.nonce || "");
      handlers.onStatus("Recording live…");
      const early = Number(msg.early_seconds) || 3.5;
      const maxSec = Number(msg.max_seconds) || MAX_CLIENT_SECONDS;
      const started = Date.now();
      frameTimer = window.setInterval(() => {
        if (ws.readyState !== WebSocket.OPEN || finished) return;
        ws.send(
          JSON.stringify({
            type: "frame",
            image_base64: grabFrame(),
            ts: Date.now(),
            nonce: sessionNonce,
          }),
        );
        const elapsed = (Date.now() - started) / 1000;
        handlers.onProgress(
          Math.min(95, (elapsed / early) * 90),
          elapsed < early ? `${Math.max(0, Math.ceil(early - elapsed))}s` : "Almost…",
        );
      }, Math.round(1000 / TARGET_FPS));

      maxTimer = window.setTimeout(() => {
        if (finished || ws.readyState !== WebSocket.OPEN) return;
        handlers.onStatus("Wrapping up…");
        ws.send(JSON.stringify({ type: "end", nonce: sessionNonce }));
      }, maxSec * 1000);
    }

    if (type === "progress") {
      const frames = Number(msg.frames || 0);
      handlers.onStatus(msg.can_finish ? "Almost done…" : `Live · ${frames} frames`);
      if (msg.can_finish) handlers.onProgress(92, "Almost…");
    }

    if (type === "status") {
      handlers.onStatus(String(msg.message || "Checking…"));
      handlers.onProgress(96, "");
    }

    if (type === "result") {
      cleanup();
      handlers.onProgress(100, "");
      handlers.onResult(msg as VerificationResult);
    }

    if (type === "error") {
      cleanup();
      handlers.onError(String(msg.message || "Unknown error"));
    }
  };

  ws.onerror = () => {
    if (finished) return;
    cleanup();
    handlers.onError("WebSocket failed");
  };

  ws.onclose = () => {
    if (!finished) {
      finished = true;
      if (frameTimer != null) window.clearInterval(frameTimer);
      if (maxTimer != null) window.clearTimeout(maxTimer);
      handlers.onError("WebSocket closed");
    }
  };

  return cleanup;
}
