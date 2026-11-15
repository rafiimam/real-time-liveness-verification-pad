import type { VerificationResult } from "./types";
import { apiBaseFrom, authHeaders } from "./types";

const RECORD_SECONDS = 8;
const TARGET_FPS_HINT = 8;

export type VerifyHandlers = {
  onStatus: (text: string) => void;
  onProgress: (pct: number, label: string) => void;
  onResult: (data: VerificationResult) => void;
  onError: (message: string) => void;
};

function dataUrlToBlob(dataUrl: string): Blob {
  const [header, b64] = dataUrl.split(",");
  const mime = /data:(.*?);/.exec(header)?.[1] || "image/jpeg";
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Blob([bytes], { type: mime });
}

function pickMimeType(): string {
  const candidates = [
    "video/webm;codecs=vp8",
    "video/webm",
    "video/mp4",
  ];
  for (const t of candidates) {
    if (typeof MediaRecorder !== "undefined" && MediaRecorder.isTypeSupported(t)) {
      return t;
    }
  }
  return "";
}

/**
 * Record a short live clip and POST to /verify.
 * Works through reverse proxies that block WebSocket upgrades.
 * Lower capture-trust than live WS; may be disabled server-side.
 */
export function startHttpVerification(opts: {
  apiUrl: string;
  accountId: string;
  photoBase64: string;
  mediaStream: MediaStream;
  handlers: VerifyHandlers;
  accessToken?: string;
}): () => void {
  const { handlers } = opts;
  const base = apiBaseFrom(opts.apiUrl);
  localStorage.setItem("liveness_api_url", base);
  const token = opts.accessToken || localStorage.getItem("liveness_api_key") || "";

  let cancelled = false;
  let recorder: MediaRecorder | null = null;
  let tickTimer: number | null = null;
  let stopTimer: number | null = null;

  const cleanup = () => {
    cancelled = true;
    if (tickTimer != null) window.clearInterval(tickTimer);
    if (stopTimer != null) window.clearTimeout(stopTimer);
    tickTimer = null;
    stopTimer = null;
    if (recorder && recorder.state !== "inactive") {
      try {
        recorder.stop();
      } catch {
        /* ignore */
      }
    }
  };

  const run = async () => {
    handlers.onStatus("Recording live…");
    handlers.onProgress(8, `${RECORD_SECONDS}s`);

    const mime = pickMimeType();
    const chunks: Blob[] = [];
    try {
      recorder = mime
        ? new MediaRecorder(opts.mediaStream, { mimeType: mime, videoBitsPerSecond: 1_200_000 })
        : new MediaRecorder(opts.mediaStream);
    } catch (err) {
      handlers.onError(
        err instanceof Error ? err.message : "MediaRecorder not supported on this browser",
      );
      return;
    }

    recorder.ondataavailable = (ev) => {
      if (ev.data && ev.data.size > 0) chunks.push(ev.data);
    };

    const started = Date.now();
    tickTimer = window.setInterval(() => {
      if (cancelled) return;
      const elapsed = (Date.now() - started) / 1000;
      const left = Math.max(0, Math.ceil(RECORD_SECONDS - elapsed));
      const pct = Math.min(70, (elapsed / RECORD_SECONDS) * 70);
      handlers.onProgress(pct, left > 0 ? `${left}s` : "Uploading…");
    }, 200);

    await new Promise<void>((resolve, reject) => {
      recorder!.onerror = () => reject(new Error("Recording failed"));
      recorder!.onstop = () => resolve();
      recorder!.start(Math.round(1000 / TARGET_FPS_HINT));
      stopTimer = window.setTimeout(() => {
        if (recorder && recorder.state === "recording") recorder.stop();
      }, RECORD_SECONDS * 1000);
    });

    if (cancelled) return;
    if (tickTimer != null) window.clearInterval(tickTimer);

    const type = recorder.mimeType || mime || "video/webm";
    const videoBlob = new Blob(chunks, { type });
    if (videoBlob.size < 1000) {
      handlers.onError("Recording was empty. Try again.");
      return;
    }

    handlers.onStatus("Uploading…");
    handlers.onProgress(78, "");

    const form = new FormData();
    form.append("photo", dataUrlToBlob(opts.photoBase64), "photo.jpg");
    const ext = type.includes("mp4") ? "mp4" : "webm";
    form.append("video", videoBlob, `live.${ext}`);
    form.append("account_id", opts.accountId.trim() || "MOBILE_USER");
    form.append("account_number", opts.accountId.trim() || "MOBILE_USER");

    try {
      handlers.onStatus("Checking…");
      handlers.onProgress(90, "");
      const resp = await fetch(`${base}/verify`, {
        method: "POST",
        body: form,
        headers: authHeaders(token),
      });
      if (!resp.ok) {
        const text = await resp.text();
        throw new Error(text || `HTTP ${resp.status}`);
      }
      const data = (await resp.json()) as VerificationResult;
      if (cancelled) return;
      handlers.onProgress(100, "");
      handlers.onResult(data);
    } catch (err) {
      if (cancelled) return;
      handlers.onError(err instanceof Error ? err.message : String(err));
    }
  };

  void run();
  return cleanup;
}
