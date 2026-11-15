import type { VerificationResult } from "./types";
import { startHttpVerification, type VerifyHandlers } from "./httpVerify";
import { startWsVerification } from "./wsVerify";

export type { VerifyHandlers };

/**
 * Prefer WebSocket for early finish; if the proxy blocks WS (GET 404),
 * fall back to short MediaRecorder + POST /verify (when server allows upload).
 */
export function startLiveVerification(opts: {
  apiUrl: string;
  accountId: string;
  photoBase64: string;
  mediaStream: MediaStream;
  handlers: VerifyHandlers;
  accessToken?: string;
  stepUpCompleted?: boolean;
}): () => void {
  let cleaned = false;
  let activeCleanup: (() => void) | null = null;
  let fallbackTimer: number | null = null;
  let wsOpened = false;
  let httpStarted = false;

  const cleanup = () => {
    cleaned = true;
    if (fallbackTimer != null) window.clearTimeout(fallbackTimer);
    fallbackTimer = null;
    activeCleanup?.();
    activeCleanup = null;
  };

  const useHttp = () => {
    if (cleaned || httpStarted || wsOpened) return;
    httpStarted = true;
    if (fallbackTimer != null) window.clearTimeout(fallbackTimer);
    fallbackTimer = null;
    activeCleanup?.();
    opts.handlers.onStatus("Using upload mode…");
    activeCleanup = startHttpVerification(opts);
  };

  activeCleanup = startWsVerification({
    ...opts,
    handlers: {
      ...opts.handlers,
      onError: (message) => {
        if (wsOpened) {
          opts.handlers.onError(message);
          return;
        }
        useHttp();
      },
    },
    onOpen: () => {
      wsOpened = true;
      if (fallbackTimer != null) window.clearTimeout(fallbackTimer);
      fallbackTimer = null;
    },
  });

  fallbackTimer = window.setTimeout(() => {
    if (cleaned || wsOpened || httpStarted) return;
    useHttp();
  }, 1800);

  return cleanup;
}

export type { VerificationResult };
