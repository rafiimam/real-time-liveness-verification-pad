export type VerifyStep = "account" | "photo" | "ready" | "streaming" | "done";

export interface VerificationResult {
  type?: string;
  is_approved?: boolean;
  face_match?: boolean;
  face_match_confidence?: number;
  nid_photo_match?: boolean;
  nid_photo_match_confidence?: number;
  account_photo_match?: boolean;
  account_photo_match_confidence?: number;
  nid_verified?: boolean;
  customer_name?: string;
  face_match_provider?: string;
  cbs_faces_detected?: number;
  liveness?: boolean;
  frames_received?: number;
  message?: string;
  reason_codes?: string[];
  capture_trust?: string;
  challenge?: { code?: string; prompt?: string } | null;
  [key: string]: unknown;
}

export interface LiveProgress {
  frames: number;
  elapsed_ms: number;
  can_finish?: boolean;
}

export function apiBaseFrom(url: string): string {
  return url.trim().replace(/\/$/, "");
}

export function wsBaseFrom(httpBase: string): string {
  if (httpBase.startsWith("https://")) return `wss://${httpBase.slice(8)}`;
  if (httpBase.startsWith("http://")) return `ws://${httpBase.slice(7)}`;
  return httpBase;
}

/** Same-origin by default when served from local-api /app. */
export function defaultApiUrl(): string {
  return window.location.origin;
}

/** Build-time API key only — never shown in the customer UI. */
export function defaultAccessToken(): string {
  return (import.meta.env.VITE_API_KEY as string | undefined) || "";
}

export function authHeaders(token: string): HeadersInit {
  if (!token) return {};
  return { Authorization: `Bearer ${token}` };
}

export function withAccessToken(url: string, token: string): string {
  if (!token) return url;
  const u = new URL(url);
  u.searchParams.set("access_token", token);
  return u.toString();
}
