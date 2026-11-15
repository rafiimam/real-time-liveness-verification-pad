import { useCallback, useEffect, useRef, useState } from "react";
import { captureFrameJpeg, openCamera, type FacingMode } from "./camera";
import { startLiveVerification } from "./liveVerify";
import { defaultAccessToken, defaultApiUrl, type VerificationResult, type VerifyStep } from "./types";
import "./App.css";

const FLOW_STEPS = [
  { id: "account", label: "Account" },
  { id: "liveness", label: "Liveness" },
  { id: "done", label: "Result" },
] as const;

function flowIndex(step: VerifyStep): number {
  if (step === "account") return 0;
  if (step === "done") return 2;
  return 1;
}

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const cleanupLiveRef = useRef<(() => void) | null>(null);
  const autoStartRef = useRef<number | null>(null);
  const photoRef = useRef<string | null>(null);
  const stepUpUsedRef = useRef(false);

  const [step, setStep] = useState<VerifyStep>("account");
  const [status, setStatus] = useState("Enter your account number to continue");
  const [guide, setGuide] = useState("Center your face, then take a photo");
  const [countdown, setCountdown] = useState("");
  const [progress, setProgress] = useState(0);
  const [photoBase64, setPhotoBase64] = useState<string | null>(null);
  const [streamReady, setStreamReady] = useState(false);
  const [result, setResult] = useState<VerificationResult | null>(null);
  const [errorDetail, setErrorDetail] = useState<string | null>(null);
  const [accountNumber, setAccountNumber] = useState("");
  const [facing, setFacing] = useState<FacingMode>("user");
  const [challengePrompt, setChallengePrompt] = useState<string | null>(null);

  const apiUrl = defaultApiUrl();
  const accessToken = defaultAccessToken();

  photoRef.current = photoBase64;

  const startCamera = useCallback(async (mode: FacingMode = facing) => {
    try {
      setStreamReady(false);
      setStatus(mode === "environment" ? "Opening back camera…" : "Opening front camera…");
      const existing = videoRef.current?.srcObject as MediaStream | null;
      existing?.getTracks().forEach((t) => t.stop());

      const stream = await openCamera(mode);
      const video = videoRef.current;
      if (!video) return;
      video.srcObject = stream;
      await video.play();
      setFacing(mode);
      setStreamReady(true);
      setStatus(mode === "environment" ? "Back camera ready" : "Front camera ready");
      setGuide(
        mode === "environment"
          ? "Point at the PC screen, then take a photo"
          : "Center your face, then take a photo",
      );
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setStatus(`Camera error: ${message}`);
      setStreamReady(false);
    }
  }, [facing]);

  useEffect(() => {
    return () => {
      cleanupLiveRef.current?.();
      if (autoStartRef.current != null) window.clearTimeout(autoStartRef.current);
      const media = videoRef.current?.srcObject as MediaStream | null;
      media?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  const goToLiveness = useCallback(async () => {
    const acct = accountNumber.trim();
    if (!acct) {
      setStatus("Please enter your account number");
      return;
    }
    setErrorDetail(null);
    setResult(null);
    setStep("photo");
    setGuide("Center your face, then take a photo");
    setStatus("Starting camera…");
    await startCamera("user");
  }, [accountNumber, startCamera]);

  const flipCamera = async () => {
    if (step === "streaming" || step === "account") return;
    const next: FacingMode = facing === "user" ? "environment" : "user";
    await startCamera(next);
  };

  const startVerify = useCallback(
    (photo?: string | null) => {
      const shot = photo ?? photoRef.current;
      if (!shot || !videoRef.current || !canvasRef.current) return;

      cleanupLiveRef.current?.();
      setStep("streaming");
      setGuide(
        challengePrompt
          ? challengePrompt
          : "Hold still — finishing as soon as it’s ready",
      );
      setStatus("Connecting…");
      setProgress(5);
      setCountdown("");
      setResult(null);
      setErrorDetail(null);

      const media = videoRef.current?.srcObject as MediaStream | null;
      if (!media) {
        setErrorDetail("Camera not ready");
        setStep("done");
        return;
      }

      const acct = accountNumber.trim();
      if (!acct) {
        setErrorDetail("Account number is required");
        setStep("account");
        setStatus("Enter your account number to continue");
        return;
      }

      const isStepUpRound = stepUpUsedRef.current;

      cleanupLiveRef.current = startLiveVerification({
        apiUrl,
        accountId: acct,
        photoBase64: shot,
        mediaStream: media,
        accessToken,
        stepUpCompleted: isStepUpRound,
        handlers: {
          onStatus: setStatus,
          onProgress: (pct, label) => {
            setProgress(pct);
            setCountdown(label);
          },
          onResult: (data) => {
            const reasons = Array.isArray(data.reason_codes) ? data.reason_codes : [];
            const needsQualityRetry = reasons.includes("CAPTURE_QUALITY");
            const needsStepUp =
              reasons.includes("STEP_UP_REQUIRED") && !stepUpUsedRef.current;

            if (needsQualityRetry) {
              setResult(data);
              setStep("photo");
              setProgress(0);
              setCountdown("");
              photoRef.current = null;
              setPhotoBase64(null);
              setGuide("Hold steady and find better lighting");
              setStatus("Image too soft — please retry with a clearer capture");
              return;
            }

            if (needsStepUp) {
              stepUpUsedRef.current = true;
              const prompt =
                (data.challenge && typeof data.challenge === "object"
                  ? String((data.challenge as { prompt?: string }).prompt || "")
                  : "") || "Complete the challenge, then take another photo";
              setChallengePrompt(prompt);
              setResult(data);
              setStep("photo");
              setProgress(0);
              setCountdown("");
              photoRef.current = null;
              setPhotoBase64(null);
              setGuide(prompt);
              setStatus("Extra check required — follow the on-screen challenge");
              return;
            }

            setResult(data);
            setStep("done");
            setProgress(100);
            setCountdown("");
            const ok = !!data.is_approved;
            setGuide(ok ? "You’re verified" : "Could not verify");
            setStatus(ok ? "All set." : "Try again with better lighting.");
            setChallengePrompt(null);
          },
          onError: (message) => {
            setErrorDetail(message);
            setStep("done");
            setGuide("Something went wrong");
            setStatus(message);
          },
        },
      });
    },
    [accountNumber, apiUrl, accessToken, challengePrompt],
  );

  const takePhoto = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !streamReady) return;

    const data = captureFrameJpeg(video, canvas, 0.92, 960);
    photoRef.current = data;
    setPhotoBase64(data);
    setStep("ready");
    setGuide("Keep your face in the oval");
    setStatus("Photo saved. Starting live check…");
    setResult(null);
    setErrorDetail(null);

    if (autoStartRef.current != null) window.clearTimeout(autoStartRef.current);
    autoStartRef.current = window.setTimeout(() => {
      startVerify(data);
    }, 500);
  }, [startVerify, streamReady]);

  const resetAll = () => {
    cleanupLiveRef.current?.();
    cleanupLiveRef.current = null;
    if (autoStartRef.current != null) window.clearTimeout(autoStartRef.current);
    photoRef.current = null;
    stepUpUsedRef.current = false;
    setChallengePrompt(null);
    setPhotoBase64(null);
    setResult(null);
    setErrorDetail(null);
    setProgress(0);
    setCountdown("");
    setStep("photo");
    setGuide(
      facing === "environment"
        ? "Point at the PC screen, then take a photo"
        : "Center your face, then take a photo",
    );
    setStatus(streamReady ? (facing === "environment" ? "Back camera ready" : "Front camera ready") : "Starting camera…");
    if (!streamReady) void startCamera(facing);
  };

  const changeAccount = () => {
    cleanupLiveRef.current?.();
    cleanupLiveRef.current = null;
    if (autoStartRef.current != null) window.clearTimeout(autoStartRef.current);
    const media = videoRef.current?.srcObject as MediaStream | null;
    media?.getTracks().forEach((t) => t.stop());
    if (videoRef.current) videoRef.current.srcObject = null;
    photoRef.current = null;
    stepUpUsedRef.current = false;
    setChallengePrompt(null);
    setPhotoBase64(null);
    setResult(null);
    setErrorDetail(null);
    setProgress(0);
    setCountdown("");
    setStreamReady(false);
    setStep("account");
    setStatus("Enter your account number to continue");
  };

  const activeFlow = flowIndex(step);
  const primaryLabel =
    step === "photo"
      ? "Start Test"
      : step === "ready"
        ? "Start verification"
        : step === "streaming"
          ? "Checking…"
          : "";

  const stageClass = [
    "stage",
    step === "streaming" ? "checking" : "",
    step === "done" && result?.is_approved ? "done-ok" : "",
    step === "done" && (result ? !result.is_approved : !!errorDetail) ? "done-bad" : "",
  ]
    .filter(Boolean)
    .join(" ");

  const previewClass = facing === "user" ? "preview mirror" : "preview";
  const flashClass = facing === "user" ? "photo-flash mirror" : "photo-flash";

  const approved = !!result?.is_approved;
  const showResult = step === "done" && (result || errorDetail);
  const onLiveness = step !== "account";

  const ynLabel = (v: boolean | null | undefined) =>
    v == null ? "n/a" : v ? "true" : "false";
  const ynClass = (v: boolean | null | undefined) =>
    v == null ? "na" : v ? "yes" : "no";

  return (
    <main className="shell">
      <header className="top">
        <p className="brand">SecureCheck</p>
        {onLiveness && accountNumber.trim() ? (
          <button type="button" className="acct-chip" onClick={changeAccount}>
            A/C {accountNumber.trim()}
          </button>
        ) : null}
      </header>

      <nav className="flow" aria-label="Verification steps">
        {FLOW_STEPS.map((s, i) => {
          const state = i < activeFlow ? "done" : i === activeFlow ? "active" : "todo";
          return (
            <div key={s.id} className={`flow-step ${state}`}>
              <span className="flow-dot" aria-hidden />
              <span className="flow-text">{s.label}</span>
              {i < FLOW_STEPS.length - 1 ? <span className="flow-line" aria-hidden /> : null}
            </div>
          );
        })}
      </nav>

      {step === "account" ? (
        <section className="account-card">
          <h1 className="account-title">Enter account number</h1>
          <p className="account-sub">We’ll verify your face against your bank records.</p>
          <label className="sr-only" htmlFor="accountNumber">
            Account number
          </label>
          <input
            id="accountNumber"
            className="account-input"
            value={accountNumber}
            onChange={(e) => setAccountNumber(e.target.value.replace(/[^\d]/g, ""))}
            placeholder="Account number"
            inputMode="numeric"
            autoComplete="off"
            autoFocus
            onKeyDown={(e) => {
              if (e.key === "Enter") void goToLiveness();
            }}
          />
          <button
            type="button"
            className="cta"
            disabled={!accountNumber.trim()}
            onClick={() => void goToLiveness()}
          >
            Continue to liveness
          </button>
          <p className="status account-status">{status}</p>
        </section>
      ) : (
        <>
          <div className={stageClass}>
            <video ref={videoRef} className={previewClass} playsInline muted autoPlay />
            <canvas ref={canvasRef} className="snap" />
            {photoBase64 && step === "ready" ? (
              <img className={flashClass} src={photoBase64} alt="Captured" />
            ) : null}
            <div className="overlay">
              <div className="oval" aria-hidden />
              <p className="guide">{guide}</p>
              <p className="countdown">{countdown}</p>
            </div>
            {step !== "streaming" && step !== "done" ? (
              <button
                type="button"
                className="cam-flip"
                onClick={() => void flipCamera()}
                disabled={!streamReady}
                aria-label="Switch camera"
              >
                {facing === "user" ? "Back cam" : "Front cam"}
              </button>
            ) : null}
          </div>

          <section className="panel">
            <p className="status">{status}</p>
            {facing === "environment" && step === "photo" ? (
              <p className="hint-line">
                Back camera — point at the PC playing the video to test screen spoof.
              </p>
            ) : null}
            {step === "streaming" ? (
              <div className="progress-bar">
                <div className="progress-fill" style={{ width: `${progress}%` }} />
              </div>
            ) : null}

            {step !== "done" ? (
              <button
                type="button"
                className="cta"
                disabled={
                  (step === "photo" && !streamReady) ||
                  step === "streaming" ||
                  (step === "ready" && !photoBase64)
                }
                onClick={() => {
                  if (step === "photo") takePhoto();
                  else if (step === "ready") startVerify();
                }}
              >
                {primaryLabel}
              </button>
            ) : (
              <div className="done-actions">
                <button type="button" className="cta" onClick={resetAll}>
                  Try again
                </button>
                <button type="button" className="ghost" onClick={changeAccount}>
                  Change account
                </button>
              </div>
            )}

            {showResult ? (
              <div className={`result ${errorDetail ? "bad" : approved ? "ok" : "bad"}`}>
                <p className="result-title">
                  {errorDetail ? "FAILED" : approved ? "APPROVED" : "REJECTED"}
                </p>
                {errorDetail ? (
                  <p className="result-detail">{errorDetail}</p>
                ) : (
                  <>
                    {result?.customer_name ? (
                      <p className="result-customer">{result.customer_name}</p>
                    ) : null}
                    <ul className="result-rows">
                      <li>
                        <span>Face matched against NID</span>
                        <strong className={ynClass(result?.nid_photo_match)}>
                          {ynLabel(result?.nid_photo_match)}
                        </strong>
                      </li>
                      <li>
                        <span>Face matched against CBS / account</span>
                        <strong className={ynClass(result?.account_photo_match)}>
                          {ynLabel(result?.account_photo_match)}
                        </strong>
                      </li>
                      <li>
                        <span>Liveness</span>
                        <strong className={ynClass(result?.liveness)}>
                          {ynLabel(result?.liveness)}
                        </strong>
                      </li>
                      {result?.nid_verified != null ? (
                        <li>
                          <span>NID name verified</span>
                          <strong className={ynClass(result.nid_verified)}>
                            {ynLabel(result.nid_verified)}
                          </strong>
                        </li>
                      ) : null}
                    </ul>
                    {Array.isArray(result?.reason_codes) && result.reason_codes.length ? (
                      <p className="result-detail">Reasons: {result.reason_codes.join(", ")}</p>
                    ) : null}
                  </>
                )}
              </div>
            ) : null}
          </section>
        </>
      )}
    </main>
  );
}
