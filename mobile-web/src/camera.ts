export type FacingMode = "user" | "environment";

/** Soft camera constraints — avoid high ideal sizes that force digital zoom. */
export async function openCamera(facing: FacingMode = "user"): Promise<MediaStream> {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      facingMode: { ideal: facing },
    },
  });

  const [track] = stream.getVideoTracks();
  if (track) {
    const caps = track.getCapabilities?.() as
      | (MediaTrackCapabilities & { zoom?: { min: number; max: number; step?: number } })
      | undefined;
    if (caps?.zoom && typeof track.applyConstraints === "function") {
      try {
        await track.applyConstraints({
          advanced: [{ zoom: caps.zoom.min } as unknown as MediaTrackConstraintSet],
        });
      } catch {
        // zoom not writable on this device
      }
    }
  }

  return stream;
}

/** @deprecated use openCamera("user") */
export async function openFrontCamera(): Promise<MediaStream> {
  return openCamera("user");
}

export function captureFrameJpeg(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
  quality = 0.92,
  maxWidth = 960,
): string {
  const w = video.videoWidth || 640;
  const h = video.videoHeight || 480;
  const scale = Math.min(1, maxWidth / w);
  const tw = Math.round(w * scale);
  const th = Math.round(h * scale);
  canvas.width = tw;
  canvas.height = th;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas unsupported");
  ctx.drawImage(video, 0, 0, tw, th);
  return canvas.toDataURL("image/jpeg", quality);
}
