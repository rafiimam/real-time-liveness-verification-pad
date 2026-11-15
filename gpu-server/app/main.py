from fastapi import FastAPI
from app.routers import detect, spoof, embedding, stream, faces

app = FastAPI(title="Liveness Verification - GPU Inference Server", version="1.0.0")

app.include_router(detect.router, prefix="/detect", tags=["detection"])
app.include_router(spoof.router, prefix="/spoof", tags=["spoof"])
app.include_router(embedding.router, prefix="/embedding", tags=["embedding"])
app.include_router(faces.router, prefix="/faces", tags=["faces"])
app.include_router(stream.router, prefix="/process-stream", tags=["stream"])


@app.get("/health")
async def health():
    return {"status": "ok", "gpu": True}
