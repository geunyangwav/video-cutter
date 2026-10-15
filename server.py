"""effecting 웹 서버 (CapCut용 자동 컷편집).

    .venv/bin/python server.py   →  http://localhost:8765
"""
from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from agent.media import content_hash
from agent.pipeline import Job, run
from agent.silence import SilenceParams

ROOT = Path(__file__).parent
UPLOADS = ROOT / "uploads"
UPLOADS.mkdir(exist_ok=True)

app = FastAPI()
jobs: dict[str, Job] = {}


@app.get("/")
def index():
    return FileResponse(ROOT / "static/index.html")


def _save_upload(file: UploadFile) -> Path:
    tmp = UPLOADS / f".tmp-{uuid.uuid4().hex}"
    with tmp.open("wb") as f:
        shutil.copyfileobj(file.file, f, length=1 << 22)
    ext = Path(file.filename or "video.mp4").suffix.lower() or ".mp4"
    dst = UPLOADS / f"{content_hash(tmp)}{ext}"
    if dst.exists():
        tmp.unlink()
    else:
        tmp.rename(dst)
    return dst


@app.post("/api/jobs")
async def create_job(file: list[UploadFile] = File(...), params: str = Form("{}"), captions: bool = Form(True)):
    """영상 1개 이상 → CapCut 프로젝트 하나 (보낸 순서대로 이어 붙임)."""
    paths = [_save_upload(f) for f in file]
    names = [f.filename or p.name for f, p in zip(file, paths)]
    known = SilenceParams.__dataclass_fields__
    p = SilenceParams(**{k: float(v) for k, v in json.loads(params).items() if k in known})
    job = Job(id=uuid.uuid4().hex[:12], paths=paths, filenames=names, params=p, captions=captions)
    jobs[job.id] = job
    return {"id": job.id}


@app.get("/api/jobs/{job_id}/events")
async def events(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404)

    async def stream():
        try:
            async for ev in run(job):
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        finally:
            # 업로드 사본은 지운다. 드래프트 폴더에는 같은 파일이 하드링크(또는 복사)로 남아 있다.
            for p in job.paths:
                p.unlink(missing_ok=True)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="warning")
