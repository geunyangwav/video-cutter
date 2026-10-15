"""ffmpeg 경로와 영상 메타데이터."""
from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

import imageio_ffmpeg
import pymediainfo


def ffmpeg_exe() -> str:
    return shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()


@dataclass
class MediaInfo:
    path: Path
    duration: float  # seconds
    width: int       # 회전 반영된 표시 크기
    height: int
    fps: float
    has_audio: bool
    hdr: bool = False  # HLG / PQ (아이폰 HDR 등)


def probe(path: str | Path) -> MediaInfo:
    path = Path(path).resolve()
    info = pymediainfo.MediaInfo.parse(str(path))
    if not info.video_tracks:
        raise ValueError(f"영상 트랙이 없습니다: {path.name}")
    v = info.video_tracks[0]
    w, h = int(v.width), int(v.height)
    rot = float(v.rotation or 0)
    if int(rot) % 180 == 90:
        w, h = h, w
    fps = float(v.frame_rate or 30)
    tc = (v.transfer_characteristics or "").upper()
    hdr = any(k in tc for k in ("HLG", "B67", "PQ", "2084"))
    return MediaInfo(path, float(v.duration) / 1000.0, w, h, fps, bool(info.audio_tracks), hdr)


def content_hash(path: str | Path, chunk: int = 1 << 20) -> str:
    """앞/중간/끝 1MB + 크기로 만드는 빠른 content hash (mtime 무관)."""
    p = Path(path)
    size = p.stat().st_size
    h = hashlib.sha1(str(size).encode())
    with p.open("rb") as f:
        for off in (0, max(0, size // 2 - chunk // 2), max(0, size - chunk)):
            f.seek(off)
            h.update(f.read(chunk))
    return h.hexdigest()[:16]
