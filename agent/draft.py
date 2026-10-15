"""pycapcut으로 CapCut 프로젝트(cut_N) 생성."""
from __future__ import annotations

import json
import os
import re
import shutil
import time
import unicodedata
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import pycapcut as cc
from pycapcut import trange

from .media import MediaInfo

DRAFT_ROOT = Path.home() / "Movies/CapCut/User Data/Projects/com.lveditor.draft"
PREFIX = "cut_"
US = 1_000_000
# CapCut 에 설치된 사용자 글꼴 (조선굴림체). 없으면 CapCut 기본 글꼴.
CAPTION_FONT = Path.home() / "Library/Containers/com.lemon.lvoverseas/Data/Library/Fonts/ChosunGu.TTF"
CAPTION_SIZE = 7.0


def next_draft_name(root: Path = DRAFT_ROOT) -> str:
    nums = [int(m.group(1)) for d in os.listdir(root)
            if (m := re.fullmatch(rf"{PREFIX}(\d+)", d))]
    return f"{PREFIX}{max(nums, default=0) + 1}"


@dataclass
class Caption:
    start: float  # 원본 영상 기준 초
    end: float
    text: str
    y: float | None = None  # 자막별 transform.y (없으면 영상 공통 위치)


def map_to_timeline(t0: float, t1: float, keeps: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """원본 시간 구간을 컷 편집 후 타임라인 구간(들)으로 변환."""
    out, offset = [], 0.0
    for ks, ke in keeps:
        s, e = max(t0, ks), min(t1, ke)
        if e > s:
            out.append((offset + s - ks, offset + e - ks))
        offset += ke - ks
    return out


@dataclass
class Part:
    """한 프로젝트에 이어 붙일 영상 하나."""
    info: MediaInfo
    keeps: list[tuple[float, float]]
    captions: list[Caption] = field(default_factory=list)
    caption_y: float | None = None  # 이 영상 자막의 transform.y


def build_draft(info: MediaInfo, keeps: list[tuple[float, float]],
                captions: list[Caption] | None = None, name: str | None = None,
                root: Path = DRAFT_ROOT, caption_y: float | None = None) -> dict:
    """영상 1개짜리 프로젝트 (build_project 의 단축형)."""
    return build_project([Part(info, keeps, captions or [], caption_y)], name, root)


def build_project(parts: list[Part], name: str | None = None, root: Path = DRAFT_ROOT) -> dict:
    """여러 영상을 순서대로 이어 붙인 CapCut 프로젝트 하나. 캔버스는 첫 영상 기준."""
    name = name or next_draft_name(root)
    first = parts[0].info
    folder = cc.DraftFolder(str(root))
    script = folder.create_draft(name, first.width, first.height, fps=round(first.fps) or 30)
    script.add_track(cc.TrackType.video)

    cursor = 0
    hdr_by_material: dict[str, bool] = {}
    placed: list[list] = []
    n_segments = 0
    for part in parts:
        info = part.info
        material = cc.VideoMaterial(str(stage_source(info.path, root / name)))
        # 아이폰 세로 영상은 가로로 저장 + 회전 표시. CapCut 은 회전을 반영한 크기(1080×1920)를 기대한다.
        # 저장된 크기(1920×1080)를 그대로 넣으면 선택 테두리가 가로로 납작하게 잡힌다.
        material.width, material.height = info.width, info.height
        script.add_material(material)
        hdr_by_material[material.material_id] = info.hdr
        mat_dur = material.duration  # μs
        start = cursor
        for s, e in part.keeps:
            src_s = int(round(s * US))
            src_e = min(int(round(e * US)), mat_dur)
            dur = src_e - src_s
            if dur <= 0:
                continue
            script.add_segment(cc.VideoSegment(material, trange(cursor, dur), source_timerange=trange(src_s, dur)))
            cursor += dur
            n_segments += 1

        default_y = part.caption_y
        if default_y is None:
            default_y = -0.22 if info.height > info.width else -0.78
        for c in part.captions:
            pieces = map_to_timeline(c.start, c.end, part.keeps)
            if not pieces:
                continue
            ts, te = pieces[0][0], pieces[-1][1]  # 컷으로 붙은 조각들은 타임라인에서 이어져 있다
            a = max(start + int(round(ts * US)), placed[-1][1] if placed else 0)
            b = min(start + int(round(te * US)), cursor)
            if b - a >= 100_000:  # 0.1s 미만은 생략
                placed.append([a, b, c.text, default_y if c.y is None else c.y, start])

    if placed:
        # 같은 영상 안에서 자막 사이 짧은 빈틈(0.6s 미만)은 앞 자막을 늘려 깜빡임 방지
        for cur, nxt in zip(placed, placed[1:]):
            if cur[4] == nxt[4] and nxt[0] - cur[1] < 600_000:
                cur[1] = nxt[0]
        script.add_track(cc.TrackType.text, "자막")
        style = cc.TextStyle(size=CAPTION_SIZE, color=(1, 1, 1), align=1, auto_wrapping=True, max_line_width=0.8)
        border = cc.TextBorder(color=(0, 0, 0), alpha=0.85, width=30)
        for a, b, text, y, _ in placed:
            script.add_segment(cc.TextSegment(text, trange(a, b - a), style=style, border=border,
                                              clip_settings=cc.ClipSettings(transform_y=y)), "자막")

    script.save()
    _finalize(root / name, name, cursor, hdr_by_material)
    return {"name": name, "path": str(root / name), "duration": cursor / US,
            "segments": n_segments, "captions": len(placed)}


def stage_source(src: Path, draft_path: Path) -> Path:
    """CapCut(샌드박스)은 ~/Movies 밖 파일을 못 읽으므로 드래프트 폴더 안에 하드링크(실패 시 복사)."""
    dst_dir = draft_path / "Resources" / "source"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / unicodedata.normalize("NFC", src.name)
    if not dst.exists():
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    return dst


def _apply_font(content: dict, font_path: str) -> None:
    for m in content.get("materials", {}).get("texts", []):
        m["font_path"] = font_path
        c = json.loads(m["content"])
        for st in c.get("styles", []):
            st["font"] = {"path": font_path, "id": ""}
        m["content"] = json.dumps(c, ensure_ascii=False)


def _finalize(path: Path, name: str, duration_us: int, hdr_by_material: dict[str, bool]) -> None:
    """CapCut(Mac 9.x)이 인식하도록 draft_info.json 과 메타 정보를 맞춘다."""
    content = json.loads((path / "draft_content.json").read_text("utf-8"))
    draft_id = str(uuid.uuid4()).upper()
    content["id"] = draft_id
    content["name"] = name
    if CAPTION_FONT.exists():
        _apply_font(content, str(CAPTION_FONT))
    # CapCut 은 HDR(HLG/PQ) 원본에 hdr_settings.mode=4, SDR 에 1 을 쓴다.
    # pycapcut 기본값은 1 이라 아이폰 HDR 영상의 색이 원본과 달라 보였다.
    for t in content.get("tracks", []):
        if t.get("type") == "video":
            for seg in t.get("segments", []):
                hdr = hdr_by_material.get(seg.get("material_id"), False)
                seg["hdr_settings"] = {"intensity": 1.0, "mode": 4 if hdr else 1, "nits": 1000}
    text = json.dumps(content, ensure_ascii=False)
    (path / "draft_info.json").write_text(text, "utf-8")
    (path / "draft_content.json").write_text(text, "utf-8")

    meta_p = path / "draft_meta_info.json"
    meta = json.loads(meta_p.read_text("utf-8"))
    now_us = int(time.time() * US)
    meta.update({
        "draft_fold_path": str(path),
        "draft_root_path": str(path.parent),
        "draft_id": draft_id,
        "draft_name": name,
        "tm_draft_create": now_us,
        "tm_draft_modified": now_us,
        "tm_duration": duration_us,
    })
    meta_p.write_text(json.dumps(meta, ensure_ascii=False), "utf-8")
