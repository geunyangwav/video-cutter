"""영상 처리 파이프라인. 여러 영상이면 각각 분석한 뒤 CapCut 프로젝트 하나로 이어 붙인다.
단계별 이벤트를 async generator로 내보낸다."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator

from .asr import captions_from, transcribe
from .draft import DRAFT_ROOT, Part, build_project
from .face import caption_y, product_fixed_y, to_transform_y
from .filler import apply_text_edits, find_marks
from .media import content_hash, probe
from .silence import SilenceParams, detect_silences, keep_ranges, load_audio, subtract

MIN_STEP = 0.5  # 캐시 hit 이어도 단계 애니메이션이 보이도록 최소 지연
ASR_LOCK = asyncio.Lock()  # mlx/numba 는 동시 호출에 안전하지 않다 → 직렬화

STEPS = [
    ("silence", "무음 감지"),
    ("asr", "음성 인식"),
    ("filler", "잔말 · NG"),
    ("draft", "드래프트 · 자막"),
]
PLACEMENT = {"face": "자막 턱 아래", "product": "자막 제품 위"}


@dataclass
class Job:
    id: str
    paths: list[Path]
    filenames: list[str]
    params: SilenceParams = field(default_factory=SilenceParams)
    captions: bool = True
    result: dict | None = None


async def _timed(fn, *args):
    t0 = time.perf_counter()
    out = await asyncio.to_thread(fn, *args)
    dt = time.perf_counter() - t0
    if dt < MIN_STEP:
        await asyncio.sleep(MIN_STEP - dt)
    return out, dt


@dataclass
class Analysis:
    part: Part
    levels: dict
    marks: list
    transcript: object
    placement: str | None


async def _analyze(path: Path, p: SilenceParams, want_captions: bool, emit) -> Analysis:
    """영상 하나: 무음 → 음성 인식 → 잔말 → 자막 위치. emit(phase, status, detail, elapsed)."""
    info = probe(path)
    audio = await asyncio.to_thread(load_audio, str(path))

    await emit("silence", "running")
    (sil, levels), dt = await _timed(detect_silences, str(path), info.duration, p, audio)
    keeps = keep_ranges(info.duration, sil, p.pad_before, p.pad_after)
    kept = sum(e - s for s, e in keeps)
    await emit("silence", "done", f"{len(keeps)}개 구간 · {info.duration - kept:.1f}s 덜어냄", dt)

    captions, transcript, marks, placement, cap_y = [], None, [], None, None
    if want_captions:
        await emit("asr", "running", "" if _model_ready() else "처음 한 번은 모델을 내려받아요")
        async with ASR_LOCK:
            (transcript, hit), dt = await _timed(transcribe, audio, content_hash(path))
        await emit("asr", "done", ("캐시 · " if hit else "") + f"{len(transcript.segments)}문장", dt)

        await emit("filler", "running")
        marks, dt = await _timed(find_marks, transcript, audio, keeps)
        apply_text_edits(transcript, marks)
        auto = [(m.start, m.end) for m in marks if m.auto]
        # 자막 줄바꿈은 무음 컷 기준 (조사 컷에서 '생각|합니다'가 갈라지지 않게)
        cuts = [(e, s) for (_, e), (s, _) in zip(keeps, keeps[1:])]
        if auto:
            keeps = [r for r in subtract(keeps, auto) if r[1] - r[0] >= 0.12]
        n_sug = sum(not m.auto for m in marks)
        await emit("filler", "done", f"{len(auto)}곳 다듬음" + (f" · 제안 {n_sug}개" if n_sug else ""), dt)

        captions = captions_from(transcript, cuts, skip=[(m.start, m.end) for m in marks if m.kind == "filler"])
        if captions:
            cy, placement = await asyncio.to_thread(caption_y, str(path), keeps)
            cap_y = to_transform_y(cy)
            if placement == "product":
                # 영상 전체에서 한 위치로 고정. 자막이 나오는 장면들의 제품을 가장 덜 가리는 높이.
                y = await asyncio.to_thread(product_fixed_y, str(path), [(c.start + c.end) / 2 for c in captions])
                cap_y = to_transform_y(y)
    return Analysis(Part(info, keeps, captions, cap_y), levels, marks, transcript, placement)


async def run(job: Job) -> AsyncIterator[dict]:
    multi = len(job.paths) > 1
    if multi:
        # 영상마다 한 줄 + 마지막에 프로젝트 만들기
        steps = [{"id": f"v{i}", "label": f"{i + 1}. {n}"} for i, n in enumerate(job.filenames)]
        steps.append({"id": "draft", "label": "CapCut 프로젝트 하나로"})
    else:
        steps = [{"id": i, "label": l} for i, l in STEPS if job.captions or i not in ("asr", "filler")]
    yield {"type": "steps", "steps": steps}

    queue: asyncio.Queue = asyncio.Queue()
    labels = dict(STEPS)

    def emitter(i: int):
        async def emit(phase, status, detail="", elapsed=None):
            if multi:
                # 한 줄 안에서 지금 단계를 보여준다
                ev = {"type": "step", "id": f"v{i}", "status": "running",
                      "detail": f"{labels[phase]}" + (f" · {detail}" if detail else "")}
            else:
                ev = {"type": "step", "id": phase, "status": status, "detail": detail}
                if elapsed is not None:
                    ev["elapsed"] = elapsed
            await queue.put(ev)
        return emit

    try:
        analyses: list[Analysis] = []
        for i, path in enumerate(job.paths):
            t0 = time.perf_counter()
            task = asyncio.create_task(_analyze(path, job.params, job.captions, emitter(i)))
            while not task.done() or not queue.empty():
                try:
                    yield await asyncio.wait_for(queue.get(), 0.2)
                except asyncio.TimeoutError:
                    pass
            a = task.result()
            analyses.append(a)
            if multi:
                info, kept = a.part.info, sum(e - s for s, e in a.part.keeps)
                where = PLACEMENT.get(a.placement or "", "")
                yield {"type": "step", "id": f"v{i}", "status": "done", "elapsed": time.perf_counter() - t0,
                       "detail": f"{fmt(info.duration)} → {fmt(kept)}" + (f" · {where}" if where else "")}

        yield {"type": "step", "id": "draft", "status": "running"}
        if not DRAFT_ROOT.exists():
            raise RuntimeError("CapCut 프로젝트 폴더를 못 찾았어요. CapCut 데스크톱 앱을 설치하고 한 번 실행해 주세요.")
        draft, dt = await _timed(build_project, [a.part for a in analyses], None, DRAFT_ROOT)
        where = "" if multi else PLACEMENT.get(analyses[0].placement or "", "")
        detail = draft["name"] + (f" · 영상 {len(analyses)}개" if multi else "") + (f" · {where}" if where else "")
        yield {"type": "step", "id": "draft", "status": "done", "elapsed": dt, "detail": detail}

        job.result = _result(job, draft, analyses)
        yield {"type": "result", **job.result}
    except RuntimeError as e:
        yield {"type": "error", "message": str(e)}
    except Exception as e:  # noqa: BLE001 — UI에 그대로 보여준다
        yield {"type": "error", "message": f"{type(e).__name__}: {e}"}


def _result(job: Job, draft: dict, analyses: list[Analysis]) -> dict:
    """여러 영상은 원본을 이어 붙인 시간축 하나로 합쳐 보여준다."""
    original = final = 0.0
    keeps, marks, transcript, bounds = [], [], [], []
    multi = len(analyses) > 1
    for i, a in enumerate(analyses):
        off = original
        tag = f"{i + 1}번 · " if multi else ""
        keeps += [(off + s, off + e) for s, e in a.part.keeps]
        marks += [{**m.dict(), "start": off + m.start, "end": off + m.end, "text": tag + m.text} for m in a.marks]
        if a.transcript:
            transcript += [{"start": off + s.start, "end": off + s.end, "text": tag + s.text}
                           for s in a.transcript.segments]
        original += a.part.info.duration
        final += sum(e - s for s, e in a.part.keeps)
        bounds.append(original)
    first = analyses[0].part.info
    return {
        "draft": draft["name"],
        "filename": ", ".join(job.filenames),
        "videos": len(analyses),
        "original": original,
        "final": final,
        "removed": original - final,
        "clips": len(keeps),
        "captions": draft["captions"],
        "size": f"{first.width}×{first.height}",
        "keeps": keeps,
        "marks": marks,
        "transcript": transcript,
        "bounds": bounds[:-1],  # 영상 사이 경계 (원본 이어 붙인 시간축)
    }


def fmt(s: float) -> str:
    m = int(s // 60)
    return f"{m}:{s % 60:04.1f}" if m else f"{s:.1f}s"


def _model_ready() -> bool:
    return (Path.home() / ".cache/huggingface/hub/models--mlx-community--whisper-large-v3-turbo").exists()
