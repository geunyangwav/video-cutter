"""mlx-whisper 음성 인식 (Apple Silicon) + content-hash 캐시 + 자막 줄 만들기."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .draft import Caption

MODEL = "mlx-community/whisper-large-v3-turbo"
CACHE = Path(__file__).resolve().parent.parent / "cache" / "asr"
# 잔말을 지우지 말고 그대로 받아쓰도록 유도 (4단 filler 감지용)
PROMPT = "음, 어, 그, 저, 아, 이제, 뭐지, 그니까 같은 말버릇도 들리는 그대로 받아 적습니다."


@dataclass
class Word:
    start: float
    end: float
    text: str
    prob: float = 1.0


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


@dataclass
class Transcript:
    language: str
    segments: list[Segment]

    @property
    def words(self) -> list[Word]:
        return [w for s in self.segments for w in s.words]

    def to_json(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_json(d: dict) -> "Transcript":
        return Transcript(d["language"], [
            Segment(s["start"], s["end"], s["text"], [Word(**w) for w in s["words"]]) for s in d["segments"]])


def transcribe(audio: np.ndarray, key: str, language: str = "ko") -> tuple[Transcript, bool]:
    """(transcript, cache_hit). audio 는 16kHz mono float32."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cp = CACHE / f"{key}-{language}.json"
    if cp.exists():
        return Transcript.from_json(json.loads(cp.read_text("utf-8"))), True

    import mlx_whisper  # 무거워서 지연 import

    r = mlx_whisper.transcribe(
        audio.astype(np.float32), path_or_hf_repo=_local_model(), language=language,
        word_timestamps=True, initial_prompt=PROMPT, condition_on_previous_text=False,
        verbose=None,
    )
    segs = []
    for s in r["segments"]:
        words = [Word(float(w["start"]), float(w["end"]), w["word"], float(w.get("probability", 1)))
                 for w in s.get("words", [])]
        text = s["text"].strip()
        if not text:
            continue
        segs.append(Segment(float(s["start"]), float(s["end"]), text, words))
    t = Transcript(r.get("language", language), segs)
    cp.write_text(json.dumps(t.to_json(), ensure_ascii=False), "utf-8")
    return t, False


def _local_model() -> str:
    """이미 받은 모델이 있으면 그 폴더를 직접 쓴다 → 영상 처리 중에는 인터넷에 접속하지 않음.
    (없을 때만 처음 한 번 모델을 내려받는다. 영상·음성은 어디에도 보내지 않는다.)"""
    from huggingface_hub import snapshot_download
    try:
        return snapshot_download(MODEL, local_files_only=True)
    except Exception:  # noqa: BLE001 — 아직 안 받았으면 받는다
        return snapshot_download(MODEL)


_TRAIL = re.compile(r"[.,!?。、…]+$")


def captions_from(t: Transcript, cuts: list[tuple[float, float]] = (), max_chars: int = 16,
                  gap: float = 0.6, skip: list[tuple[float, float]] = ()) -> list[Caption]:
    """숏츠용 짧은 자막 줄. 단어 타임스탬프로 끊고, 끝 마침표/쉼표는 뗀다.
    0.6s 넘는 쉼(컷)을 사이에 둔 단어는 같은 줄에 묶지 않는다. 짧은 컷은 타임라인에서 붙으니 그대로 묶는다."""
    out: list[Caption] = []

    def crosses_cut(a: Word, b: Word) -> bool:
        return any(cs < b.start and ce > a.start + 0.05 and ce - cs > 0.6 for cs, ce in cuts)

    for seg in t.segments:
        words = seg.words or [Word(seg.start, seg.end, seg.text)]
        # 잘려 나간 잔말은 자막에서도 뺀다
        words = [w for w in words if not any(a <= (w.start + w.end) / 2 <= b for a, b in skip)]
        line: list[Word] = []

        def flush():
            if line:
                text = _TRAIL.sub("", "".join(w.text for w in line).strip())
                if text:
                    out.append(Caption(line[0].start, line[-1].end, text))
                line.clear()

        first = len(out)
        for w in words:
            if line:
                cur = len("".join(x.text for x in line).strip())
                if (w.start - line[-1].end > gap or cur + len(w.text.strip()) > max_chars
                        or _TRAIL.search(line[-1].text.strip()) or crosses_cut(line[-1], w)):
                    flush()
            line.append(w)
        flush()
        _merge_short(out, first, max_chars, cuts)
    return out


def _merge_short(out: list[Caption], first: int, max_chars: int, cuts) -> None:
    """한 문장 안에서 3글자 이하로 떨어진 줄은 옆 줄에 붙인다 ('때 요거', '있고요' 깜빡임 방지)."""
    def blen(c): return len(c.text.replace(" ", ""))
    def long_cut(a, b): return any(a.end - 0.05 < cs and ce < b.start + 0.05 and ce - cs > 0.6 for cs, ce in cuts)
    i = first
    while i < len(out):
        c = out[i]
        if blen(c) > 3 or len(out) - first < 2:
            i += 1
            continue
        cands = []
        if i > first and not long_cut(out[i - 1], c) and blen(out[i - 1]) + blen(c) <= max_chars + 4:
            cands.append(i - 1)
        if i + 1 < len(out) and not long_cut(c, out[i + 1]) and blen(out[i + 1]) + blen(c) <= max_chars + 4:
            cands.append(i + 1)
        if not cands:
            i += 1
            continue
        j = min(cands, key=lambda k: blen(out[k]))
        a, b = (out[j], c) if j < i else (c, out[j])
        out[min(i, j)] = Caption(a.start, b.end, f"{a.text} {b.text}")
        del out[max(i, j)]
        i = max(first, min(i, j))
