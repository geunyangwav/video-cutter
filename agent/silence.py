"""무음 구간 감지 (적응형 RMS) → 유지 구간 계산.

ffmpeg silencedetect 는 고정 dB 임계값이라 방 소음/숨소리/클릭에 약하다.
여기서는 영상별로 바닥 소음(floor)과 말소리(speech) 레벨을 측정해 임계값을 정하고,
'말소리 레벨'에 충분히 닿지 못한 조각(숨, 딸깍, 부스럭)은 버린다.
"""
from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass

import numpy as np

from .media import ffmpeg_exe

Range = tuple[float, float]
SR = 16000
HOP = 0.02  # 20ms 프레임


@dataclass
class SilenceParams:
    min_silence: float = 0.30   # 이 길이 이상 쉬면 컷
    pad_before: float = 0.04    # 말 시작 전 여유
    pad_after: float = 0.02     # 말 끝난 뒤 여유 (숏츠용 타이트)
    tail_strict: float = 0.8    # 0~1, 클수록 말 끝을 바짝 자름
    sensitivity: float = 0.45   # 0~1, 클수록 작은 소리도 무음 취급 (더 많이 자름)
    min_voiced: float = 0.20    # 조각 안에 말소리가 이만큼은 있어야 유지

    def dict(self):
        return asdict(self)


def load_audio(path: str) -> np.ndarray:
    raw = subprocess.run([ffmpeg_exe(), "-v", "quiet", "-i", str(path), "-vn", "-ac", "1",
                          "-ar", str(SR), "-f", "f32le", "-"], capture_output=True).stdout
    return np.frombuffer(raw, np.float32)


def frame_db(x: np.ndarray) -> np.ndarray:
    w = int(SR * HOP)
    n = len(x) // w
    if n == 0:
        return np.full(1, -90.0)
    rms = np.sqrt((x[: n * w].reshape(n, w) ** 2).mean(1))
    return 20 * np.log10(rms + 1e-9)


def detect_silences(path: str, duration: float, p: SilenceParams,
                    audio: np.ndarray | None = None) -> tuple[list[Range], dict]:
    x = load_audio(path) if audio is None else audio
    db = frame_db(x)
    # 50ms 이동 최대값으로 자음 사이 미세한 골을 메움
    k = 3
    smooth = np.max(np.lib.stride_tricks.sliding_window_view(np.pad(db, k // 2, mode="edge"), k), axis=1)

    floor = float(np.percentile(db, 10))
    speech = float(np.percentile(db, 80))
    span = max(speech - floor, 6.0)
    gate = floor + (1 - p.sensitivity) * 0.6 * span + 4      # 소리 있음 / 없음
    voice = floor + 0.65 * span                               # 확실한 말소리

    sound = smooth > gate
    # 소리 있는 구간 → 런(run)
    runs, start = [], None
    for i, s in enumerate(np.append(sound, False)):
        if s and start is None:
            start = i
        elif not s and start is not None:
            runs.append((start, i))
            start = None
    # 짧은 쉼(min_silence 미만)으로 끊긴 런은 합침
    merged = []
    gap = int(p.min_silence / HOP)
    for a, b in runs:
        if merged and a - merged[-1][1] < gap:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    # 말소리가 부족한 조각(숨/클릭) 제거
    need = int(p.min_voiced / HOP)
    # 말 끝의 잔향/숨은 더 엄격한 기준(tail)으로 잘라낸다
    tail = gate + p.tail_strict * (voice - gate)
    speech_runs = []
    for a, b in merged:
        if int((db[a:b] > voice).sum()) < need:
            continue
        above = np.nonzero(db[a:b] > tail)[0]
        speech_runs.append((a * HOP, (a + above[-1] + 1) * HOP))

    silences, cur = [], 0.0
    for a, b in speech_runs:
        if a - cur > 1e-3:
            silences.append((cur, a))
        cur = b
    if duration - cur > 1e-3:
        silences.append((cur, duration))
    stats = {"floor_db": round(floor, 1), "speech_db": round(speech, 1),
             "gate_db": round(gate, 1), "voice_db": round(voice, 1)}
    return silences, stats


def keep_ranges(duration: float, cuts: list[Range], pad_before: float = 0.0,
                pad_after: float = 0.0, min_keep: float = 0.15) -> list[Range]:
    """전체 구간에서 cuts를 빼고, 각 유지 구간 앞뒤로 여유를 준다."""
    cuts = merge(sorted(cuts))
    keeps: list[Range] = []
    cur = 0.0
    for s, e in cuts:
        if s > cur:
            keeps.append((cur, s))
        cur = max(cur, e)
    if cur < duration:
        keeps.append((cur, duration))
    padded = [(max(0.0, s - pad_before), min(duration, e + pad_after)) for s, e in keeps]
    return [r for r in merge(padded) if r[1] - r[0] >= min_keep]


def merge(ranges: list[Range], gap: float = 0.0) -> list[Range]:
    out: list[Range] = []
    for s, e in sorted(ranges):
        if out and s <= out[-1][1] + gap:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def subtract(ranges: list[Range], holes: list[Range]) -> list[Range]:
    """ranges에서 holes 부분을 제거 (사용자가 보존 마킹한 구간을 cuts에서 빼는 용도)."""
    out = []
    for s, e in ranges:
        pieces = [(s, e)]
        for hs, he in holes:
            nxt = []
            for ps, pe in pieces:
                if he <= ps or hs >= pe:
                    nxt.append((ps, pe))
                    continue
                if hs > ps:
                    nxt.append((ps, hs))
                if he < pe:
                    nxt.append((he, pe))
            pieces = nxt
        out.extend(pieces)
    return [r for r in out if r[1] - r[0] > 1e-3]
