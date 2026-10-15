"""잔말(filler) / 조사 다듬기 / NG(다시 말하기) 찾기.

- 확실한 잔말(음, 어, 으…)과 쉼에 걸친 '그/저' → 자동 컷
- '생각을 합니다' → '생각합니다' : '을/를' 한 음절을 오디오에서 잘라냄 (자동)
- '아' 같은 애매한 말과 반복(NG)은 '제안'으로만 표시 → 사용자가 보고 고름
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import numpy as np

from .asr import Transcript, Word
from .silence import HOP, frame_db

SURE = {"음", "어", "으", "흠", "엄", "어어", "음음", "으음", "어음"}
PAUSED = {"그", "저", "그그", "저저"}          # 쉼과 함께 나오면 잔말 (그 사람 ≠ 그… 사람)
MAYBE = {"아", "에", "이제", "뭐지", "그니까", "약간"}
HADA = ("하", "합", "한", "할", "해", "했", "함")  # '하다' 활용 시작 음절
_CLEAN = re.compile(r"[\s.,!?…~\-]+")


@dataclass
class Mark:
    start: float
    end: float
    text: str
    kind: str      # "filler" | "particle" | "maybe" | "ng"
    auto: bool     # True 면 자동으로 잘림, False 면 제안만

    def dict(self):
        return asdict(self)


def _norm(w: Word) -> str:
    return _CLEAN.sub("", w.text)


def _snap_to_dip(db: np.ndarray | None, t: float, radius: float = 0.07) -> float:
    """t 근처(±radius)에서 소리가 가장 작은 프레임으로 경계를 옮긴다 (음절 사이 골)."""
    if db is None:
        return t
    a, b = max(0, int((t - radius) / HOP)), min(len(db), int((t + radius) / HOP) + 1)
    if b - a < 2:
        return t
    return (a + int(np.argmin(db[a:b]))) * HOP


def _nuclei(audio: np.ndarray, t0: float, t1: float) -> tuple[list[float], np.ndarray, float]:
    """t0~t1 구간의 음절 핵(10ms 에너지 봉우리) 시각들."""
    sr, hop = 16000, 160
    a, b = max(0, int(t0 * sr)), min(len(audio), int(t1 * sr))
    x = audio[a:b]
    n = len(x) // hop
    if n < 5:
        return [], np.zeros(0), t0
    e = 20 * np.log10(np.sqrt((x[: n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-9)
    sm = np.convolve(e, np.ones(3) / 3, mode="same")
    top = sm.max()
    peaks = []
    for k in range(1, n - 1):
        if sm[k] >= sm[k - 1] and sm[k] > sm[k + 1] and sm[k] > top - 18:
            lo_l = sm[max(0, k - 12):k].min() if k else sm[k]
            lo_r = sm[k + 1:k + 13].min()
            if sm[k] - max(lo_l, lo_r) >= 5:  # 주변보다 5dB 이상 솟은 봉우리
                if peaks and (k - peaks[-1]) * 0.01 < 0.07:
                    if sm[k] > sm[peaks[-1]]:
                        peaks[-1] = k
                    continue
                peaks.append(k)
    return [t0 + k * 0.01 for k in peaks], sm, t0


def _particle_cut(audio: np.ndarray, w: Word, nxt: Word, n_syl: int) -> tuple[float, float] | None:
    t0, t1 = w.start - 0.03, nxt.start + 0.35
    peaks, sm, base = _nuclei(audio, t0, t1)
    if len(peaks) < n_syl + 1:
        return None
    p, q = peaks[n_syl - 1], peaks[n_syl]          # '을' 봉우리, 다음 단어 첫 봉우리
    prev = peaks[n_syl - 2] if n_syl >= 2 else t0
    if not (w.end - 0.15 <= p <= w.end + 0.25):  # whisper 와 너무 다르면 포기
        return None
    idx = lambda t: int(round((t - base) / 0.01))
    dip_before = base + (idx(prev) + int(np.argmin(sm[idx(prev):idx(p) + 1]))) * 0.01
    dip_after = base + (idx(p) + int(np.argmin(sm[idx(p):idx(q) + 1]))) * 0.01
    # 받침(ㅇ·ㄴ 등)을 살리려고 앞 경계는 골과 봉우리의 중간에서 시작
    s = (dip_before + p) / 2 - 0.02
    e = dip_after
    return (s, e) if 0.05 < e - s < 0.3 else None


def find_marks(t: Transcript, audio: np.ndarray | None = None,
               keeps: list[tuple[float, float]] = ()) -> list[Mark]:  # noqa: C901
    """keeps: 무음 컷 후 남은 구간. 받아쓰기에 단어가 없는 짧은 소리 덩어리는 '음/어'로 본다
    (whisper 는 음·어를 자주 받아쓰지 않고 건너뛴다)."""
    words = t.words
    db = frame_db(audio) if audio is not None else None
    marks: list[Mark] = []

    def gap_before(i): return words[i].start - words[i - 1].end if i else 1.0
    def gap_after(i): return words[i + 1].start - words[i].end if i + 1 < len(words) else 1.0

    for i, w in enumerate(words):
        n = _norm(w)
        if not n or w.end - w.start > 1.2:
            continue
        if n in SURE or re.fullmatch(r"(음|어|으)+", n):
            marks.append(Mark(w.start, w.end, n, "filler", True))
        elif n in PAUSED and (gap_before(i) > 0.1 or gap_after(i) > 0.1
                               or (i + 1 < len(words) and _norm(words[i + 1]) in SURE | PAUSED)):
            marks.append(Mark(w.start, w.end, n, "filler", True))
        elif n in MAYBE and gap_before(i) > 0.2 and gap_after(i) > 0.2:
            marks.append(Mark(w.start, w.end, n, "maybe", False))
        elif (len(n) >= 2 and n[-1] in "을를" and i + 1 < len(words)
              and _norm(words[i + 1]).startswith(HADA)):
            # '생각을 합니다' → 마지막 음절 '을'만 잘라 '생각합니다'
            # whisper 단어 시각은 0.1~0.2s 어긋날 수 있다 → 오디오에서 음절 핵(모음 봉우리)을 세어
            # n번째 봉우리('을')만 잘라낸다. 애매하면 자르지 않는다.
            cut = _particle_cut(audio, w, words[i + 1], len(n)) if audio is not None else None
            if cut:
                cut_s, cut_e = cut
                marks.append(Mark(cut_s, cut_e, f"{n} {_norm(words[i + 1])} → {n[:-1]}{_norm(words[i + 1])}",
                                  "particle", True))

    for ks, ke in keeps:
        if ke - ks > 0.7:
            continue
        covered = sum(max(0.0, min(ke, w.end) - max(ks, w.start)) for w in words)
        if covered < 0.3 * (ke - ks):
            marks.append(Mark(ks, ke, "음·어 (소리)", "filler", True))

    # whisper 의 잔말 시각은 0.5s 가까이 어긋날 수 있다. 잔말 컷이 짧은 소리 덩어리(0.8s 이하)에
    # 걸치면 덩어리 전체를 잘라, '어'의 끝자락 같은 짧은 조각(딸깍 소리)이 남지 않게 한다.
    for m in marks:
        if m.kind != "filler":
            continue
        for ks, ke in keeps:
            if ke - ks <= 0.8 and m.start < ke and m.end > ks:
                m.start, m.end = min(m.start, ks), max(m.end, ke)

    # NG: 같은 2~6 단어 묶음이 바로 이어서 반복 → 앞쪽 시도를 제안
    norms = [_norm(w) for w in words]
    i = 0
    while i < len(words):
        for n in range(6, 1, -1):
            a, b = norms[i:i + n], norms[i + n:i + 2 * n]
            if len(b) == n and a == b and all(a):
                marks.append(Mark(words[i].start, words[i + n].start,
                                  " ".join(w.text.strip() for w in words[i:i + n]), "ng", False))
                i += n
                break
        else:
            i += 1
    return sorted(marks, key=lambda m: m.start)


def apply_text_edits(t: Transcript, marks: list[Mark]) -> None:
    """조사를 잘랐으면 자막 글자도 '생각합니다' 로 맞춘다 (in-place)."""
    words = t.words
    for m in marks:
        if m.kind != "particle":
            continue
        for i, w in enumerate(words[:-1]):
            if w.start < m.start < w.end + 0.01:
                w.text = re.sub(r"[을를]([\s.,!?…]*)$", r"\1", w.text.rstrip())
                words[i + 1].text = words[i + 1].text.lstrip()
                break
