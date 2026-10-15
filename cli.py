"""1단 검증용 CLI: 무음 컷 → effecting_N 드래프트.

    .venv/bin/python cli.py <영상> [--min-silence 0.3] [--sensitivity 0.45] [--dry]
"""
import argparse

from agent.draft import build_draft
from agent.media import probe
from agent.silence import SilenceParams, detect_silences, keep_ranges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--min-silence", type=float, default=0.30)
    ap.add_argument("--sensitivity", type=float, default=0.45)
    ap.add_argument("--dry", action="store_true", help="드래프트 생성 없이 컷만 출력")
    a = ap.parse_args()

    p = SilenceParams(min_silence=a.min_silence, sensitivity=a.sensitivity)
    info = probe(a.video)
    sil, st = detect_silences(a.video, info.duration, p)
    keeps = keep_ranges(info.duration, sil, p.pad_before, p.pad_after)
    kept = sum(e - s for s, e in keeps)
    print(f"원본 {info.duration:.2f}s · {info.width}x{info.height} · 레벨 {st}")
    for s, e in keeps:
        print(f"  keep {s:7.2f} → {e:7.2f}  ({e - s:.2f}s)")
    print(f"유지 {kept:.2f}s · {info.duration - kept:.2f}s 단축 · 클립 {len(keeps)}개")
    if not a.dry:
        r = build_draft(info, keeps)
        print(f"→ {r['name']}  {r['path']}")


if __name__ == "__main__":
    main()
