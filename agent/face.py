"""Apple Vision 으로 인물 턱 위치를 찾아 자막 높이를 정한다."""
from __future__ import annotations

import statistics
import subprocess
import tempfile
from pathlib import Path

from .media import ffmpeg_exe

CHIN_GAP = 0.062     # 턱 아래로 띄울 거리 (화면 높이 비율)
DEFAULT_Y = 0.61     # 얼굴을 못 찾을 때 자막 중심 (위에서부터 비율)
PRODUCT_Y = 0.29     # 제품 컷 기본 자막 중심 (제품 위쪽)
PRODUCT_GAP = 0.05   # 제품 윗변 위로 띄울 거리
PRODUCT_MIN = 0.12   # 이보다 위로는 안 올림 (상단 UI 자리)
PRODUCT_MAX = 0.30   # 제품 위 자막의 가장 낮은 위치


def _grab_frames(path: str, times: list[float], out: Path, height: int = 720) -> list[Path]:
    files = []
    for i, t in enumerate(times):
        f = out / f"f{i}.jpg"
        subprocess.run([ffmpeg_exe(), "-v", "quiet", "-ss", f"{t:.2f}", "-i", path, "-frames:v", "1",
                        "-vf", f"scale=-2:{height}", "-q:v", "4", "-y", str(f)], check=False)
        if f.exists():
            files.append(f)
    return files


def _chin_y(img: Path) -> float | None:
    """가장 큰 얼굴의 턱 끝 y (위에서부터 0~1). 없으면 None."""
    import Quartz
    import Vision
    from Foundation import NSURL

    url = NSURL.fileURLWithPath_(str(img))
    src = Quartz.CGImageSourceCreateWithURL(url, None)
    if not src:
        return None
    cg = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)
    req = Vision.VNDetectFaceLandmarksRequest.alloc().init()
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg, None)
    ok, _ = handler.performRequests_error_([req], None)
    faces = list(req.results() or []) if ok else []
    if not faces:
        return None
    face = max(faces, key=lambda f: f.boundingBox().size.width * f.boundingBox().size.height)
    bb = face.boundingBox()  # 좌하단 원점, 0~1
    bottom = bb.origin.y
    contour = face.landmarks().faceContour() if face.landmarks() else None
    if contour is not None and contour.pointCount() > 0:
        pts = contour.normalizedPoints()
        ys = [pts[i].y for i in range(contour.pointCount())]
        bottom = bb.origin.y + min(ys) * bb.size.height
    return 1.0 - bottom


def _product_boxes(img: Path) -> list[tuple[float, float, float, float]]:
    """눈에 띄는 물체(제품+손) 상자들. (x0, y0, x1, y1), 왼쪽 위가 원점인 0~1 좌표."""
    import Quartz
    import Vision
    from Foundation import NSURL

    src = Quartz.CGImageSourceCreateWithURL(NSURL.fileURLWithPath_(str(img)), None)
    if not src:
        return []
    cg = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)
    req = Vision.VNGenerateObjectnessBasedSaliencyImageRequest.alloc().init()
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg, None)
    ok, _ = handler.performRequests_error_([req], None)
    res = list(req.results() or []) if ok else []
    out = []
    for o in (list(res[0].salientObjects() or []) if res else []):
        bb = o.boundingBox()
        out.append((bb.origin.x, 1 - bb.origin.y - bb.size.height, bb.origin.x + bb.size.width, 1 - bb.origin.y))
    return out


def _product_top(img: Path) -> float | None:
    """눈에 띄는 물체(제품+손)의 윗변 y (위에서부터 0~1)."""
    import Quartz
    import Vision
    from Foundation import NSURL

    src = Quartz.CGImageSourceCreateWithURL(NSURL.fileURLWithPath_(str(img)), None)
    if not src:
        return None
    cg = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)
    req = Vision.VNGenerateObjectnessBasedSaliencyImageRequest.alloc().init()
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg, None)
    ok, _ = handler.performRequests_error_([req], None)
    res = list(req.results() or []) if ok else []
    objs = list(res[0].salientObjects() or []) if res else []
    if not objs:
        return None
    return 1.0 - max(o.boundingBox().origin.y + o.boundingBox().size.height for o in objs)


def caption_y(path: str, keeps: list[tuple[float, float]], samples: int = 7) -> tuple[float, str]:
    """자막 중심 y (위에서부터 0~1)와 배치 방식.

    - 인물이 나오면("face"): 턱 바로 아래
    - 제품만 나오면("product"): 제품을 가리지 않게 제품 윗변 위쪽
    """
    if not keeps:
        return DEFAULT_Y, "default"
    total = sum(e - s for s, e in keeps)
    times, acc, k = [], 0.0, 0
    for i in range(samples):  # 남긴 구간 위에 고르게 샘플
        target = total * (i + 0.5) / samples
        while k < len(keeps) - 1 and acc + (keeps[k][1] - keeps[k][0]) < target:
            acc += keeps[k][1] - keeps[k][0]
            k += 1
        times.append(keeps[k][0] + (target - acc))
    with tempfile.TemporaryDirectory() as d:
        frames = _grab_frames(path, times, Path(d))
        chins = [y for f in frames if (y := _chin_y(f)) is not None]
        if frames and len(chins) >= max(2, len(frames) // 2):
            y = statistics.median(chins) + CHIN_GAP
            return min(max(y, 0.35), 0.88), "face"
        tops = [y for f in frames if (y := _product_top(f)) is not None]
    if not tops:
        return PRODUCT_Y, "product"
    # 제품 윗변보다 조금 위. 너무 위(상태바/제목 자리)나 화면 중앙 아래로는 가지 않게
    y = statistics.median(tops) - PRODUCT_GAP
    return min(max(y, PRODUCT_MIN), PRODUCT_MAX), "product"


def to_transform_y(center_y: float) -> float:
    """화면 위→아래 0~1 을 CapCut transform.y (반 화면 단위, 위가 +) 로."""
    return 1.0 - 2.0 * center_y


# 제품 영상: 자막은 영상 전체에서 한 위치로 고정하되, 여러 장면의 제품을 가장 덜 가리는 높이를 고른다.
CAPTION_BAND = 0.05        # 자막 한 줄이 차지하는 높이 (화면 비율)
CAPTION_X = (0.15, 0.85)   # 자막이 차지하는 가로 범위 (가운데 정렬)
SAFE_TOP, SAFE_BOTTOM = 0.11, 0.80   # 숏폼 앱 UI 에 가리지 않는 범위
PREFERRED_Y = 0.25         # 비슷하게 안 가린다면 이 높이에 가까운 쪽 (화면 위쪽 1/4)


def product_fixed_y(path: str, times: list[float]) -> float:
    """자막이 나오는 순간들의 화면을 보고, 제품과 겹치는 장면이 가장 적은 높이 (위에서부터 0~1)."""
    frames: list[list[tuple[float, float, float, float]]] = []
    with tempfile.TemporaryDirectory() as d:
        step = max(1, len(times) // 40)          # 많아도 40장면만 본다
        for i, t in enumerate(times[::step]):
            f = _grab_frames(path, [t], Path(d), 480)
            if f:
                fr = Path(d) / f"c{i}.jpg"
                f[0].rename(fr)
                frames.append(_product_boxes(fr))
    frames = [b for b in frames if b]
    if not frames:
        return PRODUCT_Y

    def covered(y: float) -> float:
        """이 높이에 자막을 두면 제품과 겹치는 장면 비율."""
        y0, y1 = y - CAPTION_BAND / 2, y + CAPTION_BAND / 2
        hit = 0
        for boxes in frames:
            if any(bx0 < CAPTION_X[1] and bx1 > CAPTION_X[0] and by0 < y1 and by1 > y0
                   for bx0, by0, bx1, by1 in boxes):
                hit += 1
        return hit / len(frames)

    candidates = [round(SAFE_TOP + i * 0.01, 2) for i in range(int((SAFE_BOTTOM - SAFE_TOP) / 0.01) + 1)]
    return min(candidates, key=lambda y: covered(y) + 0.1 * abs(y - PREFERRED_Y))


def product_cover_rate(path: str, times: list[float], y: float) -> float:
    """확인용: 고른 높이에서 제품과 겹치는 장면 비율."""
    frames = []
    with tempfile.TemporaryDirectory() as d:
        for i, t in enumerate(times):
            f = _grab_frames(path, [t], Path(d), 480)
            if f:
                frames.append(_product_boxes(f[0]))
    y0, y1 = y - CAPTION_BAND / 2, y + CAPTION_BAND / 2
    hits = sum(any(b[0] < CAPTION_X[1] and b[2] > CAPTION_X[0] and b[1] < y1 and b[3] > y0 for b in bx) for bx in frames if bx)
    return hits / max(1, sum(1 for bx in frames if bx))
