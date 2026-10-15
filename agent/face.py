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


def product_caption_ys(path: str, times: list[float]) -> list[float]:
    """제품 컷: 자막마다 그 순간 화면에서 제품 윗변 바로 위 (위에서부터 0~1).
    제품이 화면 꼭대기까지 올라오면 PRODUCT_MIN 에 붙인다."""
    tops: list[float | None] = []
    with tempfile.TemporaryDirectory() as d:
        for i, t in enumerate(times):
            f = _grab_frames(path, [t], Path(d) / "", 480)
            fr = Path(d) / f"c{i}.jpg"
            if f:
                f[0].rename(fr)
                tops.append(_product_top(fr))
            else:
                tops.append(None)
    ys = [PRODUCT_Y if top is None else min(max(top - PRODUCT_GAP, PRODUCT_MIN), PRODUCT_MAX) for top in tops]
    # 같은 장면에서 자막이 들썩이지 않도록, 앞 자막과 4% 이내 차이면 그대로 둔다
    for i in range(1, len(ys)):
        if abs(ys[i] - ys[i - 1]) < 0.04:
            ys[i] = ys[i - 1]
    return ys


def fixed_product_y(ys: list[float]) -> float:
    """자막별 '제품 바로 위' 높이들 → 영상 전체 고정 높이.
    중앙값을 쓰되 너무 위(상단 UI)로는 가지 않게 한다. 제품 윗부분이 살짝 겹칠 수는 있다."""
    if not ys:
        return PRODUCT_Y
    return min(max(statistics.median(ys), 0.16), PRODUCT_MAX)
