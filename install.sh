#!/bin/bash
# Video Cutter 설치 스크립트 — 캡컷 컷편집 자동화 도구 (made by @geunyang_wav)
#
#   터미널에 붙여넣기:
#   curl -fsSL https://raw.githubusercontent.com/geunyangwav/video-cutter/main/install.sh | bash
#
# 하는 일: ~/video-cutter 에 프로그램을 받고, 필요한 도구를 설치하고, 바탕화면에 Video Cutter 아이콘을 만듭니다.
# 다시 실행하면 최신 버전으로 업데이트돼요. (만든 CapCut 프로젝트는 그대로)
set -euo pipefail

REPO="geunyangwav/video-cutter"
DEST="${EFFECTING_HOME:-$HOME/video-cutter}"
APP="${EFFECTING_APP:-$HOME/Desktop/Video Cutter.app}"
PORT=8765
DRAFTS="$HOME/Movies/CapCut/User Data/Projects/com.lveditor.draft"

step() { printf "\n\033[1m● %s\033[0m\n" "$*"; }
note() { printf "  %s\n" "$*"; }

printf "\n\033[1mVideo Cutter 설치\033[0m · 캡컷 컷편집 자동화 도구\n"

# 0. 이 맥에서 쓸 수 있는지
if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
  echo "죄송해요. Video Cutter 는 M칩(Apple Silicon) 맥에서만 쓸 수 있어요."; exit 1
fi
if [ ! -d "$DRAFTS" ]; then
  note "⚠️  CapCut 프로젝트 폴더가 아직 없어요. CapCut 데스크톱 앱을 설치하고 한 번 실행해 주세요."
  note "    (설치는 계속 진행할게요)"
fi

# 1. 프로그램 받기
step "프로그램 받는 중"
if [ -n "${EFFECTING_SRC:-}" ]; then
  SRC="$EFFECTING_SRC"                         # 로컬 폴더에서 설치 (테스트용)
elif [ -f "$(dirname "${BASH_SOURCE[0]:-$0}")/server.py" ]; then
  SRC="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"   # 받은 폴더에서 '설치하기'를 더블클릭한 경우
else
  TMP="$(mktemp -d)"
  curl -fsSL "https://codeload.github.com/$REPO/zip/refs/heads/main" -o "$TMP/video-cutter.zip"
  unzip -q "$TMP/video-cutter.zip" -d "$TMP"
  SRC="$TMP/video-cutter-main"
fi
mkdir -p "$DEST"
if [ "$SRC" != "$DEST" ]; then
  rsync -a --exclude .venv --exclude uploads --exclude cache --exclude .git --exclude server.log --exclude launcher.applescript "$SRC/" "$DEST/"
fi
note "설치 위치: $DEST"

# 2. uv (파이썬 관리 도구) — 관리자 비밀번호 필요 없음
step "도구 준비 중"
UV="$(command -v uv || true)"
if [ -z "$UV" ] && [ -x "$HOME/.local/bin/uv" ]; then UV="$HOME/.local/bin/uv"; fi
if [ -z "$UV" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
  UV="$HOME/.local/bin/uv"
fi

# 3. 파이썬 + 패키지
step "필요한 패키지 설치 중 (처음엔 몇 분 걸려요)"
cd "$DEST"
[ -x .venv/bin/python ] || "$UV" venv --python 3.11 .venv -q
"$UV" pip install --python .venv/bin/python -r requirements.txt -q

# 4. 음성 인식 모델 미리 받기 (약 1.6GB, 한 번만)
step "음성 인식 모델 받는 중 (약 1.6GB · 한 번만)"
.venv/bin/python -c "from huggingface_hub import snapshot_download as d; d('mlx-community/whisper-large-v3-turbo')" 2>/dev/null \
  || note "모델은 처음 영상을 넣을 때 받을게요."

# 5. 바탕화면 아이콘 (더블클릭하면 서버를 켜고 브라우저를 엽니다)
step "바탕화면에 Video Cutter 아이콘 만드는 중"
lsof -ti tcp:$PORT 2>/dev/null | xargs kill 2>/dev/null || true   # 예전 버전이 켜져 있으면 끔
SCRIPT="$(mktemp -d)/launcher.applescript"
cat > "$SCRIPT" <<APPLESCRIPT
-- Video Cutter 실행기: 서버가 꺼져 있으면 조용히 켜고, 브라우저로 엽니다.
set appDir to "$DEST"
set pageURL to "http://localhost:$PORT"
set httpCode to do shell script "curl -s -o /dev/null -w '%{http_code}' " & pageURL & " || true"
if httpCode is not "200" then
	do shell script "(cd " & quoted form of appDir & " && exec .venv/bin/python server.py) > " & quoted form of (appDir & "/server.log") & " 2>&1 < /dev/null &"
	repeat 80 times
		delay 0.25
		set httpCode to do shell script "curl -s -o /dev/null -w '%{http_code}' " & pageURL & " || true"
		if httpCode is "200" then exit repeat
	end repeat
end if
open location pageURL
APPLESCRIPT
rm -rf "$APP"
# 예전 이름(effecting) 아이콘이 있으면 정리
OLD="$HOME/Desktop/effecting.app"
if [ -z "${EFFECTING_APP:-}" ] && [ -d "$OLD" ] && osadecompile "$OLD/Contents/Resources/Scripts/main.scpt" 2>/dev/null | grep -q "localhost:$PORT"; then rm -rf "$OLD"; fi
osacompile -o "$APP" "$SCRIPT"
cp assets/effecting.icns "$APP/Contents/Resources/applet.icns"
rm -f "$APP/Contents/Resources/Assets.car"
/usr/libexec/PlistBuddy -c "Delete :CFBundleIconName" "$APP/Contents/Info.plist" 2>/dev/null || true
/usr/libexec/PlistBuddy -c "Set :CFBundleIconFile applet" "$APP/Contents/Info.plist"
codesign --force --deep -s - "$APP" 2>/dev/null || true
touch "$APP"

printf "\n\033[1m✓ 설치 끝!\033[0m 바탕화면의 \033[1mVideo Cutter\033[0m 아이콘을 더블클릭하세요.\n"
note "Dock 에 끌어다 두면 편해요."
[ -n "${EFFECTING_NO_OPEN:-}" ] || open "$APP"
