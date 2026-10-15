#!/bin/bash
# 더블클릭으로 설치/업데이트. (맥이 막으면 README 의 '경고창이 떠요' 참고)
cd "$(dirname "$0")" && bash ./install.sh
echo; read -n 1 -s -r -p "아무 키나 누르면 이 창을 닫아요..."
