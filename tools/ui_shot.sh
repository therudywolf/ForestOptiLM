#!/usr/bin/env bash
# Headless MD3 UI screenshot via WSL Xvfb — never touches the real screen.
#
#   tools/ui_shot.sh <screen> <out.png>      # screen: archive|workspace|components
#
# Requires (one-time, in WSL): python3-tk xvfb scrot + a venv ~/uienv with
# customtkinter + pillow. The preview imports the repo's md3.py tokens, so a
# screenshot here faithfully reflects the real app's design system.
set -euo pipefail
screen="${1:-workspace}"
out="${2:-.build/ui/${screen}.png}"
size="${3:-1280x860}"
mkdir -p "$(dirname "$out")"
PY="${UIENV_PY:-$HOME/uienv/bin/python}"
xvfb-run -a --server-args="-screen 0 ${size}x24" \
    "$PY" tools/ui_preview.py --screen "$screen" --shot "$out" --size "$size"
echo "wrote $out ($(stat -c%s "$out") bytes)"
