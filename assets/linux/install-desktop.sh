#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Register Nocturne Data Forge in the desktop application menu (per-user).
# Run from inside the unpacked one-dir build:  ./install-desktop.sh
# Undo: rm ~/.local/share/applications/nocturnedataforge.desktop and the icon.
set -euo pipefail
here="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"

apps="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
icondir="${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor/512x512/apps"
mkdir -p "$apps" "$icondir"

if [ -f "$here/NocturneDataForge.png" ]; then
    install -m644 "$here/NocturneDataForge.png" "$icondir/nocturnedataforge.png"
fi

# Resolve Exec/Icon to absolute install paths so the menu entry works anywhere.
sed -e "s|^Exec=.*|Exec=$here/NocturneDataForge %F|" \
    -e "s|^Icon=.*|Icon=nocturnedataforge|" \
    "$here/NocturneDataForge.desktop" > "$apps/nocturnedataforge.desktop"
chmod 644 "$apps/nocturnedataforge.desktop"

update-desktop-database "$apps" 2>/dev/null || true
gtk-update-icon-cache "${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor" 2>/dev/null || true

echo "Installed: 'Nocturne Data Forge' should now appear in your application menu."
