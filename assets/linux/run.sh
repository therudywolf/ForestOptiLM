#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Convenience launcher for the Linux/Fedora one-dir build.
# Unpack the archive and run this script (or ./NocturneDataForge directly).
set -euo pipefail
here="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
cd "$here"
exec "./NocturneDataForge" "$@"
