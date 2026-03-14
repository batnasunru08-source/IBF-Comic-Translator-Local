#!/usr/bin/env bash
set -euo pipefail

# Compatibility wrapper for older docs and WSL-specific commands.
# The main installer is install-linux-gpu.sh.
cd "$(dirname "$0")"
exec bash ./install-linux-gpu.sh
