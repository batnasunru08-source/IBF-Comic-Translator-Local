#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ "${OSTYPE:-}" == msys* || "${OSTYPE:-}" == cygwin* ]]; then
  echo "install-patchmatch.sh is intended for Linux or WSL."
  exit 1
fi

for cmd in git make pkg-config; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing required command: $cmd"
    echo "Install build tools first, for example:"
    echo "  sudo apt install -y build-essential git pkg-config libopencv-dev"
    exit 1
  fi
done

if ! pkg-config --exists opencv4 && ! pkg-config --exists opencv; then
  echo "OpenCV development files were not found via pkg-config."
  echo "Install them first, for example:"
  echo "  sudo apt install -y libopencv-dev"
  exit 1
fi

OPENCV_PKG="opencv"
if pkg-config --exists opencv4 && ! pkg-config --exists opencv; then
  OPENCV_PKG="opencv4"
fi

LIBS_DIR="$(pwd)/data/libs"
BUILD_DIR="$(pwd)/.cache/patchmatch-build"

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR" "$LIBS_DIR"

git clone --depth 1 https://github.com/vacancy/PyPatchMatch "$BUILD_DIR/PyPatchMatch"

if [[ "$OPENCV_PKG" == "opencv4" ]]; then
  python - <<'PY'
from pathlib import Path
import re

makefile = Path(".cache/patchmatch-build/PyPatchMatch/Makefile")
text = makefile.read_text(encoding="utf-8")
patched = re.sub(r'(?<![A-Za-z0-9_])opencv(?![A-Za-z0-9_])', 'opencv4', text)

if patched != text:
    makefile.write_text(patched, encoding="utf-8")
    print("[PatchMatch] patched Makefile to use pkg-config opencv4")
else:
    print("[PatchMatch] Makefile did not need patching")
PY
fi

pushd "$BUILD_DIR/PyPatchMatch" >/dev/null
make -j"$(nproc)"
popd >/dev/null

cp "$BUILD_DIR/PyPatchMatch/libpatchmatch.so" "$LIBS_DIR/libpatchmatch.so"

echo
echo "PatchMatch installed:"
echo "  $LIBS_DIR/libpatchmatch.so"
echo
echo "To enable it for the server:"
echo "  export COMIC_TRANSLATOR_INPAINT_BACKEND=patchmatch"
