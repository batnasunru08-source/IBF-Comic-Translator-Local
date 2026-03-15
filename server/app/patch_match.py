from __future__ import annotations

import ctypes
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

__all__ = ["inpaint", "set_random_seed", "set_verbose"]


class CShapeT(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("channels", ctypes.c_int),
    ]


class CMatT(ctypes.Structure):
    _fields_ = [
        ("data_ptr", ctypes.c_void_p),
        ("shape", CShapeT),
        ("dtype", ctypes.c_int),
    ]


_DTYPE_PYMAT_TO_CTYPES = [
    ctypes.c_uint8,
    ctypes.c_int8,
    ctypes.c_uint16,
    ctypes.c_int16,
    ctypes.c_int32,
    ctypes.c_float,
    ctypes.c_double,
]

_DTYPE_NP_TO_PYMAT = {
    "uint8": 0,
    "int8": 1,
    "uint16": 2,
    "int16": 3,
    "int32": 4,
    "float32": 5,
    "float64": 6,
}


def _server_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_library_name() -> str:
    if sys.platform == "win32":
        return "patchmatch_inpaint.dll"
    if sys.platform == "darwin":
        return "macos_libpatchmatch_inpaint.dylib"
    return "libpatchmatch.so"


def _default_opencv_sidecar() -> str | None:
    if sys.platform == "darwin":
        return "macos_libopencv_world.dylib"
    return None


def _library_candidates() -> list[Path]:
    candidates: list[Path] = []
    env_path = os.getenv("COMIC_TRANSLATOR_PATCHMATCH_LIB", "").strip()
    if env_path:
        candidates.append(Path(env_path))

    libs_dir = _server_root() / "data" / "libs"
    candidates.append(libs_dir / _default_library_name())
    return candidates


@lru_cache(maxsize=1)
def _load_library():
    opencv_sidecar = _default_opencv_sidecar()
    libs_dir = _server_root() / "data" / "libs"
    if opencv_sidecar:
        sidecar_path = libs_dir / opencv_sidecar
        if sidecar_path.exists():
            ctypes.CDLL(str(sidecar_path))

    for candidate in _library_candidates():
        if candidate.exists():
            library = ctypes.CDLL(str(candidate))
            library.PM_set_random_seed.argtypes = [ctypes.c_uint]
            library.PM_set_verbose.argtypes = [ctypes.c_int]
            library.PM_free_pymat.argtypes = [CMatT]
            library.PM_inpaint.argtypes = [CMatT, CMatT, ctypes.c_int]
            library.PM_inpaint.restype = CMatT
            library.PM_inpaint2.argtypes = [CMatT, CMatT, CMatT, ctypes.c_int]
            library.PM_inpaint2.restype = CMatT
            library.PM_inpaint_regularity.argtypes = [
                CMatT,
                CMatT,
                CMatT,
                ctypes.c_int,
                ctypes.c_float,
            ]
            library.PM_inpaint_regularity.restype = CMatT
            library.PM_inpaint2_regularity.argtypes = [
                CMatT,
                CMatT,
                CMatT,
                CMatT,
                ctypes.c_int,
                ctypes.c_float,
            ]
            library.PM_inpaint2_regularity.restype = CMatT
            return library

    searched = ", ".join(str(path) for path in _library_candidates())
    raise RuntimeError(
        "PatchMatch library not found. "
        f"Looked for: {searched}. "
        "Install it with `bash server/install-patchmatch.sh` or set "
        "`COMIC_TRANSLATOR_PATCHMATCH_LIB`."
    )


def _canonize_mask_array(mask):
    if isinstance(mask, Image.Image):
        mask = np.array(mask)
    if mask.ndim == 2 and mask.dtype == "uint8":
        mask = mask[..., np.newaxis]
    assert mask.ndim == 3 and mask.shape[2] == 1 and mask.dtype == "uint8"
    return np.ascontiguousarray(mask)


def _np_to_pymat(npmat: np.ndarray) -> CMatT:
    assert npmat.ndim == 3
    return CMatT(
        ctypes.cast(npmat.ctypes.data, ctypes.c_void_p),
        CShapeT(npmat.shape[1], npmat.shape[0], npmat.shape[2]),
        _DTYPE_NP_TO_PYMAT[str(npmat.dtype)],
    )


def _pymat_to_np(pymat: CMatT) -> np.ndarray:
    npmat = np.ctypeslib.as_array(
        ctypes.cast(
            pymat.data_ptr,
            ctypes.POINTER(_DTYPE_PYMAT_TO_CTYPES[pymat.dtype]),
        ),
        (pymat.shape.height, pymat.shape.width, pymat.shape.channels),
    )
    result = np.empty(npmat.shape, npmat.dtype)
    result[:] = npmat
    return result


def set_random_seed(seed: int):
    _load_library().PM_set_random_seed(ctypes.c_uint(seed))


def set_verbose(verbose: bool):
    _load_library().PM_set_verbose(ctypes.c_int(bool(verbose)))


def inpaint(
    image: np.ndarray | Image.Image,
    mask: Optional[np.ndarray | Image.Image] = None,
    *,
    global_mask: Optional[np.ndarray | Image.Image] = None,
    patch_size: int = 15,
) -> np.ndarray:
    if isinstance(image, Image.Image):
        image = np.array(image)
    image = np.ascontiguousarray(image)
    assert image.ndim == 3 and image.shape[2] == 3 and image.dtype == "uint8"

    if mask is None:
        mask = (image == (255, 255, 255)).all(axis=2, keepdims=True).astype("uint8")
        mask = np.ascontiguousarray(mask)
    else:
        mask = _canonize_mask_array(mask)

    library = _load_library()
    if global_mask is None:
        ret_pymat = library.PM_inpaint(
            _np_to_pymat(image),
            _np_to_pymat(mask),
            ctypes.c_int(patch_size),
        )
    else:
        global_mask = _canonize_mask_array(global_mask)
        ret_pymat = library.PM_inpaint2(
            _np_to_pymat(image),
            _np_to_pymat(mask),
            _np_to_pymat(global_mask),
            ctypes.c_int(patch_size),
        )

    result = _pymat_to_np(ret_pymat)
    library.PM_free_pymat(ret_pymat)
    return result
