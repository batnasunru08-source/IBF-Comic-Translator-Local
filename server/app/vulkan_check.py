"""Thin wrapper around vulkaninfo for runtime ICD detection.

Used by the 'vulkan' backend to log device info at startup and to fail
fast with a clear message if no Vulkan ICD is available.
"""
from __future__ import annotations

import logging
import subprocess

logger = logging.getLogger(__name__)

_SUMMARY_DEVICE_MARKERS = ("GPU id", "deviceName", "apiVersion")


class VulkanCheckError(RuntimeError):
    """Raised when vulkaninfo cannot be executed or returns an error."""


def get_vulkan_summary() -> str:
    """Run `vulkaninfo --summary` and return its stdout.

    Raises VulkanCheckError if vulkaninfo is missing or exits non-zero.
    """
    try:
        result = subprocess.run(
            ["vulkaninfo", "--summary"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError as exc:
        raise VulkanCheckError(
            "vulkaninfo not found. Install vulkan-tools (apt: vulkan-tools, "
            "or ensure the image's runtime stage includes it)."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise VulkanCheckError("vulkaninfo --summary timed out") from exc

    if result.returncode != 0:
        raise VulkanCheckError(
            f"vulkaninfo --summary failed (code={result.returncode}): "
            f"{result.stderr.strip() or '<no stderr>'}"
        )

    return result.stdout


def has_vulkan_icd() -> bool:
    """Return True if at least one Vulkan device is reported by vulkaninfo.

    Heuristic: vulkaninfo prints "GPU id = N" or "deviceName" / "apiVersion"
    lines when devices are present, and "None" / blank section when not.
    """
    summary = get_vulkan_summary()
    for marker in _SUMMARY_DEVICE_MARKERS:
        if marker in summary:
            return True
    return False


def log_vulkan_devices() -> None:
    """Log a one-line summary of detected Vulkan devices. Never raises."""
    try:
        if has_vulkan_icd():
            logger.info("[VULKAN] ICD available")
        else:
            logger.warning("[VULKAN] No Vulkan devices reported by vulkaninfo")
    except VulkanCheckError as exc:
        logger.warning("[VULKAN] %s", exc)
