"""OpenCV GUI（Qt）环境：在 import cv2 前设置字体/插件路径，避免 qt/fonts 警告。"""

from __future__ import annotations

import importlib.util
import os


def _configure_qt_env() -> None:
    spec = importlib.util.find_spec("cv2")
    if not spec or not spec.origin:
        return

    base = os.path.dirname(spec.origin)
    fonts = os.path.join(base, "qt", "fonts")
    plugins = os.path.join(base, "qt", "plugins")

    if os.path.isdir(fonts):
        os.environ.setdefault("QT_QPA_FONTDIR", fonts)
    if os.path.isdir(plugins):
        os.environ.setdefault("QT_PLUGIN_PATH", plugins)

    # 降低 Qt 字体相关刷屏（不影响 imshow）
    os.environ.setdefault(
        "QT_LOGGING_RULES",
        "qt.qpa.fonts.warning=false;qt.qpa.fonts=false",
    )


_configure_qt_env()

import cv2  # noqa: E402

__all__ = ["cv2"]
