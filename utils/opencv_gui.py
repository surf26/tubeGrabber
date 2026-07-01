"""OpenCV GUI（Qt）环境：在 import cv2 前准备 Qt 字体，避免 qt/fonts 刷屏。"""

from __future__ import annotations

import importlib.util
import os


_SYSTEM_FONT_DIRS = (
    "/usr/share/fonts/truetype/dejavu",
    "/usr/share/fonts/TTF",
    "/usr/share/fonts/truetype/liberation",
    "/usr/share/fonts/truetype/freefont",
)


def _link_system_fonts(fonts_dir: str) -> bool:
    """将系统字体软链到 cv2/qt/fonts（新版 opencv-python 不再自带字体）。"""
    linked = False
    try:
        os.makedirs(fonts_dir, exist_ok=True)
    except OSError:
        return False

    for sys_dir in _SYSTEM_FONT_DIRS:
        if not os.path.isdir(sys_dir):
            continue
        for name in os.listdir(sys_dir):
            if not name.lower().endswith((".ttf", ".otf", ".ttc")):
                continue
            src = os.path.join(sys_dir, name)
            dst = os.path.join(fonts_dir, name)
            if os.path.lexists(dst):
                linked = True
                continue
            try:
                os.symlink(src, dst)
                linked = True
            except OSError:
                continue
    return linked


def _configure_qt_env() -> None:
    spec = importlib.util.find_spec("cv2")
    if not spec or not spec.origin:
        return

    base = os.path.dirname(spec.origin)
    fonts_dir = os.path.join(base, "qt", "fonts")
    plugins_dir = os.path.join(base, "qt", "plugins")

    has_fonts = os.path.isdir(fonts_dir) and any(
        f.lower().endswith((".ttf", ".otf", ".ttc")) for f in os.listdir(fonts_dir)
    )
    if not has_fonts:
        _link_system_fonts(fonts_dir)
        has_fonts = os.path.isdir(fonts_dir) and any(
            f.lower().endswith((".ttf", ".otf", ".ttc")) for f in os.listdir(fonts_dir)
        )

    if has_fonts:
        os.environ["QT_QPA_FONTDIR"] = fonts_dir
    elif os.path.isdir(_SYSTEM_FONT_DIRS[0]):
        # 回退：直接用系统 dejavu 目录
        os.environ["QT_QPA_FONTDIR"] = _SYSTEM_FONT_DIRS[0]

    if os.path.isdir(plugins_dir):
        os.environ.setdefault("QT_PLUGIN_PATH", plugins_dir)

    os.environ.setdefault(
        "QT_LOGGING_RULES",
        "qt.qpa.fonts.warning=false;qt.qpa.fonts=false",
    )


_configure_qt_env()

import cv2  # noqa: E402

__all__ = ["cv2"]
