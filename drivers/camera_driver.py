"""奥比中光相机驱动封装（pyorbbecsdk2，import 名为 pyorbbecsdk）。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from pyorbbecsdk import (
    AlignFilter,
    Config,
    Context,
    OBFormat,
    OBFrameAggregateOutputMode,
    OBStreamType,
    OBSensorType,
    Pipeline,
    VideoFrame,
)

from utils.config_loader import load_yaml


class CameraDriverError(RuntimeError):
    """相机通信或取帧异常。"""


@dataclass
class Frame:
    color: np.ndarray  # H×W×3, BGR, uint8
    depth: np.ndarray  # H×W, uint16, 单位 mm（0 表示无效）
    timestamp: float


class CameraDriver:
    def __init__(
        self,
        serial: str,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        *,
        warmup_frames: int = 10,
        frame_timeout_ms: int = 1000,
    ) -> None:
        self._serial = serial
        self._width = width
        self._height = height
        self._fps = fps
        self._warmup_frames = warmup_frames
        self._frame_timeout_ms = frame_timeout_ms

        self._pipeline: Pipeline | None = None
        self._align_filter: AlignFilter | None = None

    def connect(self) -> None:
        """按 serial 打开设备，失败抛出 CameraDriverError。"""
        if self._pipeline is not None:
            return

        target = (
            f"serial={self._serial}, {self._width}x{self._height}@{self._fps}fps"
        )
        try:
            ctx = Context()
            device_list = ctx.query_devices()
            count = device_list.get_count()
            if count == 0:
                raise CameraDriverError(
                    f"未发现 Orbbec 设备 ({target})，请检查 USB 连接与权限"
                )

            device = device_list.get_device_by_serial_number(self._serial)
            if device is None:
                raise CameraDriverError(
                    f"找不到 serial={self._serial!r} ({target})，"
                    "请核对 config 中 camera.serial"
                )

            pipeline = Pipeline(device)
            config = Config()

            color_profile = _pick_video_profile(
                pipeline,
                OBSensorType.COLOR_SENSOR,
                self._width,
                self._height,
                OBFormat.RGB,
                self._fps,
            )
            depth_profile = _pick_video_profile(
                pipeline,
                OBSensorType.DEPTH_SENSOR,
                self._width,
                self._height,
                OBFormat.Y16,
                self._fps,
            )
            config.enable_stream(color_profile)
            config.enable_stream(depth_profile)
            config.set_frame_aggregate_output_mode(
                OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE
            )
            self._align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

            pipeline.start(config)
            self._pipeline = pipeline

            for _ in range(self._warmup_frames):
                self.capture()
        except CameraDriverError:
            self.disconnect()
            raise
        except Exception as exc:
            self.disconnect()
            raise CameraDriverError(
                f"相机连接异常 ({target}): {type(exc).__name__}: {exc}"
            ) from exc

    def disconnect(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
        self._pipeline = None
        self._align_filter = None

    def is_connected(self) -> bool:
        return self._pipeline is not None

    def capture(self) -> Frame:
        """取一帧对齐后的 BGR 彩色图与深度图（mm）。"""
        pipeline = self._require_pipeline()

        frames = pipeline.wait_for_frames(self._frame_timeout_ms)
        if frames is None:
            raise CameraDriverError("wait_for_frames 超时")

        if self._align_filter is None:
            raise CameraDriverError("AlignFilter 未初始化")

        aligned = self._align_filter.process(frames)
        if aligned is None:
            raise CameraDriverError("AlignFilter 处理失败")
        frames = aligned.as_frame_set()

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if color_frame is None or depth_frame is None:
            raise CameraDriverError("color 或 depth 帧为空")

        color = _frame_to_bgr_image(color_frame)
        if color is None:
            raise CameraDriverError(f"不支持的彩色格式: {color_frame.get_format()}")

        depth = _depth_frame_to_mm(depth_frame)
        if color.shape[:2] != depth.shape[:2]:
            raise CameraDriverError(
                f"彩色与深度尺寸不一致: color={color.shape[:2]}, depth={depth.shape[:2]}"
            )

        return Frame(color=color, depth=depth, timestamp=time.time())

    def get_intrinsics(self) -> dict[str, Any]:
        """读取 config/camera_intrinsics.yaml（Phase 2 先用离线标定）。"""
        return load_yaml("config/camera_intrinsics.yaml")

    def _require_pipeline(self) -> Pipeline:
        if self._pipeline is None:
            raise CameraDriverError("相机未连接，请先调用 connect()")
        return self._pipeline


def _pick_video_profile(
    pipeline: Pipeline,
    sensor_type: OBSensorType,
    width: int,
    height: int,
    fmt: OBFormat,
    fps: int,
):
    """优先精确匹配分辨率，否则用默认 profile。"""
    profile_list = pipeline.get_stream_profile_list(sensor_type)

    try:
        return profile_list.get_video_stream_profile(width, height, fmt, fps)
    except Exception:
        pass

    for i in range(len(profile_list)):
        profile = profile_list[i]
        try:
            if (
                profile.get_width() == width
                and profile.get_height() == height
                and profile.get_format() == fmt
                and (not fps or profile.get_fps() == fps)
            ):
                return profile
        except Exception:
            continue

    return profile_list.get_default_video_stream_profile()


def _depth_frame_to_mm(depth_frame) -> np.ndarray:
    height = depth_frame.get_height()
    width = depth_frame.get_width()
    scale = depth_frame.get_depth_scale()

    depth_raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(height, width)
    depth_mm = (depth_raw.astype(np.float32) * scale).astype(np.uint16)
    return depth_mm


def _frame_to_bgr_image(frame: VideoFrame) -> np.ndarray | None:
    """将 VideoFrame 转为 OpenCV BGR（精简版，覆盖常见格式）。"""
    width = frame.get_width()
    height = frame.get_height()
    fmt = frame.get_format()
    data = np.asanyarray(frame.get_data())

    if fmt == OBFormat.RGB:
        image = data.reshape((height, width, 3))
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if fmt == OBFormat.BGR:
        return data.reshape((height, width, 3)).copy()

    if fmt == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)

    if fmt == OBFormat.YUYV:
        image = data.reshape((height, width, 2))
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)

    if fmt == OBFormat.UYVY:
        image = data.reshape((height, width, 2))
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)

    return None
