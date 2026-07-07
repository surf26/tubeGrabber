"""采集 YOLO 训练图片，默认保存到 data/captures/data/。"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drivers.camera_driver import CameraDriver, CameraDriverError
from utils.config_loader import PROJECT_ROOT, load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="采集 YOLO 训练图片")
    parser.add_argument(
        "-n",
        "--count",
        type=int,
        default=1,
        help="自动采集张数，默认 1",
    )
    parser.add_argument(
        "-i",
        "--interval",
        type=float,
        default=1.0,
        help="自动采集间隔秒数，默认 1.0",
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="手动模式：每按一次 Enter 拍一张，输入 q 结束",
    )
    parser.add_argument(
        "--prefix",
        default="yolo",
        help="保存文件名前缀，默认 yolo",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "captures" / "data",
        help="输出目录，默认 data/captures/data",
    )
    parser.add_argument(
        "--save-depth",
        action="store_true",
        help="同时保存深度图；YOLO 训练通常只需要 RGB 图片",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="实时显示 RGB 画面；手动模式下窗口按 Space/Enter 保存，q 退出",
    )
    args = parser.parse_args()

    if args.count < 1:
        print("失败: --count 必须 >= 1")
        return 1
    if args.interval < 0:
        print("失败: --interval 必须 >= 0")
        return 1

    cfg = load_config()["camera"]
    cam = CameraDriver(
        serial=cfg["serial"],
        width=cfg["width"],
        height=cfg["height"],
        fps=cfg.get("fps", 30),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"输出目录: {args.out_dir}")
    print(f"连接相机 serial={cfg['serial']} ...")
    try:
        cam.connect()
        if args.manual:
            return _capture_manual(cam, args)
        return _capture_auto(cam, args)
    except CameraDriverError as exc:
        print(f"失败: {exc}")
        return 1
    finally:
        cam.disconnect()
        if args.show:
            cv2.destroyAllWindows()
        print("已断开相机")


def _capture_auto(cam: CameraDriver, args: argparse.Namespace) -> int:
    if args.show:
        return _capture_auto_live(cam, args)

    for index in range(1, args.count + 1):
        _capture_one(cam, args, index)
        if index < args.count and args.interval > 0:
            time.sleep(args.interval)
    print(f"采集完成: {args.count} 张")
    return 0


def _capture_auto_live(cam: CameraDriver, args: argparse.Namespace) -> int:
    print("实时预览自动采集: q 退出")
    index = 1
    next_capture_at = time.monotonic()
    last_frame = None
    while index <= args.count:
        frame = cam.capture()
        last_frame = frame
        _show_frame(frame.color, f"auto {index}/{args.count}")

        now = time.monotonic()
        if now >= next_capture_at:
            _save_frame(frame, args, index)
            index += 1
            next_capture_at = now + args.interval

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            print(f"已停止: {index - 1} 张")
            return 130

    if last_frame is not None:
        _show_frame(last_frame.color, "done")
        cv2.waitKey(1)
    print(f"采集完成: {args.count} 张")
    return 0


def _capture_manual(cam: CameraDriver, args: argparse.Namespace) -> int:
    if args.show:
        return _capture_manual_live(cam, args)

    index = 1
    print("手动采集: 按 Enter 拍照，输入 q 后 Enter 结束")
    while True:
        try:
            text = input(f"[{index}] Enter 拍照 / q 结束: ").strip().lower()
        except KeyboardInterrupt:
            print("\n已取消")
            return 130
        if text in {"q", "quit", "exit"}:
            print(f"采集完成: {index - 1} 张")
            return 0
        _capture_one(cam, args, index)
        index += 1


def _capture_manual_live(cam: CameraDriver, args: argparse.Namespace) -> int:
    index = 1
    print("实时预览手动采集: 窗口按 Space/Enter 保存，q 或 Esc 结束")
    while True:
        frame = cam.capture()
        _show_frame(frame.color, f"manual saved={index - 1}")
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            print(f"采集完成: {index - 1} 张")
            return 0
        if key in (ord(" "), 13, 10):
            _save_frame(frame, args, index)
            index += 1


def _capture_one(cam: CameraDriver, args: argparse.Namespace, index: int) -> None:
    frame = cam.capture()
    _save_frame(frame, args, index)


def _save_frame(frame, args: argparse.Namespace, index: int) -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    color_path = args.out_dir / f"{args.prefix}_{stamp}_{index:04d}.jpg"
    cv2.imwrite(str(color_path), frame.color)

    message = f"已保存: {color_path}"
    if args.save_depth:
        depth_path = args.out_dir / f"{args.prefix}_{stamp}_{index:04d}_depth.png"
        cv2.imwrite(str(depth_path), frame.depth)
        message += f" | depth: {depth_path}"
    print(message)


def _show_frame(color, status: str) -> None:
    canvas = color.copy()
    cv2.putText(
        canvas,
        f"{status} | Space/Enter: save | q: quit",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.imshow("capture_yolo_data", canvas)


if __name__ == "__main__":
    raise SystemExit(main())
