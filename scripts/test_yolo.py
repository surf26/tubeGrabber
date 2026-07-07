"""单独测试 YOLO 检测：相机抓帧或读取图片，保存标注结果。"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drivers.camera_driver import CameraDriver, CameraDriverError
from utils.config_loader import PROJECT_ROOT, load_config
from utils.perception_factory import build_detector


def main() -> int:
    parser = argparse.ArgumentParser(description="单独测试 YOLO 检测")
    parser.add_argument(
        "--image",
        type=Path,
        help="读取已有图片测试；不填则连接相机抓一帧",
    )
    parser.add_argument(
        "--refine",
        action="store_true",
        help="使用 vision.refine_conf_threshold 作为置信度阈值",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="弹窗显示检测结果",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="标注图保存路径；默认保存到 data/captures/yolo_时间.png",
    )
    args = parser.parse_args()

    cfg = load_config()
    print("加载 YOLO 模型 ...")
    detector = build_detector(cfg, refine=args.refine)

    cam: CameraDriver | None = None
    try:
        if args.image is not None:
            image = cv2.imread(str(args.image))
            if image is None:
                print(f"失败: 无法读取图片 {args.image}")
                return 1
            source_name = str(args.image)
        else:
            cam_cfg = cfg["camera"]
            cam = CameraDriver(
                serial=cam_cfg["serial"],
                width=cam_cfg["width"],
                height=cam_cfg["height"],
                fps=cam_cfg.get("fps", 30),
            )
            print(f"连接相机 serial={cam_cfg['serial']} ...")
            cam.connect()
            frame = cam.capture()
            image = frame.color
            source_name = f"camera:{cam_cfg['serial']}"

        print(f"检测来源: {source_name}")
        detections = detector.detect(image)
        print(f"检测数量: {len(detections)}")
        for i, det in enumerate(detections, start=1):
            x1, y1, x2, y2 = det.bbox
            u, v = det.center_uv
            print(
                f"{i:02d} {det.class_name:<5} conf={det.confidence:.3f} "
                f"bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}) "
                f"center=({u:.1f},{v:.1f})"
            )

        annotated = detector.draw(image, detections)
        out_path = _output_path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), annotated)
        print(f"已保存标注图: {out_path}")

        if args.show:
            cv2.imshow("YOLO detections", annotated)
            print("按任意键关闭窗口")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return 0
    except CameraDriverError as exc:
        print(f"失败: {exc}")
        return 1
    finally:
        if cam is not None:
            cam.disconnect()
            print("已断开相机")


def _output_path(path: Path | None) -> Path:
    if path is not None:
        return path
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "data" / "captures" / f"yolo_{stamp}.png"


if __name__ == "__main__":
    raise SystemExit(main())
