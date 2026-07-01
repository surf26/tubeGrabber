"""Phase 2 验收：连接相机、抓帧并保存到 data/captures/。"""

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drivers.camera_driver import CameraDriver, CameraDriverError
from utils.config_loader import load_config, PROJECT_ROOT


def main() -> int:
    cfg = load_config()["camera"]
    cam = CameraDriver(
        serial=cfg["serial"],
        width=cfg["width"],
        height=cfg["height"],
        fps=cfg.get("fps", 30),
    )

    out_dir = PROJECT_ROOT / "data" / "captures"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"连接相机 serial={cfg['serial']} ...")
    try:
        cam.connect()
        print("连续抓 5 帧（connect 时已 warmup）...")
        frame = None
        for i in range(5):
            frame = cam.capture()
            u, v = cfg["width"] // 2, cfg["height"] // 2
            d = int(frame.depth[v, u])
            print(
                f"  帧 {i + 1}: color={frame.color.shape}, depth={frame.depth.shape}, "
                f"中心深度={d} mm"
            )

        assert frame is not None
        color_path = out_dir / "test_color.png"
        depth_path = out_dir / "test_depth.png"
        cv2.imwrite(str(color_path), frame.color)
        cv2.imwrite(str(depth_path), frame.depth)
        print(f"已保存: {color_path}")
        print(f"已保存: {depth_path}")
    except CameraDriverError as exc:
        print(f"失败: {exc}")
        return 1
    finally:
        cam.disconnect()
        print("已断开")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
