"""试管抓放状态机。"""

from __future__ import annotations

import math
import time
import traceback
from enum import Enum
from typing import Any

from drivers.arm_driver import ArmDriver
from drivers.camera_driver import CameraDriver
from drivers.gripper_driver import GripperDriver, GripperDriverError
from perception.coord_transform import load_T_ee_cam, load_intrinsics, pose_6d_to_matrix
from perception.refine import RefineError, refine_pick_slot, refine_place_slot
from perception.slot_mapper import SlotMapper
from perception.yolo_detector import YoloDetector
from planning.command_validator import CommandValidator, MoveCommand
from planning.motion_planner import MotionPlanner, Waypoint, format_waypoints
from utils.cli_output import (
    AUTO_CONT,
    PROMPT_GRASP,
    err,
    move_done,
    move_line,
    ok,
    stage,
    sub,
    warn,
)
from utils.config_loader import load_yaml
from utils.vision_viz import VisionDisplay, draw_refine_annotation, draw_scan_annotation
from world.operation_verifier import OperationVerifier
from world.tube_registry import TubeRegistry, measure_rack_z


def _angle_close(a: float, b: float, tol: float) -> bool:
    """角度就近比较，处理 ±π 环绕（竖直姿态 rx≈π 时 SDK 可能返回 ±π）。"""
    d = (a - b + math.pi) % (2 * math.pi) - math.pi
    return abs(d) <= tol


class FSMError(RuntimeError):
    """状态机错误。"""


class State(Enum):
    INIT = "init"
    CHECK_HW = "check_hw"
    SCAN_GLOBAL = "scan_global"
    WAIT_CMD = "wait_cmd"
    VALIDATE_CMD = "validate_cmd"
    PICK_TRANSIT = "pick_transit"
    PICK_REFINE = "pick_refine"
    PICK_GRASP = "pick_grasp"
    VERIFY_PICK = "verify_pick"
    PLACE_TRANSIT = "place_transit"
    PLACE_REFINE = "place_refine"
    PLACE_RELEASE = "place_release"
    VERIFY_PLACE = "verify_place"
    DONE = "done"
    FAILED = "failed"


class PickPlaceFSM:
    def __init__(
        self,
        arm: ArmDriver,
        camera: CameraDriver,
        gripper: GripperDriver | None,
        detector: YoloDetector,
        refine_detector: YoloDetector,
        mapper: SlotMapper,
        registry: TubeRegistry,
        planner: MotionPlanner,
        validator: CommandValidator,
        config: dict[str, Any],
        *,
        dry_run: bool = False,
        skip_gripper: bool = False,
    ) -> None:
        self._arm = arm
        self._camera = camera
        self._gripper = gripper
        self._detector = detector
        self._refine_detector = refine_detector
        self._mapper = mapper
        self._registry = registry
        self._planner = planner
        self._validator = validator
        self._config = config
        self._dry_run = dry_run
        gripper_enabled = config.get("gripper", {}).get("enabled", True)
        self._skip_gripper = skip_gripper or dry_run or not gripper_enabled

        self._cam_cfg = config["camera"]
        self._arm_cfg = config["arm"]
        self._motion = config.get("motion", {})
        self._registry_cfg = config.get("registry", {})
        self._vision_cfg = config.get("vision", {})
        self._safety_cfg = config.get("safety", {})
        self._continuous_mode = bool(
            config.get("runtime", {}).get("continuous_mode", False)
        )
        self._verifier = OperationVerifier(config.get("verification", {}))

        self._K, self._dist = load_intrinsics(config["calib"]["camera_intrinsics"])
        self._T_ee_cam = load_T_ee_cam(config["calib"]["hand_eye"])
        self._rack_layout = load_yaml(config["calib"]["rack_layout"])
        self._viz = VisionDisplay(config)
        self._font_scale = float(config.get("vision", {}).get("font_scale", 0.28))
        self._refine_update_enabled = bool(
            self._vision_cfg.get("refine_update_enabled", True)
        )

        self._state = State.INIT
        self._cmd: MoveCommand | None = None
        self._fail_reason = ""
        self._last_pose: tuple[float, float, float, float, float, float] | None = None
        self._pick_approach: tuple[float, float, float, float, float, float] | None = None
        self._place_approach: tuple[float, float, float, float, float, float] | None = None
        self._pre_pick_place_base_xyz: tuple[float, float, float] | None = None
        self._need_direct_scan = True

    @property
    def state(self) -> State:
        return self._state

    @property
    def registry(self) -> TubeRegistry:
        return self._registry

    @property
    def fail_reason(self) -> str:
        return self._fail_reason

    def connect_and_scan(self) -> bool:
        """连接硬件并执行首次全局扫描。"""
        return self._bootstrap()

    def scan(self) -> bool:
        """全局扫描（需已 CHECK_HW）。"""
        return self._do_scan_global()

    def shutdown(self) -> None:
        """断开相机与机械臂。"""
        self._viz.close_all()
        try:
            self._camera.disconnect()
        except Exception:
            pass
        if self._arm.is_connected():
            try:
                self._arm.disconnect()
            except Exception:
                pass

    def run_dry_move(self, cmd_text: str) -> bool:
        """空跑：移动 + 精定位，不夹取。"""
        was_dry = self._dry_run
        was_skip = self._skip_gripper
        self._dry_run = True
        self._skip_gripper = True
        try:
            if not self._bootstrap():
                return False
            print(self._registry.to_table_str())
            print()
            return self.execute_move(cmd_text, dry=True)
        finally:
            self._dry_run = was_dry
            self._skip_gripper = was_skip

    def run_once(self, cmd_text: str) -> bool:
        """完整抓放（非 dry-run）。"""
        if not self._bootstrap():
            return False
        return self.execute_move(cmd_text, dry=False)

    def execute_move(self, cmd_text: str, *, dry: bool) -> bool:
        """执行单次搬运（需已 connect / scan）。"""
        was_dry = self._dry_run
        was_skip = self._skip_gripper
        self._dry_run = dry
        if dry:
            self._skip_gripper = True

        try:
            self._cmd, valid, reason = self._validator.parse_and_validate(
                cmd_text, self._registry
            )
            if not valid or self._cmd is None:
                self._fail(f"invalid command: {reason}")
                return False
            dst_state = self._registry.get(self._cmd.dst)
            self._pre_pick_place_base_xyz = dst_state.base_xyz

            label = "DRY_RUN" if dry else "MOVE"
            stage(label, f"{self._cmd.src} -> {self._cmd.dst}")

            if dry:
                sequence = [
                    State.PICK_TRANSIT,
                    State.PICK_REFINE,
                    State.PICK_GRASP,
                    State.SCAN_GLOBAL,
                    State.PLACE_TRANSIT,
                    State.PLACE_REFINE,
                    State.PLACE_RELEASE,
                    State.SCAN_GLOBAL,
                    State.DONE,
                ]
            else:
                sequence = [
                    State.PICK_TRANSIT,
                    State.PICK_REFINE,
                    State.PICK_GRASP,
                    State.VERIFY_PICK,
                    State.PLACE_TRANSIT,
                    State.PLACE_REFINE,
                    State.PLACE_RELEASE,
                    State.VERIFY_PLACE,
                    State.DONE,
                ]

            for st in sequence:
                self._state = st
                if not self.step():
                    return False
            return True
        finally:
            self._dry_run = was_dry
            self._skip_gripper = was_skip

    def step(self) -> bool:
        handlers = {
            State.CHECK_HW: self._do_check_hw,
            State.SCAN_GLOBAL: self._do_scan_global,
            State.VALIDATE_CMD: self._do_validate_cmd,
            State.PICK_TRANSIT: self._do_pick_transit,
            State.PICK_REFINE: self._do_pick_refine,
            State.PICK_GRASP: self._do_pick_grasp,
            State.VERIFY_PICK: self._do_verify_pick,
            State.PLACE_TRANSIT: self._do_place_transit,
            State.PLACE_REFINE: self._do_place_refine,
            State.PLACE_RELEASE: self._do_place_release,
            State.VERIFY_PLACE: self._do_verify_place,
            State.DONE: self._do_done,
        }
        handler = handlers.get(self._state)
        if handler is None:
            self._fail(f"unhandled state: {self._state.value}")
            return False
        try:
            return handler()
        except Exception as exc:
            traceback.print_exc()
            self._fail(str(exc))
            return False

    def _bootstrap(self) -> bool:
        self._state = State.CHECK_HW
        if not self.step():
            return False
        self._state = State.SCAN_GLOBAL
        return self.step()

    def _do_check_hw(self) -> bool:
        stage("CHECK_HW", "connecting arm, camera, gripper")
        if not self._arm.is_connected():
            self._arm.connect()
        self._camera.connect()
        self._viz.bind_camera(self._camera)
        if not self._skip_gripper:
            try:
                if self._gripper is None:
                    self._gripper = GripperDriver(self._arm, self._config["gripper"])
                self._gripper.setup_modbus()
                close_pos = int(self._config["gripper"].get("close_position", 0))
                sub(f"gripper init OK, close_position={close_pos}")
            except GripperDriverError as exc:
                if not self._dry_run:
                    self._fail(f"gripper connect failed: {exc}")
                    return False
                warn(f"gripper unavailable, skipped in dry-run: {exc}")
                self._skip_gripper = True
                self._gripper = None
        else:
            sub("gripper skipped (--no-gripper or config)")
        pose = self._arm.get_pose_6d()
        self._last_pose = pose
        sub(f"current pose xyz=({pose[0]:.1f},{pose[1]:.1f},{pose[2]:.1f}) mm")
        stage("CHECK_HW", "OK")
        return True

    def _do_scan_global(self) -> bool:
        stage("SCAN_GLOBAL", "scanning 24 slots")
        if self._need_direct_scan:
            # 冷启动：从当前任意姿态直达 scan_pose，避免先抬 Z 导致不可达
            waypoints = self._planner.plan_to_scan(None)
            self._need_direct_scan = False
        else:
            from_pose = self._last_pose or self._planner.scan_pose
            waypoints = self._planner.plan_to_scan(from_pose)
        print(format_waypoints(waypoints))
        sub("waypoints planned, moving")
        self._execute_waypoints(waypoints, phase="scan")

        self._arm.wait_motion_done()
        time.sleep(0.2)
        self._viz.stop_live()
        frame = self._camera.capture()
        pose_6d = self._arm.get_pose_6d()
        self._last_pose = pose_6d
        T_base_ee = pose_6d_to_matrix(pose_6d)

        detections = self._detector.detect(frame.color)
        working_z = self._registry.z_rack
        tube_above = float(self._rack_layout.get("tube_above_rack_mm", 30))
        observations = self._mapper.map(
            detections,
            frame.depth,
            K=self._K,
            dist=self._dist,
            T_ee_cam=self._T_ee_cam,
            T_base_ee=T_base_ee,
            depth_min_mm=self._cam_cfg.get("depth_min_mm", 100),
            depth_max_mm=self._cam_cfg.get("depth_max_mm", 800),
            z_rack_override=working_z,
        )
        measured_z, n_tubes = measure_rack_z(observations, tube_above_rack_mm=tube_above)
        z_rack = self._resolve_scan_z_rack(working_z, measured_z, n_tubes)
        if z_rack is None:
            self._fail(
                "SCAN_GLOBAL: 无法确定 z_rack（当前 tube 太少且无标定兜底），"
                "请在架上多放几支试管后重新扫描"
            )
            return False
        self._registry.update_from_scan(observations, z_rack)
        self._registry.set_rack_theta(self._mapper.last_rack_theta)
        tube_count = len(self._registry.find_tube_slots())
        sub(f"z_rack={z_rack:.1f} mm, tube_count={tube_count}")

        annotated = draw_scan_annotation(
            frame.color,
            detections,
            observations,
            self._mapper,
            title="SCAN",
            z_rack=z_rack,
            font_scale=self._font_scale,
        )
        self._viz.show_scan(annotated)
        return True

    def _resolve_scan_z_rack(
        self,
        working_z: float | None,
        measured_z: float | None,
        n_tubes: int,
    ) -> float | None:
        """
        选定本次扫描使用的 z_rack：
        - 首帧(working_z=None)：管足够则用本帧测量，否则用标定兜底；
        - 后续：复用已标定 working_z，仅当"管足够且偏差大"时告警提示重标。
        """
        min_tubes = int(self._registry_cfg.get("min_tubes_for_z_calib", 3))
        drift = float(self._registry_cfg.get("z_rack_drift_warn_mm", 5))
        default_z = self._rack_layout.get("default_rack_plane_z_mm")
        default_z = float(default_z) if default_z is not None else None
        if working_z is None:
            if measured_z is not None and n_tubes >= min_tubes:
                return measured_z
            return default_z
        if (
            measured_z is not None
            and n_tubes >= min_tubes
            and abs(measured_z - working_z) > drift
        ):
            print("[SCAN_GLOBAL][WARN] 架面高度疑似变化,建议重标")
        return working_z

    def _do_validate_cmd(self) -> bool:
        if self._cmd is None:
            self._fail("no move command")
            return False
        valid, reason = self._validator.validate(self._cmd, self._registry)
        if not valid:
            self._fail(reason)
            return False
        sub(f"cmd valid: {self._cmd.src} -> {self._cmd.dst}")
        return True

    def _do_pick_transit(self) -> bool:
        assert self._cmd is not None
        src = self._registry.get(self._cmd.src)
        from_pose = self._last_pose or self._planner.scan_pose
        waypoints = self._planner.plan_pick_transit(src, from_pose)
        stage("PICK_TRANSIT", f"transit to {self._cmd.src}")
        print(format_waypoints(waypoints))
        sub("waypoints planned, moving")
        self._execute_waypoints(waypoints, phase="pick_transit")
        self._last_pose = waypoints[-1].pose_6d
        return True

    def _do_pick_refine(self) -> bool:
        assert self._cmd is not None
        stage("PICK_REFINE", f"refine slot {self._cmd.src}")
        self._arm.wait_motion_done()
        time.sleep(0.2)
        self._viz.stop_live()
        frame = self._camera.capture()

        src = self._registry.get(self._cmd.src)
        if src.base_xyz is None:
            self._fail(f"{self._cmd.src} missing base_xyz from SCAN_GLOBAL")
            return False
        result = None
        refined_base = src.base_xyz
        if self._refine_update_enabled:
            detections = self._refine_detector.detect(frame.color)
            pose_6d = self._arm.get_pose_6d()
            T_base_ee = pose_6d_to_matrix(pose_6d)
            max_dist, ambiguity_delta = self._refine_match_params()
            try:
                result = refine_pick_slot(
                    self._cmd.src,
                    self._registry,
                    frame.color,
                    frame.depth,
                    self._refine_detector,
                    self._K,
                    self._dist,
                    self._T_ee_cam,
                    T_base_ee,
                    max_dist_xy_mm=max_dist,
                    ambiguity_min_delta_mm=ambiguity_delta,
                    depth_min_mm=self._cam_cfg.get("depth_min_mm", 100),
                    depth_max_mm=self._cam_cfg.get("depth_max_mm", 800),
                    detections=detections,
                    projected_match_max_px=float(
                        self._vision_cfg.get("pick_refine_projected_match_max_px", 80)
                    ),
                    projected_bbox_margin_px=float(
                        self._vision_cfg.get("pick_refine_projected_bbox_margin_px", 20)
                    ),
                    allow_projected_fallback=bool(
                        self._vision_cfg.get(
                            "pick_refine_projected_fallback_enabled", True
                        )
                    ),
                )
            except RefineError as exc:
                if not self._vision_cfg.get("pick_refine_fallback_to_global", True):
                    self._fail(f"{self._cmd.src} refine failed: {exc}")
                    return False
                warn(f"{self._cmd.src} refine failed, fallback to scan base_xyz: {exc}")
            else:
                refined_base = result.base_xyz
                sub(
                    f"base_xyz updated: dist_xy={result.dist_xy_mm:.2f} mm, "
                    f"xyz=({result.base_xyz[0]:.1f},{result.base_xyz[1]:.1f},"
                    f"{result.base_xyz[2]:.1f}), conf={result.confidence:.3f}"
                )
                self._registry.update_slot(
                    self._cmd.src,
                    base_xyz=result.base_xyz,
                    pixel_uv=result.pixel_uv,
                    confidence=result.confidence,
                    z_source=result.z_source,
                )
        else:
            detections = []
            sub(
                f"refine_update disabled, use scan base_xyz="
                f"({src.base_xyz[0]:.1f},{src.base_xyz[1]:.1f},{src.base_xyz[2]:.1f})"
            )

        refine_vis = draw_refine_annotation(
            frame.color,
            detections,
            self._cmd.src,
            result,
            title="PICK_REFINE" if result else "PICK_REFINE_IMAGE_ONLY",
            font_scale=self._font_scale,
        )
        self._viz.show_refine(refine_vis)
        yaw = self._planner.choose_grasp_yaw(self._cmd.src, self._registry)
        if yaw is not None:
            sub(f"grasp yaw aligned: rz={yaw:.3f} rad")
        approach = self._planner.build_approach_pose(refined_base, yaw_rad=yaw)
        sub(f"descend to pick_approach, speed={self._arm_cfg['approach_speed']}")
        self._move_pose(approach, self._arm_cfg["approach_speed"], label="pick_approach")
        self._pick_approach = approach
        self._last_pose = approach
        return True

    def _do_pick_grasp(self) -> bool:
        assert self._cmd is not None
        if self._pick_approach is None:
            self._fail("missing pick_approach pose")
            return False

        stage("PICK_GRASP", "grasp")
        if not self._dry_run and not self._skip_gripper and self._gripper is None:
            self._fail("gripper not available")
            return False
        pick_open = int(self._config["gripper"].get("pick_open_position", 135))
        pick_grip = int(self._config["gripper"].get("grip_position", 110))
        descend = float(self._motion.get("pick_descend_mm", 30))
        retreat_mm = float(self._motion.get("pick_retreat_mm", 100))
        if self._dry_run or self._skip_gripper:
            sub("dry-run: skip gripper open")
        else:
            sub(f"gripper open, pick_open_position={pick_open}")
            self._gripper.open_for_pick()
        if not self._confirm(PROMPT_GRASP):
            return False
        insert = self._planner.build_pick_insert_pose(self._pick_approach)
        sub(f"descend pick_descend_mm={descend:.1f}, speed={self._arm_cfg['approach_speed']}")
        self._move_pose(insert, self._arm_cfg["approach_speed"], label="pick_insert")
        if self._dry_run or self._skip_gripper:
            sub("dry-run: skip gripper close")
        else:
            sub(f"gripper close, grip_position={pick_grip}")
            self._gripper.move_to(pick_grip)
        retreat = self._planner.build_retreat_pose(insert, retreat_mm)
        sub(f"retreat pick_retreat_mm={retreat_mm:.1f}, speed={self._arm_cfg['default_speed']}")
        self._move_pose(retreat, self._arm_cfg["default_speed"], label="pick_retreat")
        self._registry.update_slot(
            self._cmd.src, klass="unknown", z_source="pending_verify"
        )
        self._last_pose = retreat
        return True

    def _do_verify_pick(self) -> bool:
        if self._dry_run:
            sub("dry-run: skip VERIFY_PICK")
            return True
        assert self._cmd is not None
        stage("VERIFY_PICK", "rescan and verify src slot")
        self._do_scan_global()
        result = self._verifier.verify(
            "pick",
            src=self._cmd.src,
            dst=self._cmd.dst,
            registry=self._registry,
        )
        if not result.ok:
            self._fail(result.summary())
            return False
        sub(result.summary())
        self._restore_pre_pick_place_target()
        return True

    def _do_place_transit(self) -> bool:
        assert self._cmd is not None
        self._restore_pre_pick_place_target()
        dst = self._registry.get(self._cmd.dst)
        from_pose = self._last_pose or self._planner.scan_pose
        waypoints = self._planner.plan_place_transit(dst, from_pose)
        stage("PLACE_TRANSIT", f"transit to {self._cmd.dst}")
        print(format_waypoints(waypoints))
        sub("waypoints planned, moving")
        self._execute_waypoints(waypoints, phase="place_transit")
        self._last_pose = waypoints[-1].pose_6d
        return True

    def _do_place_refine(self) -> bool:
        assert self._cmd is not None
        stage("PLACE_REFINE", f"refine slot {self._cmd.dst}")
        self._arm.wait_motion_done()
        time.sleep(0.2)
        self._viz.stop_live()
        frame = self._camera.capture()

        dst = self._registry.get(self._cmd.dst)
        if dst.base_xyz is None:
            self._fail(f"{self._cmd.dst} missing base_xyz from SCAN_GLOBAL")
            return False
        refined_base = dst.base_xyz
        result = None
        detections = self._refine_detector.detect(frame.color)
        if self._vision_cfg.get("place_refine_update_enabled", True):
            pose_6d = self._arm.get_pose_6d()
            T_base_ee = pose_6d_to_matrix(pose_6d)
            max_dist, ambiguity_delta = self._refine_match_params()
            try:
                result = refine_place_slot(
                    self._cmd.dst,
                    self._registry,
                    frame.color,
                    frame.depth,
                    self._refine_detector,
                    self._K,
                    self._dist,
                    self._T_ee_cam,
                    T_base_ee,
                    max_dist_xy_mm=max_dist,
                    ambiguity_min_delta_mm=ambiguity_delta,
                    depth_min_mm=self._cam_cfg.get("depth_min_mm", 100),
                    depth_max_mm=self._cam_cfg.get("depth_max_mm", 800),
                    detections=detections,
                )
            except RefineError as exc:
                if not self._vision_cfg.get("place_refine_fallback_to_global", True):
                    self._fail(f"{self._cmd.dst} refine failed: {exc}")
                    return False
                warn(f"{self._cmd.dst} refine failed, fallback to scan base_xyz: {exc}")
            else:
                refined_base = result.base_xyz
                sub(
                    f"base_xyz updated: dist_xy={result.dist_xy_mm:.2f} mm, "
                    f"xyz=({result.base_xyz[0]:.1f},{result.base_xyz[1]:.1f},"
                    f"{result.base_xyz[2]:.1f}), conf={result.confidence:.3f}"
                )
                self._registry.update_slot(
                    self._cmd.dst,
                    base_xyz=result.base_xyz,
                    pixel_uv=result.pixel_uv,
                    confidence=result.confidence,
                    z_source=result.z_source,
                )
        else:
            sub(
                f"place_refine_update disabled, use base_xyz="
                f"({dst.base_xyz[0]:.1f},{dst.base_xyz[1]:.1f},{dst.base_xyz[2]:.1f})"
            )

        refine_vis = draw_refine_annotation(
            frame.color,
            detections,
            self._cmd.dst,
            result,
            title="PLACE_REFINE" if result else "PLACE_REFINE_DETECT_ONLY",
            font_scale=self._font_scale,
        )
        self._viz.show_refine(refine_vis)
        yaw = self._planner.choose_grasp_yaw(self._cmd.dst, self._registry)
        if yaw is not None:
            sub(f"place yaw aligned: rz={yaw:.3f} rad")
        approach = self._planner.build_place_approach_pose(refined_base, yaw_rad=yaw)
        from_pose = self._last_pose or self._arm.get_pose_6d()
        above_refined = (
            approach[0],
            approach[1],
            from_pose[2],
            approach[3],
            approach[4],
            approach[5],
        )
        dx_high = above_refined[0] - from_pose[0]
        dy_high = above_refined[1] - from_pose[1]
        if abs(dx_high) > 0.5 or abs(dy_high) > 0.5:
            sub(f"XY correction: dx={dx_high:.1f} mm, dy={dy_high:.1f} mm")
            self._move_pose(
                above_refined,
                self._arm_cfg["default_speed"],
                label="place_above_refined",
            )
            from_pose = above_refined
        vertical_approach = (
            from_pose[0],
            from_pose[1],
            approach[2],
            from_pose[3],
            from_pose[4],
            from_pose[5],
        )
        sub(f"descend to place_approach, speed={self._arm_cfg['approach_speed']}")
        self._move_pose(
            vertical_approach,
            self._arm_cfg["approach_speed"],
            label="place_approach_vertical",
        )
        self._place_approach = vertical_approach
        self._last_pose = vertical_approach
        return True

    def _do_place_release(self) -> bool:
        assert self._cmd is not None
        if self._place_approach is None:
            self._fail("missing place_approach pose")
            return False

        stage("PLACE_RELEASE", "insert and release")
        if not self._dry_run and not self._skip_gripper and self._gripper is None:
            self._fail("gripper not available")
            return False
        insert = self._planner.build_place_insert_pose(self._place_approach)
        min_insert_z = float(self._safety_cfg.get("min_place_insert_flange_z", 150))
        if insert[2] < min_insert_z:
            self._fail(
                f"place_insert flange Z={insert[2]:.1f} mm < "
                f"min_place_insert_flange_z={min_insert_z:.1f} mm"
            )
            return False
        self._move_pose(insert, self._arm_cfg["approach_speed"], label="place_insert")
        release_open = int(self._config["gripper"].get("release_open_position", 120))
        retreat_mm = float(self._motion.get("place_retreat_mm", 100))
        if self._dry_run or self._skip_gripper:
            sub("dry-run: skip gripper release")
        else:
            sub(f"gripper open, release_open_position={release_open}")
            self._gripper.open_for_release()
        retreat = self._planner.build_retreat_pose(insert, retreat_mm)
        sub(f"retreat place_retreat_mm={retreat_mm:.1f}, speed={self._arm_cfg['default_speed']}")
        self._move_pose(retreat, self._arm_cfg["default_speed"], label="place_retreat")
        if self._dry_run or self._skip_gripper:
            sub("dry-run: skip gripper reset")
        else:
            sub("gripper reset, close_position=0")
            self._gripper.close(wait=False)
        self._registry.update_slot(
            self._cmd.dst, klass="unknown", z_source="pending_verify"
        )
        self._last_pose = retreat
        return True

    def _do_verify_place(self) -> bool:
        if self._dry_run:
            sub("dry-run: skip VERIFY_PLACE")
            return True
        assert self._cmd is not None
        stage("VERIFY_PLACE", "rescan and verify dst slot")
        self._do_scan_global()
        result = self._verifier.verify(
            "place",
            src=self._cmd.src,
            dst=self._cmd.dst,
            registry=self._registry,
        )
        if not result.ok:
            self._fail(result.summary())
            return False
        sub(result.summary())
        return True

    def _do_done(self) -> bool:
        stage("DONE", "OK")
        return True

    def _confirm(self, message: str) -> bool:
        if self._continuous_mode:
            print(f"{message}{AUTO_CONT}")
            return True
        print(message)
        try:
            input()
            return True
        except KeyboardInterrupt:
            self._fail("cancelled by user")
            return False
        except EOFError:
            return True

    def _execute_waypoints(self, waypoints: list[Waypoint], *, phase: str) -> None:
        for wp in waypoints:
            if "place_transit" in phase and wp.label.endswith(("_refine", "_approach")):
                min_approach_z = float(
                    self._safety_cfg.get("min_place_approach_flange_z", 220)
                )
                if wp.pose_6d[2] < min_approach_z:
                    raise FSMError(
                        f"{wp.label} flange Z={wp.pose_6d[2]:.1f} mm < "
                        f"min_place_approach_flange_z={min_approach_z:.1f} mm"
                    )
            speed = wp.speed or self._arm_cfg["default_speed"]
            self._move_pose(wp.pose_6d, speed, label=f"{phase}/{wp.label}")

    def _move_pose(
        self,
        pose: tuple[float, float, float, float, float, float],
        speed: int,
        *,
        label: str,
    ) -> None:
        cur = self._arm.get_pose_6d()
        delta_mm = max(abs(cur[i] - pose[i]) for i in range(3))
        move_line(
            label,
            pose,
            cur_xyz=cur[:3],
            delta_mm=delta_mm,
            speed=speed,
        )
        if self._pose_close(cur, pose, tol_mm=1.0, tol_rad=0.02):
            sub(f"move_p [{label}] already at target, skipped")
            return

        self._viz.start_live()
        try:
            self._arm.move_p(pose, speed=speed, block=True)
            deadline = time.monotonic() + 120.0
            while time.monotonic() < deadline:
                now_pose = self._arm.get_pose_6d()
                if self._pose_close(now_pose, pose, tol_mm=2.0, tol_rad=0.08):
                    moved = max(abs(now_pose[i] - cur[i]) for i in range(3))
                    move_done(label, now_pose[:3], moved)
                    return
                time.sleep(0.1)
            after = self._arm.get_pose_6d()
            raise FSMError(
                f"move_p [{label}] timeout: "
                f"target=({pose[0]:.1f},{pose[1]:.1f},{pose[2]:.1f}) "
                f"current=({after[0]:.1f},{after[1]:.1f},{after[2]:.1f})"
            )
        finally:
            self._viz.stop_live()

    @staticmethod
    def _pose_close(
        cur: tuple[float, float, float, float, float, float],
        target: tuple[float, float, float, float, float, float],
        *,
        tol_mm: float,
        tol_rad: float,
    ) -> bool:
        """位置按 mm abs 比较，姿态角用 _angle_close 处理 ±π 环绕。"""
        for i in range(3):
            if abs(cur[i] - target[i]) > tol_mm:
                return False
        for i in range(3, 6):
            if not _angle_close(cur[i], target[i], tol_rad):
                return False
        return True

    def _near_pose(
        self,
        target: tuple[float, float, float, float, float, float],
        *,
        tol_mm: float = 1.0,
        tol_rad: float = 0.02,
    ) -> bool:
        cur = self._arm.get_pose_6d()
        return self._pose_close(cur, target, tol_mm=tol_mm, tol_rad=tol_rad)

    def _refine_match_params(self) -> tuple[float, float]:
        """精定位 XY 匹配半径与歧义最小差距（mm），来自 config registry。"""
        return (
            float(self._registry_cfg.get("slot_match_max_dist_mm", 15)),
            float(self._registry_cfg.get("refine_ambiguity_min_delta_mm", 2)),
        )

    def _restore_pre_pick_place_target(self) -> None:
        """抓取后 scan 可能被夹持试管干扰，放置目标坐标优先使用抓前扫描值。"""
        if self._cmd is None or self._pre_pick_place_base_xyz is None:
            return
        try:
            self._registry.update_slot(
                self._cmd.dst,
                klass="empty",
                base_xyz=self._pre_pick_place_base_xyz,
                z_source="pre_pick_scan",
            )
        except Exception as exc:
            warn(f"restore pre-pick dst base_xyz failed: {exc}")

    def _fail(self, reason: str) -> None:
        self._fail_reason = reason
        self._state = State.FAILED
        err(reason)
