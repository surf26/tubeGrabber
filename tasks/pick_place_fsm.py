"""试管抓放状态机。"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any

from drivers.arm_driver import ArmDriver
from drivers.camera_driver import CameraDriver
from drivers.gripper_driver import GripperDriver
from perception.coord_transform import load_T_ee_cam, load_intrinsics, pose_6d_to_matrix
from perception.refine import refine_pick_slot, refine_place_slot
from perception.slot_mapper import SlotMapper
from perception.yolo_detector import YoloDetector
from planning.command_validator import CommandValidator, MoveCommand
from planning.motion_planner import MotionPlanner, Waypoint, format_waypoints
from utils.config_loader import load_yaml
from utils.vision_viz import VisionDisplay, draw_refine_annotation, draw_scan_annotation
from world.tube_registry import TubeRegistry, estimate_z_rack


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
        self._skip_gripper = skip_gripper or dry_run

        self._cam_cfg = config["camera"]
        self._arm_cfg = config["arm"]
        self._motion = config.get("motion", {})
        self._registry_cfg = config.get("registry", {})

        self._K, self._dist = load_intrinsics(config["calib"]["camera_intrinsics"])
        self._T_ee_cam = load_T_ee_cam(config["calib"]["hand_eye"])
        self._rack_layout = load_yaml(config["calib"]["rack_layout"])
        self._viz = VisionDisplay(config)
        self._font_scale = float(config.get("vision", {}).get("font_scale", 0.28))

        self._state = State.INIT
        self._cmd: MoveCommand | None = None
        self._fail_reason = ""
        self._last_pose: tuple[float, float, float, float, float, float] | None = None
        self._pick_approach: tuple[float, float, float, float, float, float] | None = None
        self._place_approach: tuple[float, float, float, float, float, float] | None = None

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

    def run_interactive(self) -> None:
        """扫描一次后循环等待用户指令。"""
        if not self._bootstrap():
            return
        self._state = State.WAIT_CMD
        print(self._registry.to_table_str())
        print("\n输入 'src dst' 执行搬运，'scan' 重扫，'quit' 退出")
        while self._state != State.FAILED:
            if self._state == State.WAIT_CMD:
                try:
                    text = input("> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n退出")
                    break
                if not text:
                    continue
                if text.lower() in ("quit", "q", "exit"):
                    break
                if text.lower() == "scan":
                    self._state = State.SCAN_GLOBAL
                    self.step()
                    print(self._registry.to_table_str())
                    self._state = State.WAIT_CMD
                    continue
                self._cmd, ok, reason = self._validator.parse_and_validate(
                    text, self._registry
                )
                if not ok:
                    print(f"指令无效: {reason}")
                    continue
                self._state = State.VALIDATE_CMD
            self.step()
            if self._state == State.DONE:
                print("完成")
                self._state = State.WAIT_CMD
            elif self._state == State.FAILED:
                print(f"失败: {self._fail_reason}")
                self._state = State.WAIT_CMD

    def run_dry_move(self, cmd_text: str) -> bool:
        """空跑：移动 + 精定位，不夹取。"""
        if not self._bootstrap():
            return False
        print(self._registry.to_table_str())
        print()
        return self.execute_move(cmd_text, dry=True)

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
            self._cmd, ok, reason = self._validator.parse_and_validate(
                cmd_text, self._registry
            )
            if not ok or self._cmd is None:
                self._fail(f"指令无效: {reason}")
                return False

            label = "dry-run" if dry else "move"
            print(f"{label}: {self._cmd.src} -> {self._cmd.dst}")

            if dry:
                sequence = [
                    State.PICK_TRANSIT,
                    State.PICK_REFINE,
                    State.PICK_GRASP,
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
                    State.SCAN_GLOBAL,
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
            self._fail(f"未处理状态: {self._state.value}")
            return False
        try:
            return handler()
        except Exception as exc:
            self._fail(str(exc))
            return False

    def _bootstrap(self) -> bool:
        self._state = State.CHECK_HW
        if not self.step():
            return False
        self._state = State.SCAN_GLOBAL
        return self.step()

    def _do_check_hw(self) -> bool:
        print("[CHECK_HW] 连接硬件...")
        if not self._arm.is_connected():
            self._arm.connect()
        self._camera.connect()
        self._viz.bind_camera(self._camera)
        if not self._skip_gripper:
            if self._gripper is None:
                self._gripper = GripperDriver(self._arm, self._config["gripper"])
            self._gripper.setup_modbus()
            self._gripper.open()
        self._last_pose = self._arm.get_pose_6d()
        print("[CHECK_HW] OK")
        return True

    def _do_scan_global(self) -> bool:
        print("[SCAN_GLOBAL] 扫描 24 槽...")
        from_pose = self._last_pose or self._planner.scan_pose
        waypoints = self._planner.plan_to_scan(from_pose)
        self._execute_waypoints(waypoints, phase="scan")

        self._arm.wait_motion_done()
        time.sleep(0.2)
        self._viz.stop_live()
        frame = self._camera.capture()
        pose_6d = self._arm.get_pose_6d()
        self._last_pose = pose_6d
        T_base_ee = pose_6d_to_matrix(pose_6d)

        detections = self._detector.detect(frame.color)
        observations = self._mapper.map(
            detections,
            frame.depth,
            K=self._K,
            dist=self._dist,
            T_ee_cam=self._T_ee_cam,
            T_base_ee=T_base_ee,
            depth_min_mm=self._cam_cfg.get("depth_min_mm", 100),
            depth_max_mm=self._cam_cfg.get("depth_max_mm", 800),
        )
        z_rack = estimate_z_rack(
            observations,
            tube_above_rack_mm=float(self._rack_layout.get("tube_above_rack_mm", 30)),
            default_z_mm=self._rack_layout.get("default_rack_plane_z_mm"),
        )
        self._registry.update_from_scan(observations, z_rack)
        print(f"[SCAN_GLOBAL] z_rack={z_rack:.1f} mm, tubes={len(self._registry.find_tube_slots())}")

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

    def _do_validate_cmd(self) -> bool:
        if self._cmd is None:
            self._fail("无搬运指令")
            return False
        ok, reason = self._validator.validate(self._cmd, self._registry)
        if not ok:
            self._fail(reason)
            return False
        print(f"[VALIDATE_CMD] {self._cmd.src} -> {self._cmd.dst} OK")
        return True

    def _do_pick_transit(self) -> bool:
        assert self._cmd is not None
        src = self._registry.get(self._cmd.src)
        from_pose = self._last_pose or self._planner.scan_pose
        waypoints = self._planner.plan_pick_transit(src, from_pose)
        print(f"[PICK_TRANSIT] -> {self._cmd.src}")
        print(format_waypoints(waypoints))
        self._execute_waypoints(waypoints, phase="pick_transit")
        self._last_pose = waypoints[-1].pose_6d
        return True

    def _do_pick_refine(self) -> bool:
        assert self._cmd is not None
        print(f"[PICK_REFINE] {self._cmd.src}")
        self._arm.wait_motion_done()
        time.sleep(0.2)
        self._viz.stop_live()
        frame = self._camera.capture()
        pose_6d = self._arm.get_pose_6d()
        T_base_ee = pose_6d_to_matrix(pose_6d)

        max_dist, ambiguity_delta = self._refine_match_params()
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
        )
        print(
            f"  refine dist_xy={result.dist_xy_mm:.2f}mm "
            f"base=({result.base_xyz[0]:.1f},{result.base_xyz[1]:.1f},{result.base_xyz[2]:.1f})"
        )
        self._registry.update_slot(
            self._cmd.src,
            base_xyz=result.base_xyz,
            pixel_uv=result.pixel_uv,
            confidence=result.confidence,
            z_source=result.z_source,
        )
        refine_vis = draw_refine_annotation(
            frame.color,
            self._refine_detector.detect(frame.color),
            self._cmd.src,
            result,
            title="PICK_REFINE",
            font_scale=self._font_scale,
        )
        self._viz.show_refine(refine_vis)
        approach = self._planner.build_approach_pose(result.base_xyz)
        self._move_pose(approach, self._arm_cfg["approach_speed"], label="pick_approach")
        self._pick_approach = approach
        self._last_pose = approach
        return True

    def _do_pick_grasp(self) -> bool:
        assert self._cmd is not None
        if self._pick_approach is None:
            self._fail("缺少 pick approach")
            return False
        if self._dry_run or self._skip_gripper:
            print("[PICK_GRASP] dry-run 跳过夹取")
            return True

        print("[PICK_GRASP] 抓取...")
        if self._gripper is None:
            self._fail("无夹爪")
            return False
        self._gripper.open()
        insert = self._planner.build_pick_insert_pose(self._pick_approach)
        self._move_pose(insert, self._arm_cfg["approach_speed"], label="pick_insert")
        self._gripper.close()
        retreat = self._planner.build_retreat_pose(
            insert,
            float(self._motion.get("pick_retreat_mm", 100)),
        )
        self._move_pose(retreat, self._arm_cfg["default_speed"], label="pick_retreat")
        self._registry.update_slot(
            self._cmd.src, klass="unknown", z_source="pending_verify"
        )
        self._last_pose = retreat
        return True

    def _do_verify_pick(self) -> bool:
        if self._dry_run:
            print("[VERIFY_PICK] dry-run 跳过")
            return True
        assert self._cmd is not None
        print("[VERIFY_PICK] 回 scan 验证...")
        self._do_scan_global()
        state = self._registry.get(self._cmd.src)
        if state.klass != "empty":
            self._fail(f"VERIFY_PICK 失败: {self._cmd.src} 仍为 {state.klass}")
            return False
        print(f"[VERIFY_PICK] {self._cmd.src} -> empty OK")
        return True

    def _do_place_transit(self) -> bool:
        assert self._cmd is not None
        dst = self._registry.get(self._cmd.dst)
        from_pose = self._last_pose or self._planner.scan_pose
        waypoints = self._planner.plan_place_transit(dst, from_pose)
        print(f"[PLACE_TRANSIT] -> {self._cmd.dst}")
        print(format_waypoints(waypoints))
        self._execute_waypoints(waypoints, phase="place_transit")
        self._last_pose = waypoints[-1].pose_6d
        return True

    def _do_place_refine(self) -> bool:
        assert self._cmd is not None
        print(f"[PLACE_REFINE] {self._cmd.dst}")
        self._arm.wait_motion_done()
        time.sleep(0.2)
        self._viz.stop_live()
        frame = self._camera.capture()
        pose_6d = self._arm.get_pose_6d()
        T_base_ee = pose_6d_to_matrix(pose_6d)

        max_dist, ambiguity_delta = self._refine_match_params()
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
        )
        print(
            f"  refine dist_xy={result.dist_xy_mm:.2f}mm "
            f"base=({result.base_xyz[0]:.1f},{result.base_xyz[1]:.1f},{result.base_xyz[2]:.1f})"
        )
        self._registry.update_slot(
            self._cmd.dst,
            base_xyz=result.base_xyz,
            pixel_uv=result.pixel_uv,
            confidence=result.confidence,
            z_source=result.z_source,
        )
        refine_vis = draw_refine_annotation(
            frame.color,
            self._refine_detector.detect(frame.color),
            self._cmd.dst,
            result,
            title="PLACE_REFINE",
            font_scale=self._font_scale,
        )
        self._viz.show_refine(refine_vis)
        approach = self._planner.build_approach_pose(result.base_xyz)
        self._move_pose(approach, self._arm_cfg["approach_speed"], label="place_approach")
        self._place_approach = approach
        self._last_pose = approach
        return True

    def _do_place_release(self) -> bool:
        assert self._cmd is not None
        if self._place_approach is None:
            self._fail("缺少 place approach")
            return False
        if self._dry_run or self._skip_gripper:
            print("[PLACE_RELEASE] dry-run 跳过放置")
            return True

        print("[PLACE_RELEASE] 放置...")
        if self._gripper is None:
            self._fail("无夹爪")
            return False
        insert = self._planner.build_place_insert_pose(self._place_approach)
        self._move_pose(insert, self._arm_cfg["approach_speed"], label="place_insert")
        self._gripper.open()
        retreat = self._planner.build_retreat_pose(
            insert,
            float(self._motion.get("place_retreat_mm", 100)),
        )
        self._move_pose(retreat, self._arm_cfg["default_speed"], label="place_retreat")
        self._registry.update_slot(
            self._cmd.dst, klass="unknown", z_source="pending_verify"
        )
        self._last_pose = retreat
        return True

    def _do_verify_place(self) -> bool:
        if self._dry_run:
            print("[VERIFY_PLACE] dry-run 跳过")
            return True
        assert self._cmd is not None
        print("[VERIFY_PLACE] 回 scan 验证...")
        self._do_scan_global()
        state = self._registry.get(self._cmd.dst)
        if state.klass != "tube":
            self._fail(f"VERIFY_PLACE 失败: {self._cmd.dst} 仍为 {state.klass}")
            return False
        print(f"[VERIFY_PLACE] {self._cmd.dst} -> tube OK")
        return True

    def _do_done(self) -> bool:
        print("[DONE]")
        return True

    def _execute_waypoints(self, waypoints: list[Waypoint], *, phase: str) -> None:
        for wp in waypoints:
            speed = wp.speed or self._arm_cfg["default_speed"]
            self._move_pose(wp.pose_6d, speed, label=f"{phase}/{wp.label}")

    def _move_pose(
        self,
        pose: tuple[float, float, float, float, float, float],
        speed: int,
        *,
        label: str,
    ) -> None:
        print(f"  move_p [{label}] z={pose[2]:.1f}")
        self._viz.start_live()
        try:
            self._arm.move_p(pose, speed=speed, block=True)
            if not self._arm.wait_motion_done():
                raise FSMError(f"move_p [{label}] 超时未到位")
        finally:
            self._viz.stop_live()

    def _refine_match_params(self) -> tuple[float, float]:
        """精定位 XY 匹配半径与歧义最小差距（mm），来自 config registry。"""
        return (
            float(self._registry_cfg.get("slot_match_max_dist_mm", 15)),
            float(self._registry_cfg.get("refine_ambiguity_min_delta_mm", 2)),
        )

    def _fail(self, reason: str) -> None:
        self._fail_reason = reason
        self._state = State.FAILED
        print(f"[FAILED] {reason}")
