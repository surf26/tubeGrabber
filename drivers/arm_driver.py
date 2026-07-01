"""机械臂驱动封装"""

from __future__ import annotations

import time

from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e


class ArmDriverError(RuntimeError):
    """机械臂通信或状态异常。"""


class ArmDriver:
    def __init__(self, ip: str, port: int) -> None:
        self._ip = ip
        self._port = port
        self._arm: RoboticArm | None = None
        self._handle = None

    def connect(self) -> None:
        """连接机械臂，失败抛出 ArmDriverError。"""
        if self._arm is not None:
            return

        target = f"{self._ip}:{self._port}"
        try:
            self._arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
            self._handle = self._arm.rm_create_robot_arm(self._ip, self._port)
            if self._handle is None or (isinstance(self._handle, int) and self._handle < 0):
                raise ArmDriverError(
                    f"rm_create_robot_arm 失败 ({target})，handle={self._handle!r}"
                )

            code, _ = self._arm.rm_get_current_arm_state()
            if code != 0:
                self.disconnect()
                raise ArmDriverError(f"连接后读取状态失败 ({target})，SDK 错误码: {code}")
        except ArmDriverError:
            raise
        except Exception as exc:
            self.disconnect()
            raise ArmDriverError(
                f"连接异常 ({target}): {type(exc).__name__}: {exc}"
            ) from exc

    def disconnect(self) -> None:
        """断开连接并释放 SDK 资源。"""
        if self._arm is not None:
            try:
                self._arm.rm_delete_robot_arm()
            except Exception:
                pass
        self._arm = None
        self._handle = None

    def is_connected(self) -> bool:
        """是否已连接机械臂。"""
        return self._arm is not None

    def get_robot(self) -> RoboticArm:
        """供 GripperDriver 等外设 Modbus 使用。"""
        return self._require_arm()

    def get_pose_6d(self) -> tuple[float, float, float, float, float, float]:
        """返回 TCP 位姿 (x, y, z, rx, ry, rz)，单位 mm + rad。"""
        arm = self._require_arm()
        code, state = arm.rm_get_current_arm_state()
        if code != 0:
            raise ArmDriverError(f"读取位姿失败，错误码: {code}")

        pose = state.get("pose")
        if pose is None:
            raise ArmDriverError(f"状态数据缺少 pose 字段: {state}")

        return _parse_pose_6d_mm_rad(pose)

    def move_p(
        self,
        pose_6d: list[float] | tuple[float, ...],
        speed: int,
        *,
        block: bool = True,
    ) -> bool:
        """
        笛卡尔空间运动（rm_movej_p）。
        pose_6d: (x,y,z,rx,ry,rz)，单位 mm + rad。
        speed: 速度比例 1~100。
        """
        arm = self._require_arm()
        if len(pose_6d) < 6:
            raise ArmDriverError(f"pose_6d 需要 6 个数，收到: {pose_6d!r}")

        sdk_pose = _pose_6d_mm_rad_to_sdk(pose_6d)
        v = _clamp_speed(speed)
        ret = arm.rm_movej_p(sdk_pose, v, 0, 0, 1 if block else 0)
        if ret != 0:
            raise ArmDriverError(f"move_p 失败，错误码: {ret}")
        return True

    def move_j(
        self,
        joints: list[float],
        speed: int,
        *,
        block: bool = True,
    ) -> bool:
        """
        关节空间运动（rm_movej）。
        joints: 各关节角，单位 deg。
        speed: 速度比例 1~100。
        """
        arm = self._require_arm()
        if not joints:
            raise ArmDriverError("joints 不能为空")

        v = _clamp_speed(speed)
        ret = arm.rm_movej(list(joints), v, 0, 0, 1 if block else 0)
        if ret != 0:
            raise ArmDriverError(f"move_j 失败，错误码: {ret}")
        return True

    def wait_motion_done(
        self,
        timeout: float = 10.0,
        poll_interval: float = 0.05,
        stable_duration: float = 0.25,
        position_tol_mm: float = 0.5,
    ) -> bool:
        """
        等待机械臂停稳（用于非阻塞运动后，或拍照前额外确认）。
        通过连续采样 TCP 位置变化量判断是否到位。
        """
        self._require_arm()

        deadline = time.monotonic() + timeout
        last_pose: tuple[float, ...] | None = None
        stable_since: float | None = None

        while time.monotonic() < deadline:
            pose = self.get_pose_6d()
            now = time.monotonic()

            if last_pose is not None:
                delta_mm = max(abs(pose[i] - last_pose[i]) for i in range(3))
                if delta_mm <= position_tol_mm:
                    if stable_since is None:
                        stable_since = now
                    elif now - stable_since >= stable_duration:
                        return True
                else:
                    stable_since = None

            last_pose = pose
            time.sleep(poll_interval)

        return False

    def stop(self, *, emergency: bool = False) -> bool:
        """
        中止运动。
        emergency=False: 轨迹减速停止（rm_set_arm_slow_stop）
        emergency=True:  急停（rm_set_arm_stop，轨迹不可恢复）
        """
        arm = self._require_arm()
        ret = arm.rm_set_arm_stop() if emergency else arm.rm_set_arm_slow_stop()
        if ret != 0:
            raise ArmDriverError(f"stop 失败，错误码: {ret}")
        return True

    def _require_arm(self) -> RoboticArm:
        if self._arm is None:
            raise ArmDriverError("机械臂未连接，请先调用 connect()")
        return self._arm


def _clamp_speed(speed: int) -> int:
    """速度范围 1~100百分比"""
    return max(1, min(100, int(speed)))


def _pose_6d_mm_rad_to_sdk(
    pose_6d: list[float] | tuple[float, ...],
) -> list[float]:
    """config/上层 mm+rad → SDK m+rad"""
    x, y, z, rx, ry, rz = (float(v) for v in pose_6d[:6])
    return [x / 1000.0, y / 1000.0, z / 1000.0, rx, ry, rz]


def _parse_pose_6d_mm_rad(pose) -> tuple[float, float, float, float, float, float]:
    """
    解析 SDK 返回的 pose，统一为 mm + rad。
    Realman SDK 位置单位为 m（见官方文档）；欧拉角为 rad。
    """
    if isinstance(pose, (list, tuple)) and len(pose) >= 6:
        x, y, z, rx, ry, rz = (float(v) for v in pose[:6])
        return x * 1000.0, y * 1000.0, z * 1000.0, rx, ry, rz

    if isinstance(pose, dict):
        position = pose.get("position", pose)
        euler = pose.get("euler", pose)

        if isinstance(position, dict):
            x = float(position["x"])
            y = float(position["y"])
            z = float(position["z"])
        else:
            raise ArmDriverError(f"无法解析 position: {position}")

        if isinstance(euler, dict):
            rx = float(euler["rx"])
            ry = float(euler["ry"])
            rz = float(euler["rz"])
        else:
            raise ArmDriverError(f"无法解析 euler: {euler}")

        # 官方文档：position 为 m
        return x * 1000.0, y * 1000.0, z * 1000.0, rx, ry, rz

    raise ArmDriverError(f"未知 pose 格式: {type(pose)!r} {pose!r}")
