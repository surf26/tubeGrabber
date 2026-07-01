"""睿尔曼末端 RS485 电动夹爪驱动（经 ArmDriver Modbus RTU）。"""

from __future__ import annotations

import time

from Robotic_Arm.rm_robot_interface import (
    rm_modbus_rtu_read_params_t,
    rm_modbus_rtu_write_params_t,
    rm_peripheral_read_write_params_t,
)

from drivers.arm_driver import ArmDriver


class GripperDriverError(RuntimeError):
    """夹爪 Modbus 通信或控制异常。"""


class GripperDriver:
    def __init__(self, arm: ArmDriver, config: dict) -> None:
        if not arm.is_connected():
            raise GripperDriverError("ArmDriver 未连接，请先 arm.connect()")

        self._arm = arm
        self._backend = str(config.get("backend", "realman_sdk")).lower()
        self._device = int(config["device_id"])
        self._baudrate = int(config["baudrate"])
        self._modbus_timeout = int(config.get("modbus_timeout", 2))  # 单位 100ms
        self._tool_voltage = int(config.get("tool_voltage", 3))  # 3 = 24V

        self._open_pos = int(config["open_position"])
        self._pick_open_pos = int(config.get("pick_open_position", self._open_pos))
        self._release_open_pos = int(config.get("release_open_position", self._open_pos))
        self._close_pos = int(config["close_position"])
        self._full_stroke = int(config.get("full_stroke", 1000))

        self._reg_zero_speed = int(config.get("reg_zero_speed", config.get("reg_position", 36)))
        self._reg_init_speed = int(config.get("reg_init_speed", config.get("reg_speed", 38)))
        self._reg_run_speed = int(config.get("reg_run_speed", config.get("reg_force", 40)))
        self._reg_command = int(config.get("reg_command", 43))
        self._reg_feedback = config.get("reg_position_feedback")

        self._zero_speed = int(config.get("zero_speed", 25600))
        self._init_speed = int(config.get("init_speed", 51200))
        self._run_speed = int(config.get("run_speed", 51200))

        self._position_reg_count = int(config.get("position_reg_count", 2))
        self._init_close_wait_s = float(config.get("init_close_wait_s", 5.0))
        self._move_timeout_s = float(config.get("move_timeout_s", 5.0))
        self._position_tolerance = int(config.get("position_tolerance", 500))
        self._settle_s = float(config.get("settle_s", 0.3))

        self._modbus_ready = False
        self._use_tool_rtu_api = True
        self._current_position = self._close_pos

    def setup_modbus(self) -> None:
        """初始化夹爪通信，并闭合到默认状态。"""
        if self._backend in ("realman_sdk", "sdk", "rm_plus"):
            self._setup_realman_sdk()
            return

        self._setup_legacy_modbus()

    def _setup_realman_sdk(self) -> None:
        """按 Realman 官方 SDK 夹爪接口初始化：上电 + RM Plus 协议。"""
        robot = self._arm.get_robot()

        ret = robot.rm_set_tool_voltage(self._tool_voltage)
        if ret != 0:
            raise GripperDriverError(f"rm_set_tool_voltage 失败，SDK 错误码: {ret}")
        time.sleep(0.5)

        if not hasattr(robot, "rm_set_rm_plus_mode"):
            raise GripperDriverError("当前 SDK 缺少 rm_set_rm_plus_mode，无法使用 realman_sdk 夹爪后端")
        ret = robot.rm_set_rm_plus_mode(self._baudrate)
        if ret != 0:
            raise GripperDriverError(
                f"rm_set_rm_plus_mode 失败，SDK 错误码: {ret} (baudrate={self._baudrate})"
            )
        time.sleep(0.3)

        # 非阻塞闭合到默认态；已闭合时部分 SDK/固件会返回 1 或 -4，按 demo 视为可继续。
        self._sdk_set_position(self._close_pos, wait=False, timeout_s=self._move_timeout_s, already_closed_ok=True)
        time.sleep(self._init_close_wait_s)

        if hasattr(robot, "rm_clear_system_err"):
            robot.rm_clear_system_err()

        self._current_position = self._close_pos
        self._modbus_ready = True

    def _setup_legacy_modbus(self) -> None:
        """旧版：配置工具端 RS485，并按寄存器协议控制夹爪。"""
        robot = self._arm.get_robot()

        ret = robot.rm_set_tool_voltage(self._tool_voltage)
        if ret != 0:
            raise GripperDriverError(f"rm_set_tool_voltage 失败，SDK 错误码: {ret}")
        time.sleep(0.5)

        self._use_tool_rtu_api = self._setup_rs485(robot)

        self._write_register(self._reg_zero_speed, self._zero_speed)
        time.sleep(0.2)
        self._write_register(self._reg_init_speed, self._init_speed)
        time.sleep(0.2)
        self._write_register(self._reg_run_speed, self._run_speed)
        time.sleep(0.2)

        self._write_register(self._reg_command, self._close_pos)
        self._current_position = self._close_pos
        time.sleep(self._init_close_wait_s)

        self._modbus_ready = True

    def get_position(self) -> int:
        """返回夹爪当前位置。SDK 后端无反馈时返回上次目标位置。"""
        self._require_modbus()
        if self._backend in ("realman_sdk", "sdk", "rm_plus"):
            return self._current_position

        if self._reg_feedback is not None:
            try:
                pos = self._read_register(int(self._reg_feedback), self._position_reg_count)
                self._current_position = pos
                return pos
            except GripperDriverError:
                pass
        return self._current_position

    def move_to(self, position: int, *, wait: bool = True) -> bool:
        """移动夹爪到目标位置。realman_sdk 后端位置范围为 0..1000。"""
        self._require_modbus()
        position = max(0, min(int(position), self._full_stroke))
        from_pos = self._current_position

        if self._backend in ("realman_sdk", "sdk", "rm_plus"):
            self._sdk_set_position(position, wait=wait, timeout_s=self._move_timeout_s)
            self._current_position = position
            if self._settle_s > 0:
                time.sleep(self._settle_s)
            return True

        self._write_register(self._reg_command, position)
        self._current_position = position

        if wait and not self._wait_until_done(from_pos, position):
            raise GripperDriverError(
                f"夹爪未在 {self._move_timeout_s}s 内到达目标 {position}，"
                f"当前 {self.get_position()}"
            )
        return True

    def open(self, *, wait: bool = True) -> bool:
        """张开夹爪。"""
        return self.move_to(self._open_pos, wait=wait)

    def open_for_pick(self, *, wait: bool = True) -> bool:
        """抓取前张开到试管套入开度。"""
        return self.move_to(self._pick_open_pos, wait=wait)

    def open_for_release(self, *, wait: bool = True) -> bool:
        """放置时松开到释放开度。"""
        return self.move_to(self._release_open_pos, wait=wait)

    def close(self, *, wait: bool = True) -> bool:
        """闭合夹爪。"""
        return self.move_to(self._close_pos, wait=wait)

    def is_ready(self) -> bool:
        """Modbus 已配置且初始化完成。"""
        return self._modbus_ready

    def _sdk_set_position(
        self,
        position: int,
        *,
        wait: bool,
        timeout_s: float,
        already_closed_ok: bool = False,
    ) -> None:
        robot = self._arm.get_robot()
        if not hasattr(robot, "rm_set_gripper_position"):
            raise GripperDriverError("当前 SDK 缺少 rm_set_gripper_position")

        timeout = max(0, int(round(timeout_s)))
        ret = robot.rm_set_gripper_position(int(position), bool(wait), timeout)
        if ret == 0:
            return
        if already_closed_ok and int(position) == self._close_pos and ret in (1, -4):
            return
        raise GripperDriverError(
            f"rm_set_gripper_position({position}, wait={wait}, timeout={timeout}) "
            f"失败，SDK 错误码: {ret}"
        )

    def _setup_rs485(self, robot) -> bool:
        """配置工具端 RS485；优先第四代 rm_set_tool_rs485_mode，失败则回退 rm_set_modbus_mode。"""
        ret = robot.rm_set_tool_rs485_mode(0, self._baudrate)
        if ret == 0:
            return True
        if ret != -4:
            raise GripperDriverError(
                f"rm_set_tool_rs485_mode 失败，SDK 错误码: {ret} "
                f"(baudrate={self._baudrate})"
            )

        port = 1
        ret = robot.rm_set_modbus_mode(port, self._baudrate, self._modbus_timeout)
        if ret != 0:
            raise GripperDriverError(
                f"rm_set_modbus_mode 失败，SDK 错误码: {ret} "
                f"(port={port}, baudrate={self._baudrate}, "
                f"timeout={self._modbus_timeout}x100ms, device_id={self._device})"
            )
        return False

    def _require_modbus(self) -> None:
        if not self._modbus_ready:
            raise GripperDriverError("Modbus 未配置，请先调用 setup_modbus()")

    def _write_register(self, address: int, value: int) -> None:
        regs = _value_to_regs(value, self._position_reg_count)
        if self._use_tool_rtu_api:
            ret = self._write_modbus_rtu_registers(address, regs)
            if ret == -4:
                self._use_tool_rtu_api = False
            elif ret != 0:
                raise GripperDriverError(f"写寄存器 {address} 失败，错误码: {ret}")
            else:
                return

        self._write_peripheral_registers(address, regs)

    def _read_register(self, address: int, count: int) -> int:
        if self._use_tool_rtu_api:
            code, regs = self._read_modbus_rtu_registers(address, count)
            if code == -4:
                self._use_tool_rtu_api = False
            elif code != 0:
                raise GripperDriverError(f"读寄存器 {address} 失败，错误码: {code}")
            else:
                return _regs_to_value(regs)

        regs = self._read_peripheral_registers(address, count)
        return _regs_to_value(regs)

    def _write_modbus_rtu_registers(self, address: int, values: list[int]) -> int:
        robot = self._arm.get_robot()
        param = rm_modbus_rtu_write_params_t(
            device=self._device,
            address=address,
            type=1,
            num=len(values),
            data=values,
        )
        return robot.rm_write_modbus_rtu_registers(param)

    def _read_modbus_rtu_registers(self, address: int, count: int) -> tuple[int, list[int]]:
        robot = self._arm.get_robot()
        param = rm_modbus_rtu_read_params_t(
            device=self._device,
            address=address,
            type=1,
            num=count,
        )
        return robot.rm_read_modbus_rtu_holding_registers(param)

    def _write_peripheral_registers(self, address: int, values: list[int]) -> None:
        robot = self._arm.get_robot()
        port = 1
        count = len(values)
        if count == 0:
            raise GripperDriverError("写入寄存器数据为空")

        if count == 1:
            params = rm_peripheral_read_write_params_t(port, address, self._device)
            ret = robot.rm_write_single_register(params, int(values[0]))
        else:
            params = rm_peripheral_read_write_params_t(port, address, self._device, count)
            ret = robot.rm_write_registers(params, [int(v) for v in values])

        if ret != 0:
            raise GripperDriverError(f"写寄存器 {address} 失败，错误码: {ret}")

    def _read_peripheral_registers(self, address: int, count: int) -> list[int]:
        robot = self._arm.get_robot()
        port = 1
        if count <= 1:
            params = rm_peripheral_read_write_params_t(port, address, self._device)
            code, value = robot.rm_read_holding_registers(params)
            if code != 0:
                raise GripperDriverError(f"读寄存器 {address} 失败，错误码: {code}")
            return [int(value)]

        params = rm_peripheral_read_write_params_t(port, address, self._device, count)
        code, values = robot.rm_read_multiple_holding_registers(params)
        if code != 0:
            raise GripperDriverError(f"读寄存器 {address}x{count} 失败，错误码: {code}")
        return [int(v) for v in values]

    def _wait_until_done(self, from_pos: int, target: int) -> bool:
        """按行程与运行速度估算等待时间，并可选轮询反馈寄存器。"""
        delta = abs(target - from_pos)
        if delta == 0:
            return True

        est_s = delta / max(self._run_speed, 1)
        wait_s = min(max(est_s, 0.1) + 0.5, self._move_timeout_s)

        if self._reg_feedback is None:
            time.sleep(wait_s)
            return True

        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            pos = self.get_position()
            if abs(pos - target) <= self._position_tolerance:
                return True
            time.sleep(0.05)
        return False


def _value_to_regs(value: int, reg_count: int) -> list[int]:
    """32 位步数 → Modbus 寄存器列表（高字在前）。"""
    value &= 0xFFFFFFFF
    if reg_count <= 1:
        return [value & 0xFFFF]
    hi = (value >> 16) & 0xFFFF
    lo = value & 0xFFFF
    return [hi, lo]


def _regs_to_value(regs: list[int]) -> int:
    """Modbus 寄存器列表 → 32 位步数。"""
    if not regs:
        raise GripperDriverError("读到的寄存器为空")

    if len(regs) == 1:
        val = regs[0]
    else:
        val = (regs[0] << 16) | (regs[1] & 0xFFFF)

    if val >= 0x80000000:
        val -= 0x100000000
    return int(val)
