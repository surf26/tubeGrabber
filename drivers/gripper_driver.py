"""睿尔曼末端 RS485 电动夹爪驱动（经 ArmDriver Modbus RTU）。"""

from __future__ import annotations

import time

from Robotic_Arm.rm_robot_interface import rm_peripheral_read_write_params_t

from drivers.arm_driver import ArmDriver


class GripperDriverError(RuntimeError):
    """夹爪 Modbus 通信或控制异常。"""


class GripperDriver:
    def __init__(self, arm: ArmDriver, config: dict) -> None:
        if not arm.is_connected():
            raise GripperDriverError("ArmDriver 未连接，请先 arm.connect()")

        self._arm = arm
        self._port = int(config["modbus_port"])
        self._device = int(config["device_id"])
        self._baudrate = int(config["baudrate"])
        self._modbus_timeout = int(config.get("modbus_timeout", 2))  # 单位 100ms

        self._open_pos = int(config["open_position"])
        self._close_pos = int(config["close_position"])
        self._reg_position = int(config["reg_position"])
        self._reg_command = int(config["reg_command"])

        self._position_reg_count = int(config.get("position_reg_count", 2))
        self._move_timeout_s = float(config.get("move_timeout_s", 5.0))
        self._position_tolerance = int(config.get("position_tolerance", 500))

        self._modbus_ready = False

    def setup_modbus(self) -> bool:
        """配置末端 RS485 为 Modbus RTU 主站。"""
        robot = self._arm.get_robot()
        ret = robot.rm_set_modbus_mode(self._port, self._baudrate, self._modbus_timeout)
        if ret != 0:
            self._modbus_ready = False
            return False
        self._modbus_ready = True
        return True

    def get_position(self) -> int:
        """读取夹爪当前位置（步数）。"""
        self._require_modbus()
        regs = self._read_holding_registers(self._reg_position, self._position_reg_count)
        return _regs_to_position(regs)

    def move_to(self, position: int, *, wait: bool = True) -> bool:
        """写入目标位置到命令寄存器。"""
        self._require_modbus()
        position = max(0, int(position))
        regs = _position_to_regs(position, self._position_reg_count)
        self._write_registers(self._reg_command, regs)

        if wait and not self._wait_until_stable(position):
            raise GripperDriverError(
                f"夹爪未在 {self._move_timeout_s}s 内到达目标 {position}，"
                f"当前 {self.get_position()}"
            )
        return True

    def open(self, *, wait: bool = True) -> bool:
        """张开夹爪。"""
        return self.move_to(self._open_pos, wait=wait)

    def close(self, *, wait: bool = True) -> bool:
        """闭合夹爪。"""
        return self.move_to(self._close_pos, wait=wait)

    def is_ready(self) -> bool:
        """Modbus 已配置且能读到位置。"""
        if not self._modbus_ready:
            return False
        try:
            self.get_position()
            return True
        except GripperDriverError:
            return False

    def _require_modbus(self) -> None:
        if not self._modbus_ready:
            raise GripperDriverError("Modbus 未配置，请先调用 setup_modbus()")

    def _read_holding_registers(self, address: int, count: int) -> list[int]:
        robot = self._arm.get_robot()
        if count <= 1:
            params = rm_peripheral_read_write_params_t(self._port, address, self._device)
            code, value = robot.rm_read_holding_registers(params)
            if code != 0:
                raise GripperDriverError(f"读寄存器 {address} 失败，错误码: {code}")
            return [int(value)]

        params = rm_peripheral_read_write_params_t(self._port, address, self._device, count)
        code, values = robot.rm_read_multiple_holding_registers(params)
        if code != 0:
            raise GripperDriverError(f"读寄存器 {address}x{count} 失败，错误码: {code}")
        return [int(v) for v in values]

    def _write_registers(self, address: int, values: list[int]) -> None:
        robot = self._arm.get_robot()
        count = len(values)
        if count == 0:
            raise GripperDriverError("写入寄存器数据为空")

        if count == 1:
            params = rm_peripheral_read_write_params_t(self._port, address, self._device)
            ret = robot.rm_write_single_register(params, int(values[0]))
        else:
            params = rm_peripheral_read_write_params_t(self._port, address, self._device, count)
            ret = robot.rm_write_registers(params, [int(v) for v in values])

        if ret != 0:
            raise GripperDriverError(f"写寄存器 {address} 失败，错误码: {ret}")

    def _wait_until_stable(self, target: int) -> bool:
        deadline = time.monotonic() + self._move_timeout_s
        while time.monotonic() < deadline:
            pos = self.get_position()
            if abs(pos - target) <= self._position_tolerance:
                return True
            time.sleep(0.05)
        return False


def _position_to_regs(position: int, reg_count: int) -> list[int]:
    """32 位步数 → Modbus 寄存器列表（高字在前）。"""
    position &= 0xFFFFFFFF
    if reg_count <= 1:
        return [position & 0xFFFF]
    hi = (position >> 16) & 0xFFFF
    lo = position & 0xFFFF
    return [hi, lo]


def _regs_to_position(regs: list[int]) -> int:
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
