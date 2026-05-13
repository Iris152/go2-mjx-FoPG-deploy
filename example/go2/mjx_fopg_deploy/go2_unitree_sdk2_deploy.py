#!/usr/bin/env python3
"""GO2 低层部署入口。

这个脚本同时服务于两种场景：
1. 本地双终端 sim2sim：通过 Unitree DDS topic 与 MuJoCo 仿真器通信。
2. 真机部署：通过 unitree_sdk2py 向 GO2 低层电机发送 LowCmd。
   真机低层控制前会先尝试释放 Go2 主运控服务 MCF，避免 sport_mode 等
   高层服务和本脚本同时控制电机。

核心数据流：
LowState -> 构造训练时一致的 observation -> ONNX policy -> 目标关节角 q_des
         -> 写入 LowCmd(q, dq, kp, kd, tau) -> rt/lowcmd

注意：policy 只输出归一化动作，最终发给电机的是“目标关节位置”。
kp/kd 是部署脚本按站立、策略、故障等阶段设置的电机 PD 增益。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import select
import signal
import struct
import sys
import threading
import time
import types
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional, Sequence


def _ensure_unitree_sdk2py_namespace() -> None:
    """兼容某些环境里 unitree_sdk2py 缺少标准 __init__.py 的情况。"""
    if "unitree_sdk2py" in sys.modules:
        return
    for entry in sys.path:
        pkg_dir = Path(entry) / "unitree_sdk2py"
        if pkg_dir.is_dir():
            pkg = types.ModuleType("unitree_sdk2py")
            pkg.__path__ = [str(pkg_dir)]
            pkg.__package__ = "unitree_sdk2py"
            pkg.__file__ = str(pkg_dir / "__init__.py")
            sys.modules["unitree_sdk2py"] = pkg
            return


try:
    import numpy as np
    import onnxruntime as ort
    _ensure_unitree_sdk2py_namespace()
    from unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize,
        ChannelPublisher,
        ChannelSubscriber,
    )
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
    try:
        # 新版 unitree_sdk2py 里如果提供了 MotionSwitcherClient，就直接使用官方封装。
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
            MotionSwitcherClient,
        )
    except ImportError:
        # 当前本机 mjx 环境里的 unitree_sdk2py 1.0.1 没带 comm.motion_switcher。
        # 这里按官方 Python 仓库同等实现，直接通过 RPC Client 调 motion_switcher 服务。
        from unitree_sdk2py.rpc.client import Client

        class MotionSwitcherClient(Client):
            """最小 MotionSwitcher 兼容实现。

            对应 Unitree C++ go2_stand_example.cpp 中的：
            CheckMode() 查询当前是否有 MCF/运动服务占用控制权；
            ReleaseMode() 释放 sport_mode/ai_sport 等主运控服务。
            """

            def __init__(self):
                super().__init__("motion_switcher", False)

            def Init(self):
                self._SetApiVerson("1.0.0.1")
                self._RegistApi(1001, 0)  # CheckMode
                self._RegistApi(1003, 0)  # ReleaseMode

            def CheckMode(self):
                code, data = self._Call(1001, json.dumps({}))
                if code == 0 and data:
                    return code, json.loads(data)
                return code, None

            def ReleaseMode(self):
                code, _ = self._Call(1003, json.dumps({}))
                return code, None
except ImportError as exc:
    missing = getattr(exc, "name", "unknown")
    raise SystemExit(
        "Missing runtime dependency: "
        f"{missing}. Activate the environment that has numpy, onnxruntime "
        "and unitree_sdk2py installed."
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent

# Unitree 低层协议里的特殊停止值：
# q=POS_STOP_F / dq=VEL_STOP_F 通常表示不下发有效位置/速度目标。
POS_STOP_F = 2.146e9
VEL_STOP_F = 16000.0

ACTION_SIZE = 12
BASELINE_OBS_SIZE = 40
FORWARD_OBS_SIZE = 52
KEYFRAME_NAME = "home"

# 训练时使用的动作缩放。策略输出在 [-1, 1] 附近，乘以 ACTION_SCALE 后
# 叠加到 command_center_policy，得到最终目标关节角 q_des。
ACTION_SCALE = np.array([0.2, 0.8, 0.8] * 4, dtype=np.float32)

# Policy order used by the MJX training code: FL, FR, RL, RR
# Unitree low-level order used by unitree_mujoco / hardware: FR, FL, RR, RL
# 训练/ONNX 内部和 Unitree 低层电机 topic 的关节顺序不同，所有 LowState/LowCmd
# 都必须经过这里的重排，否则腿会对不上。
POLICY_TO_UNITREE = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], dtype=np.int64)
UNITREE_TO_POLICY = POLICY_TO_UNITREE.copy()

# Unitree 官方示例里常见的站立角度，原始顺序是 Unitree 低层顺序。
# 默认部署使用训练 XML 的 home；只有 --command_center official_standup 时才用它。
OFFICIAL_STAND_UP_UNITREE = np.array(
    [
        0.00571868,
        0.608813,
        -1.21763,
        -0.00571868,
        0.608813,
        -1.21763,
        0.00571868,
        0.608813,
        -1.21763,
        -0.00571868,
        0.608813,
        -1.21763,
    ],
    dtype=np.float32,
)

PASSIVE_MOTOR_COUNT = 20

# LowCmd 有 20 个 motor_cmd 槽位，GO2 实际腿部只用前 12 个。
# 这里的 struct 格式用于按 Unitree C++ SDK 的内存布局重新计算 CRC。
LOWCMD_PACK_FMT = "<4B4IH2x" + "B3x5f3I" * 20 + "4B" + "55Bx2I"


def should_release_mcf(args: argparse.Namespace) -> bool:
    """根据命令行参数决定是否释放 MCF。

    auto 是默认策略：实机 domain_id=0 时释放，仿真 domain_id=1 时跳过。
    仿真没有 Go2 主运控服务，强行调用 MotionSwitcher 反而会超时。
    """
    if args.release_mcf == "always":
        return True
    if args.release_mcf == "never":
        return False
    return int(args.domain_id) == 0


def query_motion_service_name(robot_form: str, motion_name: str) -> str:
    """把 MotionSwitcher 返回的 form/name 翻译成可读的服务名。"""
    if robot_form == "0":
        if motion_name == "normal":
            return "sport_mode"
        if motion_name == "ai":
            return "ai_sport"
        if motion_name == "advanced":
            return "advanced_sport"
    else:
        if motion_name == "ai-w":
            return "wheeled_sport(go2W)"
        if motion_name == "normal-w":
            return "wheeled_sport(b2W)"
    return motion_name or "<unknown>"


def release_mcf_if_needed(args: argparse.Namespace) -> None:
    """真机低层控制前释放 Go2 主运控服务 MCF。

    Unitree 官方低层站立示例在发送 LowCmd 前会先调用 MotionSwitcher：
    1. CheckMode 查询当前是否有 sport_mode/ai_sport 等高层运控服务。
    2. 如果有，就 ReleaseMode 释放。
    3. 直到 motion_name 为空，才允许进入低层电机控制。

    这样可以避免主运控服务和本脚本同时向电机发控制指令。
    """
    if not should_release_mcf(args):
        print("[mcf] Motion-control service release skipped.", flush=True)
        return

    msc = MotionSwitcherClient()
    msc.SetTimeout(float(args.mcf_timeout))
    msc.Init()

    max_attempts = max(1, int(args.mcf_release_retries))
    for attempt in range(1, max_attempts + 1):
        code, status = msc.CheckMode()
        if code != 0:
            message = f"[mcf] CheckMode failed with code {code}."
            if args.mcf_allow_failure:
                print(message + " Continuing because --mcf_allow_failure is set.", flush=True)
                return
            raise SystemExit(message)

        status = status or {}
        robot_form = str(status.get("form", ""))
        motion_name = str(status.get("name", ""))
        if not motion_name:
            print("[mcf] Motion-control service is deactivated.", flush=True)
            return

        service_name = query_motion_service_name(robot_form, motion_name)
        print(
            f"[mcf] Active motion-control service detected: {service_name} "
            f"(form={robot_form}, name={motion_name}).",
            flush=True,
        )

        code, _ = msc.ReleaseMode()
        if code == 0:
            print(f"[mcf] ReleaseMode succeeded ({attempt}/{max_attempts}).", flush=True)
        else:
            message = f"[mcf] ReleaseMode failed with code {code} ({attempt}/{max_attempts})."
            if args.mcf_allow_failure:
                print(message + " Continuing because --mcf_allow_failure is set.", flush=True)
                return
            print(message, flush=True)

        if attempt < max_attempts:
            time.sleep(float(args.mcf_retry_delay))

    message = "[mcf] Motion-control service is still active after ReleaseMode retries."
    if args.mcf_allow_failure:
        print(message + " Continuing because --mcf_allow_failure is set.", flush=True)
        return
    raise SystemExit(message)


def crc32_core(words: Sequence[int]) -> int:
    """Unitree LowCmd 使用的 CRC32 计算核心。"""
    crc = 0xFFFFFFFF
    polynomial = 0x04C11DB7
    for current in words:
        bit = 1 << 31
        for _ in range(32):
            if crc & 0x80000000:
                crc = ((crc << 1) & 0xFFFFFFFF) ^ polynomial
            else:
                crc = (crc << 1) & 0xFFFFFFFF
            if current & bit:
                crc ^= polynomial
            bit >>= 1
    return crc & 0xFFFFFFFF


def pack_lowcmd_words(cmd) -> list[int]:
    """把 LowCmd 按协议内存布局打包为 uint32 words，用于 CRC。"""
    data: list[object] = []
    data.extend(cmd.head)
    data.append(cmd.level_flag)
    data.append(cmd.frame_reserve)
    data.extend(cmd.sn)
    data.extend(cmd.version)
    data.append(cmd.bandwidth)

    for i in range(PASSIVE_MOTOR_COUNT):
        motor = cmd.motor_cmd[i]
        data.append(motor.mode)
        data.append(motor.q)
        data.append(motor.dq)
        data.append(motor.tau)
        data.append(motor.kp)
        data.append(motor.kd)
        data.extend(motor.reserve)

    data.append(cmd.bms_cmd.off)
    data.extend(cmd.bms_cmd.reserve)
    data.extend(cmd.wireless_remote)
    data.extend(cmd.led)
    data.extend(cmd.fan)
    data.append(cmd.gpio)
    data.append(cmd.reserve)
    data.append(0)

    packed = struct.pack(LOWCMD_PACK_FMT, *data)
    words: list[int] = []
    calc_len = (len(packed) >> 2) - 1
    for i in range(calc_len):
        base = i * 4
        word = (
            (packed[base + 3] << 24)
            | (packed[base + 2] << 16)
            | (packed[base + 1] << 8)
            | packed[base]
        )
        words.append(word)
    return words


def compute_lowcmd_crc(cmd) -> int:
    """给发送前的 LowCmd 计算并返回 CRC。"""
    return crc32_core(pack_lowcmd_words(cmd))


def normalize_quaternion_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return quat / norm


def quat_wxyz_to_rotmat(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize_quaternion_wxyz(quat)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def gravity_body_from_quaternion(quat_wxyz: np.ndarray) -> np.ndarray:
    """根据 IMU 四元数得到机身坐标系下的重力方向。"""
    rot = quat_wxyz_to_rotmat(quat_wxyz)
    gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    return (rot.T @ gravity_world).astype(np.float32)


def cos_wave(step_index: np.ndarray, step_period: float, scale: float) -> np.ndarray:
    wave = -np.cos(((2.0 * np.pi) / step_period) * step_index)
    return wave * (scale / 2.0) + (scale / 2.0)


def make_kinematic_ref(step_k: int, control_dt: float, scale: float = 0.3) -> np.ndarray:
    """生成训练时使用的关节空间 trot 参考轨迹。

    baseline observation 会包含这一段参考轨迹，让部署时的输入分布和训练时一致。
    """
    steps = np.arange(step_k, dtype=np.float32)
    step_period = step_k * control_dt
    t = steps * control_dt
    wave = cos_wave(t, step_period, scale)
    leg_block = np.concatenate(
        [
            np.zeros((step_k, 1), dtype=np.float32),
            wave.reshape(step_k, 1),
            -2.0 * wave.reshape(step_k, 1),
        ],
        axis=1,
    )
    block1 = np.concatenate(
        [
            np.zeros((step_k, 3), dtype=np.float32),
            leg_block,
            leg_block,
            np.zeros((step_k, 3), dtype=np.float32),
        ],
        axis=1,
    )
    block2 = np.concatenate(
        [
            leg_block,
            np.zeros((step_k, 3), dtype=np.float32),
            np.zeros((step_k, 3), dtype=np.float32),
            leg_block,
        ],
        axis=1,
    )
    return np.concatenate([block1, block2], axis=0)


def find_home_keyframe_qpos(xml_path: Path, key_name: str = KEYFRAME_NAME) -> np.ndarray:
    """从训练 XML 读取 home keyframe 的 qpos。

    qpos[7:19] 是 12 个腿部关节角，作为训练策略的 action center。
    """
    root = ET.parse(xml_path).getroot()
    for keyframe in root.findall(".//keyframe"):
        for key in keyframe.findall("key"):
            if key.attrib.get("name") == key_name:
                qpos_text = key.attrib.get("qpos")
                if not qpos_text:
                    raise ValueError(f"keyframe '{key_name}' in {xml_path} has no qpos")
                qpos = np.fromstring(qpos_text, sep=" ", dtype=np.float32)
                if qpos.shape[0] < 19:
                    raise ValueError(f"keyframe '{key_name}' in {xml_path} has invalid qpos length")
                return qpos
    raise ValueError(f"Could not find keyframe '{key_name}' in {xml_path}")


def resolve_training_xml(candidate: Path) -> Path:
    """找到可用于读取 home keyframe 的训练 XML。"""
    if candidate.exists():
        try:
            qpos = find_home_keyframe_qpos(candidate)
            if qpos.shape[0] >= 19:
                return candidate
        except ValueError:
            pass
    fallback = candidate.parent / "go2.xml"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        "Could not resolve a training XML with a home keyframe. "
        f"Tried: {candidate} and {fallback}"
    )


def resolve_default_onnx(candidates: Sequence[Path]) -> Path:
    """按优先级查找默认 ONNX 文件，优先使用 *_ort.onnx。"""
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Could not find any of: " + ", ".join(str(p) for p in candidates))


def ort_variant_path(model_path: Path) -> Path:
    """根据普通 ONNX 路径推导 ORT 兼容版本路径。"""
    if model_path.name.endswith("_ort.onnx"):
        return model_path
    if model_path.suffix != ".onnx":
        return model_path.with_name(model_path.name + "_ort")
    return model_path.with_name(model_path.stem + "_ort.onnx")


def _load_sanitize_module():
    """动态加载本目录的 ONNX 清理脚本，避免做成包依赖。"""
    sanitize_path = SCRIPT_DIR / "sanitize_onnx_for_ort.py"
    spec = importlib.util.spec_from_file_location("sanitize_onnx_for_ort_local", sanitize_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load sanitize helper from {sanitize_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_ort_compatible_model(model_path: Path) -> Path:
    """确保 ONNX Runtime 能加载模型。

    JAX/TF 导出的 ONNX 里可能带 Expm1 / PreventGradient 等 ORT 不支持节点。
    如果没有现成的 *_ort.onnx，就自动调用 sanitize helper 生成一个。
    """
    model_path = model_path.resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    ort_path = ort_variant_path(model_path)
    if ort_path.exists():
        return ort_path

    if model_path.name.endswith("_ort.onnx"):
        return model_path

    module = _load_sanitize_module()
    import onnx

    print(f"[onnx] Sanitizing {model_path.name} -> {ort_path.name}", flush=True)
    model = onnx.load(str(model_path))
    model, replaced_expm1, replaced_pg = module.sanitize_model(model)
    print(
        f"[onnx] Replaced {replaced_expm1} Expm1 node(s), {replaced_pg} PreventGradient node(s).",
        flush=True,
    )
    onnx.checker.check_model(model)
    onnx.save(model, str(ort_path))
    return ort_path


def create_onnx_session(model_path: Path) -> ort.InferenceSession:
    """创建单线程 ONNX Runtime session，降低部署时 CPU 抢占和时延抖动。"""
    candidate_path = ensure_ort_compatible_model(model_path)
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    return ort.InferenceSession(str(candidate_path), sess_options=session_options, providers=["CPUExecutionProvider"])


def run_onnx_policy(session: ort.InferenceSession, obs: np.ndarray) -> np.ndarray:
    """执行一次策略前向推理，兼容 [obs] 和 [1, obs] 两种输入签名。"""
    obs = np.asarray(obs, dtype=np.float32)
    input_meta = session.get_inputs()[0]
    if len(input_meta.shape) == 1:
        feed = obs
    elif len(input_meta.shape) == 2:
        feed = obs.reshape(1, -1)
    else:
        raise ValueError(f"Unsupported input shape: {input_meta.shape}")
    output = session.run(None, {input_meta.name: feed})[0]
    output = np.asarray(output, dtype=np.float32)
    return output[0] if output.ndim == 2 else output


def remap_unitree_to_policy(values: np.ndarray) -> np.ndarray:
    """Unitree 低层顺序 -> 训练/策略顺序。"""
    return np.asarray(values, dtype=np.float32)[UNITREE_TO_POLICY]


def remap_policy_to_unitree(values: np.ndarray) -> np.ndarray:
    """训练/策略顺序 -> Unitree 低层顺序。"""
    values = np.asarray(values, dtype=np.float32)
    remapped = np.zeros_like(values)
    remapped[POLICY_TO_UNITREE] = values
    return remapped


@dataclass
class RobotState:
    """部署控制循环使用的机器人状态，字段都转换为训练/策略顺序。"""

    joint_angles_policy: np.ndarray
    joint_speeds_policy: np.ndarray
    gyro_body_policy: np.ndarray
    gravity_body: np.ndarray
    quat_wxyz: np.ndarray
    received_at: float


class Phase(Enum):
    """部署状态机。

    WAIT_START -> STAND_RAMP -> STAND_HOLD -> WAIT_POLICY -> POLICY_RAMP -> POLICY
    任意非等待阶段触发安全检查失败后会进入 FAULT。
    """

    WAIT_START = auto()
    STAND_RAMP = auto()
    STAND_HOLD = auto()
    WAIT_POLICY = auto()
    POLICY_RAMP = auto()
    POLICY = auto()
    FAULT = auto()


class EnterLatch:
    """非阻塞读取 Enter，用于手动确认站立和启动策略。"""

    def __init__(self, enabled: bool):
        self.enabled = bool(enabled and sys.stdin.isatty())

    def poll(self) -> bool:
        if not self.enabled:
            return False
        readable, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not readable:
            return False
        sys.stdin.readline()
        return True


class Go2DeployRunner:
    """GO2 部署主控制器。

    它负责订阅 rt/lowstate、维护状态机、调用 ONNX 策略，并把目标关节位置
    和电机 PD 参数写入 rt/lowcmd。
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.control_dt = float(args.control_dt)
        self.phase = Phase.WAIT_START
        self._shutdown_requested = False
        self._fault_reason: Optional[str] = None

        # 训练 XML 的 home keyframe 定义了策略训练时的默认站立姿态。
        train_xml = resolve_training_xml(Path(args.train_xml_path).expanduser())
        qpos_home = find_home_keyframe_qpos(train_xml)
        self.action_loc = qpos_home[7:19].astype(np.float32)

        # command_center_policy 是部署时所有 q_des 的中心：
        # 站立时直接跟踪它，策略阶段在它上面叠加 action * ACTION_SCALE。
        if args.command_center == "official_standup":
            self.command_center_policy = remap_unitree_to_policy(OFFICIAL_STAND_UP_UNITREE)
        else:
            self.command_center_policy = self.action_loc.copy()

        # 参考步态也要以训练 XML 的 home 为中心，保证 observation 与训练一致。
        kin_ref = make_kinematic_ref(step_k=args.step_k, control_dt=self.control_dt, scale=args.gait_scale)
        self.kinematic_ref_qpos = (kin_ref + self.action_loc[None, :]).astype(np.float32)
        self.l_cycle = int(self.kinematic_ref_qpos.shape[0])

        # baseline 是第一阶段 trot 策略；forward 是第二阶段前进残差策略。
        baseline_path = Path(args.baseline_onnx).expanduser()
        forward_path = Path(args.forward_onnx).expanduser() if args.forward_onnx else None
        self.baseline_session = create_onnx_session(baseline_path)
        self.forward_session = create_onnx_session(forward_path) if forward_path else None

        self.robot_state: Optional[RobotState] = None
        self.robot_state_lock = threading.Lock()
        self.lowstate_pub_seen = threading.Event()

        # last_joint_target_policy 会进入下一帧 observation，对应训练时 obs 中的 last action。
        self.last_joint_target_policy = self.command_center_policy.copy()
        self.control_steps = 0
        self.phase_started_at = time.monotonic()
        self.phase_start_joint_policy: Optional[np.ndarray] = None

        # policy loop 只更新 desired_*；低层发送 loop 按 lowcmd_dt 高频重发当前指令。
        self.desired_joint_target_policy = self.command_center_policy.copy()
        self.desired_kp = 0.0
        self.desired_kd = 0.0
        self.desired_passive = True

        # 真机和 sim2sim 都通过同一组 Unitree DDS topic 通信。
        self.cmd_pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.cmd_pub.Init()
        self.lowstate_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_sub.Init(self._lowstate_cb, 10)

        self.lowcmd = unitree_go_msg_dds__LowCmd_()
        self._init_lowcmd()

        self.start_latch = EnterLatch(not args.auto_start)
        self.policy_latch = EnterLatch(not args.auto_policy)
        self._start_prompt_shown = False
        self._policy_prompt_shown = False

    def _init_lowcmd(self) -> None:
        """初始化 LowCmd 为 passive，不给电机有效位置/速度/力矩目标。"""
        self.lowcmd.head[0] = 0xFE
        self.lowcmd.head[1] = 0xEF
        self.lowcmd.level_flag = 0xFF
        self.lowcmd.gpio = 0
        for i in range(PASSIVE_MOTOR_COUNT):
            self.lowcmd.motor_cmd[i].mode = 0x01
            self.lowcmd.motor_cmd[i].q = POS_STOP_F
            self.lowcmd.motor_cmd[i].dq = VEL_STOP_F
            self.lowcmd.motor_cmd[i].kp = 0.0
            self.lowcmd.motor_cmd[i].kd = 0.0
            self.lowcmd.motor_cmd[i].tau = 0.0

    def _transition(self, phase: Phase, note: str) -> None:
        """切换状态机阶段，并重置阶段计时器。"""
        self.phase = phase
        self.phase_started_at = time.monotonic()
        print(f"[phase] {phase.name}: {note}", flush=True)

    def _lowstate_cb(self, msg: LowState_) -> None:
        """DDS 回调：接收 LowState 并转换成策略需要的状态。"""
        q_unitree = np.array([msg.motor_state[i].q for i in range(ACTION_SIZE)], dtype=np.float32)
        dq_unitree = np.array([msg.motor_state[i].dq for i in range(ACTION_SIZE)], dtype=np.float32)
        quat_wxyz = np.array(msg.imu_state.quaternion, dtype=np.float32)
        gyro_unitree = np.array(msg.imu_state.gyroscope, dtype=np.float32)

        state = RobotState(
            joint_angles_policy=remap_unitree_to_policy(q_unitree),
            joint_speeds_policy=remap_unitree_to_policy(dq_unitree),
            gyro_body_policy=gyro_unitree.astype(np.float32),
            gravity_body=gravity_body_from_quaternion(quat_wxyz),
            quat_wxyz=normalize_quaternion_wxyz(quat_wxyz),
            received_at=time.monotonic(),
        )
        with self.robot_state_lock:
            self.robot_state = state
        self.lowstate_pub_seen.set()

    def get_robot_state(self) -> Optional[RobotState]:
        with self.robot_state_lock:
            if self.robot_state is None:
                return None
            return RobotState(
                joint_angles_policy=self.robot_state.joint_angles_policy.copy(),
                joint_speeds_policy=self.robot_state.joint_speeds_policy.copy(),
                gyro_body_policy=self.robot_state.gyro_body_policy.copy(),
                gravity_body=self.robot_state.gravity_body.copy(),
                quat_wxyz=self.robot_state.quat_wxyz.copy(),
                received_at=self.robot_state.received_at,
            )

    def wait_for_connection(self) -> None:
        """等待机器人或仿真器开始发布 rt/lowstate。"""
        print("[info] Waiting for rt/lowstate ...", flush=True)
        while not self.lowstate_pub_seen.wait(timeout=0.2):
            if self._shutdown_requested:
                return
        print("[info] Connected to rt/lowstate.", flush=True)

    def publish_passive(self) -> None:
        """发送 passive 指令：所有 motor_cmd 不含有效 q/dq/kp/kd/tau。"""
        for i in range(PASSIVE_MOTOR_COUNT):
            self.lowcmd.motor_cmd[i].mode = 0x01
            self.lowcmd.motor_cmd[i].q = POS_STOP_F
            self.lowcmd.motor_cmd[i].dq = VEL_STOP_F
            self.lowcmd.motor_cmd[i].kp = 0.0
            self.lowcmd.motor_cmd[i].kd = 0.0
            self.lowcmd.motor_cmd[i].tau = 0.0
        self.lowcmd.crc = compute_lowcmd_crc(self.lowcmd)
        self.cmd_pub.Write(self.lowcmd)

    def set_passive_command(self) -> None:
        """把当前期望指令切成 passive，实际发送由 publish_current_command 完成。"""
        self.desired_passive = True
        self.desired_kp = 0.0
        self.desired_kd = 0.0

    def set_joint_target_command(self, joint_target_policy: np.ndarray, kp: float, kd: float) -> None:
        """缓存一帧关节目标和 PD 增益，供低层发送循环重复发布。"""
        self.desired_joint_target_policy = np.asarray(joint_target_policy, dtype=np.float32).copy()
        self.desired_kp = float(kp)
        self.desired_kd = float(kd)
        self.desired_passive = False

    def publish_joint_target(self, joint_target_policy: np.ndarray, kp: float, kd: float) -> None:
        """把策略顺序的关节目标写入 LowCmd 并发布。

        LowCmd 语义：
        q   = 目标关节角 q_des
        dq  = 目标关节速度，这里固定为 0
        kp/kd = 当前阶段使用的电机 PD
        tau = 前馈力矩，这里固定为 0
        """
        joint_target_unitree = remap_policy_to_unitree(joint_target_policy)
        for i in range(PASSIVE_MOTOR_COUNT):
            self.lowcmd.motor_cmd[i].mode = 0x01
            self.lowcmd.motor_cmd[i].tau = 0.0
            if i < ACTION_SIZE:
                self.lowcmd.motor_cmd[i].q = float(joint_target_unitree[i])
                self.lowcmd.motor_cmd[i].dq = 0.0
                self.lowcmd.motor_cmd[i].kp = float(kp)
                self.lowcmd.motor_cmd[i].kd = float(kd)
            else:
                self.lowcmd.motor_cmd[i].q = POS_STOP_F
                self.lowcmd.motor_cmd[i].dq = VEL_STOP_F
                self.lowcmd.motor_cmd[i].kp = 0.0
                self.lowcmd.motor_cmd[i].kd = 0.0
        self.lowcmd.crc = compute_lowcmd_crc(self.lowcmd)
        self.cmd_pub.Write(self.lowcmd)

    def publish_current_command(self) -> None:
        """按当前 desired_* 状态发布 passive 或关节目标。"""
        if self.desired_passive:
            self.publish_passive()
        else:
            self.publish_joint_target(
                self.desired_joint_target_policy,
                self.desired_kp,
                self.desired_kd,
            )

    def _tilt_angle_rad(self, gravity_body: np.ndarray) -> float:
        """根据机身系重力方向估计机身倾斜角。"""
        aligned = float(np.clip(-gravity_body[2], -1.0, 1.0))
        return math.acos(aligned)

    def _check_safety(self, state: RobotState) -> None:
        """基础安全检查：LowState 超时和机身倾角过大。"""
        if time.monotonic() - state.received_at > self.args.lowstate_timeout:
            self._enter_fault("lowstate timeout")
            return
        tilt = self._tilt_angle_rad(state.gravity_body)
        if tilt > self.args.tilt_limit_rad:
            self._enter_fault(f"body tilt too large: {tilt:.3f} rad")

    def _enter_fault(self, reason: str) -> None:
        """进入故障阶段，后续会回到站立中心并使用 fault_kp/fault_kd。"""
        if self.phase == Phase.FAULT:
            return
        self._fault_reason = reason
        self._transition(Phase.FAULT, reason)

    def _build_inner_obs(self, state: RobotState) -> np.ndarray:
        """构造第一阶段 baseline 策略的 40 维 observation。"""
        kin_ref = self.kinematic_ref_qpos[self.control_steps % self.l_cycle]
        obs = np.concatenate(
            [
                # yaw 角速度缩放项。
                np.array([state.gyro_body_policy[2] * 0.25], dtype=np.float32),
                # 机身坐标系下的重力方向。
                state.gravity_body.astype(np.float32),
                # 当前关节角相对训练 home 的偏移。
                (state.joint_angles_policy - self.action_loc).astype(np.float32),
                # 上一帧下发的目标关节角。
                self.last_joint_target_policy.astype(np.float32),
                # 当前相位对应的参考步态关节角。
                kin_ref.astype(np.float32),
            ],
            axis=0,
        )
        if obs.shape[0] != BASELINE_OBS_SIZE:
            raise ValueError(f"baseline obs mismatch: {obs.shape[0]} != {BASELINE_OBS_SIZE}")
        return np.clip(obs, -100.0, 100.0).astype(np.float32)

    def _compute_policy_target(self, state: RobotState) -> np.ndarray:
        """运行 ONNX 策略并转换为最终目标关节角。"""
        inner_obs = self._build_inner_obs(state)
        baseline_action = run_onnx_policy(self.baseline_session, inner_obs)
        baseline_action = np.clip(np.asarray(baseline_action, dtype=np.float32), -1.0, 1.0)

        if self.args.mode == "trot":
            # 只跑第一阶段时，baseline_action 直接作为最终动作。
            final_action = baseline_action
        else:
            if self.forward_session is None:
                raise RuntimeError("forward mode requires forward onnx")
            outer_obs = np.concatenate([inner_obs, baseline_action], axis=0).astype(np.float32)
            if outer_obs.shape[0] != FORWARD_OBS_SIZE:
                raise ValueError(f"forward obs mismatch: {outer_obs.shape[0]} != {FORWARD_OBS_SIZE}")
            residual_action = run_onnx_policy(self.forward_session, outer_obs)
            residual_action = np.clip(np.asarray(residual_action, dtype=np.float32), -1.0, 1.0)
            # 第二阶段策略输出的是残差，要叠加到第一阶段 baseline action 上。
            final_action = residual_action + baseline_action
            if self.args.clip_final_action:
                final_action = np.clip(final_action, -1.0, 1.0)

        # 最终下发的 q_des = 站立中心 + 归一化动作 * 训练动作缩放。
        return (self.command_center_policy + final_action * ACTION_SCALE).astype(np.float32)

    def _phase_elapsed(self) -> float:
        return time.monotonic() - self.phase_started_at

    def request_shutdown(self) -> None:
        self._shutdown_requested = True

    def _maybe_start(self, state: RobotState) -> None:
        """根据 auto_start 或手动 Enter 决定是否进入站立流程。"""
        if self.args.auto_start:
            self.phase_start_joint_policy = state.joint_angles_policy.copy()
            self._transition(Phase.STAND_RAMP, "auto start")
            return
        if not self._start_prompt_shown:
            print("[input] Press Enter to start stand-up.", flush=True)
            self._start_prompt_shown = True
        if self.start_latch.poll():
            self.phase_start_joint_policy = state.joint_angles_policy.copy()
            self._transition(Phase.STAND_RAMP, "manual start")

    def _stand_target(self) -> np.ndarray:
        """站立渐变目标：从当前真实关节角平滑插值到站立中心。"""
        if self.phase_start_joint_policy is None:
            raise RuntimeError("phase_start_joint_policy is not initialized")
        alpha = min(self._phase_elapsed() / self.args.stand_ramp_duration, 1.0)
        return ((1.0 - alpha) * self.phase_start_joint_policy + alpha * self.command_center_policy).astype(np.float32)

    def _policy_ramp_target(self, policy_target: np.ndarray) -> np.ndarray:
        """策略渐入目标：从站立中心平滑插值到策略输出。"""
        alpha = min(self._phase_elapsed() / self.args.policy_ramp_duration, 1.0)
        return ((1.0 - alpha) * self.command_center_policy + alpha * policy_target).astype(np.float32)

    def _shutdown_release_target(self) -> np.ndarray:
        """真正 passive 前的释放姿态。

        Unitree passive 本身只是停止给电机有效 q/kp/kd/tau，并没有“目标姿态”。
        为了避免站立后突然卸力，这里先把 q_des 平滑过渡到一个较低姿态，
        再逐步降低 kp/kd，最后才发送真正 passive。
        """
        if self.args.shutdown_release_pose == "stand":
            return self.command_center_policy.copy()
        if self.args.shutdown_release_pose == "crouch":
            return np.asarray([0.0, 1.25, -2.45] * 4, dtype=np.float32)
        if self.args.shutdown_release_pose == "prone":
            return np.asarray([0.0, 1.45, -2.60] * 4, dtype=np.float32)
        raise ValueError(f"Unsupported shutdown_release_pose: {self.args.shutdown_release_pose}")

    def loop_once(self) -> None:
        """策略控制循环的一次更新，默认频率由 control_dt 决定。"""
        state = self.get_robot_state()
        if state is None:
            # 没有状态时不下发有效控制，避免电机收到 stale command。
            self.set_passive_command()
            return

        if self.phase not in (Phase.WAIT_START, Phase.FAULT):
            self._check_safety(state)

        if self.phase == Phase.WAIT_START:
            # 等待用户确认站立前保持 passive。
            self.set_passive_command()
            self._maybe_start(state)
            return

        if self.phase == Phase.STAND_RAMP:
            # 起身阶段：从当前姿态缓慢插值到站立中心，使用 stand_kp/kd。
            target = self._stand_target()
            self.set_joint_target_command(target, self.args.stand_kp, self.args.stand_kd)
            self.last_joint_target_policy = target.copy()
            if self._phase_elapsed() >= self.args.stand_ramp_duration:
                self._transition(Phase.STAND_HOLD, "stand pose reached")
            return

        if self.phase == Phase.STAND_HOLD:
            # 站稳阶段：保持站立中心，等待策略启动。
            self.set_joint_target_command(self.command_center_policy, self.args.stand_kp, self.args.stand_kd)
            self.last_joint_target_policy = self.command_center_policy.copy()
            if self._phase_elapsed() >= self.args.stand_hold_duration:
                if self.args.auto_policy:
                    self._transition(Phase.POLICY_RAMP, "auto policy start")
                else:
                    self._transition(Phase.WAIT_POLICY, "waiting for policy arm")
            return

        if self.phase == Phase.WAIT_POLICY:
            # 已站立但还未启动策略，仍然保持站立中心。
            self.set_joint_target_command(self.command_center_policy, self.args.stand_kp, self.args.stand_kd)
            self.last_joint_target_policy = self.command_center_policy.copy()
            if not self._policy_prompt_shown:
                print("[input] Press Enter to start the policy.", flush=True)
                self._policy_prompt_shown = True
            if self.policy_latch.poll():
                self._transition(Phase.POLICY_RAMP, "manual policy arm")
            return

        if self.phase == Phase.POLICY_RAMP:
            # 策略渐入阶段：逐渐从站立中心过渡到 policy target。
            target = self._compute_policy_target(state)
            blended = self._policy_ramp_target(target)
            self.set_joint_target_command(blended, self.args.policy_kp, self.args.policy_kd)
            self.last_joint_target_policy = blended.copy()
            self.control_steps += 1
            if self._phase_elapsed() >= self.args.policy_ramp_duration:
                self._transition(Phase.POLICY, "policy fully enabled")
            return

        if self.phase == Phase.POLICY:
            # 正常策略阶段：每个 control_dt 更新一次 ONNX 输出。
            target = self._compute_policy_target(state)
            self.set_joint_target_command(target, self.args.policy_kp, self.args.policy_kd)
            self.last_joint_target_policy = target.copy()
            self.control_steps += 1
            return

        if self.phase == Phase.FAULT:
            # 故障阶段不继续跑策略，回到站立中心。
            self.set_joint_target_command(self.command_center_policy, self.args.fault_kp, self.args.fault_kd)
            self.last_joint_target_policy = self.command_center_policy.copy()
            return

        raise RuntimeError(f"Unhandled phase: {self.phase}")

    def run(self) -> None:
        """主循环。

        policy loop 低频运行 ONNX 和状态机；LowCmd loop 以更高频率重发最近指令，
        避免 DDS/底层控制因为短暂延迟而认为命令超时。
        """
        self.wait_for_connection()
        next_policy_tick = time.perf_counter()
        next_cmd_tick = next_policy_tick
        while not self._shutdown_requested:
            now = time.perf_counter()

            if now >= next_policy_tick:
                loop_start = time.perf_counter()
                self.loop_once()
                next_policy_tick += self.control_dt
                if next_policy_tick <= now:
                    next_policy_tick = now + self.control_dt
                if self.args.print_timing and self.phase == Phase.POLICY:
                    elapsed_ms = (time.perf_counter() - loop_start) * 1000.0
                    print(f"[timing] {elapsed_ms:.3f} ms", flush=True)

            now = time.perf_counter()
            if now >= next_cmd_tick:
                self.publish_current_command()
                next_cmd_tick += self.args.lowcmd_dt
                if next_cmd_tick <= now:
                    next_cmd_tick = now + self.args.lowcmd_dt

            sleep_dt = min(next_policy_tick, next_cmd_tick) - time.perf_counter()
            if sleep_dt > 0:
                time.sleep(sleep_dt)

    def graceful_shutdown(self) -> None:
        """退出时平滑回站立，再逐步卸掉 PD，最后发送真正 passive。

        旧逻辑是：短暂站立 -> 直接 passive。真机上这会让电机力矩突然归零，
        机器人可能出现明显一顿或突然下沉。

        新逻辑分四段：
        1. 从当前策略目标平滑插值回 command_center_policy。
        2. 在 command_center_policy 短暂保持站立。
        3. 从站立中心平滑过渡到 shutdown_release_pose，同时把 kp/kd 线性降到 0。
        4. 降到 0 后再发送 Unitree passive 停止值。
        """
        print("[shutdown] Ramping to stand, fading PD, then passive.", flush=True)

        step_dt = max(self.control_dt, 1e-3)

        ramp_duration = max(float(self.args.shutdown_stand_ramp_duration), 0.0)
        ramp_start = self.last_joint_target_policy.astype(np.float32).copy()
        if ramp_duration > 0.0:
            t0 = time.perf_counter()
            while True:
                alpha = min((time.perf_counter() - t0) / ramp_duration, 1.0)
                target = ((1.0 - alpha) * ramp_start + alpha * self.command_center_policy).astype(np.float32)
                self.publish_joint_target(target, self.args.stand_kp, self.args.stand_kd)
                self.last_joint_target_policy = target.copy()
                if alpha >= 1.0:
                    break
                time.sleep(step_dt)

        hold_deadline = time.perf_counter() + max(float(self.args.shutdown_stand_duration), 0.0)
        while time.perf_counter() < hold_deadline:
            self.publish_joint_target(self.command_center_policy, self.args.stand_kp, self.args.stand_kd)
            self.last_joint_target_policy = self.command_center_policy.copy()
            time.sleep(step_dt)

        release_duration = max(float(self.args.shutdown_release_duration), 0.0)
        release_target = self._shutdown_release_target()
        if release_duration > 0.0:
            t0 = time.perf_counter()
            while True:
                alpha = min((time.perf_counter() - t0) / release_duration, 1.0)
                target = ((1.0 - alpha) * self.command_center_policy + alpha * release_target).astype(np.float32)
                kp = (1.0 - alpha) * float(self.args.stand_kp)
                kd = (1.0 - alpha) * float(self.args.stand_kd)
                self.publish_joint_target(target, kp, kd)
                self.last_joint_target_policy = target.copy()
                if alpha >= 1.0:
                    break
                time.sleep(step_dt)

        passive_deadline = time.perf_counter() + max(float(self.args.shutdown_passive_duration), 0.0)
        while time.perf_counter() < passive_deadline:
            self.publish_passive()
            time.sleep(step_dt)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    真机部署时常用参数：
    --network 选择连接 GO2 的网卡名
    --domain_id 真机通常用 0，仿真默认用 1
    --stand_kp/stand_kd 控制起身和等待策略阶段
    --policy_kp/policy_kd 控制策略阶段
    --release_mcf 控制是否在真机低层控制前释放 Go2 主运控服务
    """
    default_baseline = resolve_default_onnx(
        [
            SCRIPT_DIR / "exported_onnx" / "trotting_2hz_policy_ort.onnx",
            SCRIPT_DIR / "exported_onnx" / "trotting_2hz_policy.onnx",
        ]
    )
    default_forward = resolve_default_onnx(
        [
            SCRIPT_DIR / "exported_onnx" / "forward_locomotion_policy_ort.onnx",
            SCRIPT_DIR / "exported_onnx" / "forward_locomotion_policy.onnx",
        ]
    )
    default_train_xml = SCRIPT_DIR / "mujoco_menagerie" / "unitree_go2" / "scene_mjx.xml"

    parser = argparse.ArgumentParser(description="GO2 low-level deploy runner for unitree_mujoco and real robot")

    # 策略模式与模型文件。
    parser.add_argument("--mode", choices=["forward", "trot"], default="forward")
    parser.add_argument("--baseline_onnx", type=str, default=str(default_baseline))
    parser.add_argument("--forward_onnx", type=str, default=str(default_forward))
    parser.add_argument("--train_xml_path", type=str, default=str(default_train_xml))

    # 站立中心选择：默认 train_home 与训练 observation 中心一致。
    parser.add_argument(    
        "--command_center",
        choices=["official_standup", "train_home"],
        default="train_home",
        help="Reference pose used for stand / output targets. Observation center remains the training home pose.",
    )
    parser.add_argument(
        "--clip_final_action",
        action="store_true",
        help="Clip baseline+residual before scaling. Default is off to match GO2_train.ipynb/local training.",
    )

    # DDS 通信参数。仿真一般 network=lo/domain_id=1；真机一般是有线网卡/domain_id=0。
    parser.add_argument("--network", type=str, default="lo", help="Use lo for unitree_mujoco, e.g. enp5s0 for the real robot")
    parser.add_argument("--domain_id", type=int, default=1, help="Use 1 for simulation, 0 for the real robot")

    # control_dt 是策略更新周期；lowcmd_dt 是 LowCmd 重发周期。
    parser.add_argument("--control_dt", type=float, default=0.02)
    parser.add_argument("--lowcmd_dt", type=float, default=0.002, help="Low-level command resend period; keep small for unitree_mujoco / hardware bridges")

    # 训练时参考步态的相位设置。
    parser.add_argument("--step_k", type=int, default=13)
    parser.add_argument("--gait_scale", type=float, default=0.3)

    # 阶段持续时间：先起身、站稳，再渐入策略。
    parser.add_argument("--stand_ramp_duration", type=float, default=2.5)
    parser.add_argument("--stand_hold_duration", type=float, default=0.5)
    parser.add_argument("--policy_ramp_duration", type=float, default=2.0)

    # 写入 LowCmd 的 PD 增益。真机上建议先保守调小，再逐步放开。
    parser.add_argument("--stand_kp", type=float, default=60.0)
    parser.add_argument("--stand_kd", type=float, default=5.0)
    parser.add_argument("--policy_kp", type=float, default=50.0)
    parser.add_argument("--policy_kd", type=float, default=3.5)
    parser.add_argument("--fault_kp", type=float, default=60.0)
    parser.add_argument("--fault_kd", type=float, default=5.0)

    # 安全与退出参数。
    parser.add_argument("--tilt_limit_rad", type=float, default=0.9)
    parser.add_argument("--lowstate_timeout", type=float, default=0.2)
    parser.add_argument(
        "--shutdown_stand_ramp_duration",
        type=float,
        default=1.0,
        help="Shutdown ramp from the latest policy target back to stand pose.",
    )
    parser.add_argument("--shutdown_stand_duration", type=float, default=0.5)
    parser.add_argument(
        "--shutdown_release_duration",
        type=float,
        default=2.0,
        help="Shutdown duration for fading stand kp/kd to zero before passive.",
    )
    parser.add_argument(
        "--shutdown_release_pose",
        choices=["stand", "crouch", "prone"],
        default="crouch",
        help=(
            "Low pose reached before final passive. 'stand' only fades gains, "
            "'crouch' lowers the body, 'prone' lowers more aggressively."
        ),
    )
    parser.add_argument("--shutdown_passive_duration", type=float, default=0.2)

    # 自动化开关。真机首次部署建议不要加这两个参数，保留人工确认。
    parser.add_argument("--auto_start", action="store_true", help="Start stand-up without waiting for Enter")
    parser.add_argument("--auto_policy", action="store_true", help="Start policy automatically after stand-up")
    parser.add_argument("--print_timing", action="store_true")

    # MCF / 主运控服务释放。实机上默认释放，仿真自动跳过。
    # 如果 sport_mode 等高层服务不释放，可能和本脚本的 rt/lowcmd 低层控制抢电机。
    parser.add_argument(
        "--release_mcf",
        choices=["auto", "always", "never"],
        default="auto",
        help=(
            "Release Go2 motion-control services through MotionSwitcher before "
            "low-level control. auto releases only when domain_id=0."
        ),
    )
    parser.add_argument("--mcf_timeout", type=float, default=10.0)
    parser.add_argument("--mcf_release_retries", type=int, default=5)
    parser.add_argument("--mcf_retry_delay", type=float, default=2.0)
    parser.add_argument(
        "--mcf_allow_failure",
        action="store_true",
        help="Continue even if CheckMode/ReleaseMode fails. Not recommended on the real robot.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(
        "[config] network=%s domain_id=%s mode=%s command_center=%s clip_final_action=%s release_mcf=%s baseline=%s forward=%s"
        % (
            args.network,
            args.domain_id,
            args.mode,
            args.command_center,
            args.clip_final_action,
            args.release_mcf,
            args.baseline_onnx,
            args.forward_onnx,
        ),
        flush=True,
    )
    print("[config] policy order -> unitree order map:", POLICY_TO_UNITREE.tolist(), flush=True)

    ChannelFactoryInitialize(args.domain_id, args.network)
    # DDS 初始化后才能调用 MotionSwitcher RPC；实机 domain_id=0 默认先释放 MCF。
    release_mcf_if_needed(args)
    runner = Go2DeployRunner(args)

    def _signal_handler(signum, _frame):
        print(f"[signal] received {signum}", flush=True)
        runner.request_shutdown()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        runner.run()
    except KeyboardInterrupt:
        runner.request_shutdown()
    finally:
        runner.graceful_shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
