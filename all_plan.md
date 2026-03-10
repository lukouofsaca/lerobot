# Pika + Piper HIL-SERL 实验详细计划

> **目标**: 在 LeRobot 框架下，使用 Pika 遥操器 + Piper 机械臂，完成 HIL-SERL 强化学习训练流程。
> **初始任务**: 抓取并抬起方块 (Pick & Lift Cube)
> **动作空间**: 末端执行器 (EE) 空间
> **干预方案**: 键盘 + Pika 按钮混合（预留纯 command_state 方案）
> **相机**: Pika 自带 RealSense D405 + 外部 RealSense（预留仅外部相机方案）

---

## 第〇阶段：现状摘要与缺口分析

### 0.1 已有代码资产清单

#### LeRobot 中的 Pika/Piper 实现

| 组件 | 路径 | 状态 | 说明 |
|------|------|------|------|
| PikaTeleoperator | `src/lerobot/teleoperators/pika/pika_teleoperator.py` | ✅ 已实现 | Vive Tracker 姿态 → 内部 IK → 6 关节角度 + 夹爪 |
| PikaTeleoperatorConfig | `src/lerobot/teleoperators/pika/config_pika.py` | ✅ 已注册 | type="pika"，含 sense_port/tracker_device/gripper_xyzrpy/filter 参数 |
| PiperIK / PiperFK | `src/lerobot/teleoperators/pika/piper_ik.py` | ✅ 已实现 | CasADi + Pinocchio 求解器，含碰撞检测与关节跳变检测 |
| PiperFollower | `src/lerobot/robots/piper_follower/piper_follower.py` | ✅ 已实现 | CAN 总线 7 电机 (6 关节 + 夹爪)，支持 cameras dict |
| PiperFollowerConfig | `src/lerobot/robots/piper_follower/config_piper_follower.py` | ✅ 已注册 | type="piper_follower"，can_name="can_slave1" |
| PiperMotorsBus | `src/lerobot/motors/piper/piper.py` | ✅ 已实现 | CAN 协议封装，读写关节/夹爪位置 |
| 工厂注册 (teleop) | `src/lerobot/teleoperators/utils.py` | ✅ "pika" 已注册 | `make_teleoperator_from_config()` 可分发 |
| 工厂注册 (robot) | `src/lerobot/robots/utils.py` | ✅ "piper_follower" 已注册 | `make_robot_from_config()` 可分发 |

#### 外部代码资产

| 组件 | 路径 | 说明 |
|------|------|------|
| Pika SDK | `pika_sdk/pika/` | sense.py (串口通信 + Vive Tracker + 编码器)、gripper.py (电机控制)、camera/ (D405 + 鱼眼) |
| 多设备绑定工具 | `pika_sdk/tools/multi_device_detector.py` | 自动检测 USB 设备并生成 udev 规则 |
| CAN 激活脚本 | `pika_ros/scripts/can_activate.sh` | 按 USB 地址激活并命名 CAN 接口 |
| CAN 配置脚本 | `pika_ros/scripts/can_config.sh` | 配置双 CAN (can_left/can_right) |
| CAN 端口发现 | `pika_ros/scripts/find_all_can_port.sh` | 列出所有 CAN 接口及 USB 端口映射 |
| Piper URDF | `pika_ros/install/piper_description/share/piper_description/urdf/piper_description.urdf` | 6-DOF + 夹爪，含碰撞/视觉 mesh |
| test_pika_ik | `zyx/test_pika_ik` | FK/IK 验证脚本：零位 FK、单目标 IK、圆弧轨迹、工作空间边界 |
| test_pika_ik_viz | `zyx/test_pika_ik_viz.py` | Pika Vive Tracker → OneEuroFilter → IK → Meshcat 可视化 + Piper CAN 控制 |
| test_gripper | `zyx/test_gripper.py` | Piper 夹爪 CAN 开合控制测试 |

#### HIL-SERL 框架代码（LeRobot 已有）

| 组件 | 路径 | 说明 |
|------|------|------|
| GymManipulator 环境 | `src/lerobot/rl/gym_manipulator.py` | RobotEnv 封装 + make_processors 管线构建 |
| Actor | `src/lerobot/rl/actor.py` | gRPC 连接 learner，执行策略 rollout |
| Learner | `src/lerobot/rl/learner.py` | SAC 梯度更新，gRPC 服务端 |
| 处理器管线 | `src/lerobot/processor/` | VanillaObservation、Intervention、IK 步骤等 |
| HIL 协议 | `src/lerobot/processor/hil_processor.py` | HasTeleopEvents 协议定义 |
| IK 管线步骤 | `src/lerobot/robots/so_follower/robot_kinematic_processor.py` | EEReferenceAndDelta、EEBoundsAndSafety、InverseKinematicsRLStep |
| 关节边界发现 | `src/lerobot/scripts/find_joint_limits.py` | lerobot-find-joint-limits 入口点 |
| ROI 裁剪工具 | `src/lerobot/rl/crop_dataset_roi.py` | 交互式相机画面裁剪 |
| SAC 策略 | `src/lerobot/policies/sac/` | Soft Actor-Critic 实现 |
| 奖励分类器 | `src/lerobot/policies/reward_classifier/` | CNN/Transformer 视觉成功检测 |
| 环境配置 | `src/lerobot/envs/configs.py` | HILSerlRobotEnvConfig + 嵌套子配置 |

### 0.2 缺口分析：从"能跑 test_pika_ik_viz"到"能跑 HIL-SERL"还差什么

| # | 缺口 | 原因 | 解决方案 | 阶段 |
|---|------|------|----------|------|
| G1 | **PikaTeleoperator 未实现 HasTeleopEvents** | 当前无 `get_teleop_events()` 方法，HIL 处理器管线中 `AddTeleopEventsAsInfoStep` 和 `InterventionActionProcessorStep` 会拒绝该 teleoperator | 添加 `get_teleop_events()` → 键盘 `s/esc/r/space` + Pika `command_state` 混合方案 | 4.2.1 |
| G2 | **PikaTeleoperator 只输出关节角度，无 EE-delta 模式** | 当前 `get_action()` 内部调用 PiperIK 返回 `{joint_1..6.pos, gripper.pos}`；HIL-SERL 的 EE 空间训练需要 `{delta_x, delta_y, delta_z, gripper}` 输出，由处理器管线的 IK 步骤统一求解 | 新增 `ee_delta` 输出模式：返回 Pika tracker 位移增量而非关节角度 | 4.2.2 |
| G3 | **无设备端口持久化绑定方案** | 每次重启后 `/dev/ttyUSB*` 和 `can*` 编号可能变化 | 基于 `multi_device_detector.py` + `can_activate.sh` 编写一键脚本 | 1 |
| G4 | **无独立的 Pika 遥操逻辑测试** | test_pika_ik_viz.py 混合了 IK/Meshcat/Piper CAN 多个关注点，无法单独验证 Pika SDK 层是否正常 | 新建 `test_pika_teleop.py` 分层测试 | 2 |
| G5 | **无 LeRobot 框架下的遥操验证 example** | test_pika_ik_viz.py 直接调用底层 API，绕过了 LeRobot 的 Teleoperator/Robot 抽象 | 新建 `examples/pika_piper_teleop_viz.py`，在 LeRobot 抽象层复现并与旧脚本对比 | 3 |
| G6 | **PiperFollower 相机配置未填写** | `PiperFollowerConfig.cameras` 默认为空 dict，未配置 wrist D405 和外部 RealSense | JSON 配置中填入相机序列号和分辨率 | 4.2.3 |
| G7 | **IK 管线步骤（EEReferenceAndDelta 等）仅在 SO100 上验证过** | `robot_kinematic_processor.py` 中的 IK 步骤使用 `placo` 求解器和 SO100 URDF；Piper 使用 CasADi/Pinocchio 求解器和不同 URDF | 需确认 `InverseKinematicsRLStep` 是否可配置为使用 Piper URDF，或需要编写 Piper 专用 IK 步骤 | 4.2.4 |
| G8 | **无 Pika+Piper 专用 HIL-SERL 配置文件** | `env_config.json` 和 `train_config.json` 模板针对 SO100 | 编写 `env_config_pika_piper.json` + `train_config_pika_piper.json` | 4.3 |
| G9 | **Piper 的 EE workspace bounds 未标定** | `lerobot-find-joint-limits` 需要适配 Pika+Piper 组合 | 用 Pika 遥操 Piper 执行 find-joint-limits 流程 | 4.4.1 |
| G10 | **control_mode 无 "pika" 选项** | 当前仅 "gamepad" / "leader"，虽代码中未用此字段做条件分支，但语义上应匹配 | 在配置中使用 "pika" 并确认不影响管线 | 4.2.5 |

---

## 第一阶段：设备端口绑定方案

### 1.1 设备拓扑与端口清单

```
┌─────────────────────────────────────────────────────────┐
│                    主机 (Ubuntu)                         │
│                                                         │
│  USB Hub                                                │
│  ├── Pika Sense ─── /dev/ttyUSB0 → 绑定 /dev/ttyUSB80  │
│  ├── Pika Gripper ─ /dev/ttyUSB1 → 绑定 /dev/ttyUSB81  │
│  ├── RealSense D405 (wrist) ─── 按序列号识别            │
│  ├── RealSense D4xx (external) ── 按序列号识别          │
│  └── USB-CAN Adapter ─── can0 → 绑定 can_slave1        │
│                                                         │
│  Vive Lighthouse × 2 ─── pysurvive 库自动发现           │
└─────────────────────────────────────────────────────────┘
```

**物理连线清单**（由人类操作员完成）：

| 设备 | 接口 | 默认端口 | 绑定后端口 | 备注 |
|------|------|----------|------------|------|
| Pika Sense | USB-Serial (460800 baud) | `/dev/ttyUSB*` | `/dev/ttyUSB80` | 含 Vive Tracker 接收器 + 夹爪编码器 |
| Pika Gripper | USB-Serial (460800 baud) | `/dev/ttyUSB*` | `/dev/ttyUSB81` | 电机驱动，本方案暂不使用（用 Piper 夹爪） |
| Piper 机械臂 | USB-CAN (1Mbps) | `can*` | `can_slave1` | 6 关节 + 夹爪，需 `sudo` 权限激活 |
| RealSense D405 (wrist) | USB 3.0 | 自动 | 按序列号 | 安装在 Pika/Piper 腕部 |
| RealSense D4xx (external) | USB 3.0 | 自动 | 按序列号 | 固定于工作台上方/侧方 |
| Vive Base Station × 2 | 电源 | N/A | N/A | pysurvive 自动发现，无需端口配置 |

### 1.2 CAN 总线配置

**前置条件**（由人类操作员确认）：
- `sudo apt install can-utils ethtool` 已安装
- USB-CAN 适配器已插入

**步骤 1：发现 CAN 端口的 USB 地址**

```bash
cd ~/pika_ros/scripts && bash find_all_can_port.sh
# 输出示例：接口 can0 插入在 USB 端口 1-1.4:1.0
```

记录输出的 USB 端口地址（如 `1-1.4:1.0`）。

**步骤 2：激活并绑定 CAN 接口**

```bash
# 单臂场景（本方案默认）
sudo bash can_activate.sh can_slave1 1000000 "1-1.4:1.0"
```

若使用双臂，编辑 `can_config.sh`：
```bash
bash can_activate.sh can_slave1 1000000 "实际USB地址1"
bash can_activate.sh can_slave2 1000000 "实际USB地址2"
```

**步骤 3：验证**

```bash
candump can_slave1 -n 5   # 应能看到 Piper 心跳帧
ip link show can_slave1    # 状态应为 UP
```

**持久化**：将步骤 2 的命令写入 `setup_devices.sh`（见 1.5）。CAN 接口在每次重启后需重新激活，udev 规则无法自动完成此步骤。

### 1.3 USB 串口设备持久化绑定

**方案**：使用 `pika_sdk/tools/multi_device_detector.py` 生成 udev 规则，将 Pika Sense/Gripper 绑定到固定设备路径。

**步骤 1：运行检测工具**（由人类操作员交互完成）

```bash
cd ~/sda/zyx/pika_sdk && python tools/multi_device_detector.py
```

流程：
1. 输入设备数量（2：Sense + Gripper）
2. 依次插入每个设备 → 按 Enter → 工具自动检测 ttyUSB 路径和 RealSense 序列号
3. 鱼眼相机选择阶段按 `q` 跳过（或 `s` 选择）
4. 生成 `setup.bash` 和 `devices_info.conf`

**步骤 2：安装 udev 规则**

```bash
# setup.bash 中包含类似以下规则：
# SUBSYSTEM=="tty", ATTRS{serial}=="xxxx", SYMLINK+="ttyUSB80"
sudo bash setup.bash
sudo udevadm control --reload-rules && sudo udevadm trigger
```

**绑定结果**：

| 设备 | 绑定路径 | 用途 |
|------|----------|------|
| Pika Sense | `/dev/ttyUSB80` | PikaTeleoperatorConfig.sense_port |
| Pika Gripper | `/dev/ttyUSB81` | 备用，本方案暂不直接使用 |

**步骤 3：验证**

```bash
ls -la /dev/ttyUSB80 /dev/ttyUSB81  # 应存在符号链接
python -c "from pika import sense; s = sense(port='/dev/ttyUSB80'); s.connect(); print('OK'); s.disconnect()"
```

### 1.4 RealSense 相机绑定

**方案**：通过序列号在配置文件中硬编码区分。LeRobot 的 `IntelRealSenseConfig` 和 Pika SDK 的 `RealSenseCamera` 均支持按序列号打开指定相机。

**步骤 1：获取序列号**

```bash
rs-enumerate-devices | grep "Serial Number"
# 输出示例：
# Serial Number: 234322300xxx  (D405 wrist)
# Serial Number: 234322300yyy  (D435 external)
```

**步骤 2：记录并填入配置**

| 相机 | 序列号 | 配置位置 | 分辨率 |
|------|--------|----------|--------|
| D405 (wrist) | `填入实际值` | `env_config_pika_piper.json` → `env.robot.cameras.wrist` | 640×480@30fps |
| D4xx (external) | `填入实际值` | `env_config_pika_piper.json` → `env.robot.cameras.front` | 640×480@30fps |

**注意**：Pika SDK 的 `set_realsense_serial_number()` 和 LeRobot 的 `IntelRealSenseConfig.serial_number` 是两条独立路径。在 HIL-SERL 流程中，相机由 `PiperFollower` 的 cameras 配置管理，不通过 Pika SDK。

### 1.5 一键启动脚本

**AGI 任务**：创建 `pika_ros/scripts/setup_pika_piper.sh`

```bash
#!/bin/bash
set -e

echo "=== [1/4] 检查 udev 规则 ==="
if [ ! -L /dev/ttyUSB80 ]; then
    echo "ERROR: /dev/ttyUSB80 不存在，请先运行 multi_device_detector.py 并安装 udev 规则"
    exit 1
fi
echo "Pika Sense: /dev/ttyUSB80 ✓"

echo "=== [2/4] 激活 CAN 总线 ==="
# 需要 sudo 密码，或已配置 NOPASSWD
CAN_USB_ADDR="${CAN_USB_ADDR:-1-1.4:1.0}"  # 默认值，由用户覆盖
sudo bash $(dirname $0)/can_activate.sh can_slave1 1000000 "$CAN_USB_ADDR"
echo "CAN can_slave1 ✓"

echo "=== [3/4] 检查 RealSense ==="
RS_COUNT=$(rs-enumerate-devices 2>/dev/null | grep -c "Serial Number" || true)
if [ "$RS_COUNT" -lt 1 ]; then
    echo "WARNING: 未检测到 RealSense 相机"
else
    echo "RealSense 相机数量: $RS_COUNT ✓"
fi

echo "=== [4/4] CAN 心跳检测 ==="
timeout 2 candump can_slave1 -n 1 > /dev/null 2>&1 && echo "Piper CAN 心跳 ✓" || echo "WARNING: CAN 无心跳，请检查 Piper 电源"

echo "=== 设备就绪 ==="
```

**使用方式**（由人类操作员执行）：

```bash
# 首次使用需设置实际 USB 地址
export CAN_USB_ADDR="1-1.4:1.0"  # 替换为 find_all_can_port.sh 输出的地址
bash ~/pika_ros/scripts/setup_pika_piper.sh
```

---

## 第二阶段：Pika 遥操逻辑独立测试

### 2.1 测试目标与验收标准

本阶段在 **纯 pika_sdk 层面** 验证 Pika 硬件通信链路。不涉及 LeRobot、IK、Piper。

| 测试项 | 验收标准 |
|--------|----------|
| Sense 连接 | `sense.connect()` 无异常，`sense.get_encoder_data()` 返回 dict 含 `angle` 和 `rad` |
| Vive Tracker 姿态 | `sense.get_pose(device_name)` 返回 PoseData，position 三轴范围合理 (±2m)，quaternion 模长 ≈ 1.0 |
| 姿态连续性 | 连续 100 帧，相邻帧位置差 < 50mm（静止时 < 2mm） |
| 夹爪编码器 | `sense.get_gripper_distance()` 范围 0-90mm，开合响应延迟 < 100ms |
| command_state | `sense.get_command_state()` 返回 0 或 1，双击夹爪后状态切换 |
| 断开重连 | `disconnect()` 后 `connect()` 无异常 |

### 2.2 test_pika_teleop.py 设计

**AGI 任务**：创建 `zyx/test_pika_teleop.py`

```python
"""Pika Sense 遥操逻辑独立测试。
验证 Vive Tracker 姿态、夹爪编码器、command_state 按钮。
不依赖 LeRobot / IK / Piper。

用法：
    python test_pika_teleop.py --sense-port /dev/ttyUSB80 --tracker-device T20
"""

import argparse, time, math
import numpy as np
from pika import sense
from pika.tracker.pose_utils import xyzQuaternion2matrix

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sense-port", default="/dev/ttyUSB80")
    p.add_argument("--tracker-device", default="T20")
    p.add_argument("--duration", type=float, default=10.0, help="测试运行时长(秒)")
    p.add_argument("--max-retries", type=int, default=10, help="等待 Tracker 最大重试次数")
    return p.parse_args()

def test_connection(s: sense):
    """T1: 基本连接"""
    s.connect()
    enc = s.get_encoder_data()
    assert "angle" in enc and "rad" in enc, f"编码器数据格式异常: {enc}"
    print(f"[PASS] 连接成功, encoder angle={enc['angle']:.2f}°, rad={enc['rad']:.4f}")

def wait_for_tracker(s: sense, device: str, max_retries: int):
    """T2: 等待 Vive Tracker 设备出现"""
    for i in range(max_retries):
        devices = s.get_tracker_devices()
        if device in devices:
            print(f"[PASS] Tracker {device} 已发现 (第 {i+1} 次尝试)")
            return
        print(f"  等待 Tracker... ({i+1}/{max_retries})")
        time.sleep(1.0)
    raise TimeoutError(f"Tracker {device} 未在 {max_retries}s 内出现")

def test_pose_validity(s: sense, device: str):
    """T3: 单帧姿态有效性"""
    pose = s.get_pose(device)
    pos = np.array(pose.position)
    quat = np.array(pose.rotation)
    quat_norm = np.linalg.norm(quat)
    assert np.all(np.abs(pos) < 2.0), f"位置超范围: {pos}"
    assert abs(quat_norm - 1.0) < 0.01, f"四元数模长异常: {quat_norm}"
    mat = xyzQuaternion2matrix(*pos, *quat)
    assert mat.shape == (4, 4), f"变换矩阵形状异常: {mat.shape}"
    print(f"[PASS] 姿态有效: pos={pos}, |q|={quat_norm:.4f}")

def test_pose_continuity(s: sense, device: str, n_frames=100, dt=0.033):
    """T4: 姿态连续性（静止状态）"""
    positions = []
    for _ in range(n_frames):
        pose = s.get_pose(device)
        positions.append(np.array(pose.position))
        time.sleep(dt)
    positions = np.array(positions)
    diffs = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    max_diff = diffs.max() * 1000  # mm
    mean_diff = diffs.mean() * 1000
    print(f"[INFO] 帧间位移: max={max_diff:.1f}mm, mean={mean_diff:.1f}mm")
    assert max_diff < 50.0, f"帧间跳变过大: {max_diff:.1f}mm"
    print(f"[PASS] 姿态连续性 OK")

def test_gripper(s: sense):
    """T5: 夹爪编码器"""
    dist = s.get_gripper_distance()
    assert 0.0 <= dist <= 90.0, f"夹爪距离超范围: {dist}mm"
    print(f"[PASS] 夹爪距离: {dist:.1f}mm")

def test_command_state(s: sense, timeout=5.0):
    """T6: command_state（需人工双击夹爪触发）"""
    state0 = s.get_command_state()
    print(f"[INFO] 当前 command_state={state0}")
    print(f"  请在 {timeout}s 内双击夹爪触发状态切换...")
    t0 = time.time()
    while time.time() - t0 < timeout:
        state = s.get_command_state()
        if state != state0:
            print(f"[PASS] command_state 已切换: {state0} -> {state}")
            return
        time.sleep(0.05)
    print(f"[SKIP] command_state 未在 {timeout}s 内变化（可能API不支持此检测方式）")

def test_reconnect(s: sense, port: str):
    """T7: 断开重连"""
    s.disconnect()
    time.sleep(0.5)
    s2 = sense(port=port)
    s2.connect()
    enc = s2.get_encoder_data()
    assert "angle" in enc
    s2.disconnect()
    print("[PASS] 断开重连成功")

def main():
    args = parse_args()
    s = sense(port=args.sense_port)

    print("=" * 60)
    print("Pika Sense 遥操逻辑独立测试")
    print("=" * 60)

    test_connection(s)
    wait_for_tracker(s, args.tracker_device, args.max_retries)
    test_pose_validity(s, args.tracker_device)
    test_pose_continuity(s, args.tracker_device)
    test_gripper(s)
    test_command_state(s)
    test_reconnect(s, args.sense_port)

    print("=" * 60)
    print("所有测试通过 ✓")
    print("=" * 60)

if __name__ == "__main__":
    main()
```

**设计要点**：
- 每个 `test_*` 函数对应一个验收标准，独立可执行
- `test_command_state` 需要人工交互，设超时后 SKIP 不阻塞
- 不依赖任何 LeRobot 代码，仅用 `pika` 和 `numpy`

### 2.3 测试用例清单

| ID | 函数 | 前置 | 输入 | 预期结果 |
|----|------|------|------|----------|
| T1 | `test_connection` | Sense 已插入 | 无 | `connect()` 无异常，encoder dict 含 `angle`+`rad` |
| T2 | `wait_for_tracker` | Vive Lighthouse 已开机、Tracker 已开机 | device="T20" | 在 max_retries 内发现设备 |
| T3 | `test_pose_validity` | T2 通过 | 单帧姿态 | pos 各轴 < 2m，四元数模长 ≈ 1.0，变换矩阵 4×4 |
| T4 | `test_pose_continuity` | T2 通过，设备静止 | 100帧@30Hz | 相邻帧位置差 < 50mm |
| T5 | `test_gripper` | T1 通过 | 无 | 返回 0-90mm 范围内浮点数 |
| T6 | `test_command_state` | T1 通过 | 人工双击夹爪 | 状态从 0→1 或 1→0，超时则 SKIP |
| T7 | `test_reconnect` | 任意 | 无 | 断开后重连成功，编码器可读 |

### 2.4 执行步骤

**AGI 执行**：
1. 创建 `zyx/test_pika_teleop.py`，内容如 2.2 所示
2. 确认 `pika` 包已安装：`pip show agx-pypika`，若未安装则 `cd ~/sda/zyx/pika_sdk && pip install -e .`

**人类操作员执行**：
1. 确保 Pika Sense 已通过 USB 连接，Vive Lighthouse 已开机
2. 运行测试：
   ```bash
   cd ~/sda/zyx && python test_pika_teleop.py --sense-port /dev/ttyUSB80 --tracker-device T20
   ```
3. 在 T6 提示时双击夹爪触发 command_state
4. 确认所有测试 PASS（T6 允许 SKIP）

**失败处理**：
- T1 失败 → 检查端口路径、USB 线、`ls /dev/ttyUSB*`
- T2 失败 → 检查 Lighthouse 电源、Tracker 电量、pysurvive 安装
- T4 失败 → Tracker 报点报号不对，或设备未固定静止
- T5 失败 → 编码器硬件问题或未开机

---

## 第三阶段：LeRobot 框架下 Pika→Piper 遥操 + Meshcat 对比验证

### 3.1 目标：在 lerobot example 中复现 test_pika_ik_viz 的效果

在 LeRobot 的 `Teleoperator` / `Robot` 抽象层下跑通 Pika→Piper 遥操，同时开 Meshcat 可视化对比。

**为什么需要这一步**：
- `test_pika_ik_viz.py` 直接调用 `pika.sense` + `PiperIK` + `C_PiperInterface_V2`，绕过了 LeRobot 抽象
- LeRobot 的 `PikaTeleoperator` 封装了相同逻辑但增加了校准/滤波/错误处理
- 需要验证 `PikaTeleoperator.get_action()` → `PiperFollower.send_action()` 全链路与裸调用行为一致
- Meshcat 可视化提供直观的对比手段

### 3.2 PikaTeleoperator 现有实现审查

**当前 `get_action()` 数据流**：

```
Pika Tracker pose (SE3)
    → initial_arm @ inv(initial_pika) @ current_pika    # 坐标变换
    → OneEuroFilter (6D: xyz+rpy)                       # 滤波
    → PiperIK.ik_fun(filtered_matrix, gripper)          # CasADi IK 求解
    → {joint_1..6.pos: float, gripper.pos: encoder_rad} # 返回关节角度
```

**关键发现**：
1. **当前输出是关节角度**，不是 EE delta。这对"关节空间遥操"是正确的。
2. **IK 在 teleoperator 内部完成**，而 HIL-SERL 的 EE 空间训练模式需要 IK 在处理器管线（`InverseKinematicsRLStep`）中完成。
3. **第三阶段先用关节空间模式**验证全链路，第四阶段再改为 EE-delta 模式。

**本阶段无需修改 PikaTeleoperator**：直接使用现有 `get_action()` → 关节角度 → `PiperFollower.send_action()`。

### 3.3 新增 examples/pika_piper_teleop_viz.py

**AGI 任务**：创建 `lerobot/examples/pika_piper_teleop_viz.py`

```python
"""LeRobot 框架下 Pika→Piper 遥操 + Meshcat 可视化验证。

使用 LeRobot 的 Teleoperator/Robot 抽象层，验证与 test_pika_ik_viz.py 行为一致。

用法：
    python -m lerobot.examples.pika_piper_teleop_viz \
        --teleop.sense_port /dev/ttyUSB80 \
        --teleop.tracker_device T20 \
        --teleop.piper_description_dir /path/to/piper_description \
        --robot.motors.can_name can_slave1 \
        --fps 30 \
        --duration 60 \
        --viz                    # 启用 Meshcat
        --no-send-to-robot       # 可选：只看可视化，不发送到真实 Piper
"""
import argparse
import time
import numpy as np

def parse_args():
    p = argparse.ArgumentParser()
    # teleop 参数
    p.add_argument("--sense-port", default="/dev/ttyUSB80")
    p.add_argument("--tracker-device", default="T20")
    p.add_argument("--piper-description-dir", required=True)
    p.add_argument("--home-joint-state", nargs=6, type=float, default=[0.0]*6)
    p.add_argument("--lift", action="store_true")
    # robot 参数
    p.add_argument("--can-name", default="can_slave1")
    # 运行参数
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--viz", action="store_true", help="启用 Meshcat 可视化")
    p.add_argument("--no-send-to-robot", action="store_true", help="不发送命令到 Piper")
    return p.parse_args()

def setup_meshcat(urdf_path: str, package_dirs: list[str]):
    """初始化 Meshcat + Pinocchio 可视化"""
    import pinocchio as pin
    from pinocchio.visualize import MeshcatVisualizer

    model, collision_model, visual_model = pin.buildModelsFromUrdf(
        urdf_path, package_dirs=package_dirs
    )
    viz = MeshcatVisualizer(model, collision_model, visual_model)
    viz.initViewer(open=True)
    viz.loadViewerModel()
    return viz, model

def add_frame_marker(viz_handle, name: str, scale: float = 0.1):
    """在 Meshcat 中添加坐标轴标记"""
    import meshcat.geometry as g
    import meshcat.transformations as tf

    colors = [(0xff0000, [scale,0,0]), (0x00ff00, [0,scale,0]), (0x0000ff, [0,0,scale])]
    for color, direction in colors:
        cyl = g.Cylinder(scale * 0.02, scale)
        viz_handle[name][f"axis_{color:06x}"].set_object(
            cyl, g.MeshLambertMaterial(color=color)
        )

def main():
    args = parse_args()

    # ---- 1. 初始化 LeRobot Teleoperator ----
    from lerobot.teleoperators.pika.config_pika import PikaTeleoperatorConfig
    from lerobot.teleoperators.pika.pika_teleoperator import PikaTeleoperator

    teleop_cfg = PikaTeleoperatorConfig(
        sense_port=args.sense_port,
        tracker_device=args.tracker_device,
        piper_description_dir=args.piper_description_dir,
        home_joint_state=args.home_joint_state,
        lift=args.lift,
    )
    teleop = PikaTeleoperator(teleop_cfg)
    teleop.connect(calibrate=True)
    print("[OK] PikaTeleoperator 已连接并校准")

    # ---- 2. 初始化 LeRobot Robot (可选) ----
    robot = None
    if not args.no_send_to_robot:
        from lerobot.robots.piper_follower.config_piper_follower import PiperFollowerConfig
        from lerobot.robots.piper_follower.piper_follower import PiperFollower

        robot_cfg = PiperFollowerConfig()
        robot_cfg.motors.can_name = args.can_name
        robot = PiperFollower(robot_cfg)
        robot.connect()
        print("[OK] PiperFollower 已连接")

    # ---- 3. 初始化 Meshcat (可选) ----
    viz = None
    if args.viz:
        from pathlib import Path
        desc_dir = Path(args.piper_description_dir)
        urdf_name = "piper_description-lift.urdf" if args.lift else "piper_description.urdf"
        urdf_path = str(desc_dir / "urdf" / urdf_name)
        package_dirs = [str(desc_dir), str(desc_dir.parent)]
        viz, pin_model = setup_meshcat(urdf_path, package_dirs)
        print("[OK] Meshcat 已启动")

    # ---- 4. 主循环 ----
    dt = 1.0 / args.fps
    t_start = time.time()
    step = 0

    print(f"开始遥操 ({args.duration}s, {args.fps}Hz)...")
    try:
        while time.time() - t_start < args.duration:
            t_loop = time.time()

            # 获取 teleop 动作 (关节角度)
            action = teleop.get_action()
            joint_positions = [action[f"joint_{i}.pos"] for i in range(1, 7)]
            gripper_pos = action["gripper.pos"]

            # 发送到 Piper
            if robot is not None:
                robot.send_action(action)

            # Meshcat 可视化
            if viz is not None:
                import pinocchio as pin
                q_viz = np.array(joint_positions + [0.0, 0.0])  # +2 locked gripper joints
                viz.display(q_viz)

            # 日志
            if step % args.fps == 0:
                pos_str = ", ".join(f"{v:.3f}" for v in joint_positions)
                print(f"[{step:5d}] joints=[{pos_str}], gripper={gripper_pos:.3f}")

            step += 1
            elapsed = time.time() - t_loop
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        teleop.disconnect()
        if robot is not None:
            robot.disconnect()
        print("[OK] 已断开所有设备")

if __name__ == "__main__":
    main()
```

**设计要点**：
- 使用 LeRobot 的 `PikaTeleoperator` 和 `PiperFollower` 类，而非直接调用底层 API
- `--no-send-to-robot` 模式允许先只验证可视化，不操作真实机械臂
- `--viz` 模式开启 Meshcat，与 `test_pika_ik_viz.py` 的可视化效果直接对比
- 日志每秒打印一次关节状态

### 3.4 验收标准

| 检查项 | 标准 | 验证方法 |
|--------|------|----------|
| 全链路连通 | example 脚本启动无报错，日志打印关节值 | 运行 `--no-send-to-robot` 模式 |
| Meshcat 目标帧 | 移动 Pika 时，机械臂模型跟随运动，无明显延迟 | 目视 Meshcat 页面 |
| 与裸脚本一致性 | 同一姿态下，两个脚本输出的关节角度差异 < 0.01 rad | 同时运行两个脚本，比较日志 |
| Piper 真机逼近 | 发花给 Piper 后实际末端位置与 Meshcat 目标误差 < 5mm | 视觉对比 + FK 计算 |
| 夹爪响应 | Pika 夹爪开合与 Piper 夹爪同步 | 目视 |

### 3.5 执行步骤

**AGI 执行**：
1. 创建 `lerobot/examples/pika_piper_teleop_viz.py`，内容如 3.3 所示
2. 确认依赖已安装：`pip install meshcat pinocchio`

**人类操作员执行**：

```bash
# 1. 确保设备就绪
bash ~/pika_ros/scripts/setup_pika_piper.sh

# 2. 纯可视化模式（不操作 Piper）
cd ~/sda/zyx/lerobot
python examples/pika_piper_teleop_viz.py \
    --sense-port /dev/ttyUSB80 \
    --tracker-device T20 \
    --piper-description-dir ~/pika_ros/install/piper_description/share/piper_description \
    --viz --no-send-to-robot

# 3. 打开 Meshcat URL (http://127.0.0.1:7001/static/)
# 4. 移动 Pika 观察机械臂模型跟随

# 5. 确认无误后，启动真机模式
python examples/pika_piper_teleop_viz.py \
    --sense-port /dev/ttyUSB80 \
    --tracker-device T20 \
    --piper-description-dir ~/pika_ros/install/piper_description/share/piper_description \
    --can-name can_slave1 \
    --viz
```

**对比验证**：同时在另一终端运行 `test_pika_ik_viz.py`，比较两者的 Meshcat 显示和日志输出。

---

## 第四阶段：HIL-SERL 全流程适配

### 4.1 架构总览

**HIL-SERL 训练时的完整数据流**：

```
┌────────────────────── Actor 进程 ──────────────────────────────────────────────────────┐
│                                                                                        │
│  ┌─ PikaTeleoperator (改造后) ────────────────────────────────┐                        │
│  │  Vive Tracker → 坐标变换 → OneEuroFilter → EE pose (SE3)  │                        │
│  │  get_action() → {delta_x, delta_y, delta_z, gripper}      │ ← 新增 EE-delta 模式  │
│  │  get_teleop_events() → {is_intervention, success, ...}    │ ← 新增 HasTeleopEvents │
│  └───────────────────┬────────────────────────────────────────┘                        │
│                      │                                                                  │
│                      ▼ action_processor 管线                                            │
│  ┌─────────────────────────────────────────────────────────────────┐                    │
│  │ 1. AddTeleopActionAsComplimentaryDataStep                      │                    │
│  │ 2. AddTeleopEventsAsInfoStep                                   │                    │
│  │ 3. InterventionActionProcessorStep                             │                    │
│  │    └─ 若 is_intervention=True → 用 teleop 动作替换 policy 动作 │                    │
│  │ 4. MapTensorToDeltaActionDictStep    ← EE delta 解包           │                    │
│  │ 5. MapDeltaActionToRobotActionStep   ← delta→enabled/target    │                    │
│  │ 6. EEReferenceAndDelta               ← FK + 增量 → 绝对 EE     │                    │
│  │ 7. EEBoundsAndSafety                 ← 工作空间裁剪             │                    │
│  │ 8. GripperVelocityToJoint            ← 夹爪速度→位置            │                    │
│  │ 9. InverseKinematicsRLStep           ← EE→关节 (Piper URDF)    │                    │
│  │ 10. RobotActionToPolicyActionProcessorStep                     │                    │
│  └───────────────────┬────────────────────────────────────────────┘                    │
│                      │ {joint_1..6.pos, gripper.pos}                                    │
│                      ▼                                                                  │
│  ┌─ PiperFollower ──────────────────────────────────────────┐                          │
│  │  CAN bus → 6 关节 + 夹爪位置命令                          │                          │
│  │  get_observation() → {joint states, camera images}       │                          │
│  │  cameras: wrist (D405) + front (external D4xx)           │                          │
│  └───────────────────┬──────────────────────────────────────┘                          │
│                      │ transition = {obs, action, reward, done, info}                   │
│                      ▼                                                                  │
│               gRPC → Learner 进程                                                       │
└────────────────────────────────────────────────────────────────────────────────────────┘

┌────────────────────── Learner 进程 ──────────────────────────┐
│  replay buffer ← transition                                  │
│  SAC policy update (batch gradient descent)                  │
│  policy weights → gRPC → Actor 进程                          │
│  WandB logging                                               │
└──────────────────────────────────────────────────────────────┘
```

**关键适配点**（标注 ← 的步骤）：

1. PikaTeleoperator 需要新增 `get_teleop_events()` 和 EE-delta 输出模式
2. IK 管线步骤 (4-9) 需要配置 Piper 的 URDF 和工作空间参数
3. PiperFollower 需要配置相机

### 4.2 适配任务分解

#### 4.2.1 PikaTeleoperator 实现 HasTeleopEvents 协议

**要改的文件**：`src/lerobot/teleoperators/pika/pika_teleoperator.py`

**HasTeleopEvents 协议要求**（定义在 `src/lerobot/processor/hil_processor.py`）：

```python
@runtime_checkable
class HasTeleopEvents(Protocol):
    def get_teleop_events(self) -> dict[str, Any]:
        # 必须返回:
        # TeleopEvents.IS_INTERVENTION: bool      - 是否正在干预
        # TeleopEvents.TERMINATE_EPISODE: bool     - 是否终止 episode
        # TeleopEvents.SUCCESS: bool               - 是否标记成功
        # TeleopEvents.RERECORD_EPISODE: bool      - 是否重录
```

**实现方案：键盘 + Pika command_state 混合**

```python
# === 新增导入 ===
import threading
from lerobot.processor.hil_processor import TeleopEvents

# === 在 PikaTeleoperator.__init__ 中新增 ===
self._keyboard_listener = None
self._intervention_active = False
self._episode_success = False
self._episode_terminate = False
self._rerecord = False

# === 新增方法 ===
def _start_keyboard_listener(self):
    """非阻塞键盘监听线程（使用 pynput 或 select/termios）"""
    try:
        from pynput import keyboard

        def on_press(key):
            try:
                if key == keyboard.Key.space:
                    self._intervention_active = not self._intervention_active
                elif hasattr(key, 'char'):
                    if key.char == 's':
                        self._episode_success = True
                        self._episode_terminate = True
                    elif key.char == 'r':
                        self._rerecord = True
                    elif key.char == 'q':
                        self._episode_terminate = True
            except AttributeError:
                pass

        self._keyboard_listener = keyboard.Listener(on_press=on_press)
        self._keyboard_listener.daemon = True
        self._keyboard_listener.start()
    except ImportError:
        logger.warning("pynput not installed, keyboard events disabled. Install with: pip install pynput")

def get_teleop_events(self) -> dict[str, bool]:
    """HasTeleopEvents 协议实现。"""
    # Pika command_state 作为备用干预信号
    pika_cmd = False
    if self.sense_device is not None:
        try:
            pika_cmd = bool(self.sense_device.get_command_state())
        except Exception:
            pass

    events = {
        TeleopEvents.IS_INTERVENTION: self._intervention_active or pika_cmd,
        TeleopEvents.TERMINATE_EPISODE: self._episode_terminate,
        TeleopEvents.SUCCESS: self._episode_success,
        TeleopEvents.RERECORD_EPISODE: self._rerecord,
    }

    # 读取后重置一次性事件
    self._episode_success = False
    self._episode_terminate = False
    self._rerecord = False

    return events
```

**在 `connect()` 末尾启动键盘监听**：

```python
def connect(self, calibrate: bool = True) -> None:
    # ... 现有代码 ...
    self._start_keyboard_listener()
    logger.info(f"{self} connected")
```

**在 `disconnect()` 中停止监听**：

```python
def disconnect(self) -> None:
    if self._keyboard_listener is not None:
        self._keyboard_listener.stop()
        self._keyboard_listener = None
    # ... 现有代码 ...
```

**按键映射**：

| 按键 | 功能 | 说明 |
|------|------|------|
| `Space` | Toggle intervention | 按一次开始干预，再按一次结束 |
| `s` | 标记成功 | 同时触发 terminate |
| `Esc` / `q` | 标记失败 | 仅 terminate |
| `r` | 重录当前 episode | |
| Pika 双击夹爪 | 备用 intervention | 通过 `command_state` 检测 |

**依赖**：`pip install pynput`（已在 lerobot hilserl extras 中）

#### 4.2.2 PikaTeleoperator 新增 EE-delta 输出模式

**要改的文件**：
- `src/lerobot/teleoperators/pika/config_pika.py`（新增配置项）
- `src/lerobot/teleoperators/pika/pika_teleoperator.py`（新增 EE-delta 逻辑）

**设计思路**：

当前 `get_action()` 在 teleoperator 内部完成 IK → 输出关节角度。HIL-SERL EE 空间训练需要：
- teleoperator 输出 EE 位移增量 `{delta_x, delta_y, delta_z, gripper}`
- 处理器管线的 IK 步骤 (`InverseKinematicsRLStep`) 完成 EE→关节的求解

新增一个 `output_mode` 配置项控制切换。

**config_pika.py 修改**：

```python
@TeleoperatorConfig.register_subclass("pika")
@dataclass
class PikaTeleoperatorConfig(TeleoperatorConfig):
    # ... 现有字段 ...
    output_mode: str = "joint"  # 新增: "joint" (关节角度) 或 "ee_delta" (EE 增量)
```

**pika_teleoperator.py 修改**：

```python
# === action_features 根据模式切换 ===
@property
def action_features(self) -> dict[str, type]:
    if self.config.output_mode == "ee_delta":
        return {"delta_x": float, "delta_y": float, "delta_z": float, "gripper": float}
    return {**{f"joint_{i}.pos": float for i in range(1, 7)}, "gripper.pos": float}

# === __init__ 中新增 ===
self._prev_ee_position = None  # 用于计算位移增量

# === get_action() 中新增 ee_delta 分支 ===
def get_action(self) -> dict[str, float]:
    # ... 现有坐标变换和滤波代码 ...
    # 已得到 filtered_target_matrix (4x4 SE3)

    if self.config.output_mode == "ee_delta":
        return self._get_ee_delta_action(filtered_target_matrix)
    else:
        return self._get_joint_action(filtered_target_matrix)  # 现有逻辑

def _get_ee_delta_action(self, target_matrix: np.ndarray) -> dict[str, float]:
    """EE-delta 模式：返回位移增量"""
    target_pos = target_matrix[:3, 3]  # x, y, z

    if self._prev_ee_position is None:
        self._prev_ee_position = target_pos.copy()
        delta = np.zeros(3)
    else:
        delta = target_pos - self._prev_ee_position
        self._prev_ee_position = target_pos.copy()

    # 夹爪：0=闭合, 1=打开 (归一化)
    gripper_mm = self.sense_device.get_gripper_distance()
    gripper_normalized = float(gripper_mm) / 90.0  # 0-90mm → 0-1

    return {
        "delta_x": float(delta[0]),
        "delta_y": float(delta[1]),
        "delta_z": float(delta[2]),
        "gripper": gripper_normalized,
    }

def _get_joint_action(self, filtered_target_matrix: np.ndarray) -> dict[str, float]:
    """关节模式：现有逻辑提取为方法"""
    # ... 原 get_action() 的 IK 求解部分 ...
```

**在 `calibrate()` 中重置 delta 基准**：

```python
def calibrate(self) -> None:
    # ... 现有代码 ...
    self._prev_ee_position = None  # 重置 delta 基准
```

**注意事项**：
- EE-delta 模式下不需要 IK solver，但仍需 FK solver 计算初始 EE 位置
- delta 值的量级约为 ±0.02m/帧@30Hz（正常手部移动速度）
- `gripper` 输出归一化到 [0, 1] 以匹配 `GripperVelocityToJoint` 步骤预期

#### 4.2.3 PiperFollower 相机集成

**要改的文件**：无代码修改，仅配置文件

`PiperFollowerConfig.cameras` 已支持 `dict[str, CameraConfig]`，只需在 JSON 配置中填写。

LeRobot 的 `IntelRealSenseConfig` 注册为 `type="intel_realsense"`，字段包括：
- `serial_number: str | None`
- `width: int = 640`
- `height: int = 480`
- `fps: int = 30`

**配置示例**（将在 4.3.1 中完整给出）：

```json
{
  "robot": {
    "type": "piper_follower",
    "cameras": {
      "wrist": {
        "type": "intel_realsense",
        "serial_number": "填入D405序列号",
        "width": 640,
        "height": 480,
        "fps": 30
      },
      "front": {
        "type": "intel_realsense",
        "serial_number": "填入外部相机序列号",
        "width": 640,
        "height": 480,
        "fps": 30
      }
    },
    "motors": {
      "can_name": "can_slave1"
    }
  }
}
```

**预留方案：仅外部相机**

去掉 `wrist` 相机配置即可。原始 HIL-SERL 论文 (hil-serl-sim / gym-hil) 多使用固定外部相机。wrist 相机在某些 pick-and-place 任务上有优势但增加了遮挡风险。

**验证**：在第三阶段 example 中加 `--camera` 参数测试相机图像获取。

#### 4.2.4 IK 管线适配

**问题**：LeRobot 的 IK 管线步骤（`InverseKinematicsRLStep`）使用 `placo` 求解器和 `RobotKinematics` 类（见 `src/lerobot/model/kinematics.py`），而现有 PikaTeleoperator 中的 IK 使用 `CasADi + Pinocchio`。需要确认两者兼容性。

**分析 `RobotKinematics` 接口**（placo 求解器）：

```python
class RobotKinematics:
    def __init__(self, urdf_path, target_frame_name, joint_names):
        ...
    def forward_kinematics(self, joint_pos_deg) -> np.ndarray:   # 4x4 SE3
        ...
    def inverse_kinematics(self, current_joint_pos, desired_ee_pose,
                           position_weight, orientation_weight) -> np.ndarray:  # joint deg
        ...
```

**方案 A（推荐）：让 `InverseKinematicsRLStep` 使用 Piper URDF + placo 求解器**

- `placo` 是通用 IK 求解器，Piper 的 URDF 格式标准，应可直接加载
- 配置方式（在 `InverseKinematicsConfig` 中指定）：

```json
{
  "inverse_kinematics": {
    "urdf_path": "/path/to/piper_description.urdf",
    "target_frame_name": "link6",
    "end_effector_bounds": {
      "min": [0.10, -0.20, 0.02],
      "max": [0.40, 0.20, 0.30]
    },
    "end_effector_step_sizes": {
      "x": 0.02,
      "y": 0.02,
      "z": 0.02
    }
  }
}
```

**需要确认的参数**：
- `target_frame_name`：Piper URDF 中末端执行器的 link 名称（可能是 `"link6"` 或 `"gripper_base_link"`，需检查 URDF）
- `end_effector_bounds`：由第 4.4.1 步 `lerobot-find-joint-limits` 确定
- 上述 bounds 为初步预估值，**必须经真机标定后替换**

**方案 B（备选）：编写 PiperKinematicsRLStep 替换 InverseKinematicsRLStep**

若 placo 与 Piper URDF 不兼容（格式、关节命名等），则：
1. 新建 `src/lerobot/robots/piper_follower/piper_kinematic_processor.py`
2. 复用 `PiperIK` / `PiperFK` (CasADi + Pinocchio)
3. 实现与 `InverseKinematicsRLStep` 相同的 `ProcessorStep` 接口

**AGI 执行策略**：先尝试方案 A，在配置 JSON 中指定 Piper URDF。若 `placo` 加载 Piper URDF 失败（导入错误或求解不收敛），再切换方案 B。

**验证命令**（AGI 可提前运行以确认）：

```python
from lerobot.model.kinematics import RobotKinematics
kin = RobotKinematics(
    urdf_path="/home/zhbs/pika_ros/install/piper_description/share/piper_description/urdf/piper_description.urdf",
    target_frame_name="link6",  # 需要确认此名称
    joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]  # 需要确认
)
fk = kin.forward_kinematics([0.0]*6)
print(fk)  # 应输出 4x4 SE3 矩阵
```

#### 4.2.5 处理器管线中 control_mode 的扩展

**分析**：`control_mode` 在 `HILSerlProcessorConfig` 中定义为字符串，默认 `"gamepad"`。经代码审查，`make_processors()` 函数中**并未** 对 `control_mode` 值做条件分支——管线步骤的选择取决于配置中的 `inverse_kinematics` 是否为 None、teleop 是否实现了 `HasTeleopEvents` 等结构性条件。

**结论：无需代码修改**，仅在 JSON 配置中将 `control_mode` 设为 `"pika"` 作为语义标记。

```json
{
  "processor": {
    "control_mode": "pika"
  }
}
```

**管线步骤的启用逻辑**（在 `make_processors` 中）：

| 条件 | 启用的步骤 |
|------|-----------|
| teleop 实现了 HasTeleopEvents | AddTeleopEventsAsInfoStep, InterventionActionProcessorStep |
| `processor.inverse_kinematics is not None` | EEReferenceAndDelta, EEBoundsAndSafety, InverseKinematicsRLStep 等 |
| `processor.reward_classifier is not None` | RewardClassifierProcessorStep |
| `processor.gripper.gripper_penalty > 0` | GripperPenaltyProcessorStep |

因此，只要 PikaTeleoperator 实现了 HasTeleopEvents（4.2.1），并在配置中提供了 `inverse_kinematics` 参数（4.2.4），完整的 EE 空间 HIL-SERL 管线就会自动启用。

### 4.3 配置文件编写

#### 4.3.1 env_config_pika_piper.json（环境配置）

**AGI 任务**：创建 `lerobot/src/lerobot/configs/env_config_pika_piper.json`

```json
{
  "env": {
    "type": "gym_manipulator",
    "name": "real_robot",
    "fps": 10,
    "robot": {
      "type": "piper_follower",
      "motors": {
        "can_name": "can_slave1",
        "motors": {
          "joint_1": [1, "agilex_piper"],
          "joint_2": [2, "agilex_piper"],
          "joint_3": [3, "agilex_piper"],
          "joint_4": [4, "agilex_piper"],
          "joint_5": [5, "agilex_piper"],
          "joint_6": [6, "agilex_piper"],
          "gripper": [7, "agilex_piper"]
        }
      },
      "cameras": {
        "wrist": {
          "type": "intel_realsense",
          "serial_number": "TODO_FILL_D405_SERIAL",
          "width": 640,
          "height": 480,
          "fps": 30
        },
        "front": {
          "type": "intel_realsense",
          "serial_number": "TODO_FILL_EXTERNAL_SERIAL",
          "width": 640,
          "height": 480,
          "fps": 30
        }
      }
    },
    "teleop": {
      "type": "pika",
      "sense_port": "/dev/ttyUSB80",
      "tracker_device": "T20",
      "piper_description_dir": "/home/zhbs/pika_ros/install/piper_description/share/piper_description",
      "gripper_xyzrpy": [0.19, 0.0, 0.2, 0.0, 0.0, 0.0],
      "home_joint_state": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      "filter_min_cutoff": 1.0,
      "filter_beta_pos": 1.0,
      "filter_beta_rot": 0.5,
      "lift": false,
      "output_mode": "ee_delta"
    },
    "processor": {
      "control_mode": "pika",
      "observation": {
        "add_joint_velocity_to_observation": false,
        "add_current_to_observation": false,
        "display_cameras": false
      },
      "image_preprocessing": {
        "crop_params_dict": {},
        "resize_size": [128, 128]
      },
      "gripper": {
        "use_gripper": true,
        "gripper_penalty": 0.0
      },
      "reset": {
        "fixed_reset_joint_positions": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "reset_time_s": 5.0,
        "control_time_s": 15.0,
        "terminate_on_success": true
      },
      "inverse_kinematics": {
        "urdf_path": "/home/zhbs/pika_ros/install/piper_description/share/piper_description/urdf/piper_description.urdf",
        "target_frame_name": "link6",
        "end_effector_bounds": {
          "min": [0.10, -0.20, 0.02],
          "max": [0.40, 0.20, 0.30]
        },
        "end_effector_step_sizes": {
          "x": 0.02,
          "y": 0.02,
          "z": 0.02
        }
      },
      "reward_classifier": null,
      "max_gripper_pos": 90.0
    }
  },
  "dataset": {
    "repo_id": "zhbs/pika_piper_pick_lift",
    "task": "pick_and_lift_cube",
    "root": null,
    "num_episodes_to_record": 15,
    "replay_episode": null,
    "push_to_hub": false
  },
  "mode": null,
  "device": "cuda"
}
```

**TODO 标记**（由人类操作员填充）：
- `TODO_FILL_D405_SERIAL`：wrist D405 的序列号（`rs-enumerate-devices`）
- `TODO_FILL_EXTERNAL_SERIAL`：外部相机序列号
- `end_effector_bounds`：须由 4.4.1 标定后替换
- `fixed_reset_joint_positions`：Piper 的安全复位关节角度
- `target_frame_name`：须由 4.2.4 验证脚本确认

#### 4.3.2 train_config_pika_piper.json（训练配置）

**AGI 任务**：创建 `lerobot/src/lerobot/configs/train_config_pika_piper.json`

```json
{
  "policy": {
    "type": "sac",
    "device": "cuda",
    "storage_device": "cuda",
    "temperature_init": 0.01,
    "actor_learner_config": {
      "policy_parameters_push_frequency": 2.0
    },
    "input_features": {
      "observation.state": {
        "type": "STATE",
        "shape": [7]
      },
      "observation.images.wrist": {
        "type": "VISUAL",
        "shape": [3, 128, 128]
      },
      "observation.images.front": {
        "type": "VISUAL",
        "shape": [3, 128, 128]
      }
    },
    "output_features": {
      "action": {
        "type": "ACTION",
        "shape": [4]
      }
    }
  },
  "dataset": {
    "repo_id": "zhbs/pika_piper_pick_lift",
    "task": "pick_and_lift_cube",
    "root": null
  },
  "env": {
    "$ref": "env_config_pika_piper.json#/env"
  },
  "training": {
    "steps": 100000,
    "batch_size": 256,
    "learning_rate": 3e-4,
    "log_freq": 100,
    "eval_freq": 5000,
    "save_freq": 10000
  },
  "wandb": {
    "enable": true,
    "project": "pika-piper-hilserl",
    "entity": null
  }
}
```

**说明**：
- `observation.state` shape=[7]：6 关节角度 + 1 夹爪位置
- `action` shape=[4]：delta_x, delta_y, delta_z, gripper（EE 空间）
- `temperature_init=0.01`：hil-serl-guide 建议值，避免过高播放使干预无效
- `policy_parameters_push_frequency=2.0`：2秒推送一次权重，平衡新鲜度与网络开销
- `$ref` 语法表示复用 env 配置，实际实现中可能需内联或指定单独路径
- 具体 SAC 超参详见 `src/lerobot/policies/sac/configuration_sac.py`，训练过程中可调

#### 4.3.3 reward_classifier_config.json（奖励分类器配置）

**首轮实验可跳过**：手动按 `s` 键标注成功。

待流程跑通后，用以下配置训练奖励分类器：

```json
{
  "policy": {
    "type": "reward_classifier",
    "model_name": "helper2424/resnet10",
    "model_type": "cnn",
    "num_cameras": 2,
    "num_classes": 2,
    "hidden_dim": 256,
    "dropout_rate": 0.1,
    "learning_rate": 1e-4,
    "device": "cuda",
    "use_amp": true,
    "input_features": {
      "observation.images.wrist": {
        "type": "VISUAL",
        "shape": [3, 128, 128]
      },
      "observation.images.front": {
        "type": "VISUAL",
        "shape": [3, 128, 128]
      }
    }
  },
  "dataset": {
    "repo_id": "zhbs/pika_piper_reward_data",
    "task": "reward_classifier_pick_lift"
  }
}
```

**采集奖励分类器数据集时**，在 env_config 中设置：
- `mode: "record"`
- `processor.reset.terminate_on_success: false`（让 episode 继续，收集更多正例）
- `dataset.repo_id: "zhbs/pika_piper_reward_data"`

### 4.4 HIL-SERL 工作流执行步骤

#### 4.4.1 Step 1: 确定工作空间边界 (lerobot-find-joint-limits)

**前置**：第三阶段遥操验证通过

**人类操作员执行**：

```bash
cd ~/sda/zyx/lerobot

# 使用 Pika 遥操 Piper，在任务工作空间内移动
lerobot-find-joint-limits \
    --robot.type=piper_follower \
    --robot.motors.can_name=can_slave1 \
    --teleop.type=pika \
    --teleop.sense_port=/dev/ttyUSB80 \
    --teleop.tracker_device=T20 \
    --teleop.piper_description_dir=/home/zhbs/pika_ros/install/piper_description/share/piper_description \
    --urdf_path=/home/zhbs/pika_ros/install/piper_description/share/piper_description/urdf/piper_description.urdf \
    --target_frame_name=link6 \
    --teleop_time_s=60 \
    --warmup_time_s=5
```

**操作流程**：
1. 脚本启动后先等待 5s 预热（热身期数据不记录）
2. 用 Pika 引导 Piper 在"抓取方块"所需的工作空间内充分移动
3. 覆盖：方块可能出现的位置、抓取高度、抬升高度、放下位置
4. 60s 后脚本输出 EE bounds 和关节限位

**输出示例**：
```
max_ee = [0.35, 0.15, 0.25]
min_ee = [0.15, -0.15, 0.03]
max_pos = [1.2, 0.8, 1.5, 0.5, 1.2, 0.3]
min_pos = [-1.2, -0.3, -0.2, -0.5, -0.1, -0.3]
```

**AGI 任务**：将输出值填入 `env_config_pika_piper.json` 的 `inverse_kinematics.end_effector_bounds`。

> **注意**：若 `lerobot-find-joint-limits` 尚不支持 Pika teleop（因缺少 EE 输出的 FK 集成），可退回到手动方案：在第三阶段 example 脚本中收集 EE 位置并记录 min/max。
#### 4.4.2 Step 2: 采集演示数据 (record 模式)

**前置**：EE bounds 已填入配置

**AGI 任务**：将 `env_config_pika_piper.json` 中 `mode` 设为 `"record"`

**人类操作员执行**：

```bash
cd ~/sda/zyx/lerobot

python -m lerobot.rl.gym_manipulator \
    --config_path src/lerobot/configs/env_config_pika_piper.json
```

**操作流程**：
1. 机械臂自动复位到 `fixed_reset_joint_positions`
2. 用 Pika 遥操完成"抓取并抬起方块"任务
3. 成功后按 `s` 键标记成功（reward=1，episode 终止）
4. 失败（超时/掉落）按 `q` 或等待超时（reward=0）
5. 若需重录按 `r`
6. 重复直到采集完 15 个 episode
7. 数据集保存到本地（`push_to_hub: false`）

**关键配置**：
- `num_episodes_to_record: 15`（hil-serl-guide 建议 10-20 个）
- `control_time_s: 15.0`（pick-and-lift 应在 10-15s 内完成）
- `crop_params_dict: {}`（先不裁剪，Step 3 确定）
#### 4.4.3 Step 3: 裁剪 ROI (crop_dataset_roi)

**人类操作员执行**：

```bash
python -m lerobot.rl.crop_dataset_roi --repo-id zhbs/pika_piper_pick_lift
```

**操作流程**：
1. 工具依次显示每个相机视角的第一帧
2. 拖动矩形框选取包含方块和机械臂末端的工作区域
3. 按 `c` 确认
4. 重复直到所有相机完成

**输出示例**：
```
Selected Rectangular Regions of Interest (top, left, height, width):
observation.images.wrist: [100, 120, 200, 280]
observation.images.front: [150, 180, 180, 240]
```

**AGI 任务**：将输出的 crop 参数填入 `env_config_pika_piper.json`：

```json
{
  "image_preprocessing": {
    "crop_params_dict": {
      "observation.images.wrist": [100, 120, 200, 280],
      "observation.images.front": [150, 180, 180, 240]
    },
    "resize_size": [128, 128]
  }
}
```
#### 4.4.4 Step 4: （可选）训练奖励分类器

**首轮跳过**。使用键盘 `s` 手动标注成功。

待流程跑通后，按以下步骤训练自动奖励检测：

1. **采集奖励分类器数据集**：修改 env_config 中 `mode: "record"`, `terminate_on_success: false`, `repo_id: "zhbs/pika_piper_reward_data"`，采集 20 个 episode
2. **训练**：`lerobot-train --config_path src/lerobot/configs/reward_classifier_config.json`
3. **部署**：在 env_config 中填入 `reward_classifier.pretrained_path`
4. **测试**：`python -m lerobot.rl.gym_manipulator --config_path ...`，观察自动奖励检测是否准确
#### 4.4.5 Step 5: 启动 Learner

**AGI 任务**：确认 `env_config_pika_piper.json` 中 `mode: null`（训练模式）

**人类操作员执行**（终端 1）：

```bash
cd ~/sda/zyx/lerobot

python -m lerobot.rl.learner \
    --config_path src/lerobot/configs/train_config_pika_piper.json
```

Learner 启动后会：
1. 初始化 SAC 策略网络
2. 加载离线演示数据集到 replay buffer
3. 开启 gRPC 服务端等待 actor 连接
4. 控制台输出类似：`Learner gRPC server started on port 50051`
#### 4.4.6 Step 6: 启动 Actor + 人类干预

**人类操作员执行**（终端 2）：

```bash
cd ~/sda/zyx/lerobot

python -m lerobot.rl.actor \
    --config_path src/lerobot/configs/train_config_pika_piper.json
```

Actor 启动后会：
1. 连接 learner gRPC 服务
2. 初始化环境（PiperFollower + PikaTeleoperator + 处理器管线）
3. 开始执行策略 rollout

**干预策略**（参考 hil-serl-guide “Guide to Human Interventions”）：

| 训练阶段 | 干预策略 |
|----------|----------|
| 前 5 个 episode | 允许策略自由探索，仅在危险时干预 |
| 5-20 episode | 在策略偏离时短暂干预纠正方向 |
| 20-50 episode | 减少干预，仅在关键时刻辅助（如抓取瞬间） |
| 50+ episode | 基本不干预，观察策略表现 |

**操作指南**：
- `Space` 键：开始/结束干预。按下后 Pika 动作替代策略动作。
- `s` 键：任务成功时按下＊episode 结束并记录 reward=1
- `q` 键：失败/放弃，episode 结束 reward=0
- **避免长时间干预**，每次干预尽量 < 2s
- 干预率应随训练进展逐渐下降（WandB 中监控）
#### 4.4.7 Step 7: 监控与调参 (WandB)

**关注指标**：

| 指标 | 期望趋势 | 异常信号 |
|------|----------|----------|
| episodic_reward | 逐步上升趋近 1.0 | 长期停滞在 0 |
| intervention_rate | 逐步下降趋近 0 | 长期保持高位 |
| actor_loss | 收敛 | 发散 |
| critic_loss | 收敛 | 发散 |
| temperature | 稳定在低值 | 持续上升 |
| episode_length | 稳定或缩短 | 极度不稳定 |

**常见调参场景**：

| 现象 | 可能原因 | 调整 |
|------|----------|------|
| 策略完全随机，干预无效 | temperature 过高 | 降低 `temperature_init`（如 1e-3） |
| 权重更新太慢 | push_frequency 过高 | 降低 `policy_parameters_push_frequency` 到 1s |
| IK 解不收敛 | EE bounds 过大或 URDF 问题 | 缩小 bounds、检查 target_frame_name |
| 夹爪抢动 | gripper_penalty=0 | 增加 `gripper_penalty`（如 0.1） |
| 训练早期完全不动 | 演示数据不足 | 增加演示 episode 数量 |

---

## 第五阶段：验证与调试检查清单

### 5.1 各阶段验收标准汇总表

| 阶段 | 检查点 | 验收标准 | 执行者 |
|------|--------|----------|--------|
| 1 | 设备绑定 | `setup_pika_piper.sh` 全部 ✓ | 人类 |
| 2 | Pika 独立测试 | `test_pika_teleop.py` 所有 PASS（T6 允许 SKIP） | 人类+AGI |
| 3 | LeRobot 遥操 | Meshcat 跟随、关节输出与裸脚本一致 (< 0.01rad) | 人类 |
| 3 | Piper 真机 | 发送关节命令后末端误差 < 5mm | 人类 |
| 4.2 | 代码适配 | `get_errors()` 无类型错误、HasTeleopEvents 兼容性检查通过 | AGI |
| 4.2.4 | IK 验证 | `RobotKinematics` 加载 Piper URDF 成功、FK 输出合理 | AGI |
| 4.4.1 | workspace bounds | find-joint-limits 输出 bounds 范围合理 | 人类 |
| 4.4.2 | 数据采集 | 15 个 episode 录制完成、数据集可加载 | 人类 |
| 4.4.3 | ROI 裁剪 | crop 参数已填入配置 | 人类 |
| 4.4.5-6 | 训练启动 | learner + actor 均无报错启动、WandB 有日志 | 人类 |
| 4.4.7 | 训练效果 | 100 episode 后 intervention_rate 下降、episodic_reward 上升 | 人类 |
### 5.2 常见问题与排查手册

| 问题 | 症状 | 排查步骤 |
|------|------|----------|
| CAN 连接失败 | `PiperMotorsBus.connect()` 超时 | 1) `ip link show can_slave1` 确认 UP 2) `candump can_slave1 -n 1` 看心跳 3) 检查 Piper 电源 4) 重新运行 `can_activate.sh` |
| Vive Tracker 丢失 | `get_pose()` 返回 None | 1) 检查 Lighthouse 电源和绿灯 2) Tracker 电量 3) `pysurvive` 进程是否存在 4) 遮挡物 |
| IK 发散 | 关节跳变、机械臂抽搐 | 1) 检查 URDF target_frame_name 2) 缩小 EE bounds 3) 降低 step_sizes 4) 增大滤波器 min_cutoff |
| 夹爪抖动 | 夹爪在开/合之间快速切换 | 1) 检查编码器数据稳定性 2) 增大夹爪死区 3) 增加 gripper_penalty |
| 相机帧率不足 | FPS < 设定值 | 1) USB 3.0 端口 2) 降低分辨率到 640×480 3) 减少同时开启的相机数 |
| gRPC 连接失败 | Actor 无法连接 Learner | 1) 确认 Learner 已启动 2) 检查端口 50051 未被占用 3) 防火墙 |
| GPU 显存不足 | CUDA OOM | 1) `storage_device` 改为 "cpu" 2) 降低 batch_size 3) resize 改为 [64, 64] |
| Episode 总是超时 | control_time_s 内未完成任务 | 1) 增大 control_time_s 2) 简化任务 3) 检查 IK 是否限制了运动范围 |
### 5.3 回滚方案

每个阶段完成后建议创建 git 检查点：

```bash
# 阶段 1 完成后
git add -A && git commit -m "stage1: device port binding scripts"

# 阶段 2 完成后
git add -A && git commit -m "stage2: pika teleop standalone test"

# 阶段 3 完成后
git add -A && git commit -m "stage3: lerobot pika-piper teleop example"

# 阶段 4 代码适配完成后
git add -A && git commit -m "stage4: hilserl pika-piper adaptation"

# 阶段 4 配置文件完成后
git add -A && git commit -m "stage4: hilserl configs with calibrated bounds"
```

若某阶段失败需回滚：`git stash` 或 `git checkout <commit>` 回到上一个检查点。

---

## 附录

### A. 文件修改清单（按文件路径索引）

#### 新增文件

| 文件路径 | 阶段 | 说明 |
|----------|------|------|
| `pika_ros/scripts/setup_pika_piper.sh` | 1.5 | 一键设备初始化脚本 |
| `zyx/test_pika_teleop.py` | 2.2 | Pika 遥操独立测试 |
| `lerobot/examples/pika_piper_teleop_viz.py` | 3.3 | LeRobot 框架遥操 + Meshcat 对比 |
| `lerobot/src/lerobot/configs/env_config_pika_piper.json` | 4.3.1 | HIL-SERL 环境配置 |
| `lerobot/src/lerobot/configs/train_config_pika_piper.json` | 4.3.2 | HIL-SERL 训练配置 |
| `lerobot/src/lerobot/configs/reward_classifier_config.json` | 4.3.3 | 奖励分类器配置（可选） |

#### 修改文件

| 文件路径 | 阶段 | 变更内容 |
|----------|------|----------|
| `lerobot/src/lerobot/teleoperators/pika/config_pika.py` | 4.2.2 | 新增 `output_mode: str = "joint"` 字段 |
| `lerobot/src/lerobot/teleoperators/pika/pika_teleoperator.py` | 4.2.1, 4.2.2 | 新增 `get_teleop_events()`、`_start_keyboard_listener()`、EE-delta 输出模式、`_get_ee_delta_action()`、`_get_joint_action()` |

#### 无需修改的文件（仅确认兼容）

| 文件路径 | 原因 |
|----------|------|
| `lerobot/src/lerobot/teleoperators/utils.py` | "pika" 已注册 |
| `lerobot/src/lerobot/robots/utils.py` | "piper_follower" 已注册 |
| `lerobot/src/lerobot/rl/gym_manipulator.py` | `make_processors` 基于配置自动适配 |
| `lerobot/src/lerobot/processor/hil_processor.py` | HasTeleopEvents 为 Protocol，无需注册 |
| `lerobot/src/lerobot/envs/configs.py` | control_mode 为字符串字段，"pika" 直接可用 |

### B. 依赖与环境要求

#### Python 包

| 包 | 用途 | 安装 |
|--|------|------|
| lerobot[hilserl] | LeRobot HIL-SERL 全套 | `pip install -e ".[hilserl]"` |
| agx-pypika | Pika SDK | `cd pika_sdk && pip install -e .` |
| piper_sdk | Piper CAN 控制 | 已安装（编译包） |
| pinocchio | FK/IK 求解 + Meshcat 可视化 | `pip install pin` |
| casadi | IK 优化求解器 | `pip install casadi` |
| meshcat | 3D 可视化 | `pip install meshcat` |
| pynput | 键盘监听 | `pip install pynput` |
| pysurvive | Vive Tracker | 已安装（编译包） |
| pyrealsense2 | RealSense 相机 | `pip install pyrealsense2` |
| wandb | 训练监控 | `pip install wandb` |

#### 系统包

```bash
sudo apt install can-utils ethtool
```

#### 硬件连线拓扑

```
主机 USB Hub
 ├─ Pika Sense (USB-Serial)
 ├─ Pika Gripper (USB-Serial, 备用)
 ├─ USB-CAN Adapter ── Piper 机械臂 (CAN bus)
 ├─ RealSense D405 (USB 3.0) ── 安装在腕部
 └─ RealSense D4xx (USB 3.0) ── 固定在工作台

Vive Lighthouse × 2 ── 对角放置，霮盖工作区域
Pika Sense 内置 Vive Tracker ── 佩戴在手上
```

### C. 参考资料

| 资料 | 路径/链接 |
|------|----------|
| HIL-SERL 实验指南 | `lerobot/hil-serl-guide.md` |
| Pika SDK API 文档 | `pika_sdk/Pika SDK API 文档.md` |
| PikaTeleoperator 源码 | `lerobot/src/lerobot/teleoperators/pika/` |
| PiperFollower 源码 | `lerobot/src/lerobot/robots/piper_follower/` |
| HIL 处理器协议 | `lerobot/src/lerobot/processor/hil_processor.py` |
| IK 管线步骤 | `lerobot/src/lerobot/robots/so_follower/robot_kinematic_processor.py` |
| SAC 配置 | `lerobot/src/lerobot/policies/sac/configuration_sac.py` |
| test_pika_ik_viz 参考实现 | `zyx/test_pika_ik_viz.py` |
| CAN 脚本 | `pika_ros/scripts/can_activate.sh`, `can_config.sh` |
| 多设备检测工具 | `pika_sdk/tools/multi_device_detector.py` |
| LeRobot 示例配置 | https://huggingface.co/datasets/lerobot/config_examples |
| HIL-SERL 论文 | Luo et al. 2024, arXiv:2410.21845 |
