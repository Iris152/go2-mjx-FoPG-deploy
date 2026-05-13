# Go2 MJX/FoPG Low-Level Deploy

这个目录是 MJX/FoPG Go2 策略的 Unitree SDK2 Python 低层部署入口。

当前 GitHub 部署仓库已经把 Unitree 官方 `unitree_sdk2_python` 源码一起并入，所以不需要再把本目录手动复制到另一个 SDK 仓库里。保持这个目录结构的目的，是让部署代码和官方 SDK2 Python 的包结构、DDS/RPC 客户端、低层示例放在同一个工作区里，方便按 Unitree 示例习惯运行和排查。

## 目录内容

- `go2_unitree_sdk2_deploy.py`：当前实机部署入口。
- `exported_onnx/`：已经导出的 trot baseline 和 forward residual ONNX。
- `sanitize_onnx_for_ort.py`：ONNX Runtime 兼容清理工具。
- `mujoco_menagerie/unitree_go2/`：读取训练 `home` keyframe 所需的 XML。

## 环境

建议在当前已经验证过的 `mjx` 环境中运行：

```bash
conda activate mjx
cd /path/to/go2-mjx-FoPG-deploy
python -m pip install -e .
```

如果 `pip install -e .` 报 CycloneDDS 相关错误，优先回到已经能导入 `unitree_sdk2py` 的 `mjx` 环境，不要临时切到 `base`。

## 实机前检查

1. Go2 按 Unitree Quick Start 接好有线网络。
2. 电脑有线网卡设为 `192.168.123.222/24`。
3. 用 `ip -br addr` 找到连接机器狗的网卡名，例如 `eno1`、`enp2s0` 或 `enx...`。
4. 确认能 ping 到 Go2。
5. 第一次运行不要加 `--auto_start` 或 `--auto_policy`。

## 运行

在部署仓库根目录运行：

```bash
conda activate mjx
cd /path/to/go2-mjx-FoPG-deploy

python example/go2/mjx_fopg_deploy/go2_unitree_sdk2_deploy.py \
  --network eno1 \
  --domain_id 0
```

如果实际网卡不是 `eno1`，把 `--network` 改成 `ip -br addr` 里看到的有线网卡名。

## MCF / 主运控服务

脚本默认 `--release_mcf auto`：

- `--domain_id 0` 实机时，会先调用 `MotionSwitcher.CheckMode()`。
- 如果检测到 `sport_mode` / `ai_sport` / `advanced_sport` 等主运控服务仍在运行，会调用 `ReleaseMode()`。
- MCF 释放后才进入低层 `rt/lowcmd` 控制。
- `--domain_id 1` 仿真时自动跳过。

这对应 Unitree 官方低层示例 `example/go2/low_level/go2_stand_example.py` / `go2_stand_example.cpp` 的做法。

## 手动阶段

默认运行时需要两次人工确认：

1. 看到 `Press Enter to start stand-up.` 后，按 Enter 开始站立。
2. 站稳后看到 `Press Enter to start the policy.`，再按 Enter 启动策略。

第一次实机建议只测试站立和短时策略，不要自动启动。

## 可选保守参数

第一次上机可以先降低策略 PD：

```bash
python example/go2/mjx_fopg_deploy/go2_unitree_sdk2_deploy.py \
  --network eno1 \
  --domain_id 0 \
  --policy_kp 25 \
  --policy_kd 1.0
```

如果只是做本机仿真，应使用原项目里的双终端 menagerie 仿真流程，而不是在这里连接真机网卡。
