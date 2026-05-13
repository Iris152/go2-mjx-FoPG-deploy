# go2-mjx-FoPG-deploy

这是 GO2 MJX/FoPG 策略的实机部署版仓库。当前仓库已经把 Unitree 官方 `unitree_sdk2_python` 源码和官方示例一起并入，因此使用者不需要再单独克隆 Unitree SDK。

完整训练、导出和仿真验证项目在：

```text
https://github.com/Iris152/go2-mjx-FoPG
```

本仓库用于实机部署和 SDK2 Python 环境复现；训练、FoPG 修改、MJX 训练脚本、双终端仿真验证仍以完整训练仓库为准。

## 目录

```text
.
├── unitree_sdk2py/                         # Unitree SDK2 Python 源码
├── example/                                # Unitree 官方示例 + 本项目部署示例
│   └── go2/mjx_fopg_deploy/
│       ├── go2_unitree_sdk2_deploy.py
│       ├── sanitize_onnx_for_ort.py
│       ├── exported_onnx/
│       └── mujoco_menagerie/unitree_go2/
├── setup.py
├── pyproject.toml
├── LICENSE
├── UNITREE_SDK2_PYTHON_README.md
└── UNITREE_SDK2_PYTHON_README_zh.md
```

其中：

- `unitree_sdk2py/`、`example/go2/low_level/` 等来自 Unitree 官方 `unitree_sdk2_python`。
- `go2_unitree_sdk2_deploy.py` 是实机低层部署入口。
- `exported_onnx/` 是已导出的 trot baseline 和 forward residual 策略。
- `mujoco_menagerie/unitree_go2/` 用来读取训练时的 `home` 关节中心。
- `sanitize_onnx_for_ort.py` 用于在需要时生成 ONNX Runtime 兼容模型。
- `UNITREE_SDK2_PYTHON_README*.md` 是 Unitree 官方 README 的副本，方便对照 SDK 原始用法。

## 安装

直接克隆本仓库即可：

```bash
git clone https://github.com/Iris152/go2-mjx-FoPG-deploy.git
cd go2-mjx-FoPG-deploy
```

建议使用已经验证过的 `mjx` 环境：

```bash
conda activate mjx
python -m pip install -e .
```

如果在新机器上安装，依赖以 `setup.py` 为准，核心包括：

```text
cyclonedds==0.10.2
numpy
opencv-python
```

## 实机运行

在本仓库根目录运行：

```bash
conda activate mjx
cd /path/to/go2-mjx-FoPG-deploy

python example/go2/mjx_fopg_deploy/go2_unitree_sdk2_deploy.py \
  --network <robot_nic> \
  --domain_id 0
```

`<robot_nic>` 换成连接 Go2 的有线网卡名，例如 `eno1`、`enp2s0` 或 `enx...`。

第一次实机建议不要加 `--auto_start` 或 `--auto_policy`，保留人工确认。脚本启动后会先等待 `rt/lowstate`，然后手动确认起身，再手动确认进入策略。

## MCF

脚本默认 `--release_mcf auto`：

- `--domain_id 0` 实机时自动调用 `MotionSwitcher.CheckMode()`。
- 如果检测到 `sport_mode` / `ai_sport` / `advanced_sport` 等主运控服务，会调用 `ReleaseMode()`。
- MCF 释放后才进入低层 `rt/lowcmd` 控制。
- `--domain_id 1` 仿真时自动跳过。

这个逻辑对应 Unitree 官方低层示例 `go2_stand_example.py` / `go2_stand_example.cpp`。

## 上机前检查

1. `conda activate mjx`
2. `python -m pip install -e .`
3. `ip -br addr` 确认有线网卡名
4. 电脑有线网卡配置为 `192.168.123.222/24`
5. `ping 192.168.123.161` 确认能连到 Go2
6. 机器人处于可低层控制的安全状态
7. 首次只测试站立和短时策略，确认可随时 `Ctrl+C` 平滑退出

## 说明

仓库里不包含原始 `unitree_sdk2_python` 的 `.git` 历史，只保留源码、示例、安装文件和官方 README 副本。这样可以直接从本仓库部署，同时避免嵌套 Git 仓库带来的提交和拉取问题。
