# go2-mjx-FoPG-deploy

这是 GO2 MJX/FoPG 策略的实机部署版仓库，按 Unitree 官方 `unitree_sdk2_python` 的示例目录结构组织。

完整训练、导出和仿真验证项目在：

```text
https://github.com/Iris152/go2-mjx-FoPG
```

本仓库只保留实机部署所需的最小内容。

## 目录

```text
example/go2/mjx_fopg_deploy/
├── go2_unitree_sdk2_deploy.py
├── sanitize_onnx_for_ort.py
├── exported_onnx/
└── mujoco_menagerie/unitree_go2/
```

其中：

- `go2_unitree_sdk2_deploy.py` 是实机低层部署入口。
- `exported_onnx/` 是已导出的 trot baseline 和 forward residual 策略。
- `mujoco_menagerie/unitree_go2/` 用来读取训练时的 `home` 关节中心。
- `sanitize_onnx_for_ort.py` 用于在需要时生成 ONNX Runtime 兼容模型。

## 推荐使用方式

先克隆 Unitree 官方 SDK2 Python：

```bash
cd /home/a
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
```

然后把本仓库的部署目录放到官方 SDK 仓库下：

```bash
cp -a /home/a/go2-mjx-FoPG-deploy/example/go2/mjx_fopg_deploy \
  /home/a/unitree_sdk2_python/example/go2/
```

当前本机已经准备好了这个目录：

```text
/home/a/unitree_sdk2_python/example/go2/mjx_fopg_deploy
```

## 实机运行

进入 Unitree SDK2 Python 仓库根目录运行：

```bash
conda activate mjx
cd /home/a/unitree_sdk2_python

python example/go2/mjx_fopg_deploy/go2_unitree_sdk2_deploy.py \
  --network <robot_nic> \
  --domain_id 0
```

`<robot_nic>` 换成连接 Go2 的有线网卡名，例如 `eno1`、`enp2s0` 或 `enx...`。

第一次实机不要加 `--auto_start` 或 `--auto_policy`，保留人工确认。

## MCF

脚本默认 `--release_mcf auto`：

- `--domain_id 0` 实机时自动调用 `MotionSwitcher.CheckMode()`。
- 如果检测到 `sport_mode` / `ai_sport` / `advanced_sport` 等主运控服务，会调用 `ReleaseMode()`。
- MCF 释放后才进入低层 `rt/lowcmd` 控制。
- `--domain_id 1` 仿真时自动跳过。

这个逻辑对应 Unitree 官方低层示例 `go2_stand_example.py` / `go2_stand_example.cpp`。

## 上机前检查

1. `conda activate mjx`
2. 电脑有线网卡配置为 `192.168.123.222/24`
3. `ip -br addr` 确认有线网卡名
4. `ping 192.168.123.161` 确认能连到 Go2
5. 机器人处于可低层控制的安全状态
