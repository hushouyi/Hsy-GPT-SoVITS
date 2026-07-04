# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GPT-SoVITS 逐句录音 + 一键训练 WebUI — 基于 GPT-SoVITS v2Pro 的定制 Web 界面，支持逐句录音、一键训练、零样本/少样本 TTS 推理。

## ⚠️ 核心约束

**`GPT-SoVITS-v2pro-20250604-nvidia50/` 是上游开源项目，严禁修改其内部任何文件。** 所有自定义代码必须放在 repo 根目录，与上游目录同级。上游文件更新后直接替换目录，定制代码不受影响。

## 目录结构

```
voice/
├── CLAUDE.md                      # 本文件
├── sentence_webui.py              # 主入口（Gradio 应用，端口 7860）
├── start_recorder.bat             # 启动脚本（设 PYTHONIOENCODING=utf-8）
├── sentence_recorder/             # 核心逻辑模块
│   ├── recorder.py                # sounddevice 录音管理（单例）
│   ├── state.py                   # 全局状态（训练/录制/推理互斥）
│   ├── mapping.py                 # mapping.txt 持久化
│   ├── script_reader.py           # 脚本文件读写（含分组标题解析）
│   ├── model_utils.py             # API 启动/停止、模型列表扫描
│   └── training_pipeline.py       # S1+S2 训练流水线
├── sentence_tabs/                 # Tab 页面模块
│   ├── tab_recording.py           # 逐句录音 Tab
│   ├── tab_training.py            # 一键训练 Tab
│   └── tab_inference.py           # 推理 Tab
├── reference.txt                  # 默认语料（只读，160句/8类别）
├── projects/                      # 运行时生成的项目数据
│   └── default/
│       ├── script.txt
│       ├── mapping.txt
│       └── recorded/              # WAV 录音文件
└── GPT-SoVITS-v2pro-20250604-nvidia50/  # ⛔ 上游 OSS，不改
    ├── webui.py                   # 上游官方 WebUI
    ├── api.py / api_v2.py         # 推理 API（FastAPI，端口 9880）
    ├── config.py                  # 模型路径、GPU 检测、端口定义
    ├── GPT_SoVITS/                # 核心模型包
    ├── tools/                     # ASR、切片、降噪、UVR5、i18n
    ├── GPT_weights_v*/            # 各版本训练好的 GPT 权重
    ├── SoVITS_weights_v*/         # 各版本训练好的 SoVITS 权重
    ├── logs/                      # 训练日志（exp: hsy_02, moyan）
    ├── runtime/                   # 捆绑的 Python 解释器 + site-packages
    └── requirements.txt           # Python 依赖
```

## 技术架构

### 训练管线（双模型）

| 阶段 | 模型 | 框架 | 配置文件 |
|------|------|------|---------|
| S1 | GPT (AR, 文本→语义) | PyTorch Lightning | `GPT_SoVITS/configs/s1.yaml` / `s1big.yaml` |
| S2 | SoVITS (VITS, 语义→波形) | PyTorch | `GPT_SoVITS/configs/s2*.json` |

S1 训练入口：`s1_train.py`（多版本共存）。S2 训练入口：`s2_train.py`（v1/v2）、`s2_train_v3.py`、`s2_train_v3_lora.py`（v3/v4 LoRA）。

当前 `weight.json` 选中的是 **v2ProPlus SoVITS + v4 GPT**。

### 推理 API

`api.py`（v1）/ `api_v2.py`（v2，YAML 配置驱动）运行在端口 9880。自定义 WebUI 通过调用本地 API 进行推理。

### 端口分配

| 端口 | 用途 |
|------|------|
| 7860 | 自定义录音+训练 WebUI（`sentence_webui.py`） |
| 17860 | 退出监听器（HTTP `/quit` 端点，关闭浏览页自动退出） |
| 9874 | 上游官方 WebUI |
| 9880 | 推理 API |

## 技术约束与坑

### Gradio 4.24.0 组件 ID Bug

`gr.State()` 必须在 `with gr.Blocks():` **顶层**创建，不能在 `with gr.TabItem():` 内部创建。子 tab 文件的 `gr.State` 由主文件创建后传入（`train_running`, `gpt_cache`, `sovits_cache`）。

### Windows GBK 编码

控制台默认 GBK 编码无法输出 emoji。所有 `print()` 中的 emoji 必须替换为 ASCII（`🎙️→[MIC]`、`✅→[OK]`、`💾→[SAVE]`、`⚠️→[WARN]`）。启动时需设置 `PYTHONIOENCODING=utf-8`。

### Windows 子进程管理

- `subprocess.run([...], shell=True)` + 列表参数 → 命令不生效，必须用字符串命令
- 端口清理：`netstat -ano` 找 PID → `taskkill /f /pid {pid}` → `tasklist` 确认
- `taskkill //f //im python.exe` 杀死所有 Python 子进程（退出清理用）

### 状态互斥

- `state.training`：训练中 → 推理/录音不可用
- `state.inference_api`：推理 API 运行中 → 训练不可用
- `state.recording`：录音中（通过 `RecordingManager.is_recording` 判断）

### 翻页策略

`PAGE_SIZE=10`，翻页时**从磁盘重新读取 mapping.txt**，不依赖内存缓存。

## 启动方式

```bash
set PYTHONIOENCODING=utf-8
.\GPT-SoVITS-v2pro-20250604-nvidia50\runtime\python.exe sentence_webui.py
```

或双击 `start_recorder.bat`。关闭浏览器标签页后，`sendBeacon()` → `/quit` 端点自动触发进程清理退出。

## 开发环境

- Python: 3.10–3.12（捆绑在 `runtime/` 中）
- 依赖: `GPT-SoVITS-v2pro-20250604-nvidia50/requirements.txt`
- 无需额外安装（捆绑 runtime 已有全部依赖）
- Git 仓库（GitHub: https://github.com/hushouyi/Hsy-GPT-SoVITS）
- **上游 `GPT-SoVITS-v2pro-20250604-nvidia50/` 已被 `.gitignore` 排除，不上传 GitHub**
