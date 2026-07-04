# Hsy-GPT-SoVITS — 逐句录音 + 一键训练 WebUI

**v1.0.4** — [GitHub](https://github.com/hushouyi/Hsy-GPT-SoVITS)

基于 GPT-SoVITS v2Pro 的定制 Web 界面，支持**逐句录音**、**一键训练**、**零样本/少样本 TTS 推理**。

> ⚠️ 本仓库**不包含**上游 GPT-SoVITS 项目代码。使用时需自行下载 `GPT-SoVITS-v2pro-20250604-nvidia50` 放到项目根目录。
>
> 上游项目: [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)

---

## 功能

### 🎙 逐句录音
- 每页 10 句，支持 REC/PLAY/DEL 操作
- 状态圆点标记：⚪未录 / 🔴录音中 / 🟢已录
- 总体进度条 + 按类别的进度统计
- 每个类别需录制 ≥5 条方可训练

### 🚀 一键训练
- 完整训练管线：数据收集 → 特征提取 → S2 SoVITS → S1 GPT → 模型保存
- 支持多项目合并作为训练数据源
- 实时进度条 + 日志输出
- 训练完成后自动保存到 `models/` 目录

### 🎧 本地推理
- 三种参考音频模式：从已录语句选择 / 上传自定义 / 现场录制
- GPT + SoVITS 模型选择
- 可调节 TTS 参数（Top-K, Top-P, 温度, 速度等）

### 📁 项目系统
- 多项目管理：新建 / 切换 / 导入 / 删除 / 锁定
- 每个项目独立维护 script.txt + mapping.txt + recorded/
- 训练时可选择单项目或多项目合并

---

## 快速开始

### 1. 下载上游项目

```bash
# 下载 GPT-SoVITS-v2pro-20250604-nvidia50 放到本项目根目录
# 目录结构：
# voice/
# ├── GPT-SoVITS-v2pro-20250604-nvidia50/   ← 上游（从官网下载）
# ├── sentence_webui.py                      ← 本项目的入口
# └── ...
```

### 2. 启动

双击 `start_recorder.bat`，或命令行：

```bash
set PYTHONIOENCODING=utf-8
.\GPT-SoVITS-v2pro-20250604-nvidia50\runtime\python.exe sentence_webui.py
```

启动后自动打开浏览器到 http://127.0.0.1:7860

### 3. 使用流程

```
1. 选项目 → 2. 逐句录音（每类≥5条） → 3. 点"完成录制" → 4. 配置训练参数 → 5. 开始训练 → 6. 切换到推理 → 7. 选模型 + 参考音频 → 8. 合成语音
```

---

## 端口分配

| 端口  | 用途               |
|-------|--------------------|
| 7860  | 录音 + 训练 WebUI  |
| 9880  | 推理 API (FastAPI) |
| 17860 | /quit 退出监听     |

---

## 项目结构

```
voice/
├── sentence_webui.py              # 主入口（Gradio Blocks）
├── start_recorder.bat             # Windows 启动脚本
├── reference.txt                  # 默认语料（160句，8类别）
├── CLAUDE.md                      # AI 辅助开发配置
├── DESIGN.md                      # 完整设计文档
├── .gitignore
├── README.md                      # ← 本文件
│
├── sentence_recorder/             # 核心逻辑模块
│   ├── recorder.py                # sounddevice 录音管理
│   ├── state.py                   # 全局状态互斥
│   ├── script_reader.py           # 脚本解析（含类别标题）
│   ├── mapping.py                 # mapping.txt 读写 + 缓存
│   ├── project_manager.py         # 项目 CRUD
│   ├── model_utils.py             # 模型扫描、API 管理
│   └── training_pipeline.py       # 训练管线编排
│
├── sentence_tabs/                 # Tab 页面
│   ├── tab_recording.py           # 录音页面
│   ├── tab_training.py            # 训练页面
│   └── tab_inference.py           # 推理页面
│
├── projects/                      # 运行时项目数据（.gitignore 排除 recorded/）
│   └── default/
│       ├── script.txt
│       ├── mapping.txt
│       └── recorded/
│
└── GPT-SoVITS-v2pro-20250604-nvidia50/  # ⛔ 上游，需自行下载
```

---

## 技术栈

- **框架**: Gradio 4.24.0
- **录音**: sounddevice (24000Hz, mono, 16-bit WAV)
- **训练**: PyTorch Lightning (S1 GPT) + PyTorch (S2 SoVITS)
- **推理**: FastAPI (api_v2.py, 端口 9880)
- **平台**: Windows 11 (捆绑 Python 3.10–3.12)

---

## License

本项目基于 MIT License 开源。上游 GPT-SoVITS 项目遵循其自身许可证。
