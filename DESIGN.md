# GPT-SoVITS 逐句录音 + 一键训练 WebUI — 完整设计文档

> 版本：v1.1
> 基于：GPT-SoVITS-v2pro-20250604-nvidia50（上游项目，**不修改其源码**）
> 框架：Gradio 4.24.0
> 端口：7860（主UI）、9880（推理API）、17860（退出监听）

---

## 目录

1. [整体架构](#1-整体架构)
2. [启动与关闭生命周期](#2-启动与关闭生命周期)
3. [全局状态栏](#3-全局状态栏)
4. [项目系统](#4-项目系统)
5. [页面一：训练 > 录音](#5-页面一训练--录音)
6. [页面一：训练 > 训练](#6-页面一训练--训练)
7. [页面二：推理](#7-页面二推理)
8. [数据层设计](#8-数据层设计)
9. [模型组织与保存](#9-模型组织与保存)
10. [技术架构](#10-技术架构)
11. [实施计划](#11-实施计划)

---

## 1. 整体架构

### 1.1 应用结构

```
┌──────────────────────────────────────────────────────────────────┐
│  Gradio App (sentence_webui.py)  :7860                          │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Tab Bar: [训练]  [推理]          ← gr.Tabs()              │  │
│  │  ────────────────────────────────────────────────────────  │  │
│  │  📊 全局状态栏: 🟢 空闲 | 项目: 我的声音v1 | 录制: 45/160│  │
│  │  ────────────────────────────────────────────────────────  │  │
│  │                                                             │  │
│  │  训练 Tab (gr.Tab())                                        │  │
│  │  ┌─────────┬──────────┐                                    │  │
│  │  │ 录音 ←默认│ 训练     │  ← gr.Tabs() (子Tabs)             │  │
│  │  ├─────────┴──────────┤                                    │  │
│  │  │ tab_recording.py   │  ← 录音页面                        │  │
│  │  │ tab_training.py    │  ← 训练页面                        │  │
│  │  └────────────────────┘                                    │  │
│  │                                                             │  │
│  │  推理 Tab (gr.Tab())                                        │  │
│  │  ┌────────────────────┐                                    │  │
│  │  │ tab_inference.py   │  ← 推理页面                        │  │
│  │  └────────────────────┘                                    │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  后台服务                                                         │
│  ├── HTTP /quit 端点 (:17860) — 浏览器关闭时自动清理进程         │
│  └── 推理 API 进程 (api_v2.py :9880) — 按需启动                   │
└──────────────────────────────────────────────────────────────────┘
```

### 1.2 端口分配

| 端口  | 用途               | 生命周期                     |
|-------|--------------------|------------------------------|
| 7860  | 自定义录音+训练UI  | 始终运行                     |
| 9880  | 推理 API (FastAPI) | 首次推理时启动，退出时关闭   |
| 17860 | /quit 退出监听      | 始终运行                     |

---

## 2. 启动与关闭生命周期

### 2.1 启动流程

```
用户双击 start_recorder.bat
       │
       ▼
┌──────────────────────────────────────────┐
│  1. 端口检查与清理                        │
│     port_list = [7860, 9880, 17860]      │
│     for port in port_list:               │
│       if port_in_use(port):              │
│         pid = find_pid_by_port(port)     │
│         taskkill /f /pid {pid}           │
│     └─ 确保所有端口都释放了才继续         │
└──────────────────┬───────────────────────┘
                   ▼
┌──────────────────────────────────────────┐
│  2. 项目初始化                            │
│     扫描 projects/ 目录下所有子目录       │
│     projects/                            │
│     ├── default/    ← 不存在则自动创建    │
│     │   ├── script.txt ← reference.txt   │
│     │   ├── mapping.txt ← 空映射         │
│     │   └── recorded/  ← 空目录          │
│     └── ...(其他项目，如果有)             │
│                                           │
│     加载上次使用的项目（记录在 config）   │
│     读取 mapping.txt → 内存缓存           │
└──────────────────┬───────────────────────┘
                   ▼
┌──────────────────────────────────────────┐
│  3. 启动 Gradio                           │
│     app.queue()                           │
│     app.launch(server_port=7860)          │
│     webbrowser.open(http://127.0.0.1:7860)│
│     └─ 自动打开默认浏览器                  │
└──────────────────────────────────────────┘
```

**端口清理实现（Python 函数）：**
```python
import subprocess, re

def kill_process_on_port(port):
    """强制释放指定端口"""
    try:
        # netstat -ano 找端口对应的 PID
        result = subprocess.run(
            f'netstat -ano | findstr ":{port} "',
            capture_output=True, text=True, shell=True
        )
        for line in result.stdout.split('\n'):
            parts = re.split(r'\s+', line.strip())
            if len(parts) >= 5 and 'LISTENING' in line:
                pid = parts[-1]
                subprocess.run(f'taskkill /f /pid {pid}', shell=True)
    except:
        pass
```

### 2.2 关闭流程

```
用户关闭浏览器标签页
       │
       │ sendBeacon() → GET http://127.0.0.1:17860/quit
       ▼
┌──────────────────────────────────────────┐
│  1. 停止训练（如果有子进程在跑）          │
│     if training_active:                  │
│       training_pipeline.cancel()         │
│       taskkill /f /pid {train_pid}       │
│                                           │
│  2. 停止推理 API                         │
│     try: requests.get(:9880/control?cmd=exit)│
│     except: taskkill /f /pid {api_pid}     │
│                                           │
│  3. 清理所有端口                          │
│     kill_process_on_port(7860)  ← 自身    │
│     kill_process_on_port(9880)            │
│     kill_process_on_port(17860)           │
│                                           │
│  4. 退出 Python 进程                      │
│     os._exit(0)                           │
└──────────────────────────────────────────┘
```

**关键保证**：下次启动时端口检查会清理残留，无论上次如何关闭。

---

## 3. 全局状态栏

位于主 Tab 栏下方，所有页面顶部固定显示：

```
┌──────────────────────────────────────────────────────────────────┐
│  [训练]  [推理]                                                    │
│  ──────────────────────────────────────────────────────────────  │
│  📊 🟢 空闲  |  项目: 我的声音v1  |  已录制: 45/160  |  ✅ 可训练│
│  ──────────────────────────────────────────────────────────────  │
│  [项目 ▼] [新建] [导入]              ← 项目操作栏                 │
└──────────────────────────────────────────────────────────────────┘
```

### 状态定义

| 状态显示 | 图标 | 说明 |
|---------|------|------|
| 🟢 **空闲** | 绿色 | 无任何操作进行中 |
| 🔴 **录音中** | 红色闪烁 | 正在录制音频 |
| 🟡 **训练中 (S2: 3/5)** | 黄色 | 训练进行中，显示阶段和进度 |
| 🔵 **推理中** | 蓝色 | 正在合成语音 |
| ⏸ **项目锁定** | 灰色锁 | 当前项目被锁定，不可编辑 |

### 状态栏数据结构

```python
status_data = {
    "icon": "🟢",                    # 状态图标
    "state": "空闲",                 # 状态文字
    "project": "我的声音v1",         # 当前项目名
    "recorded": "45/160",            # 录制进度
    "trainable": "✅ 可训练",        # 是否满足训练条件
    "message": ""                    # 附加消息（如"类别4还需4条"）
}
```

### 更新时机

- 录音开始/停止 → 更新图标和录制计数
- 训练进度变化 → 更新训练阶段
- 项目切换 → 更新项目名和录制计数
- 推理开始/结束 → 更新推理状态

---

## 4. 项目系统

### 4.1 项目概念

项目是一组**独立的录音数据**：一个脚本文件 + 一批录音文件 + 映射文件。每个项目完全自包含。

```
projects/
├── default/                     # 默认项目，首次启动自动创建
│   ├── script.txt               ← 从 reference.txt 复制
│   ├── mapping.txt              ← 录音状态
│   ├── recorded/                ← WAV 录音文件
│   └── .locked                  ← 空文件，存在=锁定
│
├── 我的声音v1/                  # 用户新建的项目
│   ├── script.txt               ← 创建时选择脚本来源
│   ├── mapping.txt
│   ├── recorded/
│   └── .locked
│
└── 英语句子/                    # 另一个项目
    ├── script.txt
    ├── mapping.txt
    ├── recorded/
    └── .locked                  ← 无此文件=可删除
```

### 4.2 项目操作

#### 新建项目

```
用户点击 [新建]
  → 弹出对话框:
     项目名称: [_______________]  (字母数字中文下划线)
     脚本来源: [● 使用默认脚本(reference.txt)]
              [○ 从文件导入       ] → 文件选择器
              [○ 从现有项目复制   ] → 项目下拉列表
     初始录音: [○ 空项目]
              [● 从已有项目导入录音] → 项目下拉列表
  → 点击 [创建]
  → 校验: 名称不重复、脚本文件存在
  → 创建目录 projects/{name}/
  → 复制脚本、初始化 mapping.txt
  → 如果选择导入录音: 复制 mapping 和 WAV
  → 自动切换到新项目
```

#### 切换项目

```
用户从下拉框选择项目
  → 检查当前是否有录音进行中 → 有则阻止切换
  → 当前 mapping 缓存写回磁盘（异步）
  → 读取新项目的 mapping.txt → 加载到内存缓存
  → 刷新录音页面所有UI（进度区、句子列表、类别检查）
  → 更新全局状态栏
```

#### 删除项目

```
用户选择项目 → 点击 [删除]
  → 检查 .locked 文件 → 存在则拒绝删除
  → 弹出确认: "确定删除项目「xxx」？录音文件将永久丢失"
  → 确认后: 删除整个 projects/{name}/ 目录
  → 从下拉列表移除
  → 如果删除的是当前项目，切换到 default
```

#### 锁定项目

```
用户右键/菜单选择 [锁定项目] 或 [解锁项目]
  → 创建/删除 projects/{name}/.locked 文件
  → 加锁: 录音按钮禁用，项目右侧显示 🔒
  → 解锁: 恢复正常操作
```

#### 导入项目

```
用户点击 [导入]
  → 文件选择器: 选择已有项目的目录或 zip 包
  → 校验目录结构是否合法（必须有 mapping.txt）
  → 复制到 projects/{原名}/
  → 如果名称重复则加数字后缀
  → 刷新项目列表
```

### 4.3 项目与训练的关联

#### 训练数据源

训练页面可以**选择数据源**：

```
数据源:
  ● 当前项目 "我的声音v1" (45句已确认)
  
  ○ 多项目合并:
     已选项目:
     [我的声音v1] 45句  [×]  ← 可移除
     [英语句子  ] 20句  [×]
     [+ 添加项目 ▼]
     合计: 65句
```

**多项目合并的实现：**
```
训练时:
1. 读取每个选中项目的 mapping.txt
2. 收集所有 confirmed=yes 的条目
3. 复制所有 WAV 到 logs/{exp_name}/recorded/ 并统一重命名
4. 生成合并的 train.list
5. 后续流程不变
```

#### 训练与项目的关系

```
训练完成后:
  - 训练只是读取了项目中的录音数据（复制过去）
  - 项目本身不变，可以继续录制
  - 删除项目 → 不影响已训练完成的模型（训练数据已复制到 logs/）
  - 删除模型 → 不影响项目（模型是模型，项目是项目）
```

---

## 5. 页面一：训练 > 录音

### 5.1 页面整体布局（含双子Tab导航）

```
┌──────────────────────────────────────────────────────────────────────┐
│  Tab栏: [训练]  [推理]                                               │
│  状态栏: 🟢 空闲 | 项目: 我的声音v1 | 已录制: 45/160 | ✅ 可训练    │
│  项目栏: [我的声音v1          ▼] [新建] [导入] [删除] [🔒锁定]      │
│  ──────────────────────────────────────────────────────────          │
│  子Tab:  [录音]  [训练]          ← 录音为默认打开                    │
│  ──────────────────────────────────────────────────────────          │
│                                                                       │
│  总体进度: 45/160 句 [█████████████████████░░░░░░░] 28%            │
│  ──────────────────────────────────────────────────────               │
│  [类1:声母韵母] 8/20 ✅  [类2:多音字] 6/20 ✅  [类3:轻声] 5/20 ✅  │
│  [类4:长句] 1/20 ❌  [类5:情感] 2/20 ❌  [类6:数字] 0/20 ❌        │
│  [类7:绕口令] 3/20 ❌  [类8:散文] 0/20 ❌                           │
│  ──────────────────────────────────────────────────────────          │
│  第 1/16 页 · 当前: 类别1：声母韵母全覆盖（基础音素）— 20条        │
│  ──────────────────────────────────────────────────────────          │
│                                                                       │
│  🟢 #1  妈妈买了一条灰色的围巾...  [REC] [PLAY▶] [DEL]   ← 绿色底  │
│  🟢 #2  哥哥踢足球的技术非常厉害..  [REC] [PLAY▶] [DEL]   ← 绿色底  │
│  ⚪ #3  我在超市买了苹果...         [REC] [PLAY▶] [DEL]   ← 无背景  │
│  🔴 #4  天空飘着几朵洁白的云彩..    [STOP] [PLAY▶] [DEL]  ← 红色底  │
│  ⚪ #5  我们一家人去附近的公园..    [REC] [PLAY▶] [DEL]   ← 无背景  │
│  ⚪ #6  早晨起床后我先刷牙洗脸..    [REC] [PLAY▶] [DEL]   ← 无背景  │
│  🟢 #7  他在图书馆里找到了一本..    [REC] [PLAY▶] [DEL]   ← 绿色底  │
│  ⚪ #8  我打算这个周末去爬梧桐山..  [REC] [PLAY▶] [DEL]   ← 无背景  │
│  ⚪ #9  我们约好下午三点钟...        [REC] [PLAY▶] [DEL]   ← 无背景  │
│  🟢 #10 父亲在教儿子骑自行车...     [REC] [PLAY▶] [DEL]   ← 绿色底  │
│                                                                       │
│  ──────────────────────────────────────────────────────────          │
│  本页: 4/10  |  [◀ 上一页]  第 1/16 页  [下一页 ▶]                  │
│  跳转到第 [   ] 页 [跳转]                                            │
│  ──────────────────────────────────────────────────────────          │
│  达标检查: 类1✅8 类2✅6 类3✅5 类4❌1 类5❌2 类6❌0 类7❌3 类8❌0 │
│  提示: 🔊 共录制45句 | 还需录制: 类4缺4 | 类5缺3 | ...              │
│  [❌ 完成录制 (需各类≥5条，当前5类未达标)]          ← 灰色禁用      │
└──────────────────────────────────────────────────────────────────────┘
```

### 5.2 句子行状态定义

| 状态名称     | has_audio | confirmed | 状态圆点 | 行背景色 | 录音按钮状态 | 播放按钮状态 | 清空按钮状态 |
|-------------|-----------|-----------|---------|---------|-------------|-------------|-------------|
| **未录制**   | false     | false     | ⚪ 灰色  | 无      | [REC] 可点击 | ⛔ 禁用(无文件) | ⛔ 禁用(无文件) |
| **录音中**   | -         | -         | 🔴 红色  | 浅红色  | [STOP] 可点击 | ⛔ 禁用(正在录制) | ⛔ 禁用(正在录制) |
| **已录制已确认** | true  | true      | 🟢 绿色  | 浅绿色  | [REC] 可点击(重录) | [PLAY▶] 可点击 | [DEL] 可点击 |

**关键规则**：
- 无音频文件 → Play 和 Del **按钮置灰禁用**
- 没有点击清空 = 自动确认可用于训练
- 已录制的句子再次点击 [REC] = 重新录制（覆盖旧文件）

### 5.3 按钮逻辑规范

#### 5.3.1 录音按钮 [REC] / [STOP]

**状态**：3种（未录制→点REC、已录制→点REC、已录制→重录）

**点击 [REC]（未录制/已录制状态）：**
```
前置检查:
  └ 检查 state.recording 是否已锁定 → 如果是，提示"请先停止当前录音"
  └ 检查项目是否锁定 → 如果锁定，提示"项目已锁定"
  
执行:
  1. 设置该行 → 🔴 红色背景，按钮变为 [STOP]
  2. 生成文件路径: recorded/p{page:03d}_i{idx:02d}_{timestamp}.wav
  3. recorder.start_recording(path) → 启动 sounddevice 线程(24000Hz, mono)
  4. state.recording = True
  5. 全局状态栏 → 🔴 录音中
  6. 禁用其他行的 [REC] 按钮
```

**点击 [STOP]（录音中状态）：**
```
执行:
  1. recorder.stop_recording() → 停止录音，保存WAV
  2. 设置该行 → 🟢 绿色背景，按钮恢复 [REC]
  3. 更新 mapping 内存缓存:
     mapping_data[idx] = {
       "text": text, "wav_path": saved_path,
       "confirmed": True, "duration": duration, "recorded_at": now
     }
  4. 异步写回 mapping.txt（写入队列，1秒内写入）
  5. state.recording = False
  6. 更新全局状态栏 → 🟢 空闲
  7. 更新进度区、类别检查、底部按钮状态
```

**边界情况**：
| 情况 | 行为 |
|------|------|
| 录音时翻页 | 阻止，提示"录音未保存，请先停止" |
| 录音时切项目 | 阻止，提示同上 |
| 录音时关浏览器 | 丢弃当前录音（未保存），已保存的不影响 |
| 录音时切到训练/推理Tab | ✅ 允许（只切视图），录音继续 |
| 文件写入失败 | 状态回退到"未录制"，显示错误提示 |
| 重录(已录制再点REC) | 先暂停播放(如果有)、删除旧WAV、开始新录音 |

#### 5.3.2 播放按钮 [PLAY▶]

**状态**：2种（无音频→禁用，有音频→可点击）

**点击 [PLAY▶]：**
```
前置检查:
  └ mapping_data[idx].wav_path 是否存在 → 不存在则状态回退到"未录制"
  
执行:
  1. 读取 WAV 文件
  2. 使用音频播放器播放（Gradio 内置或 HTML5 <audio>）
  3. 按钮变为 [▶播放中...] 并禁用
  4. 播放完成 → 恢复 [PLAY▶]
```

**播放期间的互斥**：
- 播放时点 [REC] → 停止播放，开始录音
- 播放时点 [DEL] → 停止播放，删除文件
- 同时只能播放一个音频

#### 5.3.3 清空按钮 [DEL]

**状态**：2种（无音频→禁用，有音频→可点击）

**点击 [DEL]：**
```
执行:
  1. 停止当前播放（如果有）
  2. 删除 wav_path 对应的文件
  3. 更新 mapping 内存缓存:
     mapping_data[idx] = {
       "text": text, "wav_path": "",
       "confirmed": False, "duration": 0, "recorded_at": 0
     }
  4. 异步写回 mapping.txt
  5. 行状态回退 → ⚪ 未录制（Play/Del 禁用）
  6. 更新进度区、类别检查
```

#### 5.3.4 翻页导航

**数据加载策略（内存缓存）：**
```
翻页触发 → 从内存缓存读取映射数据（不读磁盘）
  └ 启动时：mapping_data = load_mapping_from_disk("projects/当前项目/mapping.txt")
  └ 录音/清空后：更新 mapping_data，异步写回磁盘
  └ 翻页时：从 mapping_data 查询页内句子状态，生成UI
```

**上一页/下一页：**
```
[◀ 上一页]:
  current_page > 1 → page--, 刷新页面
  current_page = 1 → 按钮禁用

[下一页 ▶]:
  current_page < total_pages → page++, 刷新页面
  current_page = total_pages → 按钮禁用
```

**页码跳转：**
```
输入页码 → 校验 1 ≤ page ≤ total_pages
  → 不合法: 提示"请输入有效页码(1-{total_pages})"
  → 合法: 跳转，刷新页面
```

**翻页刷新内容：**
```
1. 从 script_data 获取本页10句文本
2. 从 mapping_data（内存）查询每句的录音状态
3. 生成句子列表HTML（含状态圆点、按钮状态）
4. 更新顶部分类名
5. 更新本页统计: "本页: X/10"
```

#### 5.3.5 类别达标检查 + 完成录制按钮

**类别达标检查（每次录音/清空后执行）：**
```
1. 遍历 mapping_data（内存），统计每个类别的 confirmed 数量
2. 生成检查结果:
   {
     "类别1：声母韵母": {"total": 20, "confirmed": 8, "met": True},
     "类别2：多音字与声调校准": {"total": 20, "confirmed": 6, "met": True},
     ...
   }
3. 更新底部状态文字:
   全部达标 → "所有类别均已满足条件 ✅"
   有未达标 → "还需录制: 类4缺4 | 类5缺3 | ..."
4. 更新 btn_done 状态
```

**完成录制按钮 [✅ 完成录制 → 开始训练]：**

| 条件 | 按钮状态 | 提示文字 |
|------|---------|---------|
| 全部8个类别 confirmed ≥ 5 | 绿色可点击 | **[✅ 完成录制 → 开始训练]** |
| 有类别 confirmed < 5 | 灰色禁用 | **[❌ 完成录制 (需各类≥5条，当前N类未达标)]** |
| 有录音进行中 | 灰色禁用 | **[❌ 请先停止录音]** |
| 项目已锁定 | 灰色禁用 | **[❌ 项目已锁定，不可训练]** |

**点击 [✅ 完成录制 → 开始训练]：**
```
1. 弹窗确认:
   "已录制 {N} 句，8个类别均已达标（各类≥5条）。
    是否跳转到训练页面开始训练？[确定] [取消]"
2. 用户点 [确定]:
   → 如有未录制的句子: 追加提示"还有 {M} 句未录制，确定跳过？"
   → 自动切换到"训练"子Tab
   → 将确认的录音数传递到训练页面
```

---

## 6. 页面一：训练 > 训练

### 6.1 完整页面布局

```
┌──────────────────────────────────────────────────────────────┐
│  Tab栏: [训练]  [推理]                                        │
│  状态栏: 🟡 训练中 (S2: 3/5) | 项目: 我的声音v1 | ...        │
│  ──────────────────────────────────────────────────────       │
│  子Tab: [录音]  [训练]          ← 当前在"训练"               │
│  ──────────────────────────────────────────────────────       │
│                                                                 │
│  ┌── [数据源] ─────────────────────────────────────────────┐  │
│  │  ● 当前项目 "我的声音v1" (45句已确认)                   │  │
│  │  ○ 多项目合并:  [我的声音v1] 45句 [英语句子] 20句 += 65│  │
│  │   进度: 🔊 共 65 句录音可用于训练                        │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌── [训练配置] ──────────────────────────────────────────┐  │
│  │  模型版本:  [v2Pro ▼]    实验名称: [my_voice_20260704 ]│  │
│  │                                                    │  │  │
│  │  S1训练轮数:  ██████●████████████████  15   (2~50) │  │  │
│  │  S2训练轮数:  ██████●████████████████   5   (1~20) │  │  │
│  │  批次大小:    █████●█████████████████   8   (1~32) │  │  │
│  │                                                    │  │  │
│  │  [▼ 高级设置]  ← gr.Accordion                    │  │  │
│  │  ┌────────────────────────────────────────────┐  │  │  │
│  │  │ 学习率比例: 0.40 (0.2~0.6)                │  │  │  │
│  │  │ 精度模式: [32-bit ▼]  |  保存频率: [1]轮  │  │  │  │
│  │  │ ☑ 保存中间权重  ☑ 仅保留最新检查点        │  │  │  │
│  │  │ ☐ 启用 DPO (实验性)                       │  │  │  │
│  │  └────────────────────────────────────────────┘  │  │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌── [训练控制] ────────────────────────────────────────────┐  │
│  │              [🚀 开始训练]                                │  │
│  ├──────────────────────────────────────────────────────────┤  │
│  │  整体进度: [███████████████████░░░░░░░░░]  56%            │  │
│  │  当前阶段: [S2训练中] 第3轮/共5轮  损失: 0.342           │  │
│  │  运行时间: 02:34  预计剩余: 03:12                        │  │
│  │  日志:                                                    │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │ 10:30:00 [数据准备] Step 1/4: 生成 train.list...   │  │  │
│  │  │ 10:30:05 [数据准备] Step 2/4: 提取音素特征...     │  │  │
│  │  │ 10:33:12 [S2训练]  开始训练，共5轮...             │  │  │
│  │  │ 10:33:45 [S2训练]  Epoch 1/5 loss_G=0.89...       │  │  │
│  │  │ 10:37:20 [S2训练]  Epoch 2/5 loss_G=0.56...       │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### 6.2 Gradio 组件清单

| # | 组件 ID            | Gradio 类型    | 说明 |
|---|-------------------|----------------|------|
| 1 | `data_source`     | gr.Radio       | "当前项目" / "多项目合并" |
| 2 | `project_selector`| gr.Dropdown    | 多项目合并时选择项目 |
| 3 | `total_count`     | gr.Markdown    | 显示录音总数 |
| 4 | `model_version`   | gr.Dropdown    | v1/v2/v3/v4/v2Pro/v2ProPlus |
| 5 | `exp_name`        | gr.Textbox     | 默认: {project_name}_{date} |
| 6 | `s1_epochs`       | gr.Slider      | 2~50, 默认15 |
| 7 | `s2_epochs`       | gr.Slider      | 1~20, 默认5 |
| 8 | `batch_size`      | gr.Slider      | 1~32, 默认auto |
| 9 | `advanced`        | gr.Accordion   | 高级设置面板 |
| 10 | `text_low_lr_rate`| gr.Slider      | 0.2~0.6, 默认0.4 |
| 11 | `precision`       | gr.Dropdown    | 32-bit / 16-mixed |
| 12 | `save_every_epoch`| gr.Number      | 默认1 |
| 13 | `if_save_every_weights` | gr.Checkbox | 默认勾选 |
| 14 | `if_save_latest`  | gr.Checkbox    | 默认勾选 |
| 15 | `if_dpo`          | gr.Checkbox    | 默认不勾选 |
| 16 | `btn_train`       | gr.Button      | 开始训练 |
| 17 | `progress_bar`    | gr.HTML        | 进度条 |
| 18 | `phase_text`      | gr.Markdown    | 阶段和轮次 |
| 19 | `time_text`       | gr.Markdown    | 时间信息 |
| 20 | `log_output`      | gr.Textbox     | 日志（只读） |

### 6.3 训练管线（自动串行执行）

#### 管线总览

```
[开始训练]
    │
    ▼
┌──────────────────────────────────────────────┐
│  Step 0: 数据收集 (5%)                        │
│  ├ 读取选中项目/多项目的 confirmed 录音        │
│  ├ 复制 WAV 到 logs/{exp_name}/recorded/      │
│  └ 生成 train.list                            │
├──────────────────────────────────────────────┤
│  Step 1: 数据集准备 (15%)                      │
│  ├ 1-get-text.py → 2-name2text.txt            │
│  ├ 2-get-hubert-wav32k.py → 4-cnhubert/*.pt   │
│  ├ [v2Pro] 2-get-sv.py → sv_emb/*.pt          │
│  └ 3-get-semantic.py → 6-name2semantic.tsv    │
├──────────────────────────────────────────────┤
│  Step 2: S2 SoVITS 训练 (50%)                  │
│  ├ 加载 s2{version}.json 配置模板              │
│  ├ 覆盖用户参数 → 临时配置文件                 │
│  ├ 执行 s2_train.py --config {tmp}.json        │
│  └ 监控 stdout 解析进度                        │
├──────────────────────────────────────────────┤
│  Step 3: S1 GPT 训练 (30%)                     │
│  ├ 加载 s1{version}.yaml 配置模板              │
│  ├ 覆盖用户参数 → 临时配置文件                 │
│  ├ 执行 s1_train.py --config_file {tmp}.yaml   │
│  └ 监控 stdout 解析进度                        │
├──────────────────────────────────────────────┤
│  Step 4: 完成处理 (5%)                          │
│  ├ 复制权重到 models/{exp_name}_{epochs}/      │
│  ├ 保存 meta.json                              │
│  ├ 更新 weight.json                            │
│  └ 弹窗 → 用户确认 → 切换到推理页面            │
└──────────────────────────────────────────────┘
```

#### 模型版本配置映射

| 版本      | S2 配置模板                 | S1 配置模板        | S2 训练脚本              | 额外步骤 |
|-----------|----------------------------|--------------------|--------------------------|---------|
| v1        | `configs/s2.json`          | `s1.yaml`          | `s2_train.py`            | 无      |
| v2        | `configs/s2.json`          | `s1longer-v2.yaml` | `s2_train.py`            | 无      |
| v3        | `configs/s2v2Pro.json`     | `s1longer-v2.yaml` | `s2_train_v3_lora.py`    | LoRA    |
| v4        | `configs/s2v2Pro.json`     | `s1longer-v2.yaml` | `s2_train_v3_lora.py`    | LoRA    |
| **v2Pro** | `configs/s2v2Pro.json`     | `s1longer-v2.yaml` | `s2_train.py`            | 2-get-sv |
| v2ProPlus | `configs/s2v2ProPlus.json` | `s1longer-v2.yaml` | `s2_train.py`            | 2-get-sv |

权重保存目录：
- `SoVITS_weights_{version}/` 和 `GPT_weights_{version}/`（上游规范）
- 同时复制一份到 `models/{exp_name}_{epochs}/`（统一管理）

#### 进度监控方式

训练脚本通过 `subprocess.Popen` 执行，实时捕获 stdout/stderr：

```
S2 输出解析:
  re: epoch[: ]*(\d+)[/:, ]+(\d+)  → 当前轮/总轮数
  re: loss[_ ]?[Gg][:\s]*([\d.]+)  → Generator Loss
  re: step[: ]+(\d+)                → 步数

S1 输出解析:
  re: Epoch (\d+)/(\d+)            → 当前轮/总轮数
  re: Loss:\s*([\d.]+)             → 损失值
  re: train_loss_epoch:\s*([\d.]+)  → 轮次平均损失
```

#### 训练完成处理

```python
def on_training_complete(training_result):
    # 1. 创建模型目录
    model_dir = f"models/{config.exp_name}_e{config.s1_epochs}_s{config.s2_epochs}"
    os.makedirs(model_dir, exist_ok=True)
    
    # 2. 复制权重
    shutil.copy(f"GPT_weights_{version}/{exp_name}_e{s1}.ckpt", f"{model_dir}/gpt.ckpt")
    shutil.copy(f"SoVITS_weights_{version}/{exp_name}_e{s2}.pth", f"{model_dir}/sovits.pth")
    
    # 3. 保存元信息
    meta = {
        "exp_name": exp_name,
        "version": version,
        "s1_epochs": s1_epochs,
        "s2_epochs": s2_epochs,
        "batch_size": batch_size,
        "data_source": "我的声音v1(45句) + 英语句子(20句)",
        "total_recordings": total_count,
        "trained_at": datetime.now().isoformat(),
        "s1_loss": final_s1_loss,
        "s2_loss": final_s2_loss,
    }
    json.dump(meta, open(f"{model_dir}/meta.json", "w"), ensure_ascii=False, indent=2)
    
    # 4. 更新 weight.json
    update_weight_json(f"GPT_weights_{version}/{exp_name}_e{s1}.ckpt",
                       f"SoVITS_weights_{version}/{exp_name}_e{s2}.pth", version)
    
    # 5. 弹窗
    gr.Info(f"训练完成！模型已保存至 {model_dir}")
```

### 6.4 [开始训练] 按钮逻辑

**点击前置检查：**
```
□ state.training == False                   → 否则拒绝
□ state.inference_api == False              → 否则提示"请先关闭推理服务"
□ state.recording == False                  → 否则提示"请先停止录音"
□ 数据源有至少5条 confirmed 录音             → 否则提示"数据不足"
□ exp_name 合法（字母数字下划线）             → 否则提示
□ 预训练模型文件存在                         → 否则提示
□ 磁盘空间 > 10GB                           → 否则提示
```

**检查不通过 → 按钮禁用并显示对应提示文字**

**点击后：**
```
state.training = True
禁用所有参数组件和项目切换
按钮变为 [⏳ 训练中...] 禁用
启动后台线程执行训练管线
```

**训练结束后：**
```
state.training = False
恢复参数组件和项目切换
按钮恢复 [🚀 开始训练]
弹窗完成提示
```

**错误处理：**
- 训练中途失败 → 显示红色错误信息，解锁状态
- S2成功S1失败 → 回滚 weight.json，提示 S1 失败
- 显存不足 → 提示降低批次大小
- 用户关浏览器 → 训练子进程被 `/quit` 清理

---

## 7. 页面二：推理

### 7.1 完整页面布局

```
┌──────────────────────────────────────────────────────────────┐
│  Tab栏: [训练]  [推理]                                        │
│  状态栏: 🔵 推理中 | 项目: 我的声音v1 | ...                   │
│  ──────────────────────────────────────────────────────       │
│                                                                 │
│  ┌── [参考音频] ────────────────────────────────────────────┐  │
│  │  ◉ 从已录语句中选择                                      │  │
│  │     ┌────────────────────────────────────────────────┐  │  │
│  │     │ [妈妈买了一条灰色的围巾...               ▼]    │  │  │
│  │     └────────────────────────────────────────────────┘  │  │
│  │     参考文字已自动填入 ✅                                 │  │
│  │                                                           │  │
│  │  ○ 上传自定义参考音频                                    │  │
│  │     音频: [选择文件] (.wav)                                │  │
│  │     文字: [_____________________________] [🎤ASR识别]    │  │
│  │                                                           │  │
│  │  ○ 现场录制参考音频  ← 本次新增                           │  │
│  │     [🔴 点击录音] → 录3秒自动停止  [▶播放] [重新录制]   │  │
│  │     文字: [_____________________________]                 │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌── [模型选择] ────────────────────────────────────────────┐  │
│  │  GPT模型:    [my_voice_20260704_e15.ckpt           ▼]   │  │
│  │  SoVITS模型: [my_voice_20260704_e5.pth            ▼]   │  │
│  │              [🔄 刷新模型列表]                            │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌── [推理参数] ────────────────────────────────────────────┐  │
│  │  合成文本:                                                 │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │ 请输入要合成语音的文字内容...                       │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  │                                                           │  │
│  │  文本语言: [中文 ▼]   参考语言: [中文 ▼]                 │  │
│  │                                                           │  │
│  │  Top-K: 15    Top-P: 1.00    温度: 1.00   速度: 1.00    │  │
│  │  分割方式: [按标点符号切 ▼]  种子: [-1 (随机)   ]      │  │
│  │                                                           │  │
│  │  ☐ 流式输出  ☐ 超采样  ☐ 并行推理                        │  │
│  │                                                           │  │
│  │  [🎧 合成语音]                                           │  │
│  │                                                           │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │  ▶ ═══════════════════════════════════             │  │  │
│  │  │  00:03 / 00:08              [💾 保存到文件]        │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### 7.2 参考音频三种模式

| 模式 | 说明 | 适用场景 |
|------|------|---------|
| **从已录语句中选择** | 下拉框显示当前项目所有 confirmed 句子，选中后自动填入文字 | 最常用，和录音流程无缝衔接 |
| **上传自定义参考音频** | 上传任意 WAV，手动或 ASR 识别文字 | 想用其他来源的音频做参考 |
| **现场录制参考音频** | 点击录音，快速录一句作为参考 | 临时起意，不想去录音页面 |

### 7.3 现场录制逻辑

```
用户选择 [现场录制参考音频]
  → 显示 [🔴 点击录音]
  → 点击后开始录音（与录音页共用 recorder.py）
  → 录音 3~5 秒后自动停止（或点[停止]提前结束）
  → 保存到 projects/{当前项目}/reference/{timestamp}.wav
  → 自动设为当前参考音频
  → 用户可手动输入文字
```

---

## 8. 数据层设计

### 8.1 script.txt 格式

```
1	妈妈买了一条灰色的围巾...
2	哥哥踢足球的技术非常厉害...
...
类别1：声母韵母全覆盖（基础音素）— 20条
...
21	类别2：多音字与声调校准（准确度）— 20条
22	银行门口排队的行人正在等待办理业务...
...
```

**解析规则：**
- 每行格式：`序号\t内容`
- 类别标题行以"类别"开头 → 不计入句子数，作为分类信息
- 空行跳过
- 序号从1递增

### 8.2 mapping.txt 格式

```
idx|text|wav_path|confirmed|duration_sec|recorded_at
1|妈妈买了一条灰色的围巾...|recorded/p001_i01_20260704_103000.wav|yes|4.2|20260704_103000
2|哥哥踢足球的技术非常厉害...|recorded/p001_i02_20260704_103105.wav|yes|3.8|20260704_103105
3|我在超市买了苹果...||no|0|0
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `idx` | int | 句子序号，与 script.txt 对应 |
| `text` | str | 句子原文（冗余存储，保证配对不丢失） |
| `wav_path` | str | 录音文件相对路径，空=未录制 |
| `confirmed` | str | `yes`=确认, `no`=未确认 |
| `duration_sec` | float | 录音时长(秒)，0=未录制 |
| `recorded_at` | str | 录制时间戳 `YYYYMMDD_HHmmss`，0=未录制 |

### 8.3 音频↔文字配对保证

```
录音时:
  mapping.txt 一行同时保存 wav_path + text
  ↓
翻页时:
  从内存缓存读，不读盘
  ↓
训练收集时:
  读 mapping → 逐行生成 train.list（每行 = 音频路径|说话人|语言|文字）
  ↓
数据集准备:
  train.list 是后续所有处理脚本的唯一输入源
  ↓
结论: 每个 WAV ↔ 对应文字 从录音那一刻就绑定，永不分离
```

**如果 script.txt 修改了某句文字：**
- mapping.txt 中已有录音的 text 字段**不会变**
- 如果要更新文字，需要清空该条重新录

### 8.4 内存缓存策略

```python
# 全局变量（不在 gr.State 中，在模块级）
mapping_cache = {}        # Dict[int, MappingEntry] — 全部 mapping 数据
mapping_dirty = False    # 是否有未写盘的变更
script_data = None        # ScriptData — 脚本解析结果

# 写盘策略：变更后立即标记 dirty，1秒内批量写入
```

---

## 9. 模型组织与保存

### 9.1 目录结构

```
models/                              # 统一管理所有训练完成的模型
├── my_voice_e15_s5_20260704/        # 目录名: {exp_name}_e{s1轮}_s{s2轮}_{日期}
│   ├── gpt.ckpt                     # GPT 权重（复制，非软链）
│   ├── sovits.pth                   # SoVITS 权重
│   ├── meta.json                    # 训练元信息
│   ├── .locked                      # 可选，存在=锁定（不可删除）
│   └── sample/                      # 可选，录音样本
│       └── ref_001.wav
│
├── my_voice_v2_e20_s8_20260705/
│   ├── gpt.ckpt
│   ├── sovits.pth
│   ├── meta.json
│   └── .locked
│
└── my_voice_e15_s5/                 # ← 兼容旧命名（无日期）
    └── ...
```

### 9.2 meta.json 格式

```json
{
  "exp_name": "my_voice_20260704",
  "model_version": "v2Pro",
  "s1_epochs": 15,
  "s2_epochs": 5,
  "batch_size": 8,
  "data_source": [
    {"project": "我的声音v1", "recordings": 45},
    {"project": "英语句子", "recordings": 20}
  ],
  "total_recordings": 65,
  "final_s1_loss": 0.234,
  "final_s2_loss": 0.089,
  "trained_at": "2026-07-04T10:30:00",
  "trained_on_gpu": "NVIDIA RTX 4060",
  "inference_params": {
    "top_k": 15,
    "top_p": 1.0,
    "temperature": 1.0,
    "speed": 1.0
  }
}
```

### 9.3 模型操作

| 操作 | 说明 |
|------|------|
| **查看** | 在推理页模型下拉框选择，meta.json 信息可展开 |
| **删除** | 在模型管理弹窗中，未锁定模型可删除 |
| **锁定** | 加 `.locked` 文件，防止误删 |
| **加载** | 自动扫描 `models/` 和 `GPT_weights_*/SoVITS_weights_*/` 目录 |
| **设为默认** | 选中后自动更新 `weight.json` |
| **导出** | 将模型目录打包为 zip（可选功能） |

### 9.4 模型管理入口

在推理页顶部加一个 **[📦 模型管理]** 按钮：

```
点击 → 弹窗显示模型列表:

┌────────────────────────────────────────────────────┐
│  📦 模型管理                    [关闭]              │
│  ──────────────────────────────────────────────     │
│  ☐ my_voice_e15_s5_20260704  v2Pro  7月4日  🔒    │
│  ☐ my_voice_v2_e20_s8       v4     7月2日         │
│  ☐ prototype_lora            v3     6月28日 🔒    │
│                                                     │
│  [设为默认] [删除选中] [导出选中] [刷新列表]        │
└────────────────────────────────────────────────────┘
```

---

## 10. 技术架构

### 10.1 文件结构

```
voice/
├── sentence_webui.py              # 主入口，Gradio Blocks 应用
├── start_recorder.bat             # Windows 启动脚本
├── reference.txt                  # 默认语料（不修改）
├── CLAUDE.md
│
├── sentence_recorder/             # 核心逻辑模块
│   ├── __init__.py
│   ├── recorder.py                # 录音管理（sounddevice 单例）
│   ├── state.py                   # 全局状态互斥
│   ├── script_reader.py           # 脚本文件解析
│   ├── mapping.py                 # mapping.txt 读写 + 内存缓存
│   ├── project_manager.py         # 项目创建/切换/删除/锁定
│   ├── model_utils.py             # 模型扫描、API 管理
│   └── training_pipeline.py       # 训练管线编排
│
├── sentence_tabs/                 # Tab 页面模块
│   ├── __init__.py
│   ├── tab_recording.py           # 录音页面
│   ├── tab_training.py            # 训练页面
│   └── tab_inference.py           # 推理页面
│
├── projects/                      # 项目数据
│   └── default/
│       ├── script.txt
│       ├── mapping.txt
│       └── recorded/
│
├── models/                        # 训练完成的模型（训练时自动创建）
│
└── GPT-SoVITS-v2pro-20250604-nvidia50/  # ⛔ 不动
```

### 10.2 模块职责与关键方法

#### `sentence_recorder/recorder.py`

```python
class RecordingManager:
    """全局录音管理器（单例）"""
    def start_recording(self, save_path: str) -> None
    def stop_recording(self) -> Optional[Dict]  # {path, duration, sr}
    @property
    def is_recording(self) -> bool
```

#### `sentence_recorder/state.py`

```python
class AppState:
    training: bool = False
    inference_api: bool = False
    recording: bool = False  # 通过 RecordingManager
    current_project: str = "default"
```

#### `sentence_recorder/script_reader.py`

```python
class ScriptReader:
    def read_script(self, path: str) -> ScriptData
    def get_page(self, page: int, page_size: int = 10) -> PageData
    def get_categories(self) -> List[CategoryInfo]
    def get_sentence_category(self, idx: int) -> str
```

#### `sentence_recorder/mapping.py`

```python
class MappingManager:
    def load(self, path: str) -> Dict[int, MappingEntry]  # 磁盘→内存
    def flush(self) -> None                                # 内存→磁盘
    def update(self, idx: int, entry: MappingEntry) -> None
    def get_confirmed(self) -> List[MappingEntry]
    def get_category_stats(self, category_map: Dict) -> Dict[str, int]
    def auto_flush_loop(self) -> None                      # 1秒延迟批量写
```

#### `sentence_recorder/project_manager.py`

```python
class ProjectManager:
    def list_projects(self) -> List[str]
    def create(self, name: str, script_source: str, import_from: str) -> bool
    def switch(self, name: str) -> bool
    def delete(self, name: str) -> bool
    def lock(self, name: str) -> None
    def unlock(self, name: str) -> None
    def is_locked(self, name: str) -> bool
    def import_project(self, path: str) -> str
```

#### `sentence_recorder/model_utils.py`

```python
def scan_gpt_weights(version: str) -> List[str]
def scan_sovits_weights(version: str) -> List[str]
def scan_model_dirs() -> List[Dict]            # 扫描 models/
def start_inference_api(port: int = 9880) -> bool
def stop_inference_api(port: int = 9880) -> None
def is_api_running(port: int = 9880) -> bool
def update_weight_json(gpt_path: str, sovits_path: str, version: str) -> None
```

#### `sentence_recorder/training_pipeline.py`

```python
class TrainingPipeline:
    def __init__(self, config: TrainingConfig)
    def run(self, progress_callback: Callable) -> TrainingResult
    def cancel(self) -> None

class TrainingConfig:
    exp_name: str
    model_version: str
    data_sources: List[str]          # 项目名列表
    s1_epochs: int
    s2_epochs: int
    batch_size: int
    text_low_lr_rate: float
    precision: str
    save_every_epoch: int
    if_save_every_weights: bool
    if_save_latest: bool
    if_dpo: bool
```

### 10.3 状态互斥规则

| 操作 \ 当前状态 | 🟢 空闲 | 🔴 录音中 | 🟡 训练中 | 🔵 推理中 |
|----------------|--------|----------|----------|----------|
| 开始录音       | ✅      | ❌        | ❌        | ✅       |
| 开始训练       | ✅      | ❌        | ❌        | ❌       |
| 开始推理       | ✅      | ✅        | ❌        | ❌       |
| 切项目         | ✅      | ❌        | ❌        | ✅       |
| 删项目         | ✅      | ❌        | ❌        | ✅       |
| 关浏览器       | ✅      | 丢弃当前  | 强制停止  | 强制停止  |

### 10.4 平台约束

| 约束 | 解决方案 |
|------|---------|
| Gradio gr.State 必须在 Blocks 顶层 | 所有 State 在 `sentence_webui.py` 顶层定义 |
| Windows GBK 不能输出 emoji | print 中 emoji 替换为 ASCII 标记 |
| subprocess.([list], shell=True) 不生效 | 用字符串命令 |
| 端口残留 | 启动时 `kill_process_on_port()` |
| 长路径 > 260 字符 | 使用相对路径 |
| GPU 互斥 | `state.py` 管理 |

### 10.5 emoji → ASCII 映射

```python
EMOJI_MAP = {
    "🔴": "[REC]", "▶": "[PLAY]", "🗑": "[DEL]",
    "✅": "[OK]", "❌": "[NO]", "🚀": "[START]",
    "🎧": "[TTS]", "⏳": "[WAIT]", "⚪": "[EMPTY]",
    "🟢": "[DONE]", "🔄": "[REFRESH]", "💾": "[SAVE]",
    "🎤": "[ASR]", "◀": "[PREV]", "▶▶": "[NEXT]",
    "⚠": "[WARN]", "🔒": "[LOCK]", "📦": "[MODEL]",
    "🟡": "[TRAIN]", "🔵": "[INFER]",
}
```

---

## 11. 实施计划

### Step 1: 基础设施

- [ ] 创建目录结构
- [ ] `start_recorder.bat` — 设置编码、启动
- [ ] `sentence_webui.py` — 骨架（Gradio Blocks + 端口清理 + 自动打开浏览器 + /quit 端点）

### Step 2: 核心模块

- [ ] `state.py` — 全局状态
- [ ] `recorder.py` — 录音管理
- [ ] `script_reader.py` — 脚本解析
- [ ] `mapping.py` — mapping 读写 + 内存缓存
- [ ] `project_manager.py` — 项目管理

### Step 3: 录音页面

- [ ] `tab_recording.py` — 页面布局、句子列表、录音/播放/清空、翻页、进度区、类别检查、完成按钮

### Step 4: 训练管线 + 训练页面

- [ ] `training_pipeline.py` — 训练编排、子进程管理、进度监控
- [ ] `tab_training.py` — 页面布局、参数配置、多项目数据源

### Step 5: 推理页面

- [ ] `tab_inference.py` — 页面布局、三种参考音频、API 管理、合成调用
- [ ] `model_utils.py` — 模型扫描、API 启停

### Step 6: 模型管理

- [ ] 模型保存逻辑（meta.json）
- [ ] 模型管理弹窗
- [ ] weight.json 自动更新

---

## 附录：完整数据流（录音→训练→推理）

```
用户选择项目 → 加载 script.txt + mapping.txt
       │
       ▼ 逐句录制
每个句子: 点[REC] → 录WAV → mapping更新(音频↔文字绑定)
       │
       ▼ 各类别≥5条
点[完成录制] → 弹窗确认 → 自动切换到训练Tab
       │
       ▼ 选择数据源
点[开始训练] → 收集录音 → train.list
       │
       ▼
logs/{exp_name}/
├── recorded/*.wav        ← 录入音频
├── train.list            ← 配对表（永久绑定）
├── 2-name2text.txt       ← 音素
├── 4-cnhubert/*.pt       ← 语义特征
├── 6-name2semantic.tsv   ← 语义token
├── logs_s1_{v}/          ← S1检查点
└── logs_s2_{v}/          ← S2检查点
       │
       ▼ 训练完成
GPT_weights_{v}/{exp_name}_e{s1}.ckpt
SoVITS_weights_{v}/{exp_name}_e{s2}.pth
       │
       ├→ models/{exp_name}_{epochs}/  (复制)
       │   ├── gpt.ckpt
       │   ├── sovits.pth
       │   └── meta.json
       │
       └→ weight.json 更新
       │
       ▼ 用户点确定
切换到推理Tab → 模型下拉框选中新模型
选参考音频 → 输入文字 → 合成语音
```

---

*文档结束*
