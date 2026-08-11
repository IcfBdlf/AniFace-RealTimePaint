# 🎨 AniFace 项目完全指南

> 本文档面向初学者，逐文件、逐行解释整个项目。
> 适合作为面试项目展示的参考材料。

---

## 目录

1. [项目是干什么的](#1-项目是干什么的)
2. [用到了哪些技术](#2-用到了哪些技术)
3. [项目文件结构](#3-项目文件结构)
4. [逐文件详解](#4-逐文件详解)
   - [config.py — 全局配置](#41-configpy--全局配置)
   - [canvas_stream.py — 程序入口](#42-canvas_streampy--程序入口)
   - [src/pipeline.py — 模型加载](#43-srcpipelinepy--模型加载与管线初始化)
   - [src/renderer.py — 渲染管理器](#44-srcrendererpy--渲染管理器)
   - [src/app.py — GUI 应用](#45-srcapppy--gui-应用界面)
   - [test_thread_safety.py — 线程安全测试](#46-test_thread_safetypy--线程安全单元测试)
   - [test_stream.py — 集成测试](#47-test_streampy--集成测试)
   - [requirements.txt — 依赖清单](#48-requirementstxt--依赖清单)
   - [.gitignore — Git 忽略规则](#49-gitignore--git-忽略规则)
   - [CHANGELOG.md — 工作日志](#410-changelogmd--工作日志)
   - [README.md — 项目说明](#411-readmemd--项目说明)
5. [项目的核心设计思想](#5-项目的核心设计思想)
6. [面试可能会问的问题](#6-面试可能会问的问题)

---

## 1. 项目是干什么的

**AniFace** 是一个"手绘线稿实时转AI动漫图"的桌面应用。

- 你在**左边的画布**上用鼠标画一个简笔画
- **右边实时**看到AI把它变成精美的动漫风格头像
- 松手后自动触发**25步高清渲染**，出成品质量的图

就像一个非常智能的"自动上色+风格化"工具。你只需画个轮廓，AI帮你补全所有细节。

---

## 2. 用到了哪些技术

| 技术 | 是干什么的 |
|------|-----------|
| **Python** | 主语言 |
| **Tkinter** | Python内置的GUI库，用来做窗口、按钮、画布 |
| **Stable Diffusion** | AI画画的核心模型。把随机噪声"去噪"成图片 |
| **ControlNet (Scribble)** | SD的"插件"，让AI照着你的线稿来画，而不是乱画 |
| **StreamDiffusion** | 对SD做了"流水线"加速，每步推理之间可以同时处理不同帧，实现低延迟实时预览 |
| **PIL (Pillow)** | Python图像处理库，负责画布上的像素操作 |
| **PyTorch + CUDA** | GPU深度学习框架，所有AI推理都在RTX 4080显卡上跑 |
| **LoRA** | 模型微调技术。可以挂载不同风格的小文件（如某个画师的风格），不用重新训练整个模型 |
| **多线程** | GUI主线程 + AI推理后台线程，互不阻塞 |
| **unittest + Mock** | 单元测试，不启动GPU也能验证线程安全 |

### 数据流图

```
用户画线（鼠标）
    │
    ▼
Tkinter Canvas (GUI层)        PIL Image (内存画布)
    │                               │
    │  同步更新两处                  │  .copy() 快照
    │                               ▼
    │                      RenderManager (渲染管理器)
    │                               │
    │              ┌────────────────┼────────────────┐
    │              ▼                                 ▼
    │      StreamDiffusion                 Stable Diffusion
    │      (实时预览, ~100ms)              (25步超清, ~几秒)
    │              │                                 │
    │              └────────────────┬────────────────┘
    │                               ▼
    │                      root.after() 回调
    │                               │
    └───────────────────────────────►│
                                    ▼
                          右侧输出区域显示结果
```

---

## 3. 项目文件结构

```
AniFaceProject/
├── canvas_stream.py          # 🚪 程序入口（30行，双击/命令行运行它）
├── config.py                 # ⚙️ 所有可调整的参数
├── requirements.txt          # 📦 依赖包清单
├── README.md                 # 📖 项目简介
├── PROJECT_GUIDE.md          # 📚 本文件 — 详细学习指南
├── CHANGELOG.md              # 📝 工作日志
├── .gitignore                # 🚫 Git忽略规则
├── src/                      # 📁 源代码包
│   ├── __init__.py           # 标记src为Python包（空文件）
│   ├── pipeline.py           # 🔧 模型加载 & 管线初始化
│   ├── renderer.py           # 🧵 渲染线程管理
│   └── app.py                # 🖥️ GUI界面 & 事件处理
├── assets/                   # 🖼️ 示例素材
│   └── sample_sketch.png     #   测试用线稿图片
├── test_stream.py            # 🧪 集成测试（需要GPU）
└── test_thread_safety.py     # 🔒 线程安全测试（无需GPU）
```

### 为什么拆成 src/ 目录

最初所有代码都在 `canvas_stream.py`（350行），后来按**单一职责原则**拆成了三个模块：

- **pipeline.py** — 只管"模型从哪来"
- **renderer.py** — 只管"怎么调度渲染线程"
- **app.py** — 只管"界面长什么样、用户怎么交互"

这样每个文件只做一件事，改界面不影响渲染，改渲染不影响模型加载。面试时这是一个很好的谈点。

---

## 4. 逐文件详解

### 4.1 `config.py` — 全局配置

**作用**：把所有"魔法数字"集中到一个地方管理。

```python
# 模型从哪里加载
CONTROLNET_MODEL_ID = "lllyasviel/sd-controlnet-scribble"
BASE_MODEL_ID = "stablediffusionapi/counterfeit-v30"

# 预览用2步快速推理，超清用25步精细推理
PREVIEW_NUM_INFERENCE_STEPS = 2
HQ_NUM_INFERENCE_STEPS = 25

# 画布512x512像素
IMAGE_SIZE = (512, 512)

# 画笔粗细和颜色
PEN_WIDTH = 4
PEN_COLOR = "black"

# 优雅退出最多等5秒
SHUTDOWN_MAX_ATTEMPTS = 25  # 25次 × 200ms = 5秒
```

**设计意图**：如果你想把画布改成 768×768，或者把笔刷改成红色，只需改这一个文件。不会因为在代码里到处找 `512` 而漏掉某个地方。

**知识点**：这体现了 **DRY原则（Don't Repeat Yourself）** 和 **关注点分离**。

---

### 4.2 `canvas_stream.py` — 程序入口

**作用**：整个项目的启动文件，双击或命令行运行它。

```python
"""AniFaceProject 入口 — 实时线稿转动漫交互画板

用法:
    PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 python canvas_stream.py
"""
import logging
import sys
import tkinter as tk

from src.pipeline import create_pipelines    # ← 加载模型
from src.app import RealTimePaintApp         # ← 创建界面

# ---- 日志配置 ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),           # 输出到终端
        logging.FileHandler("aniface.log", encoding="utf-8"),  # 同时写入文件
    ],
)


def main():
    base_pipe, stream = create_pipelines()   # 1. 加载模型
    root = tk.Tk()                           # 2. 创建窗口
    RealTimePaintApp(root, base_pipe, stream) # 3. 初始化应用
    root.mainloop()                          # 4. 进入事件循环


if __name__ == "__main__":
    main()
```

**逐行讲解**：

| 行 | 做了什么 | 为什么 |
|----|---------|--------|
| `logging.basicConfig(...)` | 配置日志系统 | 把 `print()` 升级为专业的日志：有**时间戳**、有**级别**（INFO/WARNING/ERROR）、**同时输出到终端和文件**。用户双击运行看不到终端时，可以从 `aniface.log` 文件排查问题 |
| `StreamHandler(sys.stdout)` | 日志显示在终端 | 开发/调试时能实时看到 |
| `FileHandler("aniface.log")` | 日志写入文件 | 用户双击GUI时终端不可见，但日志持久化到文件 |
| `create_pipelines()` | 加载AI模型 | 见下节 pipeline.py |
| `tk.Tk()` | 创建GUI窗口 | Tkinter 的核心，一个Tk实例就是一个窗口 |
| `RealTimePaintApp(root, ...)` | 初始化应用 | 把所有控件（画布、按钮、输入框）放到这个窗口里 |
| `root.mainloop()` | 进入事件循环 | 程序停在这里，等待用户点击、画线、输入。每个操作触发一个回调函数 |

**知识点**：
- `if __name__ == "__main__"`：Python 的"入口守卫"，只有直接运行这个文件才执行 `main()`，被 `import` 时不执行
- `mainloop()`：GUI 程序的核心概念——不是像普通脚本那样从上到下跑完就退出，而是进入一个**死循环**等待事件
- `PYTHONIOENCODING=utf-8`：Windows 中文系统默认 GBK 编码，但代码里有 emoji（🎨），不设这个会报错

---

### 4.3 `src/pipeline.py` — 模型加载与管线初始化

**作用**：加载所有 AI 模型，并实现一个关键的优化——**两个管线共享同一份模型**。

```python
import logging
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from streamdiffusion import StreamDiffusion
import config

logger = logging.getLogger("AniFace.pipeline")


def create_pipelines(device=None):
    """加载模型，返回 (base_pipe, stream)。两个管线共享底层 UNet/VAE/TextEncoder。"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"正在配置高性能设备: {device}...")

    # 1. ControlNet（只加载一次）
    #    ControlNet 是一个"外挂"，告诉 SD "照着这张线稿来画"
    controlnet = ControlNetModel.from_pretrained(
        config.CONTROLNET_MODEL_ID, torch_dtype=torch.float16
    )

    # 2. 基础 Pipeline（UNet / VAE / TextEncoder 唯此一份）
    #    from_pretrained 从 HuggingFace 下载/加载预训练模型
    base_pipe = StableDiffusionControlNetPipeline.from_pretrained(
        config.BASE_MODEL_ID,
        controlnet=controlnet,
        torch_dtype=torch.float16,  # 半精度浮点，节省一半显存
    ).to(device)
    base_pipe.safety_checker = None  # 关闭NSFW过滤器（动漫图片用不上）

    # 3. 共享组件，创建 StreamDiffusion 专用 Pipeline（节省 ~5GB 显存）
    #    ⚠️ 这里是关键优化！
    #    如果再用 from_pretrained，会把 UNet/VAE/TextEncoder 再加载一遍
    #    每份模型约 5GB，两份就是 10GB，RTX 4080只有 12GB 显存，根本放不下
    #    所以这里直接复用 base_pipe 的组件
    stream_pipe = StableDiffusionControlNetPipeline(
        vae=base_pipe.vae,
        text_encoder=base_pipe.text_encoder,
        tokenizer=base_pipe.tokenizer,
        unet=base_pipe.unet,
        controlnet=base_pipe.controlnet,
        scheduler=base_pipe.scheduler,
        safety_checker=None,
        feature_extractor=base_pipe.feature_extractor,
        requires_safety_checker=False,
    )

    # 4. 把 stream_pipe 包进 StreamDiffusion 加速器
    #    t_index_list=[0,1] 表示只走2步去噪（正常是50步）
    #    虽然步数少，但每步之间可以"流水线"处理多个帧，实现实时
    stream = StreamDiffusion(
        stream_pipe,
        t_index_list=config.PREVIEW_T_INDEX_LIST,
        torch_dtype=torch.float16,
    )
    # 预热：提前把 prompt 编码成向量，后续推理不再重复编码
    stream.prepare(
        prompt=config.DEFAULT_PROMPT,
        num_inference_steps=config.PREVIEW_NUM_INFERENCE_STEPS,
    )

    return base_pipe, stream
```

**核心概念讲解**：

#### 什么是 Pipeline？
Stable Diffusion 不是一个单一模型，而是多个模型的"流水线"：

```
文字 → TextEncoder → 文本向量
                       ↓
随机噪声 → UNet（去噪） → 潜在表示 → VAE解码 → 图片
            ↑
      ControlNet（控制生成方向）
```

每个组件都是独立的神经网络。`StableDiffusionControlNetPipeline` 把这些组件打包成一个方便调用的对象。

#### 什么是模型共享？
`base_pipe` 和 `stream_pipe` 是两个 Pipeline 实例，但它们**共享同一份** UNet、VAE、TextEncoder、ControlNet。这就像两个方向盘操控同一台发动机，而不是两台发动机。

- `base_pipe`：用于 25 步高清渲染
- `stream_pipe`：用于 2 步实时预览
- 底层模型的权重数据只有**一份**在 GPU 显存里

**知识点**：
- `torch.float16`：半精度（16位浮点数）。正常是 float32（32位），砍一半精度，显存也砍一半，但画质几乎无差别
- `torch.device("cuda")`：把计算分配到 GPU
- `.to(device)`：把模型参数从 CPU 内存搬到 GPU 显存

---

### 4.4 `src/renderer.py` — 渲染管理器

**作用**：负责所有 AI 推理的后台线程调度。它是 GUI 和模型之间的"中间人"。

```python
import threading
import logging
from streamdiffusion.image_utils import postprocess_image
import config

logger = logging.getLogger("AniFace.renderer")


class RenderManager:
    """管理 AI 渲染管线、后台线程和渲染状态。
    
    通过回调与 GUI 层解耦：
    - schedule(fn): 将 fn 调度到主线程执行（如 root.after）
    - on_frame(img): 右侧输出区域显示一帧结果
    - get_sketch(): 返回当前画布的副本（线程安全）
    """

    def __init__(self, base_pipe, stream, schedule_fn, on_frame_fn, get_sketch_fn):
        self.pipe = base_pipe          # 高清渲染用的pipeline
        self.stream = stream           # 实时预览用的stream
        self._schedule = schedule_fn    # 回调1: 调度主线程任务（=root.after）
        self._on_frame = on_frame_fn    # 回调2: 显示渲染结果
        self._get_sketch = get_sketch_fn # 回调3: 获取画布快照

        # 渲染状态
        self.is_rendering = False       # 是否正在渲染
        self.need_update_again = False  # 渲染中是否有新请求需要排队
        self._preview_thread = None     # 预览线程引用
        self._last_prompt = ""          # 最后一次请求的 prompt 快照

        self.state_lock = threading.Lock()        # 状态锁（保护以上变量）
        self._shutdown_event = threading.Event()  # 退出信号
```

**核心设计：三个回调**

`RenderManager` 不知道界面长什么样，也不直接操作画布。它只通过三个回调与外界交互：

| 回调 | 谁提供 | 作用 |
|------|--------|------|
| `schedule_fn` | `root.after` | 让渲染结果在**主线程**显示（Tkinter 只能在主线程操作UI） |
| `on_frame_fn` | `_on_frame_ready` | 把渲染好的 PIL 图片显示在右侧输出区 |
| `get_sketch_fn` | `_get_sketch_snapshot` | 安全地获取当前画布的副本（加锁） |

这叫做**依赖注入**和**控制反转**——`RenderManager` 不依赖具体的 GUI 类，只依赖接口。如果将来把 Tkinter 换成 PyQt，只需改回调函数，渲染逻辑不用动。

**方法讲解**：

```python
def request_preview(self, sketch_snapshot):
    """发起流式预览渲染（后台线程）。"""
    t = threading.Thread(
        target=self._run_preview, args=(sketch_snapshot,), daemon=True
    )
    t.start()
    with self.state_lock:
        self._preview_thread = t
```
- 在**新线程**中运行 `_run_preview`，主线程不会被阻塞
- `daemon=True`：守护线程，主程序退出时自动结束
- `state_lock` 保护 `_preview_thread` 的赋值

```python
def request_hq(self, sketch_snapshot, prompt):
    """请求 25 步超清重绘；若正在渲染则排队。"""
    with self.state_lock:
        if self.is_rendering:
            # 正在渲染中 → 只更新 prompt 和标志位，排队等待
            self.need_update_again = True
            self._last_prompt = prompt
            return
        self.is_rendering = True
        self._last_prompt = prompt

    # 空闲 → 立即启动渲染
    t = threading.Thread(
        target=self._run_hq, args=(sketch_snapshot, prompt), daemon=True
    )
    t.start()
```
- **排队机制**：如果上一笔还在渲染，新请求不丢弃，标记 `need_update_again=True`，等当前渲染完自动重试
- 所有状态检查和修改都在 `state_lock` 内，防止竞态

```python
def _run_hq(self, img_snapshot, prompt_snapshot):
    """后台线程：执行 25 步高清渲染。"""
    try:
        if self._shutdown_event.is_set():
            return  # 退出中，不做无用功
        result = self.pipe(
            prompt=prompt_snapshot,
            image=img_snapshot,
            num_inference_steps=config.HQ_NUM_INFERENCE_STEPS,
            controlnet_conditioning_scale=config.HQ_CONTROLNET_CONDITIONING_SCALE,
            controlnet_ending_step=config.HQ_CONTROLNET_ENDING_STEP,
        )
        output_image = result.images[0]
        if not self._shutdown_event.is_set():
            # ⚠️ 用 _schedule 而不是直接更新UI
            # 因为这是在后台线程，不能直接操作 Tkinter 控件
            self._schedule(lambda img=output_image: self._on_frame(img))
    except Exception as e:
        logger.error(f"⚠️ 超清重绘失败: {e}")
    finally:
        # 处理排队：如果渲染期间有新请求，立即再渲一次
        need_again = False
        with self.state_lock:
            self.is_rendering = False
            if self.need_update_again:
                self.need_update_again = False
                self.is_rendering = True  # 原子操作：清除旧flag + 标记开始
                need_again = True

        if need_again and not self._shutdown_event.is_set():
            latest_snapshot = self._get_sketch()
            latest_prompt = self._last_prompt
            t = threading.Thread(
                target=self._run_hq,
                args=(latest_snapshot, latest_prompt),
                daemon=True,
            )
            t.start()
```

**关键细节**：

1. **`finally` 块里的排队重试**：渲染完成后检查 `need_update_again`。如果用户在渲染期间画了新一笔，自动用最新的画布再渲一次。这保证了用户停下来时看到的一定是最新的画面。

2. **TOCTOU 修复**：`need_update_again=False` 和 `is_rendering=True` 在**同一个 `state_lock` 临界区**内完成。如果分开做（先清标志、再设渲染），中间可能被另一线程插入，导致状态混乱。这是 P0 线程安全修复的核心。

3. **`self._schedule(lambda img=output_image: ...)`**：为什么用 lambda？因为直接写 `self._schedule(lambda: self._on_frame(output_image))` 会有一个 Python 闭包陷阱——lambda 被调用时 `output_image` 可能已经被后续循环覆盖。用默认参数 `img=output_image` 把值"捕获"在定义时。

4. **`controlnet_ending_step=0.35`**：ControlNet 只在去噪的前 35% 步骤起作用，后面的步骤让 SD 自由发挥。这样即使导入的线稿有瑕疵，最终效果也不会被线稿"绑架"。

---

### 4.5 `src/app.py` — GUI 应用界面

**作用**：所有用户看到的东西——窗口、画布、按钮、输入框，以及用户操作的响应。

这是项目中最长的文件（约290行），因为 GUI 代码天然比较啰嗦——每个按钮、标签、输入框都需要指定位置、大小、文字、事件绑定。

```python
class RealTimePaintApp:
    """实时线稿转动漫交互画板。"""

    def __init__(self, root, base_pipe, stream):
        self.root = root
        self.root.title(config.WINDOW_TITLE)

        # ---- 创建渲染管理器 ----
        # 把三个回调传进去：怎么调度UI线程、怎么显示图片、怎么获取画布
        self.renderer = RenderManager(
            base_pipe=base_pipe,
            stream=stream,
            schedule_fn=self.root.after,      # Tkinter提供的线程安全调度函数
            on_frame_fn=self._on_frame_ready,  # 我们自己写的显示方法
            get_sketch_fn=self._get_sketch_snapshot,  # 我们自己写的画布快照方法
        )

        # ---- 画布状态 ----
        self.sketch_img = Image.new("RGB", config.IMAGE_SIZE, "white")
        self.draw_buffer = ImageDraw.Draw(self.sketch_img)
        self.canvas_lock = threading.Lock()

        # ---- 构建界面 ----
        self._build_ui()

        # ---- 注册窗口关闭回调 ----
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
```

#### 界面布局（`_build_ui()`）

```
┌─────────────────────────────────────────────────────┐
│ 🎨 AniFace 交互完全体                               │
├──────────────────┬──────────────────────────────────┤
│                  │                                  │
│   左侧画布       │     右侧输出                      │
│   (512x512)     │     (显示渲染结果)                  │
│                  │                                  │
├──────────────────┴──────────────────────────────────┤
│ 🎛️ AI 实时引导控制面板                               │
│ 提示词: [1girl, masterpiece...]  [💎25步超清] [📥导入] [🧹擦除] │
│ LoRA路径: [________] 权重: [===]  [📦挂载/更新LoRA]  │
└─────────────────────────────────────────────────────┘
```

#### 核心方法讲解

**1. 画线（`_on_paint`）**：用户按住鼠标在画布上拖动时，每移动一个像素触发一次。

```python
def _on_paint(self, event):
    x, y = event.x, event.y
    if self.last_x and self.last_y:
        # 在 Tkinter 画布上画一条线段（用户看到的）
        self.canvas.create_line(
            self.last_x, self.last_y, x, y,
            width=config.PEN_WIDTH, fill=config.PEN_COLOR,
            capstyle=tk.ROUND, smooth=True,
        )
        # 在 PIL Image 上面也画同样的线段（AI的输入）
        with self.canvas_lock:
            self.draw_buffer.line(
                [self.last_x, self.last_y, x, y],
                fill=config.PEN_COLOR, width=config.PEN_WIDTH,
            )

        # 节流（throttle）：0.15秒内只触发一次预览渲染
        now = time.time()
        if now - self.last_render_time > self.render_interval:
            preview_busy = self.renderer.check_busy()[0]
            if not preview_busy and not self.renderer.is_rendering:
                self.last_render_time = now
                with self.canvas_lock:
                    snapshot = self.sketch_img.copy()
                self.renderer.request_preview(snapshot)

    self.last_x, self.last_y = x, y
```

**为什么要画两次**？Tkinter Canvas（用户看到）和 PIL Image（AI读到）是两套独立的图像系统。必须同步更新。

**为什么需要节流**？鼠标移动事件非常密集（每秒几十上百次），如果每次移动都触发AI推理，GPU根本来不及。所以用 `RENDER_INTERVAL = 0.15` 秒做节流，每 0.15 秒最多触发一次。

**2. 松手触发高清（`_on_mouse_up`）**：

```python
def _on_mouse_up(self, event):
    self.last_x, self.last_y = None, None
    self._request_hq()
```

松手时重置连线起点（否则下一笔会从莫名其妙的位置连过来），然后触发 25 步高清渲染。

**3. 高清渲染请求（`_request_hq`）**：

```python
def _request_hq(self):
    prompt = self.prompt_var.get()     # 从GUI输入框读取当前prompt
    with self.canvas_lock:
        snapshot = self.sketch_img.copy()  # 线程安全地获取画布副本
    self.renderer.request_hq(snapshot, prompt)
```

注意：`prompt` 必须**在主线程**读取（因为 `StringVar` 不是线程安全的），然后作为参数传给渲染线程。这是 P0 修复的关键点。

**4. 导入线稿（`_import_sketch`）**：

```python
def _import_sketch(self):
    file_path = filedialog.askopenfilename(
        title="选择本地线稿图片",
        filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp *.webp")],
    )
    if not file_path:
        return

    try:
        imported = Image.open(file_path).convert("RGB").resize(config.IMAGE_SIZE)
        with self.canvas_lock:
            self.sketch_img = imported
            self.draw_buffer = ImageDraw.Draw(self.sketch_img)

        # 更新Tkinter画布显示
        self.sketch_tk_img = ImageTk.PhotoImage(self.sketch_img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.sketch_tk_img)

        self._request_hq()  # 自动触发渲染
    except Exception as e:
        logger.error(f"❌ 读取线稿失败: {e}")
        messagebox.showerror("读取错误", f"无法解析该图片文件。\n错误: {e}")
```

**`self.canvas.delete("all")`**：清除之前的水笔画痕，用导入的线稿图替换。

**5. LoRA 加载（`_load_lora`）**：

LoRA（Low-Rank Adaptation）是一种轻量级模型微调技术。一个 LoRA 文件通常只有几十MB，但它能让模型画风偏向某个特定风格（比如"新海诚风格"或"某个画师风格"）。

```python
def _load_lora(self):
    lora_path = self.lora_var.get().strip()

    # 检查：路径不能为空
    if not lora_path:
        messagebox.showwarning("提示", "请先填入 LoRA 文件路径！")
        return

    # 检查：渲染中不能加载
    if self.renderer.is_rendering:
        messagebox.showwarning("提示", "当前正在渲染中，请稍后再加载 LoRA")
        return

    try:
        self.renderer.load_lora(lora_path, self.lora_weight_var.get(),
                                self.prompt_var.get())
        messagebox.showinfo("成功", "LoRA 补丁挂载成功！")
    except Exception as e:
        messagebox.showerror("加载失败", f"错误信息: {e}")
    finally:
        self._request_hq()  # 无论如何都重绘一次，让用户看到新风格效果
```

**6. 优雅退出（`_on_closing` + `_check_shutdown`）**：

```python
def _on_closing(self):
    logger.info("🛑 正在安全关闭应用...")
    self.renderer.shutdown()              # 设置退出信号
    self._shutdown_attempts = 0
    self._shutdown_max_attempts = config.SHUTDOWN_MAX_ATTEMPTS
    self._check_shutdown()                # 开始轮询

def _check_shutdown(self):
    self._shutdown_attempts += 1
    preview_alive, hq_busy = self.renderer.check_busy()

    if (preview_alive or hq_busy) and self._shutdown_attempts < self._shutdown_max_attempts:
        # 还有线程在跑，200ms后再检查
        self.root.after(config.SHUTDOWN_CHECK_INTERVAL_MS, self._check_shutdown)
    else:
        # 所有线程结束 或 超时 → 关窗口
        self.root.destroy()
```

**为什么不能直接 `root.destroy()`**？如果后台线程正在用 CUDA 推理，直接销毁窗口会让 CUDA 的资源被意外释放，产生 `CUDA error` 异常。正确的做法是先通知线程"该停了"（`shutdown_event.set()`），等线程自己结束，再关窗口。

---

### 4.6 `test_thread_safety.py` — 线程安全单元测试

**作用**：验证多线程代码不会出问题——不会死锁、不会竞态、不会在关闭窗口时崩溃。**不需要GPU**，用 Mock 替代所有真实模型。

```python
class _MockRenderer:
    """模拟 RenderManager，暴露与真实 renderer 相同的接口和状态属性。"""
    
    def __init__(self, *args, **kwargs):
        self.is_rendering = False
        self.need_update_again = False
        self._preview_thread = None
        self.state_lock = threading.Lock()
        self._shutdown_event = threading.Event()
        # 方法全是 Mock，记录调用但不做真实推理
        self.request_preview = Mock()
        self.request_hq = Mock()
        self.load_lora = Mock()
        self.shutdown = Mock()
    
    def check_busy(self):
        """返回真实状态，与 RenderManager 行为一致。"""
        return (
            self._preview_thread is not None and self._preview_thread.is_alive(),
            self.is_rendering,
        )
```

**为什么需要 Mock**？
- 加载真实模型需要 GPU + 5GB 显存 + 下载 10GB 权重
- Mock 让测试可以在任何机器上跑，1秒内完成

**13 项测试覆盖了什么**：

| 测试 | 验证什么 |
|------|---------|
| `test_01_canvas_lock` | canvas_lock 创建成功且可正常获取/释放 |
| `test_02_renderer_state_lock` | RenderManager 持有独立的 state_lock |
| `test_03_shutdown_event` | 关闭事件初始为未触发 |
| `test_04_request_hq` | `_request_hq()` 正确委托给 `renderer.request_hq()` |
| `test_05_prompt_snapshot` | prompt 快照在主线程捕获后，不受 GUI 修改影响 |
| `test_06_load_lora` | `_load_lora()` 正确委托给 `renderer.load_lora()` |
| `test_07_on_closing` | `_on_closing()` 发出 shutdown 信号并启动轮询 |
| `test_08_clear_canvas_lock` | `_clear_canvas()` 后 canvas_lock 已释放 |
| `test_09_get_sketch_lock` | `_get_sketch_snapshot()` 后 canvas_lock 已释放 |
| `test_10_paint_no_deadlock` | `_on_paint()` 不会死锁 |
| `test_11_concurrent` | 并发 `_on_paint` + `_request_hq` 不产生死锁（压力测试） |
| `test_12_locks_independent` | canvas_lock 和 state_lock 互不干扰（无交叉持有） |
| `test_canvas_lock_free` | 初始化后 canvas_lock 未被持有 |

运行命令：`python test_thread_safety.py`

---

### 4.7 `test_stream.py` — 集成测试

**作用**：用代码画一个简笔画，让AI实时渲染10次，验证端到端流程能跑通。**需要GPU**。

```python
def create_mock_sketch():
    """用代码绘制一个简单的五官线条图，模拟前端传来的不完整草图"""
    img = Image.new("RGB", (512, 512), "white")
    draw = ImageDraw.Draw(img)
    # 画椭圆脸型
    draw.ellipse([120, 100, 390, 420], outline="black", width=3)
    # 画眼睛
    draw.ellipse([180, 210, 230, 250], outline="black", width=4)
    draw.ellipse([280, 210, 330, 250], outline="black", width=4)
    # 画嘴巴
    draw.arc([210, 310, 300, 350], start=0, end=180, fill="black", width=4)
    return img
```

然后循环10次：
```python
for i in range(10):
    start_time = time.time()
    x_output = stream(mock_sketch)   # 送入StreamDiffusion
    output_image = postprocess_image(x_output, output_type="pil")[0]
    latency = (time.time() - start_time) * 1000
    print(f"第 {i+1} 笔刷新成功 | 本帧延迟: {latency:.2f} ms")
```

运行命令：`python test_stream.py`（需要先启动 conda 环境）

---

### 4.8 `requirements.txt` — 依赖清单

列出项目需要的所有 Python 包及版本号。

```txt
torch>=2.5.0              # PyTorch 深度学习框架
diffusers==0.24.0         # HuggingFace Diffusers 库
transformers==4.36.2      # HuggingFace Transformers
accelerate>=1.13.0        # 分布式推理加速
Pillow>=12.0.0            # PIL 图像处理
numpy>=2.0.0              # 数值计算
streamdiffusion==0.1.1    # 实时流式推理引擎
fire>=0.5.0               # streamdiffusion 的依赖
omegaconf>=2.3.0          # streamdiffusion 的依赖
```

安装命令：`pip install -r requirements.txt`

---

### 4.9 `.gitignore` — Git 忽略规则

告诉 Git 哪些文件不要追踪：

```
*.safetensors   # 模型权重文件（太大，不应提交到Git）
*.bin           # 同上
__pycache__/    # Python缓存目录
*.log           # 日志文件
```

---

### 4.10 `CHANGELOG.md` — 工作日志

记录了项目的迭代历史：

- **2026-08-10**：P0 线程安全修复 — 修复了7个线程安全问题，新增17项测试
- **2026-08-10**：清理无关文件 — 删除了不属于这个项目的 `transformer_test.py`
- **2026-08-11**（未记录）：P1+P2+P3 改进 — 配置提取、模型共享、logging、目录重构

---

### 4.11 `README.md` — 项目说明

项目门面，包含：
- 功能介绍 (带emoji的表格)
- 环境要求
- 安装步骤
- 使用方法
- 项目结构
- 技术栈
- 测试说明

适合放在 GitHub 首页。

---

## 5. 项目的核心设计思想

### 5.1 分层架构

```
┌─────────────────────────┐
│   app.py  (GUI层)        │  ← 只管界面
├─────────────────────────┤
│   renderer.py (业务层)   │  ← 只管渲染逻辑
├─────────────────────────┤
│   pipeline.py (模型层)   │  ← 只管模型加载
├─────────────────────────┤
│   config.py  (配置层)    │  ← 所有可调参数
└─────────────────────────┘
```

每层只依赖下一层，不跨层调用。改界面不影响模型，改模型不影响界面。

### 5.2 GUI 和 AI 推理分离

这是 **生产者-消费者模式** 的变体：

- GUI 线程是"生产者"：用户操作 → 产生渲染请求
- 后台线程是"消费者"：接收请求 → 执行推理 → 回调显示结果

用 `state_lock` 保护共享状态，用 `_shutdown_event` 实现优雅退出，用 `root.after` 做线程安全的 UI 更新。

### 5.3 防御性编程

- **canvas_lock**：防止 GUI 线程和渲染线程同时读写画布
- **state_lock**：防止多个线程同时修改渲染状态
- **prompt 快照**：不在后台线程读取 GUI 控件
- **shutdown_event**：优雅退出，避免 CUDA 崩溃
- **节流**：防止鼠标事件洪水淹没 GPU

### 5.4 关注点分离

- 配置与代码分离（config.py）
- 模型加载与应用逻辑分离（pipeline.py）
- 渲染逻辑与界面分离（renderer.py ↔ app.py，通过回调解耦）
- 入口与实现分离（canvas_stream.py → src/）

---

## 6. 面试可能会问的问题

### Q1: 为什么用 Tkinter 而不是 PyQt/Electron？

**答**：Tkinter 是 Python 标准库的一部分，不需要额外安装，零依赖。对于这个工具型应用的简单界面需求（两个画布+几个按钮）来说足够用了。而且 AI 模型加载和推理才是性能瓶颈，GUI 框架的选择对整体性能影响微乎其微。

### Q2: 线程安全怎么保证的？

**答**：三个层面：
1. **canvas_lock**：保护 `sketch_img` 和 `draw_buffer`。画线、导入、获取快照都要先获取这个锁。
2. **state_lock**：保护 `is_rendering`、`need_update_again` 等渲染状态。状态检查和修改必须在同一个锁临界区内完成，防止 TOCTOU 竞态。
3. **prompt 快照**：后台线程不读取 Tkinter `StringVar`（线程不安全），而是在主线程把它转成普通 Python 字符串再传给线程。

### Q3: 为什么要做模型共享？

**答**：`base_pipe` 和 `stream_pipe` 如果各自加载一份模型，UNet + VAE + TextEncoder 约 5GB × 2 = 10GB，加上 ControlNet 和后处理显存开销，RTX 4080 的 12GB 显存根本装不下。共享底层组件后只需要一份模型在显存里。

### Q4: StreamDiffusion 和普通 Stable Diffusion 有什么区别？

**答**：普通 SD 需要约 50 步去噪才能出一张好图（几秒钟）。StreamDiffusion 用两个技巧加速：
1. **减少步数**：只需 2-4 步，配合 LCM（Latent Consistency Model）调度器
2. **流水线批处理**：把多帧的去噪步骤交错执行，类似 CPU 的指令流水线

实现约 100ms 的实时预览延迟。

### Q5: 如果用户疯狂画线，程序会崩吗？

**答**：不会。有三层防护：
1. **节流**：0.15 秒内只触发一次预览
2. **排队**：高清渲染中来了新请求，不丢弃，标记 `need_update_again`，完成后自动重试
3. **守护线程**：线程是 daemon 模式，即使堆积了，关闭程序也能退出

### Q6: LoRA 是什么？在项目里怎么用的？

**答**：LoRA（Low-Rank Adaptation）是一种在已有大模型上叠加小型权重矩阵的技术。一个 LoRA 文件通常只有几十MB，但能让模型偏向特定风格（如某个画师、某种光影）。本项目通过 `pipe.load_lora_weights()` 动态挂载，修改 `set_adapters()` 的权重可以控制风格强度。因为 base_pipe 和 stream_pipe 共享模型，只需加载一次就全局生效。

### Q7: 如果要在面试中展示这个项目，重点讲什么？

1. **先展示效果**（如果有录屏）：手绘 → 实时预览 → 高清输出
2. **讲架构**：四层分离（配置 → 模型 → 渲染 → 界面），为什么这么拆
3. **讲线程安全**：这是最有技术深度的部分——锁、竞态、TOCTOU、优雅退出
4. **讲模型共享优化**：从"把 12GB 显存用爆"到"省 5GB"的思考过程
5. **讲测试策略**：用 Mock 做无 GPU 的线程安全测试，13项全部通过

### Q8: 这个项目还有什么可以改进的？

- 可以加**撤销/重做**功能
- 可以加**图层系统**（线稿层、参考图层等）
- 可以把 Tkinter 换成 Web 前端（FastAPI + WebSocket）
- 可以支持**批量处理**（对一个文件夹的线稿全部渲染）
- ControlNet 可以用 `controlnet_conditioning_scale` 做**动态调节**（滑条控制 AI 多大程度跟随线稿）

---

## 附录：运行项目

```bash
# 1. 激活 conda 环境
conda activate aniface_diff

# 2. 进入项目目录
cd "D:\code project\AniFaceProject"

# 3. 运行主程序（Windows 需要设置编码）
PYTHONIOENCODING=utf-8 python canvas_stream.py

# 4. 运行测试（不需要 GPU）
python test_thread_safety.py

# 5. 运行集成测试（需要 GPU）
python test_stream.py
```
