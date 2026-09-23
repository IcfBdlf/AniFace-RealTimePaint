# AniFace 核心源码逐函数学习手册

> 2026-09-23 更新：本文正文与代码摘录是迁移前 SD1.5 版本的学习快照（回退标签 `backup/pre-sdxl-20260923`），不再代表全部当前源码。当前模型、双编码器管线、采样器、生成尺寸及新增参数请先读 [SDXL 迁移与代码变化](SDXL_MIGRATION.md)。下文旧性能数字不适用于 SDXL，旧行号可能偏移。

核对日期：2026-09-21。代码片段直接从当日工作区抽取，保留原实现；本次只新增学习文档，不修改应用逻辑。

这份文档用于坐在编辑器前逐段读代码；技术原理和面试表达请配合 [完整技术与面试指南](INTERVIEW_GUIDE.md)。每个核心函数都有实际代码、源码跳转和执行说明。长方法按逻辑阶段解释，避免机械地翻译每个括号。

## 先弄懂几个读代码符号

- `def` 定义函数；只有调用才执行函数体。`self` 表示当前实例，例如每个画板自己的历史记录。
- `self.method` 是方法对象，适合交给按钮或线程；`self.method()` 是立即执行并取得返回值。
- 名字前面的单下划线表示内部使用约定，不是 Python 强制访问权限。
- `None` 表示没有值；坐标 0、权重 0 是合法数值，不能随便当成“没有”。
- `*args` 接收多个位置参数；调用中的 `*pair` 展开二元组；`**options` 将字典展开成关键字参数。
- `@property` 把方法变成属性式访问；`@classmethod` 收到的是类；`@dataclass` 为数据对象生成构造等方法。
- `with` 管理作用域内资源/状态，比如锁、图像文件和推理模式；`try/finally` 确保离开时执行清理。
- `return` 离开当前函数；`raise` 抛出异常，交给上层对应的 `except`；取消不一定是错误。

## 建议阅读顺序

第一遍先读入口、config 和 pipeline，知道窗口与模型怎么起来。第二遍只追踪画一笔：`_on_mouse_down → _draw_to → _request_preview → RenderManager._request → _run → render_sketch → poll_events → _poll_renderer`。第三遍再读 LoRA、历史保存和关闭。

无需立刻理解所有模型数学；先给每个函数标上“谁调用、在哪个线程、读什么、改什么、下一步去哪”。

## 文件目录

- [1. canvas_stream.py：入口与日志](#file-1)
- [2. config.py：集中配置](#file-2)
- [3. src/startup.py：启动、加载与失败重试](#file-3)
- [4. src/pipeline.py：模型加载与单次推理](#file-4)
- [5. src/app.py：画板交互与结果展示](#file-5)
- [6. src/renderer.py：后台调度与适配器管理](#file-6)
- [7. src/prompts.py：提示词组合](#file-7)
- [8. src/profiles.py：角色与画风配置](#file-8)
- [9. src/references.py：历史参考与保存](#file-9)
- [10. 三个完整调用过程](#traces)
- [11. 测试与工具的阅读方法](#tests-tools)
- [12. 动手练习与答案](#practice)

所有源码链接按当前行号定位；后续代码变动时，函数名比旧行号更可靠。以下摘录中的方法缩进已去除，便于阅读，并非独立脚本。

<a id="file-1"></a>
## 1. `canvas_stream.py`：入口与日志

入口导入界面类，创建日志目录，设置控制台与文件日志。注意 logging.basicConfig 等模块级语句在 import 时也会执行；只有末尾 main 守卫限制直接启动窗口。

[打开源文件](<D:/code project/AniFaceProject/canvas_stream.py>)

### 模块级代码

```python
import logging

import sys

import tkinter as tk

from pathlib import Path

from src.app import RealTimePaintApp

from src.startup import StartupWindow

LOG_DIR = Path(__file__).resolve().parent / "artifacts" / "logs"

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "aniface.log", encoding="utf-8"),
    ],
)

```

#### `main`

[定位到实现](<D:/code project/AniFaceProject/canvas_stream.py:28>)

```python
def main():
    root = tk.Tk()
    StartupWindow(root, RealTimePaintApp)
    root.mainloop()
```

1. 主线程入口。tk.Tk() 创建一个根窗口；StartupWindow(root, RealTimePaintApp) 把画板类作为将来加载成功时的回调传入，此时没有立刻创建画板。
2. root.mainloop() 进入事件循环，后续鼠标、定时器和显示都依靠它；函数通常直到窗口退出才继续返回。
3. 阅读顺序：main → StartupWindow.__init__ → ModelLoader.start。

### 文件末尾的启动守卫

```python
if __name__ == "__main__":
    main()
```

直接执行此文件时 __name__ 为 "__main__"，因此创建窗口；被其他模块导入时不执行 main。源码中 main 的定义位于这段之前。Path(__file__).resolve().parent 以入口文件位置确定日志目录，因此不依赖用户从哪个文件夹打开终端；mkdir 的 parents=True 创建缺少的上层目录，exist_ok=True 允许目录已经存在。

<a id="file-2"></a>
## 2. `config.py`：集中配置

本文件没有函数，只有设置值。看它时要区分模型质量参数、交互时间参数和界面参数；修改全局配置不是 GUI 中的角色切换机制。

[打开源文件](<D:/code project/AniFaceProject/config.py>)

### 模块级代码

```python
CONTROLNET_MODEL_ID = "lllyasviel/sd-controlnet-scribble"

BASE_MODEL_ID = "stablediffusionapi/counterfeit-v30"

PREVIEW_NUM_INFERENCE_STEPS = 10

SEED = 42

HQ_IDLE_DELAY_MS = 800

UI_POLL_INTERVAL_MS = 50

HISTORY_LIMIT = 30

HQ_NUM_INFERENCE_STEPS = 25

HQ_CONTROLNET_CONDITIONING_SCALE = 1.1

HQ_CONTROLNET_ENDING_STEP = 0.35  # 提前结束 ControlNet 引导，防止导入线稿的细微瑕疵被放大

RENDER_INTERVAL = 0.15

MAX_PREVIEW_AGE_SECONDS = 5.0  # 超过此输入年龄的预览不再展示；按设备性能调整

REFERENCE_HISTORY_LIMIT = 12

IMAGE_SIZE = (512, 512)  # 画布宽高（像素）

PEN_WIDTH = 4

PEN_COLOR = "black"

DEFAULT_LORA_WEIGHT = 0.8

LORA_WEIGHT_MIN = 0.0

LORA_WEIGHT_MAX = 1.5

DEFAULT_PROMPT = (
    "1girl, masterpiece, hyper detailed, anime portrait, high quality, sharp focus"
)

SHUTDOWN_CHECK_INTERVAL_MS = 200  # 轮询间隔（毫秒）

WINDOW_TITLE = "AniFace — 线稿转动漫画板"
```

配置逐组理解：

| 设置 | 读法与作用 | 主要读取者 |
|---|---|---|
| BASE_MODEL_ID / CONTROLNET_MODEL_ID | 模型仓库标识；从缓存或仓库读取权重 | create_pipelines |
| PREVIEW/HQ_NUM_INFERENCE_STEPS | 10/25 步，两次独立采样，不是先 10 再补 15 | RenderManager._run |
| SEED | 默认 42，控制随机起点，不保证人物身份不变 | render_sketch |
| HQ_CONTROLNET_CONDITIONING_SCALE | 控制残差强度 1.1，不是 LoRA 权重 | render_sketch |
| HQ_CONTROLNET_ENDING_STEP | 名字带 STEP，但实际值 0.35 是进度比例，不是整数步号 | render_sketch |
| RENDER_INTERVAL | 移动时请求节流 0.15 秒，不等于生成帧率 | _draw_to |
| MAX_PREVIEW_AGE_SECONDS | 快照提交到消费的预览年龄上限 5 秒 | _cancel_request |
| HQ_IDLE_DELAY_MS | 自动 HQ 开启时的 800ms 延后 | _defer_hq |
| UI_POLL_INTERVAL_MS | 50ms 轮询间隔，after 不保证精确实时 | 启动页和画板轮询 |
| SHUTDOWN_CHECK_INTERVAL_MS | 200ms 非阻塞检查退出 | _check_shutdown |
| HISTORY_LIMIT / REFERENCE_HISTORY_LIMIT | 30 张撤销快照 / 12 张生成参考，两个不同历史 | app |
| IMAGE_SIZE | 原图和模型 GUI 输入为 512×512，缩放不改变它 | app / pipeline |
| PEN_WIDTH / PEN_COLOR | 默认笔宽 4 与黑色；运行时可调笔宽或白色橡皮 | _draw_to |
| DEFAULT_LORA_WEIGHT / MIN / MAX | 默认 0.8，允许 0–1.5 | GUI / 预设和调度校验 |
| DEFAULT_PROMPT | 单人头像文本，可能与全身或双人线稿冲突 | 初始输入框 |
| WINDOW_TITLE | 只是窗口标题，不影响模型 | 启动页与画板 |


<a id="file-3"></a>
## 3. `src/startup.py`：启动、加载与失败重试

ModelLoader 负责后台工作，StartupWindow 负责主线程界面，LoadEvent 连接二者。

[打开源文件](<D:/code project/AniFaceProject/src/startup.py>)

### 模块级代码

```python
from dataclasses import dataclass

import gc

import logging

import queue

import threading

import tkinter as tk

from tkinter import ttk

import config

from src.pipeline import create_pipelines, LoadCancelled

logger = logging.getLogger("AniFace.startup")
```

### 类 `LoadEvent`

```python
@dataclass(frozen=True)
class LoadEvent:
    kind: str
    value: object = None
```

字段类型用于表达约定；类型注解本身不自动做运行时验证。`frozen=True`（若有）限制字段重新赋值，不会递归冻结字段中的 PIL 图片或字典，所以输入快照仍要复制。专用异常类则用于让调用方区分取消、拦截和真正失败。

### 类 `ModelLoader`

#### `ModelLoader.__init__`

[定位到实现](<D:/code project/AniFaceProject/src/startup.py:25>)

```python
def __init__(self, factory=create_pipelines):
    self._factory = factory
    self._events = queue.Queue()
    self._cancel = threading.Event()
    self._thread = None
```

1. factory 默认是真实 create_pipelines 函数，测试可以传入假函数，这叫依赖注入。
2. Queue 存后台发给主线程的事件；Event 存取消状态；_thread 初始为 None，表示尚未启动。这里只准备状态，不加载模型。

#### `ModelLoader.busy`

[定位到实现](<D:/code project/AniFaceProject/src/startup.py:32>)

```python
@property
def busy(self):
    return self._thread is not None and self._thread.is_alive()
```

1. @property 让调用者写 loader.busy 而不是 loader.busy()。
2. 先判断线程对象存在，再调用 is_alive()；and 的短路求值避免对 None 调用方法。返回是否仍有加载线程运行。

#### `ModelLoader.start`

[定位到实现](<D:/code project/AniFaceProject/src/startup.py:35>)

```python
def start(self):
    if self.busy:
        return False
    self.poll_events()
    # start 由 GUI 主线程调用；不要在后台主动收集 Tkinter 对象。
    gc.collect()
    self._cancel.clear()
    self._thread = threading.Thread(target=self._run, name="AniFace-load", daemon=True)
    self._thread.start()
    return True
```

1. 由主线程调用。busy 为真就返回 False，拒绝同时启动第二次加载。
2. poll_events() 丢弃上轮遗留事件；gc.collect() 在主线程执行垃圾回收；_cancel.clear() 清除之前的取消标记。
3. Thread(target=self._run) 传的是方法，不是调用结果；start() 才真正启动线程。daemon=True 不等于自动完成资源清理。成功发起返回 True。

#### `ModelLoader.cancel`

[定位到实现](<D:/code project/AniFaceProject/src/startup.py:46>)

```python
def cancel(self):
    self._cancel.set()
```

1. 设置 Event，告诉后台下次检查时退出。这不会直接杀死线程，也不能立即中止一个尚未返回的第三方下载调用。

#### `ModelLoader.join`

[定位到实现](<D:/code project/AniFaceProject/src/startup.py:49>)

```python
def join(self, timeout=None):
    if self._thread is not None:
        self._thread.join(timeout)
    return not self.busy
```

1. 若曾启动线程，就调用线程的 join(timeout)。timeout=0 表示只检查，不等待。
2. 返回 not self.busy，把底层 join 本身不提供的“是否已结束”转换成布尔值，供 GUI 决定是否销毁窗口。

#### `ModelLoader.poll_events`

[定位到实现](<D:/code project/AniFaceProject/src/startup.py:54>)

```python
def poll_events(self):
    events = []
    while True:
        try:
            events.append(self._events.get_nowait())
        except queue.Empty:
            return events
```

1. 由主线程调用。get_nowait() 不等待新消息，持续取出已有事件；队列空时捕获 queue.Empty 并返回列表。
2. 空队列是正常控制流程，不作为错误；主线程拿到列表后自行更新控件。

#### `ModelLoader._run`

[定位到实现](<D:/code project/AniFaceProject/src/startup.py:62>)

```python
def _run(self):
    pipe = None
    try:
        pipe = self._factory(
            on_progress=lambda message: self._events.put(LoadEvent("progress", message)),
            should_stop=self._cancel.is_set,
        )
        if not self._cancel.is_set():
            self._events.put(LoadEvent("ready", pipe))
    except LoadCancelled:
        pass
    except Exception as exc:
        logger.exception("模型加载失败")
        if not self._cancel.is_set():
            self._events.put(LoadEvent("error", f"{type(exc).__name__}: {exc}"))
    finally:
        pipe = None
```

1. 运行在 AniFace-load 线程。给 factory 传两个回调：进度回调把文字入队，should_stop 提供实时取消查询。
2. factory 返回管线后再次检查取消标记，再发 ready 事件，避免关闭期间继续进入画板。
3. LoadCancelled 是正常取消，直接忽略；其他异常写日志，且只将异常类型与文字传给 GUI，避免在队列中长期保留 traceback 对模型的引用。
4. finally 把局部 pipe 引用清空；若已发 ready，队列仍持有管线，不会因为这行就把模型从接收者那里删除。

### 类 `StartupWindow`

#### `StartupWindow.__init__`

[定位到实现](<D:/code project/AniFaceProject/src/startup.py:82>)

```python
def __init__(self, root, on_ready, loader=None):
    self.root = root
    self.on_ready = on_ready
    self.loader = loader if loader is not None else ModelLoader()
    self._closing = False
    self._done = False
    self._error = None
    root.title(config.WINDOW_TITLE)
    self.frame = ttk.Frame(root, padding=24)
    self.frame.grid(sticky="nsew")
    ttk.Label(self.frame, text="AniFace", font=("Segoe UI", 20)).grid(row=0, column=0, sticky="w")
    self.status = tk.StringVar(value="准备加载模型…")
    ttk.Label(self.frame, textvariable=self.status, wraplength=560).grid(
        row=1, column=0, sticky="w", pady=12)
    self.progress = ttk.Progressbar(self.frame, mode="indeterminate", length=560)
    self.progress.grid(row=2, column=0, sticky="ew")
    ttk.Label(self.frame, text="首次使用可能下载模型。关闭请求会在当前加载步骤结束后生效。",
              wraplength=560).grid(row=3, column=0, sticky="w", pady=12)
    self.details = tk.Text(self.frame, width=72, height=7, wrap="word", state="disabled")
    self.details.grid(row=4, column=0, sticky="ew")
    self.details.grid_remove()
    buttons = ttk.Frame(self.frame)
    buttons.grid(row=5, column=0, sticky="e", pady=(12, 0))
    self.retry_button = ttk.Button(buttons, text="重试", command=self._retry, state="disabled")
    self.retry_button.pack(side="left", padx=6)
    self.close_button = ttk.Button(buttons, text="关闭", command=self._close)
    self.close_button.pack(side="left")
    root.protocol("WM_DELETE_WINDOW", self._close)
    self._retry()
    root.after(config.UI_POLL_INTERVAL_MS, self._poll)
```

1. 主线程创建启动页，把 on_ready 保存起来。loader 可注入，默认创建真实 ModelLoader。
2. Frame/Label 显示信息；Progressbar 使用 indeterminate，只表示仍在工作；Text 起初隐藏，专门放错误详情。
3. 按钮 command 绑定方法，WM_DELETE_WINDOW 绑定 _close。调用 _retry 发起加载，after 安排第一次轮询；这里没有在主线程调用重型模型加载。

#### `StartupWindow._retry`

[定位到实现](<D:/code project/AniFaceProject/src/startup.py:113>)

```python
def _retry(self):
    if self._closing or self._done or not self.loader.start():
        return
    self._error = None
    self.details.grid_remove()
    self.status.set("正在加载模型…")
    self.retry_button.configure(state="disabled")
    self.progress.start(12)
```

1. 关闭中、已交接完成，或 loader.start() 拒绝重复启动时，立即返回。
2. 成功启动后清空旧错误，隐藏详情，禁用重试按钮并启动进度动画。progress.start(12) 是动画间隔，不是模型每 12ms 完成一部分。

#### `StartupWindow._show_error`

[定位到实现](<D:/code project/AniFaceProject/src/startup.py:122>)

```python
def _show_error(self, message):
    self._error = message
    self.progress.stop()
    self.status.set("加载失败。请查看下面的原因，处理后点击重试。")
    self.details.configure(state="normal")
    self.details.delete("1.0", "end")
    self.details.insert("1.0", message)
    self.details.configure(state="disabled")
    self.details.grid()
```

1. 保存错误文字，停止动画并更新状态。Text 原本 disabled，需要先改为 normal，清空旧内容、插入新文字，再恢复 disabled，防止用户误编辑错误报告。
2. grid() 恢复之前隐藏的详情区域；是否允许重试由 _poll 在加载线程真正结束后决定。

#### `StartupWindow._close`

[定位到实现](<D:/code project/AniFaceProject/src/startup.py:132>)

```python
def _close(self):
    if self._done or self._closing:
        return
    self._closing = True
    self.loader.cancel()
    self.retry_button.configure(state="disabled")
    self.close_button.configure(state="disabled")
    self.status.set("正在关闭，等待当前加载步骤结束…")
```

1. 用 _done/_closing 防止重复关闭。设置关闭状态、通知 loader 取消并禁用按钮，避免等待期间又触发重试。
2. 这里没有马上 destroy，因为第三方加载调用可能尚未返回。之后由 _poll 确认线程结束再关闭。

#### `StartupWindow._poll`

[定位到实现](<D:/code project/AniFaceProject/src/startup.py:141>)

```python
def _poll(self):
    if self._done:
        return
    events = self.loader.poll_events()
    if self._closing:
        # 释放已经排队的管线；关闭请求不再进入画板。
        events.clear()
        if self.loader.join(0):
            self._done = True
            self.progress.stop()
            self.root.destroy()
            return
    else:
        for event in events:
            if event.kind == "progress":
                self.status.set(event.value)
            elif event.kind == "error":
                self._show_error(event.value)
            elif event.kind == "ready":
                self._done = True
                self.progress.stop()
                self.frame.destroy()
                self.on_ready(self.root, event.value)
                return
        if self._error is not None and not self.loader.busy:
            self.retry_button.configure(state="normal")
    self.root.after(config.UI_POLL_INTERVAL_MS, self._poll)
```

1. 这是主线程的事件消费者。先取消息，但关闭中优先处理：清空已到达事件，即使有 ready 也不再打开画板。
2. 正常情况下 progress 更新文字，error 展示错误，ready 停动画并销毁启动 Frame，然后执行 on_ready(root, pipe) 创建画板。
3. ready 分支直接 return，不再安排启动页轮询。错误分支需要等 loader.busy 为假才启用重试；其余情况继续 after(50ms)。

<a id="file-4"></a>
## 4. `src/pipeline.py`：模型加载与单次推理

本模块不接触 Tk。创建管线与单次推理分开，便于模型复用和工具复用。

[打开源文件](<D:/code project/AniFaceProject/src/pipeline.py>)

### 模块级代码

```python
import logging

from PIL import ImageOps

import config

logger = logging.getLogger("AniFace.pipeline")

LINEART_MODEL_ID = "lllyasviel/control_v11p_sd15_lineart"

ANIME_LINEART_MODEL_ID = "lllyasviel/control_v11p_sd15s2_lineart_anime"
```

#### `control_model_id`

[定位到实现](<D:/code project/AniFaceProject/src/pipeline.py:12>)

```python
def control_model_id(control_type):
    return {"scribble": config.CONTROLNET_MODEL_ID, "lineart": LINEART_MODEL_ID,
            "lineart_anime": ANIME_LINEART_MODEL_ID}[control_type]
```

1. 用字典把逻辑类型 scribble/lineart/lineart_anime 映射为模型仓库 ID，并返回字符串。
2. create_pipelines 在调用前验证支持类型；不应把这个仓库 ID 当成远程出图 API 地址。

### 类 `RenderCancelled`

```python
class RenderCancelled(Exception):
    """关闭应用或输入过时时在去噪步骤边界退出。"""
```

字段类型用于表达约定；类型注解本身不自动做运行时验证。`frozen=True`（若有）限制字段重新赋值，不会递归冻结字段中的 PIL 图片或字典，所以输入快照仍要复制。专用异常类则用于让调用方区分取消、拦截和真正失败。

### 类 `RenderBlocked`

```python
class RenderBlocked(Exception):
    """模型检查器未放行结果；不将占位黑图作为正常图像显示。"""
```

字段类型用于表达约定；类型注解本身不自动做运行时验证。`frozen=True`（若有）限制字段重新赋值，不会递归冻结字段中的 PIL 图片或字典，所以输入快照仍要复制。专用异常类则用于让调用方区分取消、拦截和真正失败。

### 类 `LoadCancelled`

```python
class LoadCancelled(Exception):
    """在模型加载阶段边界取消启动。"""
```

字段类型用于表达约定；类型注解本身不自动做运行时验证。`frozen=True`（若有）限制字段重新赋值，不会递归冻结字段中的 PIL 图片或字典，所以输入快照仍要复制。专用异常类则用于让调用方区分取消、拦截和真正失败。

#### `create_pipelines`

[定位到实现](<D:/code project/AniFaceProject/src/pipeline.py:29>)

```python
def create_pipelines(device=None, local_files_only=False, *, on_progress=None,
                     should_stop=lambda: False, control_type="scribble", control_cache_dir=None):
    """返回唯一的 ControlNet 管线。延迟导入重型依赖以支持无 GPU 测试。"""
    if control_type not in {"scribble", "lineart", "lineart_anime"}:
        raise ValueError("不支持的线稿控制类型")
    def stage(message):
        if should_stop():
            raise LoadCancelled()
        if on_progress is not None:
            on_progress(message)

    stage("正在初始化推理环境…")
    import torch
    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    logger.info("加载 ControlNet 管线，设备=%s，精度=%s", device, dtype)
    stage("正在加载线稿控制模型；首次使用可能需要下载…")
    controlnet = ControlNetModel.from_pretrained(
        control_model_id(control_type),
        torch_dtype=dtype, local_files_only=local_files_only,
        **({"cache_dir": control_cache_dir} if control_cache_dir is not None else {})
    )
    stage("正在加载基础绘画模型；首次使用可能需要下载…")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        config.BASE_MODEL_ID, controlnet=controlnet, torch_dtype=dtype,
        local_files_only=local_files_only,
    )
    stage(f"正在将模型准备到 {device} 设备…")
    pipe = pipe.to(device)
    stage("正在配置推理管线…")
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    pipe._aniface_control_type = control_type
    stage("模型已就绪")
    return pipe
```

1. 默认选择 CUDA（若可用），否则 CPU；CUDA 用 float16，CPU 用 float32。* 后的参数要求按名称传入，减少调用时位置混淆。
2. 重型导入放在函数内部，因此仅导入模块或运行假模型测试不必加载整个深度学习环境。
3. 先加载 ControlNet，再将它传入基础绘画管线，随后 pipe.to(device) 准备到设备；local_files_only 控制能否只使用本地缓存。
4. 以原 scheduler.config 构造 UniPC 采样器，关闭进度条；给 pipe 记录控制类型，后续按此决定线稿反色。返回唯一的管线。
5. 其中 stage 是嵌套函数：每次阶段报告前检查取消，具体定义见下一节。

#### `create_pipelines.stage`

[定位到实现](<D:/code project/AniFaceProject/src/pipeline.py:34>)

```python
def stage(message):
    if should_stop():
        raise LoadCancelled()
    if on_progress is not None:
        on_progress(message)
```

1. 这个内部函数捕获外层 should_stop 和 on_progress，属于闭包。
2. 先检查取消，有需要就抛 LoadCancelled；然后仅在提供了进度回调时发消息。它并不自己访问 Tk 控件。

#### `prepare_control_image`

[定位到实现](<D:/code project/AniFaceProject/src/pipeline.py:68>)

```python
def prepare_control_image(sketch, control_type="scribble"):
    """Scribble 使用黑底白线；Lineart 保留白底黑线。"""
    gray = sketch.convert("L")
    if control_type == "scribble":
        gray = ImageOps.invert(gray)
    elif control_type not in {"lineart", "lineart_anime"}:
        raise ValueError("不支持的线稿控制类型")
    return gray.convert("RGB")
```

1. 输入 Pillow 图像，输出供 ControlNet 使用的 RGB 条件图。convert("L") 先转灰度；scribble 反色，lineart 两种模式保留原极性；未知类型报错。
2. 最后 convert("RGB") 将灰度复制为三个通道，是匹配输入格式，不是给图片上色。

#### `render_sketch`

[定位到实现](<D:/code project/AniFaceProject/src/pipeline.py:78>)

```python
def render_sketch(pipe, sketch, prompt, steps, should_stop=lambda: False, *, control_end=None, clip_skip=None):
    control_end = config.HQ_CONTROLNET_ENDING_STEP if control_end is None else control_end
    if not 0 < control_end <= 1:
        raise ValueError("线稿控制结束比例必须大于 0 且不超过 1")
    import torch

    def on_step_end(_pipe, _step, _timestep, callback_kwargs):
        if should_stop():
            raise RenderCancelled()
        return callback_kwargs

    if should_stop():
        raise RenderCancelled()
    # 两个档位从相同噪声开始；步数差异仍可能导致图像差异。
    generator = torch.Generator(device="cpu").manual_seed(config.SEED)
    with torch.inference_mode():
        result = pipe(
            prompt=prompt, image=prepare_control_image(
                sketch, pipe.__dict__.get("_aniface_control_type", "scribble")),
            width=sketch.width, height=sketch.height,
            num_inference_steps=steps, generator=generator,
            controlnet_conditioning_scale=config.HQ_CONTROLNET_CONDITIONING_SCALE,
            control_guidance_end=control_end,
            callback_on_step_end=on_step_end,
            **({"clip_skip": clip_skip} if clip_skip is not None else {}),
        )
    if should_stop():
        raise RenderCancelled()
    flags = getattr(result, "nsfw_content_detected", None)
    if flags is not None and any(flags):
        raise RenderBlocked("本次结果被模型安全检查拦截，请调整提示词或线稿后重试。")
    return result.images[0]
```

1. 一次完整生成的入口，调用者是模型 worker 或显式运行的评估工具。先解析并验证控制结束比例，定义步骤结束回调，再在计算前检查取消。
2. 每次新建相同 seed 的 CPU Generator；这控制噪声随机数来源，不代表整个模型在 CPU 推理。
3. inference_mode 避免为推理构建梯度记录。pipe(...) 收到提示词、条件图、尺寸、步数、控制强度、控制结束比例和回调。clip_skip 只有非 None 时才传入。
4. 计算结束再检查取消，覆盖最后一次回调之后用户改变输入的情况。检查返回对象的 nsfw_content_detected，有被标记项就抛 RenderBlocked；否则返回第一张 PIL 图。
5. 该函数没有训练、没有上一帧 latent 复用，也没有在这里实现去噪循环；循环由 Diffusers 管线完成。

#### `render_sketch.on_step_end`

[定位到实现](<D:/code project/AniFaceProject/src/pipeline.py:84>)

```python
def on_step_end(_pipe, _step, _timestep, callback_kwargs):
    if should_stop():
        raise RenderCancelled()
    return callback_kwargs
```

1. 每次扩散步骤结束时由管线调用，四个参数符合本机管线回调接口。
2. 前面的下划线表示参数在这个实现里不使用，不是禁止访问。should_stop 是现在才求值，因此能看见后续取消状态。
3. 取消时抛专用异常；正常时返回原 callback_kwargs，让管线继续使用预期数据。

<a id="file-5"></a>
## 5. `src/app.py`：画板交互与结果展示

这里所有方法通常运行在主线程。不要试图一次背完：先读绘画链，再读结果链，最后读预设与关闭。

[打开源文件](<D:/code project/AniFaceProject/src/app.py>)

### 模块级代码

```python
import logging

import time

import tkinter as tk

from dataclasses import replace

from tkinter import ttk, messagebox, filedialog

from PIL import Image, ImageDraw, ImageTk, ImageOps

import config

from src.renderer import RenderManager

from src.prompts import EXPRESSION_PRESETS, compose_prompt

from src.profiles import load_profile

from src.references import Reference

logger = logging.getLogger("AniFace.app")
```

#### `fit_sketch`

[定位到实现](<D:/code project/AniFaceProject/src/app.py:18>)

```python
def fit_sketch(image):
    """保留宽高比、EXIF 方向；透明区域合成到白底。"""
    image = ImageOps.exif_transpose(image).convert("RGBA")
    fitted = ImageOps.contain(image, config.IMAGE_SIZE, Image.Resampling.LANCZOS)
    result = Image.new("RGB", config.IMAGE_SIZE, "white")
    result.paste(fitted, ((result.width - fitted.width) // 2,
                          (result.height - fitted.height) // 2), fitted)
    return result
```

1. 按 EXIF 纠正图片方向后转 RGBA，保留 Alpha 透明度；contain 在目标尺寸内等比缩放，不拉伸人物。
2. 创建白底 RGB 图，把缩放后的图居中贴上，并使用图本身的 Alpha 作为遮罩；返回固定 512×512 输入。
3. 图的实际内容可能只占方形画布的一部分；等比缩小仍会损失非常细的线条。

### 类 `RealTimePaintApp`

#### `RealTimePaintApp.__init__`

[定位到实现](<D:/code project/AniFaceProject/src/app.py:29>)

```python
def __init__(self, root, pipe):
    self.root = root
    root.title(config.WINDOW_TITLE)
    self.sketch_img = Image.new("RGB", config.IMAGE_SIZE, "white")
    self.draw_buffer = ImageDraw.Draw(self.sketch_img)
    self.last_x = self.last_y = None
    self.last_render_time = 0
    self._hq_timer = None
    self._poll_timer = None
    self._closing = False
    self._undo_stack = []
    self._redo_stack = []
    self._result = None
    self._notice = ""
    self._lora_notice = ""
    self._profile = None
    self._pending_profile = None
    self._loading_lora = False
    self._reset_profile_prompt = False
    self._references = []
    self._reference_index = -1
    self._reference = None
    self._canvas_revision = 0
    self._build_ui()
    self.renderer = RenderManager(pipe)
    self.prompt_var.trace_add("write", self._on_prompt_changed)
    self.expression_var.trace_add("write", self._on_prompt_changed)
    self.follow_sketch_var.trace_add("write", self._on_prompt_changed)
    root.protocol("WM_DELETE_WINDOW", self._on_closing)
    self._display_image(Image.new("RGB", config.IMAGE_SIZE, "white"))
    self._poll_renderer()
    logger.info("AniFace 画板已就绪")
```

1. 由启动页成功交接后在主线程调用。建立 PIL 白底画布、画笔对象、计时器 ID、撤销重做栈、参考历史和 LoRA/预设状态。
2. _build_ui 创建控件，然后创建 RenderManager；之后才注册变量 trace，因此初次设置控件变量不会误向尚不存在的 renderer 提交任务。
3. 注册关闭回调、显示空白输出、启动结果轮询。self.xxx 表示这个画板实例持续保存的状态，不是每次事件都重新创建的局部变量。

#### `RealTimePaintApp._build_ui`

[定位到实现](<D:/code project/AniFaceProject/src/app.py:62>)

```python
def _build_ui(self):
    frame = ttk.Frame(self.root, padding=10)
    frame.grid(sticky="nsew")
    drawing = ttk.Frame(frame)
    drawing.grid(row=0, column=0, padx=5, pady=5)
    self.canvas = tk.Canvas(drawing, width=config.IMAGE_SIZE[0],
                            height=config.IMAGE_SIZE[1], bg="white", cursor="pencil",
                            highlightthickness=0)
    self.canvas.grid(row=0, column=0)
    xbar = ttk.Scrollbar(drawing, orient="horizontal", command=self.canvas.xview)
    ybar = ttk.Scrollbar(drawing, orient="vertical", command=self.canvas.yview)
    xbar.grid(row=1, column=0, sticky="ew")
    ybar.grid(row=0, column=1, sticky="ns")
    self.canvas.configure(xscrollcommand=xbar.set, yscrollcommand=ybar.set)
    self.output_label = ttk.Label(frame)
    self.output_label.grid(row=0, column=1, padx=5, pady=5)
    controls = ttk.LabelFrame(frame, text="生成设置", padding=10)
    controls.grid(row=1, column=0, columnspan=2, sticky="ew")
    controls.columnconfigure(1, weight=1)
    ttk.Label(controls, text="提示词").grid(row=0, column=0, padx=5)
    self.prompt_var = tk.StringVar(value=config.DEFAULT_PROMPT)
    entry = ttk.Entry(controls, textvariable=self.prompt_var, width=65)
    entry.grid(row=0, column=1, columnspan=3, sticky="ew", pady=5)
    entry.bind("<Return>", lambda _e: self._request_hq())
    ttk.Label(controls, text="LoRA 路径/ID").grid(row=1, column=0, padx=5)
    self.lora_var = tk.StringVar()
    ttk.Entry(controls, textvariable=self.lora_var, width=45).grid(row=1, column=1, sticky="ew")
    self.lora_weight_var = tk.DoubleVar(value=config.DEFAULT_LORA_WEIGHT)
    ttk.Scale(controls, from_=config.LORA_WEIGHT_MIN, to=config.LORA_WEIGHT_MAX,
              variable=self.lora_weight_var, length=110).grid(row=1, column=2, padx=10)
    ttk.Button(controls, text="加载 LoRA", command=self._load_lora).grid(row=1, column=3)
    ttk.Label(controls, text="表情").grid(row=2, column=0, padx=5, pady=5)
    self.expression_var = tk.StringVar(value="不附加表情")
    ttk.Combobox(controls, textvariable=self.expression_var,
                 values=list(EXPRESSION_PRESETS), state="readonly", width=18).grid(
                     row=2, column=1, sticky="w", pady=5)
    ttk.Label(controls, text="附加到提示词；自定义表情可直接写入提示词").grid(
        row=2, column=2, columnspan=2, sticky="w")
    self.follow_sketch_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(controls, text="尽量跟随线稿", variable=self.follow_sketch_var).grid(
        row=3, column=0, columnspan=2, sticky="w", padx=5, pady=5)
    ttk.Label(controls, text="更重视轮廓和姿势；细节仍可能改变").grid(
        row=3, column=2, columnspan=2, sticky="w")
    self.profile_var = tk.StringVar(value="未使用角色/画风预设")
    ttk.Button(controls, text="打开角色/画风预设", command=self._open_profile).grid(row=4, column=0)
    ttk.Label(controls, textvariable=self.profile_var, wraplength=620).grid(row=4, column=1, columnspan=2, sticky="w")
    ttk.Button(controls, text="卸载 LoRA / 预设", command=self._unload_lora).grid(row=4, column=3)
    self.active_lora_var = tk.StringVar(value="实际生效：基础模型")
    ttk.Label(controls, textvariable=self.active_lora_var, wraplength=950).grid(row=5, column=0, columnspan=4, sticky="w")
    toolbar = ttk.Frame(frame, padding=(0, 8))
    toolbar.grid(row=2, column=0, columnspan=2, sticky="w")
    for title, command in [("精细重绘", self._request_hq), ("导入线稿", self._import_sketch),
                           ("清空", self._clear_canvas), ("撤销", self._undo),
                           ("重做", self._redo), ("保存结果", self._save_result)]:
        ttk.Button(toolbar, text=title, command=command).pack(side=tk.LEFT, padx=4)
    self.status_var = tk.StringVar(value="就绪")
    options = ttk.Frame(frame)
    options.grid(row=3, column=0, columnspan=2, sticky="w")
    self.eraser_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(options, text="橡皮擦", variable=self.eraser_var).pack(side=tk.LEFT)
    ttk.Label(options, text="笔刷大小").pack(side=tk.LEFT)
    self.brush_var = tk.IntVar(value=config.PEN_WIDTH)
    ttk.Spinbox(options, from_=1, to=64, textvariable=self.brush_var, width=4).pack(side=tk.LEFT)
    self.zoom_var = tk.StringVar(value="100%")
    zoom = ttk.Combobox(options, textvariable=self.zoom_var, values=["100%", "150%", "200%", "300%"], state="readonly", width=6)
    zoom.pack(side=tk.LEFT, padx=8)
    zoom.bind("<<ComboboxSelected>>", lambda _e: self._redraw_canvas())
    ttk.Label(options, text="放大后用滚动条或中键平移").pack(side=tk.LEFT)
    self.paused_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(options, text="暂停生成", variable=self.paused_var, command=self._toggle_pause).pack(side=tk.LEFT, padx=8)
    self.pinned_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(options, text="固定参考", variable=self.pinned_var, command=self._toggle_pin).pack(side=tk.LEFT)
    self.auto_hq_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(options, text="停笔后自动精细重绘", variable=self.auto_hq_var).pack(side=tk.LEFT)
    history = ttk.Frame(frame)
    history.grid(row=4, column=0, columnspan=2, sticky="w", pady=5)
    ttk.Button(history, text="上一张", command=lambda: self._browse_reference(-1)).pack(side=tk.LEFT)
    ttk.Button(history, text="下一张", command=lambda: self._browse_reference(1)).pack(side=tk.LEFT)
    self.reference_var = tk.StringVar(value="尚无参考图")
    ttk.Label(history, textvariable=self.reference_var, wraplength=820).pack(side=tk.LEFT, padx=8)
    ttk.Label(frame, textvariable=self.status_var, wraplength=1000).grid(
        row=5, column=0, columnspan=2, sticky="w")
    self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
    self.canvas.bind("<B1-Motion>", self._on_paint)
    self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
    self.canvas.bind("<ButtonPress-2>", lambda e: self.canvas.scan_mark(e.x, e.y))
    self.canvas.bind("<B2-Motion>", lambda e: self.canvas.scan_dragto(e.x, e.y, gain=1))
```

1. 本函数只负责创建界面和绑定事件，值得按区域分块读。drawing 包含画布与滚动条，另一列 output_label 显示生成结果；Canvas 的 x/y scrollcommand 与滚动条双向连接。
2. controls 中提示词 StringVar 绑定 Entry；LoRA 用文本路径和 DoubleVar 权重，按钮触发真正加载；表情 readonly 下拉框限制输入到已知选项。
3. “尽量跟随线稿”控制 BooleanVar，角色按钮选择 JSON，active_lora_var 显示已经生效的状态，与正在编辑的路径区别开。
4. toolbar 的 for 循环批量创建按钮，列表中的方法都是回调对象，没有括号，所以创建按钮时不会直接执行这些操作。
5. options 创建橡皮擦、笔刷尺寸、缩放、暂停、固定与自动 HQ。history 创建前后浏览；status_var 放运行信息。
6. grid 用于外层的行列布局，pack 用于不同子 Frame 内的一排控件；没有在同一个父容器中混用两种布局。
7. 最后 bind 鼠标按下/移动/松开和中键平移。lambda 把事件参数转给对应操作；StringVar 等变量只是存状态，不自动执行模型推理。

#### `RealTimePaintApp._remember`

[定位到实现](<D:/code project/AniFaceProject/src/app.py:150>)

```python
def _remember(self):
    self._undo_stack.append(self.sketch_img.copy())
    del self._undo_stack[:-config.HISTORY_LIMIT]
    self._redo_stack.clear()
```

1. 在新的一笔或清空、导入前保存图像副本。del stack[:-HISTORY_LIMIT] 删除超过保留数量的旧项；数量不足时切片为空，不误删。
2. 新的编辑路径建立后清空 redo，符合撤销后再编辑不能沿旧分支重做的约定。

#### `RealTimePaintApp._cancel_hq_timer`

[定位到实现](<D:/code project/AniFaceProject/src/app.py:155>)

```python
def _cancel_hq_timer(self):
    if self._hq_timer is not None:
        self.root.after_cancel(self._hq_timer)
        self._hq_timer = None
```

1. 若已经安排了停笔 HQ，用 root.after_cancel(timer_id) 撤销，并把 ID 设回 None。
2. 这只取消尚未触发的 Tk 定时器，不负责中断已经运行的 GPU 任务；后者靠 renderer 的取消规则。

#### `RealTimePaintApp._changed`

[定位到实现](<D:/code project/AniFaceProject/src/app.py:160>)

```python
def _changed(self, *, soft=False):
    self._cancel_hq_timer()
    self.renderer.invalidate(soft=soft)
    self._canvas_revision += 1
    self._notice = ""
    if self._reference is not None:
        self.reference_var.set("保留的参考图对应较早输入；新结果生成后更新（固定时保持）")
```

1. 统一的输入改变入口：取消停笔计时，通知 renderer 失效规则，增加 GUI 自己的 canvas_revision，清掉普通提示。
2. soft=True 用于普通落笔；默认 False 用于较大上下文变化。已有参考时更新说明文字，但不马上清掉图片。
3. canvas_revision 是 GUI 输入/状态标识，和 renderer 的 _version 不是同一个计数器；提示词改变也会走这里。

#### `RealTimePaintApp._defer_hq`

[定位到实现](<D:/code project/AniFaceProject/src/app.py:168>)

```python
def _defer_hq(self):
    self._cancel_hq_timer()
    if self._can_generate() and self.auto_hq_var.get():
        self._hq_timer = self.root.after(config.HQ_IDLE_DELAY_MS, self._request_hq)
```

1. 先取消已有定时器，只有允许生成且用户开启自动 HQ 时，才安排 800ms 后调用 _request_hq。
2. 默认自动 HQ 关闭，所以松手不会必然触发精细重绘。新笔画调用 _changed 会取消这个计时。

#### `RealTimePaintApp._on_prompt_changed`

[定位到实现](<D:/code project/AniFaceProject/src/app.py:173>)

```python
def _on_prompt_changed(self, *_args):
    if not self._closing:
        self._changed()
        self._request_preview()
        self._defer_hq()
```

1. 作为 Tk trace 回调使用，*_args 接收并忽略 Tk 传来的变量名等参数。
2. 未关闭时做硬失效、立即请求预览，并按自动选项安排 HQ。它没有文本防抖：每个输入变化都可能提交。

#### `RealTimePaintApp._point`

[定位到实现](<D:/code project/AniFaceProject/src/app.py:179>)

```python
def _point(self, event):
    zoom = int(self.zoom_var.get().rstrip("%")) / 100
    return (max(0, min(self.canvas.canvasx(event.x) / zoom, self.sketch_img.width - 1)),
            max(0, min(self.canvas.canvasy(event.y) / zoom, self.sketch_img.height - 1)))
```

1. 输入鼠标事件，返回原图坐标。把 "200%" 去掉百分号、转数值并除以 100 得到 2.0。
2. canvasx/canvasy 加入滚动偏移，再除以 zoom，最后把坐标夹在原图范围内；所以放大显示不会把推理图尺寸也放大。

#### `RealTimePaintApp._current_prompt`

[定位到实现](<D:/code project/AniFaceProject/src/app.py:184>)

```python
def _current_prompt(self):
    prompt = compose_prompt(self.prompt_var.get(), self.expression_var.get())
    return f"{self._profile.trigger_words}, {prompt}" if self._profile else prompt
```

1. 先将用户文本和表情通过 compose_prompt 组合；有生效预设时在前面加 trigger_words。
2. 仅返回字符串，不修改用户 Entry 的原文本，因此切换表情不会反复向输入框累积标签。

#### `RealTimePaintApp._render_options`

[定位到实现](<D:/code project/AniFaceProject/src/app.py:188>)

```python
def _render_options(self):
    end = 1.0 if self.follow_sketch_var.get() else (self._profile.control_end if self._profile else None)
    return {"control_end": end, "metadata": {"profile": self._profile.metadata() if self._profile else None,
                                             "canvas_revision": self._canvas_revision}}
```

1. 决定控制结束比例：跟随模式优先给 1.0；否则用预设值；没有预设就传 None，让管线采用全局默认。
2. 同时打包当前预设字典和 canvas_revision。它返回的是稍后通过 ** 展开的参数字典，不是立即调用模型。

#### `RealTimePaintApp._can_generate`

[定位到实现](<D:/code project/AniFaceProject/src/app.py:193>)

```python
def _can_generate(self):
    return not self._closing and not self.paused_var.get() and not self._loading_lora
```

1. 返回是否未关闭、未暂停、且未在等待 LoRA 操作完成。
2. 这是请求入口的门禁；不读取 GPU 状态。worker 正忙时仍可以提交，待处理槽位会保留最新输入。

#### `RealTimePaintApp._request_preview`

[定位到实现](<D:/code project/AniFaceProject/src/app.py:196>)

```python
def _request_preview(self):
    if self._can_generate():
        self.renderer.request_preview(self.sketch_img, self._current_prompt(), **self._render_options())
```

1. 门禁通过才把当前线稿、组合好的字符串和控制选项传给 renderer。
2. renderer 内部会复制图像；GUI 不在这里执行 pipe。**self._render_options() 将字典字段展开为具名参数。

#### `RealTimePaintApp._on_mouse_down`

[定位到实现](<D:/code project/AniFaceProject/src/app.py:200>)

```python
def _on_mouse_down(self, event):
    if self._closing:
        return
    self._remember()
    self.last_x, self.last_y = self._point(event)
    self._draw_to(self.last_x, self.last_y)
```

1. 关闭中忽略事件。否则先记录撤销快照，再将鼠标转换为原图坐标，保存 last_x/last_y，并调用 _draw_to。
2. 按下时也画一个点，因此只点击不拖动也能留下一笔；一整笔通常只保存一次撤销快照。

#### `RealTimePaintApp._on_paint`

[定位到实现](<D:/code project/AniFaceProject/src/app.py:207>)

```python
def _on_paint(self, event):
    if not self._closing and self.last_x is not None and self.last_y is not None:
        self._draw_to(*self._point(event))
```

1. 只在尚未关闭且有正在进行的笔画时绘制；用 is not None 判断，避免把合法坐标 0 当成“没有起点”。
2. *self._point(event) 把二元组展开成 _draw_to(x, y) 的两个位置参数。

#### `RealTimePaintApp._draw_to`

[定位到实现](<D:/code project/AniFaceProject/src/app.py:211>)

```python
def _draw_to(self, x, y):
    self._changed(soft=True)
    try:
        width = max(1, min(64, self.brush_var.get()))
    except (tk.TclError, ValueError):
        width = config.PEN_WIDTH
    color = "white" if self.eraser_var.get() else config.PEN_COLOR
    self.draw_buffer.line((self.last_x, self.last_y, x, y),
                          fill=color, width=width)
    radius = width / 2
    self.draw_buffer.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)
    self.last_x, self.last_y = x, y
    # 同一 PIL 图像既用于显示也用于推理，避免两套笔刷边缘不一致。
    self._redraw_canvas()
    now = time.monotonic()
    if now - self.last_render_time >= config.RENDER_INTERVAL:
        self.last_render_time = now
        self._request_preview()
```

1. 每次先 soft 失效，使旧 HQ 过期但允许近期预览完成。笔刷宽度限制到 1–64，输入非法时退回默认；橡皮擦选择白色。
2. 从旧点画线到新点，再画端点圆形，使连续事件之间连接起来；更新 last_x/last_y。
3. _redraw_canvas 立即更新左侧显示；只有距离上次提交达到 0.15 秒，才请求模型预览。绘图刷新和模型请求不是同一个频率。

#### `RealTimePaintApp._on_mouse_up`

[定位到实现](<D:/code project/AniFaceProject/src/app.py:230>)

```python
def _on_mouse_up(self, _event):
    if self._closing or self.last_x is None:
        return
    self.last_x = self.last_y = None
    self._request_preview()
    self._defer_hq()
```

1. 如果没有有效笔画或正在关闭，就返回。正常结束时把两个起点变量设为 None，阻止后续移动继续画。
2. 再提交一次预览，确保最后一段笔画进入模型输入；然后按用户选项延迟安排 HQ。

#### `RealTimePaintApp._request_hq`

[定位到实现](<D:/code project/AniFaceProject/src/app.py:237>)

```python
def _request_hq(self):
    self._cancel_hq_timer()
    if self._can_generate():
        self._notice = ""
        self.renderer.request_hq(self.sketch_img, self._current_prompt(), **self._render_options())
```

1. 先撤销可能重复的 HQ 定时器；门禁通过后清除普通提示并请求 hq。
2. 这会以 25 步重新生成，不是接着已有预览 latent 再跑 15 步。

#### `RealTimePaintApp._redraw_canvas`

[定位到实现](<D:/code project/AniFaceProject/src/app.py:243>)

```python
def _redraw_canvas(self):
    zoom = int(self.zoom_var.get().rstrip("%")) / 100
    size = tuple(int(x * zoom) for x in self.sketch_img.size)
    self.sketch_tk_img = ImageTk.PhotoImage(self.sketch_img.resize(size))
    self.canvas.configure(scrollregion=(0, 0, *size))
    self.canvas.delete("all")
    self.canvas.create_image(0, 0, anchor=tk.NW, image=self.sketch_tk_img)
```

1. 根据缩放值计算仅用于显示的像素尺寸，resize 后创建 PhotoImage 并保存在实例属性中，避免被垃圾回收。
2. scrollregion 定义可滚动范围；删除旧 Canvas 显示项再放新的图，anchor=NW 把图左上角放在原点。
3. 原始 sketch_img 尺寸没有被这里重新赋值，所以模型仍使用原分辨率。

#### `RealTimePaintApp._replace_sketch`

[定位到实现](<D:/code project/AniFaceProject/src/app.py:251>)

```python
def _replace_sketch(self, image):
    self._changed()
    self.sketch_img = image
    self.draw_buffer = ImageDraw.Draw(image)
    self.last_x = self.last_y = None
    self._redraw_canvas()
```

1. 画布整体替换时做硬失效，更新图像并重新创建指向新图像的 ImageDraw 对象。
2. 不能只换 sketch_img 而继续用旧 draw_buffer，否则以后笔画仍会画到旧对象上。最后清除笔画起点并刷新画布。

#### `RealTimePaintApp._clear_canvas`

[定位到实现](<D:/code project/AniFaceProject/src/app.py:258>)

```python
def _clear_canvas(self):
    if self._closing:
        return
    self._remember()
    self._replace_sketch(Image.new("RGB", config.IMAGE_SIZE, "white"))
    self._result = None
    if self.pinned_var.get():
        self._result = self._reference.image if self._reference else None
        return
    self._reference = None
    self.reference_var.set("尚无参考图；历史记录仍可浏览")
    self._display_image(Image.new("RGB", config.IMAGE_SIZE, "white"))
```

1. 记录可撤销快照，替换成白图并清空 _result。
2. 如果固定了参考，则把 _result 恢复为固定图并提前返回，保留右侧；没有固定时才清空当前 Reference 并显示白色输出。
3. 参考历史列表没有清空，所以可以稍后浏览；画布清空与历史清空是不同操作。

#### `RealTimePaintApp._undo`

[定位到实现](<D:/code project/AniFaceProject/src/app.py:271>)

```python
def _undo(self):
    if not self._closing and self._undo_stack:
        self._redo_stack.append(self.sketch_img.copy())
        self._replace_sketch(self._undo_stack.pop())
        self._request_preview()
        self._defer_hq()
```

1. 有撤销记录时，把当前画布副本压到 redo，再弹出最近 undo 图做整体替换。
2. 替换后请求预览并按选项安排 HQ。它恢复位图快照，不是倒着执行一串笔画命令。

#### `RealTimePaintApp._redo`

[定位到实现](<D:/code project/AniFaceProject/src/app.py:278>)

```python
def _redo(self):
    if not self._closing and self._redo_stack:
        self._undo_stack.append(self.sketch_img.copy())
        self._replace_sketch(self._redo_stack.pop())
        self._request_preview()
        self._defer_hq()
```

1. 有 redo 记录时先保存当前画布到 undo，再恢复 redo 的栈顶。
2. 同样重新生成参考。它没有调用 _remember，因为该方法会清空 redo，破坏重做链。

#### `RealTimePaintApp._import_sketch`

[定位到实现](<D:/code project/AniFaceProject/src/app.py:285>)

```python
def _import_sketch(self):
    if self._closing:
        return
    path = filedialog.askopenfilename(title="选择白底黑线的线稿",
                filetypes=[("图片", "*.jpg *.jpeg *.png *.bmp *.webp")])
    if not path:
        return
    try:
        with Image.open(path) as image:
            imported = fit_sketch(image)
        self._remember()
        self._replace_sketch(imported)
        self._request_hq()
    except Exception as exc:
        messagebox.showerror("导入失败", str(exc))
```

1. 对话框取消时返回；选中文件后用 with Image.open 读取，并在文件关闭前完成 fit_sketch。
2. 先记住旧画布，再整体替换，并显式请求 HQ；因此“导入”与普通停笔的默认生成档位不同。
3. 读取/转换等异常由消息框提示。暂停或加载 LoRA 时，_request_hq 的门禁仍会阻止提交。

#### `RealTimePaintApp._load_lora`

[定位到实现](<D:/code project/AniFaceProject/src/app.py:301>)

```python
def _load_lora(self):
    if self._closing or self._loading_lora:
        return
    path = self.lora_var.get().strip()
    if not path:
        messagebox.showwarning("提示", "请填写 LoRA 文件路径或模型 ID")
        return
    self._changed()
    self._loading_lora = True
    self._reset_profile_prompt = False
    self._pending_profile = (replace(self._profile, weight=self.lora_weight_var.get())
                             if self._profile and path == self._profile.lora_path else None)
    self.renderer.load_lora(path, self.lora_weight_var.get())
```

1. 已经关闭或有待完成 LoRA 操作时拒绝重复请求；去掉路径首尾空白，空路径提示用户。
2. 硬失效当前输入后设置 _loading_lora；同一个预设路径调权重时，用 dataclasses.replace 产生更新权重的新预设，保留自由提示词。
3. 这里只提交到 worker，界面不会立即宣称已生效；_poll_renderer 收到成功信息后才完成配置提交。

#### `RealTimePaintApp._open_profile`

[定位到实现](<D:/code project/AniFaceProject/src/app.py:315>)

```python
def _open_profile(self):
    if self._closing or self._loading_lora:
        return
    path = filedialog.askopenfilename(title="选择角色/画风预设", filetypes=[("预设", "*.json")])
    if not path:
        return
    try:
        profile = load_profile(path)
    except Exception as exc:
        messagebox.showerror("预设无效", str(exc))
        return
    self._changed()
    self._pending_profile = profile
    self._reset_profile_prompt = True
    self._loading_lora = True
    self.renderer.load_lora(profile.lora_path, profile.weight)
```

1. 选择 JSON 后先调用 load_profile 做校验，失败只提示并退出，不修改模型。
2. 成功解析不代表成功加载：先把它存 _pending_profile，设置加载门禁及需要重设默认提示词的标志，再提交 LoRA 加载。
3. 生效动作发生在事件消费者，便于加载失败时保留旧预设。

#### `RealTimePaintApp._unload_lora`

[定位到实现](<D:/code project/AniFaceProject/src/app.py:332>)

```python
def _unload_lora(self):
    if self._closing or self._loading_lora:
        return
    self._changed()
    self._loading_lora = True
    self._pending_profile = None
    self._reset_profile_prompt = False
    self.renderer.load_lora(None, 0)
```

1. 硬失效旧输出，设置加载门禁，清空待提交预设，用 load_lora(None, 0) 表达卸载请求。
2. None 是这里明确约定的操作标记，不是“空文件路径”；实际卸载在 worker 中完成。

#### `RealTimePaintApp._toggle_pause`

[定位到实现](<D:/code project/AniFaceProject/src/app.py:341>)

```python
def _toggle_pause(self):
    self._changed()
    if not self.paused_var.get():
        self._request_preview()
```

1. 先调用 _changed 使当前生成任务失效；如果新状态是恢复，就请求当前画布的预览。
2. 暂停控制计算，固定参考控制显示。即使仍能画笔，也不代表后台一定在生成。

#### `RealTimePaintApp._toggle_pin`

[定位到实现](<D:/code project/AniFaceProject/src/app.py:346>)

```python
def _toggle_pin(self):
    if not self.pinned_var.get() and self._references:
        self._show_reference(len(self._references) - 1)
```

1. 切到固定时不替换图；取消固定且有历史时，把最新一项显示出来。
2. 它没有取消后台推理。固定期间收到的新图仍可加入历史列表。

#### `RealTimePaintApp._browse_reference`

[定位到实现](<D:/code project/AniFaceProject/src/app.py:350>)

```python
def _browse_reference(self, direction):
    if not self._references:
        return
    self.pinned_var.set(True)
    self._show_reference(max(0, min(len(self._references) - 1, self._reference_index + direction)))
```

1. 无历史时返回；有历史则自动固定，计算索引加方向，并用 max/min 限制边界。
2. 方向通常为 -1 或 1，由上一张/下一张按钮给出；再交给 _show_reference 统一更新显示。

#### `RealTimePaintApp._show_reference`

[定位到实现](<D:/code project/AniFaceProject/src/app.py:356>)

```python
def _show_reference(self, index):
    self._reference_index = index
    self._reference = self._references[index]
    self._result = self._reference.image
    self._display_image(self._result)
    meta = self._reference.metadata
    profile = meta.get("profile")
    name = profile["name"] if profile else "自定义配置"
    age = "当前输入" if meta.get("canvas_revision") == self._canvas_revision else "较早输入"
    self.reference_var.set(f"参考 {index + 1}/{len(self._references)} · {name} · {meta['mode']} · {age} · "
                           f"快照到显示 {meta['input_to_display_seconds']:.2f}s")
```

1. 设置当前索引、Reference 和 _result，再更新右侧图片。
2. 从这张参考自己的 metadata 取预设名、生成档位和时间；canvas_revision 与当前状态比较，用于标注较早输入。
3. f-string 中 :.2f 将秒数格式化为两位小数，只影响展示，不改变实际数据精度。

#### `RealTimePaintApp._poll_renderer`

[定位到实现](<D:/code project/AniFaceProject/src/app.py:368>)

```python
def _poll_renderer(self):
    if self._closing:
        return
    for event in self.renderer.poll_events():
        if event.kind == "frame":
            metadata = dict(event.metadata)
            metadata["input_to_display_seconds"] = time.monotonic() - event.request.submitted_at
            self._references.append(Reference.capture(event.value, event.request.image, metadata))
            if len(self._references) > config.REFERENCE_HISTORY_LIMIT:
                self._references.pop(0)
                self._reference_index = max(-1, self._reference_index - 1)
            if not self.pinned_var.get():
                self._show_reference(len(self._references) - 1)
            self._notice = ""
        elif event.kind == "blocked":
            retained = "已保留上一张有效结果。" if self._result is not None else "尚无可显示的结果。"
            self._notice = f"{event.value} {retained}"
        elif event.kind in {"lora_error", "info"}:
            self._lora_notice = str(event.value)
            if self._loading_lora:
                if event.kind == "info":
                    self._profile = self._pending_profile
                    self.profile_var.set(self._profile.name if self._profile else "未使用角色/画风预设")
                    if self._profile:
                        if self._reset_profile_prompt:
                            self.prompt_var.set(self._profile.default_prompt)
                            self.follow_sketch_var.set(False)
                            self.expression_var.set("不附加表情")
                        self.lora_var.set(self._profile.lora_path)
                        self.lora_weight_var.set(self._profile.weight)
                self._pending_profile = None
                self._loading_lora = False
                if event.kind == "info":
                    self._request_preview()
            active = self.renderer.active_lora
            self.active_lora_var.set(f"实际生效：{active['path']} · 权重 {active['weight']}" if active["path"] else "实际生效：基础模型")
        else:
            # 非模态提示，不在处理结果过程中进入嵌套 Tk 事件循环。
            self._notice = str(event.value)
    status = "已暂停生成" if self.paused_var.get() else self.renderer.status
    if self.pinned_var.get():
        status += " · 当前参考已固定"
    self.status_var.set("  |  ".join(part for part in (status, self._lora_notice, self._notice) if part))
    self._poll_timer = self.root.after(config.UI_POLL_INTERVAL_MS, self._poll_renderer)
```

1. 主线程每轮读取 renderer 已过滤的事件。frame 分支复制 metadata，记下从提交到消费的时间，用事件自带的线稿创建 Reference。
2. 历史超出 12 项就删除最早项，并调整浏览索引；没有固定时显示最新参考，固定时仅更新历史。
3. blocked 分支保留上一张有效图；info/lora_error 分支管理 LoRA 信息与待提交预设。只有 info 成功时才将 pending 设为当前预设，并在需要时重设默认提示词。
4. 程序设置 StringVar 会触发 trace，但 _loading_lora 此时仍为真，阻止中间配置发起生成；之后解除门禁再请求预览。
5. 最后组合状态文字，并保存下一次 after 的 ID，关闭时可以取消。这里记录时间发生在实际显示函数之前，不等于精确的屏幕物理延迟。

#### `RealTimePaintApp._display_image`

[定位到实现](<D:/code project/AniFaceProject/src/app.py:413>)

```python
def _display_image(self, image):
    self.tk_img = ImageTk.PhotoImage(image)
    self.output_label.configure(image=self.tk_img)
```

1. 把 PIL 图像转成 Tk 图片，保存在 self.tk_img，然后配置输出 Label。
2. 保存 Python 引用是必要的，否则 PhotoImage 被回收可能让控件看起来没有图。仅显示，不修改参考的生成参数。

#### `RealTimePaintApp._save_result`

[定位到实现](<D:/code project/AniFaceProject/src/app.py:417>)

```python
def _save_result(self):
    if self._closing:
        return
    if self._result is None:
        messagebox.showinfo("提示", "还没有生成结果")
        return
    reference = self._reference
    path = filedialog.asksaveasfilename(title="保存当前显示结果", defaultextension=".png",
                                       filetypes=[("PNG", "*.png")])
    if path:
        try:
            reference.save(path)
        except Exception as exc:
            messagebox.showerror("保存失败", str(exc))
```

1. 没有有效图时提示用户；有图时先保存当前 Reference 到局部变量，再打开保存对话框。
2. 对话框可能处理其他 Tk 事件；事先捕获对象，保证保存的是用户发起保存时选中的参考，而不是之后刚更新的图。
3. 写文件交给 Reference.save，失败显示错误，不从当前输入框重新拼参数。

#### `RealTimePaintApp._on_closing`

[定位到实现](<D:/code project/AniFaceProject/src/app.py:432>)

```python
def _on_closing(self):
    if self._closing:
        return
    self._closing = True
    self._cancel_hq_timer()
    if self._poll_timer is not None:
        self.root.after_cancel(self._poll_timer)
    self.renderer.shutdown()
    self._check_shutdown()
```

1. _closing 防止重复执行。取消 HQ 定时器与轮询定时器，然后通知 renderer 关闭。
2. 不在这里无限等待线程，立即进入 _check_shutdown，让窗口等待期间仍可处理事件。

#### `RealTimePaintApp._check_shutdown`

[定位到实现](<D:/code project/AniFaceProject/src/app.py:442>)

```python
def _check_shutdown(self):
    if self.renderer.join(timeout=0):
        self.root.destroy()
    else:
        self.status_var.set(self.renderer.status)
        self.root.after(config.SHUTDOWN_CHECK_INTERVAL_MS, self._check_shutdown)
```

1. join(timeout=0) 只检查后台是否退出。结束后销毁根窗口；尚未结束则更新状态，200ms 后再次检查。
2. 这是协作式退出，不保证第三方调用能在固定毫秒内立即停止。

<a id="file-6"></a>
## 6. `src/renderer.py`：后台调度与适配器管理

请求入口和事件消费由主线程调用；_run/_switch_lora 由 worker 执行；_cancel_request 可在这两条路径中使用。

[打开源文件](<D:/code project/AniFaceProject/src/renderer.py>)

### 模块级代码

```python
from dataclasses import dataclass

import logging

import queue

import threading

import time

import math

from copy import deepcopy

import config

from src.pipeline import RenderBlocked, RenderCancelled, render_sketch

logger = logging.getLogger("AniFace.renderer")
```

### 类 `RenderRequest`

```python
@dataclass(frozen=True)
class RenderRequest:
    version: int
    kind: str
    image: object
    prompt: str
    control_end: float | None = None
    epoch: int = 0
    submitted_at: float = 0
    metadata: dict | None = None
```

字段类型用于表达约定；类型注解本身不自动做运行时验证。`frozen=True`（若有）限制字段重新赋值，不会递归冻结字段中的 PIL 图片或字典，所以输入快照仍要复制。专用异常类则用于让调用方区分取消、拦截和真正失败。

### 类 `RenderEvent`

```python
@dataclass(frozen=True)
class RenderEvent:
    kind: str
    value: object
    version: int | None = None
    request: RenderRequest | None = None
    metadata: dict | None = None
```

字段类型用于表达约定；类型注解本身不自动做运行时验证。`frozen=True`（若有）限制字段重新赋值，不会递归冻结字段中的 PIL 图片或字典，所以输入快照仍要复制。专用异常类则用于让调用方区分取消、拦截和真正失败。

### 类 `LoraSwitchError`

```python
class LoraSwitchError(Exception):
    """LoRA 切换失败，消息包含恢复结果。"""
```

字段类型用于表达约定；类型注解本身不自动做运行时验证。`frozen=True`（若有）限制字段重新赋值，不会递归冻结字段中的 PIL 图片或字典，所以输入快照仍要复制。专用异常类则用于让调用方区分取消、拦截和真正失败。

### 类 `RenderManager`

#### `RenderManager.__init__`

[定位到实现](<D:/code project/AniFaceProject/src/renderer.py:48>)

```python
def __init__(self, pipe, render_fn=render_sketch):
    self.pipe = pipe
    self._render = render_fn
    self._condition = threading.Condition()
    self._shutdown_event = threading.Event()
    self._version = 0
    self._epoch = 0
    self._displayed_version = -1
    self._last_input = None
    self._render_settings = None
    self._pending_render = None
    self._pending_lora = None
    self._active = None
    self._current_lora = None
    self._lora_path = None
    self._lora_sequence = 0
    self._retired_adapters = set()
    self._model_error = None
    self._events = queue.Queue()
    self._worker = threading.Thread(target=self._run, name="AniFace-render", daemon=True)
    self._worker.start()
```

1. 保存管线与 render_fn，后者可由测试替换。Condition 保护调度状态，Event 表示关闭，Queue 存结果事件。
2. _pending_render/_pending_lora 各自仅保存最新任务；_active 表示当前操作。epoch 隔离上下文，version 标记输入，_displayed_version 防止展示倒退。
3. 初始化适配器、待清理集合和模型错误状态，最后启动唯一模型 worker。正式计算和适配器修改都交给它。

#### `RenderManager.is_shutting_down`

[定位到实现](<D:/code project/AniFaceProject/src/renderer.py:71>)

```python
@property
def is_shutting_down(self):
    return self._shutdown_event.is_set()
```

1. 只读 property，返回关闭 Event 是否已设置。Event 是状态，不是只发一次就消失的通知。

#### `RenderManager.invalidate`

[定位到实现](<D:/code project/AniFaceProject/src/renderer.py:74>)

```python
def invalidate(self, *, soft=False):
    """丢弃待处理输入；soft 仅使 HQ 失效，保留同配置的近期预览。"""
    with self._condition:
        self._version += 1
        if not soft:
            self._epoch += 1
            self._render_settings = None
        self._last_input = None
        self._pending_render = None
```

1. 在 Condition 锁内增加输入版本并清空待处理图与 identity。
2. 默认硬失效同时增加 epoch 并清掉 render_settings；soft=True 则保持 epoch，让同上下文近期预览仍可完成。
3. 这个函数本身不提交新画布，也不会直接删除已经显示的参考图。

#### `RenderManager._request`

[定位到实现](<D:/code project/AniFaceProject/src/renderer.py:84>)

```python
def _request(self, kind, image, prompt, control_end=None, metadata=None):
    if control_end is not None and not 0 < control_end <= 1:
        raise ValueError("线稿控制结束比例必须大于 0 且不超过 1")
    snapshot = image.copy()
    identity = (snapshot.mode, snapshot.size, snapshot.tobytes(), prompt, control_end)
    with self._condition:
        if self.is_shutting_down:
            return False
        # 同一输入从预览升级为精细重绘时，仍允许先显示预览。
        settings = (prompt, control_end)
        if self._render_settings is not None and settings != self._render_settings:
            self._epoch += 1
        self._render_settings = settings
        if identity != self._last_input:
            self._version += 1
            self._last_input = identity
        self._pending_render = RenderRequest(self._version, kind, snapshot, prompt, control_end,
                                             self._epoch, time.monotonic(), deepcopy(metadata))
        self._condition.notify()
    return True
```

1. 先验证可选控制比例，然后复制 PIL 输入，用模式、尺寸、像素字节、提示词和控制比例构造 identity。
2. 进入锁后若已关闭返回 False；提示词或控制配置改变就增加 epoch；不同 identity 增加 version，相同输入升档可共用版本。
3. 构造 RenderRequest 时记录单调提交时刻，metadata 深复制，再直接覆盖 _pending_render。notify 唤醒等待的 worker，最后返回 True 表示已接受，而不是已生成。

#### `RenderManager.request_preview`

[定位到实现](<D:/code project/AniFaceProject/src/renderer.py:105>)

```python
def request_preview(self, image, prompt, *, control_end=None, metadata=None):
    return self._request("preview", image, prompt, control_end, metadata)
```

1. 薄包装：将 kind 固定为 preview，其他参数原样交给 _request。
2. 控制参数在 * 后，必须用名称传入；它不选择步数，具体步数由 worker 根据 kind 决定。

#### `RenderManager.request_hq`

[定位到实现](<D:/code project/AniFaceProject/src/renderer.py:108>)

```python
def request_hq(self, image, prompt, *, control_end=None, metadata=None):
    return self._request("hq", image, prompt, control_end, metadata)
```

1. 与预览包装相同，但 kind 为 hq。
2. 统一进入 _request，避免预览和精细模式各自维护一套互相不一致的版本规则。

#### `RenderManager.load_lora`

[定位到实现](<D:/code project/AniFaceProject/src/renderer.py:111>)

```python
def load_lora(self, path, weight):
    """异步切换；最新 LoRA 在下一次图像推理前应用。"""
    path = str(path) if path is not None else None
    if not math.isfinite(weight) or not config.LORA_WEIGHT_MIN <= weight <= config.LORA_WEIGHT_MAX:
        raise ValueError("LoRA 权重超出范围")
    with self._condition:
        if self.is_shutting_down:
            return False
        self._version += 1
        self._epoch += 1
        self._last_input = None
        self._pending_render = None
        self._pending_lora = (path, weight)
        self._condition.notify()
    return True
```

1. 把非 None 路径规范为字符串，验证权重为有限值且在范围内。
2. 锁内拒绝关闭后的新任务，然后增加 version 与 epoch，清空待处理图，覆盖最新 LoRA 请求并唤醒 worker。
3. 不会立刻执行读取文件。None 路径用于卸载；正在运行的旧图会通过 epoch 检查失效。

#### `RenderManager.poll_events`

[定位到实现](<D:/code project/AniFaceProject/src/renderer.py:127>)

```python
def poll_events(self):
    events = []
    while True:
        try:
            event = self._events.get_nowait()
        except queue.Empty:
            break
        with self._condition:
            valid = event.version is None or event.version == self._version
            if event.request is not None:
                valid = not self._cancel_request(event.request) and event.version >= self._displayed_version
            if not self.is_shutting_down and valid:
                if event.kind == "frame":
                    self._displayed_version = event.version
                events.append(event)
    return events
```

1. 由 GUI 调用，逐个非阻塞取结果。普通带版本事件先与当前版本比较；带 request 的事件改用 _cancel_request 规则和单调显示版本检查。
2. 因此同 epoch 的近期 preview 可以不是最新 version；HQ 必须最新。
3. 有效 frame 会推进 _displayed_version。完成时有效但取出前已过期的事件，在这里再次丢弃。

#### `RenderManager.status`

[定位到实现](<D:/code project/AniFaceProject/src/renderer.py:145>)

```python
@property
def status(self):
    with self._condition:
        if self.is_shutting_down:
            return "正在关闭，等待当前推理结束…"
        labels = {"preview": "正在生成预览…", "hq": "正在精细重绘…", "lora": "正在加载 LoRA…"}
        if self._active:
            return labels[self._active]
        if self._pending_render or self._pending_lora:
            return "等待渲染…"
        return "就绪"
```

1. 锁内读取状态，按“关闭 → 正在执行 → 等待任务 → 就绪”的优先级返回文字。
2. 它提供界面状态，不调用模型，也不保证下一微秒队列不会发生变化。

#### `RenderManager.shutdown`

[定位到实现](<D:/code project/AniFaceProject/src/renderer.py:156>)

```python
def shutdown(self):
    with self._condition:
        self._shutdown_event.set()
        self._version += 1
        self._pending_render = None
        self._pending_lora = None
        self._condition.notify_all()
```

1. 锁内设置关闭 Event，增加版本，清空两种待处理任务并 notify_all。
2. 如果 worker 正在 wait，就让它醒来发现关闭；如果在计算，就让回调发现关闭。已经显示的 GUI 对象由主线程处理。

#### `RenderManager.join`

[定位到实现](<D:/code project/AniFaceProject/src/renderer.py:164>)

```python
def join(self, timeout=None):
    self._worker.join(timeout)
    return not self._worker.is_alive()
```

1. 等待渲染线程至多 timeout 秒，并返回是否真正退出。
2. 主线程使用 timeout=0 轮询，测试通常给有限等待上限，避免线程泄漏。

#### `RenderManager._should_cancel`

[定位到实现](<D:/code project/AniFaceProject/src/renderer.py:168>)

```python
def _should_cancel(self, version):
    with self._condition:
        return self.is_shutting_down or version != self._version
```

1. 这是按单一 version 判断的辅助方法，目前仍被部分兼容性测试使用。
2. 真正的渲染回调和消费过滤使用 _cancel_request，而不是用这个方法否定全部较旧预览；学习时不要把它误当成当前主取消规则。

#### `RenderManager._cancel_request`

[定位到实现](<D:/code project/AniFaceProject/src/renderer.py:172>)

```python
def _cancel_request(self, request):
    with self._condition:
        return (self.is_shutting_down or request.epoch != self._epoch or
                (request.kind == "hq" and request.version != self._version) or
                (request.kind == "preview" and
                 time.monotonic() - request.submitted_at > config.MAX_PREVIEW_AGE_SECONDS))
```

1. 锁内按三个层次判断：关闭或 epoch 不同一律取消；hq 的 version 不同取消；preview 的提交年龄超过阈值取消。
2. or 会短路，any 一项成立就返回 True。使用 time.monotonic 减 submitted_at 得到秒数，包含队列等待时间。

#### `RenderManager.active_lora`

[定位到实现](<D:/code project/AniFaceProject/src/renderer.py:180>)

```python
@property
def active_lora(self):
    with self._condition:
        return {"path": self._lora_path, "weight": self._current_lora[1] if self._current_lora else None}
```

1. 返回一个新字典，包括当前加载路径和当前适配器权重；没有适配器时权重为 None。
2. 供 GUI 标注实际状态和 worker 记录结果元数据，不代表输入框正在编辑的值已生效。

#### `RenderManager._switch_lora`

[定位到实现](<D:/code project/AniFaceProject/src/renderer.py:184>)

```python
def _switch_lora(self, path, weight):
    # 仅由模型工作线程调用。新适配器成功启用前，保留旧权重在内存中。
    if path is None:
        try:
            self.pipe.unload_lora_weights()
        except Exception as exc:
            self._model_error = "LoRA 卸载失败，模型状态不确定；请重新加载或重启"
            raise LoraSwitchError(self._model_error) from exc
        self._current_lora = None
        self._lora_path = None
        self._retired_adapters.clear()
        self._model_error = None
        self._events.put(RenderEvent("info", "已恢复基础模型"))
        return
    if self._model_error is not None:
        try:
            self.pipe.unload_lora_weights()
        except Exception as exc:
            raise LoraSwitchError(f"无法清理模型状态，请重启应用：{exc}") from exc
        self._current_lora = None
        self._lora_path = None
        self._retired_adapters.clear()
        self._model_error = None
    previous = self._current_lora
    if previous is not None and path == self._lora_path:
        try:
            self.pipe.set_adapters([previous[0]], adapter_weights=[weight])
        except Exception as exc:
            try:
                self.pipe.set_adapters([previous[0]], adapter_weights=[previous[1]])
            except Exception as recovery:
                self._model_error = "权重恢复失败，已暂停推理，请重新加载或重启"
                raise LoraSwitchError(self._model_error) from recovery
            raise LoraSwitchError(f"权重更新失败，已恢复原权重：{exc}") from exc
        self._current_lora = (previous[0], weight)
        self._events.put(RenderEvent("info", "LoRA 权重已更新"))
        return
    name = "paint_lora" if self._lora_sequence == 0 else f"paint_lora_{self._lora_sequence}"
    self._lora_sequence += 1
    try:
        self.pipe.load_lora_weights(path, adapter_name=name)
        self.pipe.set_adapters([name], adapter_weights=[weight])
    except Exception as exc:
        try:
            if previous is not None:
                self.pipe.delete_adapters(name)
                self.pipe.set_adapters([previous[0]], adapter_weights=[previous[1]])
            else:
                self.pipe.unload_lora_weights()
        except Exception as recovery:
            self._model_error = f"LoRA 恢复失败，已暂停推理，请重新加载 LoRA 或重启应用：{recovery}"
            raise LoraSwitchError(f"加载失败：{exc}；{self._model_error}") from recovery
        restored = "原来的 LoRA 和权重" if previous is not None else "基础模型"
        raise LoraSwitchError(f"LoRA 加载失败，已恢复{restored}：{exc}") from exc
    self._current_lora = (name, weight)
    self._lora_path = path
    if previous is not None:
        self._retired_adapters.add(previous[0])
    # 清理失败不会撤销已成功的切换；保留名字，在下一次成功切换后重试清理。
    for retired in tuple(self._retired_adapters):
        try:
            self.pipe.delete_adapters(retired)
            self._retired_adapters.remove(retired)
        except Exception:
            logger.exception("旧 LoRA 释放失败: %s", retired)
    suffix = "；旧适配器尚未完全释放显存" if self._retired_adapters else ""
    self._events.put(RenderEvent("info", f"LoRA 已加载{suffix}"))
```

1. 仅在 worker 执行，按分支阅读。① path 为 None：卸载所有 LoRA，清理内部状态并发成功信息；卸载失败标记模型不可用。
2. ② 已有 _model_error：先尝试清理到基础状态，再允许恢复加载；失败继续阻止推理。
3. ③ 同一路径：直接 set_adapters 更新权重，失败恢复旧权重；恢复也失败就记入 _model_error。
4. ④ 新路径：生成唯一适配器名称，加载并激活；失败删除不完整适配器，恢复旧适配器及其权重，没有旧适配器则卸载回基础。
5. ⑤ 成功后才把新适配器记为当前，将旧名称加入 retired 集合；逐个尝试删除，失败保留名字以便下次重试。清理旧资源失败不撤销已经成功的新配置。
6. 最后发 info 事件，GUI 收到后才提交待生效预设。这个过程是事务式设计，不是数据库提供的原子事务。

#### `RenderManager._run`

[定位到实现](<D:/code project/AniFaceProject/src/renderer.py:252>)

```python
def _run(self):
    while True:
        with self._condition:
            self._condition.wait_for(lambda: self.is_shutting_down or
                                     self._pending_lora is not None or
                                     self._pending_render is not None)
            if self.is_shutting_down:
                return
            lora = self._pending_lora
            if lora is not None:
                self._pending_lora = None
                request = None
                self._active = "lora"
            else:
                request = self._pending_render
                self._pending_render = None
                self._active = request.kind
        try:
            if lora is not None:
                self._switch_lora(*lora)
            else:
                if self._model_error is not None:
                    raise RuntimeError(self._model_error)
                steps = (config.PREVIEW_NUM_INFERENCE_STEPS if request.kind == "preview"
                         else config.HQ_NUM_INFERENCE_STEPS)
                options = {} if request.control_end is None else {"control_end": request.control_end}
                image = self._render(self.pipe, request.image, request.prompt, steps,
                                     lambda: self._cancel_request(request), **options)
                metadata = dict(request.metadata or {})
                metadata.update(prompt=request.prompt, seed=config.SEED, steps=steps,
                                mode=request.kind, base_model=config.BASE_MODEL_ID,
                                control_type=self.pipe.__dict__.get("_aniface_control_type", "scribble"),
                                control_scale=config.HQ_CONTROLNET_CONDITIONING_SCALE,
                                control_end=request.control_end if request.control_end is not None else config.HQ_CONTROLNET_ENDING_STEP,
                                lora=self.active_lora, version=request.version,
                                submitted_at=request.submitted_at)
                self._events.put(RenderEvent("frame", image, request.version, request, metadata))
        except LoraSwitchError as exc:
            logger.error("%s", exc)
            self._events.put(RenderEvent("lora_error", str(exc)))
        except RenderBlocked as exc:
            logger.warning("%s", exc)
            self._events.put(RenderEvent("blocked", str(exc), request.version, request))
        except RenderCancelled:
            pass
        except Exception as exc:
            logger.exception("模型操作失败")
            prefix = "LoRA 操作失败" if lora is not None else "渲染失败"
            version = None if request is None else request.version
            self._events.put(RenderEvent("error", f"{prefix}: {exc}", version, request))
        finally:
            with self._condition:
                self._active = None
```

1. worker 的永久循环。持锁 wait_for 等待关闭/LoRA/图像，关闭就 return；有 LoRA 时优先取它，否则取最新图并清空待处理槽。
2. 离开 with 块后才进行耗时模型操作，所以 GUI 能继续提交和取消。LoRA 走 _switch_lora，图像先检查模型是否可用，再根据 kind 选 10/25 步。
3. 传给 render 的 lambda 在每次调用时检查这个 request 的当前有效性；不是提交时求出一次布尔值后就固定。
4. 成功结果附带请求快照和元数据入队。RenderCancelled 正常忽略，RenderBlocked 发拦截事件，LoRA 错误单独提示，其他异常保留版本/request 供消费端过滤。
5. 无论哪个分支成功或失败，finally 都锁内清空 _active，避免状态永远卡在“正在生成”。

<a id="file-7"></a>
## 7. `src/prompts.py`：提示词组合

这是纯字符串逻辑，无 GUI 和 GPU 副作用，很适合先动手练习。

[打开源文件](<D:/code project/AniFaceProject/src/prompts.py>)

### 模块级代码

```python
EXPRESSION_PRESETS = {
    "不附加表情": "",
    "微笑": "smiling",
    "闭眼微笑": "closed eyes, smiling",
    "惊讶": "surprised, open mouth",
}
```

#### `compose_prompt`

[定位到实现](<D:/code project/AniFaceProject/src/prompts.py:11>)

```python
def compose_prompt(prompt, expression):
    suffix = EXPRESSION_PRESETS[expression]
    if not suffix:
        return prompt
    # 避免手动填写的同名标签重复；不删除用户写入的其他表情标签。
    tags = {tag.strip().casefold() for tag in prompt.split(",")}
    additions = [tag for tag in suffix.split(", ") if tag.casefold() not in tags]
    base = prompt.rstrip(" ,")
    return ", ".join([part for part in [base, *additions] if part])
```

1. 输入用户字符串和表情选项名称，返回组合后的字符串。EXPRESSION_PRESETS[expression] 取附加标签，没有标签就直接返回原提示词。
2. 把用户现有逗号标签 strip 后 casefold 放进集合，用于不区分大小写的重复检查；集合只是用于判断，不拿来重排用户文本。
3. 列表推导式筛选缺少的表情标签；rstrip(" ,") 清掉末尾空格/逗号，然后 join 拼起来。
4. 例：输入 portrait, SMILING 加“闭眼微笑”，只补 closed eyes；不会删除用户自己写的冲突标签，也不会写回输入框。

<a id="file-8"></a>
## 8. `src/profiles.py`：角色与画风配置

这里只解析和验证配置。成功读出 JSON 还没有使模型应用该 LoRA。

[打开源文件](<D:/code project/AniFaceProject/src/profiles.py>)

### 模块级代码

```python
from dataclasses import asdict, dataclass

import json

import math

from pathlib import Path

import config
```

### 类 `ReferenceProfile`

```python
@dataclass(frozen=True)
class ReferenceProfile:
    name: str
    kind: str
    base_model: str
    lora_path: str
    weight: float
    trigger_words: str
    default_prompt: str = "anime illustration, high quality"
    control_end: float = 1.0
```

字段类型用于表达约定；类型注解本身不自动做运行时验证。`frozen=True`（若有）限制字段重新赋值，不会递归冻结字段中的 PIL 图片或字典，所以输入快照仍要复制。专用异常类则用于让调用方区分取消、拦截和真正失败。

#### `ReferenceProfile.metadata`

[定位到实现](<D:/code project/AniFaceProject/src/profiles.py:21>)

```python
def metadata(self):
    return asdict(self)
```

1. asdict(self) 将 dataclass 字段转换为普通字典，供请求 metadata 和 JSON 保存使用。
2. 它不加载权重、不验证效果，只输出配置的数据表示。

#### `load_profile`

[定位到实现](<D:/code project/AniFaceProject/src/profiles.py:25>)

```python
def load_profile(path):
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    profile = ReferenceProfile(**data)
    for key in ("name", "base_model", "lora_path", "trigger_words", "default_prompt"):
        if not isinstance(getattr(profile, key), str) or not getattr(profile, key).strip():
            raise ValueError(f"预设 {key} 不能为空")
    if profile.kind not in {"character", "style"}:
        raise ValueError("kind 必须是 character 或 style")
    if profile.base_model != config.BASE_MODEL_ID:
        raise ValueError("预设的基础模型与当前管线不一致；请使用匹配的预设")
    if not math.isfinite(profile.weight) or not config.LORA_WEIGHT_MIN <= profile.weight <= config.LORA_WEIGHT_MAX:
        raise ValueError("LoRA 权重超出范围")
    if not math.isfinite(profile.control_end) or not 0 < profile.control_end <= 1:
        raise ValueError("control_end 必须在 (0, 1] 范围内")
    # 预设使用本地适配器，以免拼写错误触发隐式下载。
    lora = Path(profile.lora_path)
    if not lora.is_absolute():
        lora = path.parent / lora
    if not lora.exists():
        raise ValueError(f"找不到预设 LoRA：{lora}")
    return ReferenceProfile(**{**data, "lora_path": str(lora.resolve())})
```

1. 将 JSON 文件路径变成 Path，使用 utf-8-sig 兼容 BOM，再把字典展开到 ReferenceProfile(**data)。此处 ** 是关键字参数展开，不是乘方。
2. 依次检查文字字段、kind、基础模型 ID、有限数值与允许范围；配置匹配只比较当前基础模型 ID，不证明权重结构一定兼容。
3. 相对 LoRA 路径以 JSON 所在目录为基准；确认存在后 resolve 成绝对路径，返回新预设对象。
4. ReferenceProfile(**{**data, "lora_path": ...}) 中后面的同名键覆盖原路径；整个函数还没有修改当前 GUI 或模型。

<a id="file-9"></a>
## 9. `src/references.py`：历史参考与保存

把输出、原输入与参数绑定，是正确保存固定参考的基础。

[打开源文件](<D:/code project/AniFaceProject/src/references.py>)

### 模块级代码

```python
from copy import deepcopy

from dataclasses import dataclass

import base64

from io import BytesIO

import json

from PIL.PngImagePlugin import PngInfo
```

### 类 `Reference`

```python
@dataclass
class Reference:
    image: object
    sketch: object
    metadata: dict
```

字段类型用于表达约定；类型注解本身不自动做运行时验证。`frozen=True`（若有）限制字段重新赋值，不会递归冻结字段中的 PIL 图片或字典，所以输入快照仍要复制。专用异常类则用于让调用方区分取消、拦截和真正失败。

#### `Reference.capture`

[定位到实现](<D:/code project/AniFaceProject/src/references.py:18>)

```python
@classmethod
def capture(cls, image, sketch, metadata):
    return cls(image.copy(), sketch.copy(), deepcopy(metadata))
```

1. @classmethod 的首参 cls 是类，不是已有实例。调用 Reference.capture(...) 会构造一个新的 Reference。
2. 输出图和线稿分别 copy，metadata 深复制，防止之后的 GUI 操作改变已经保存的历史内容。

#### `Reference.save`

[定位到实现](<D:/code project/AniFaceProject/src/references.py:21>)

```python
def save(self, path):
    buffer = BytesIO()
    self.sketch.save(buffer, format="PNG")
    info = PngInfo()
    info.add_itxt("aniface.recipe", json.dumps(self.metadata, ensure_ascii=False))
    info.add_itxt("aniface.sketch.png.base64", base64.b64encode(buffer.getvalue()).decode("ascii"))
    self.image.save(path, format="PNG", pnginfo=info)
```

1. 先把对应线稿写入 BytesIO 内存流；PNG 压缩得到二进制，再 Base64 编码成可写入文本字段的 ASCII。
2. PngInfo.add_itxt 写入生成配方 JSON 和线稿编码；ensure_ascii=False 保留中文。
3. 最后将当前参考的输出图片写成带这些信息的 PNG。Base64 是编码，不是加密；其他软件重存 PNG 时可能删除附加字段。

<a id="coverage"></a>
## 核心函数覆盖说明

本手册覆盖上述核心模块的全部 **76 个函数/方法**（包含 stage 和 on_step_end 两个内部函数）；config 的全部设置另表解释。`src/__init__.py` 当前为空，只作为包结构文件，没有遗漏的业务逻辑。

## 源码快照校验

以下 SHA-256 对应生成本手册时的文件原始字节；它不是 Git 提交编号。以后源码改变，摘录需要同步更新。

| 文件 | SHA-256 |
|---|---|
| `canvas_stream.py` | `beab4de541ed342b19a546def1ae6cb6e1eec0a4a27a06a329e878c7195f0ce3` |
| `config.py` | `40dc881563dffb05124d93cd158c2b103559184e0baf99c6f795c256947c9ed6` |
| `src/startup.py` | `c5a9e146a1954adf59edbd9c6dc6a9bf04445b474ea1fa6beca3c35b29e3c182` |
| `src/pipeline.py` | `14c1c8415ea7cfa8df2c3d5e42242c94c6bec37ca5969f35844739f507010706` |
| `src/app.py` | `02c8abd4ad0e4b376504306007828defaf993cceecdd509c020b74d0ce65db10` |
| `src/renderer.py` | `75b723559f82b974062ecd34598a0f4d7e0a1301730fb027ccf311bcfec5163c` |
| `src/prompts.py` | `20b920654719fee275290bdd3b8c3ef39b27b70664332308ddbcff158555e416` |
| `src/profiles.py` | `8a870e7e09baae1f484ed7daf476bc3a1d044eaa03fbfef2b07139645f9fa12e` |
| `src/references.py` | `59fa504498ebe3af8979f7c83d3c121a6e27783d5a19bc71c0f700c7d8072d44` |


<a id="traces"></a>
## 10. 把函数串起来：三个完整例子

### 10.1 从白纸上画一条线

假设模型已加载、未暂停、自动 HQ 关闭，GUI 此前没有提交任务。下面的版本数字用于说明当前实现中“失效”和“提交”都会改变计数，实际运行还会受其他操作影响。

| 顺序 | 调用/事件 | 数据变化 | 执行者 |
|---|---|---|---|
| 1 | 按下 (20, 30) | 保存白纸撤销快照，last_x/y 变成原图坐标 | 主线程 |
| 2 | _draw_to → _changed(soft=True) | renderer version 从 0 到 1，epoch 仍 0 | 主线程 |
| 3 | PIL 画点并刷新 | sketch_img 已改变，左侧马上可见 | 主线程 |
| 4 | 请求满足节流条件 | _request 复制图像、记录提示词，version 到 2，放进 pending | 主线程 |
| 5 | worker 取走请求 | active=preview，pending=None，释放调度锁 | worker |
| 6 | 用户很快移动到 (40, 30) | 再画线，soft invalidation 将 version 到 3；此时可能还没到下一次预览提交间隔 | 主线程 |
| 7 | 第一个请求完成 | request.version=2，epoch=0，若年龄未超限，仍可显示 | worker → 事件队列 |
| 8 | _poll_renderer | renderer 过滤后，GUI 保存请求对应的参考，更新右侧 | 主线程 |
| 9 | 松手 | 清掉 last_x/y，再提交最终画布预览；自动 HQ 关闭则不安排精细 | 主线程 |

注意第 7 步允许略旧的同上下文预览，所以“当前版本已经变成 3”不等于一定取消版本 2 的预览。若第 6 步改成清空或切换提示词，epoch 会变化，旧图就不能显示。

画布状态一直在变，模型每次只看到某个时刻的副本。参考 metadata 的 canvas_revision 不等于 renderer 的 version，它用于把参考与 GUI 输入状态对应起来。

### 10.2 从角色 A 切换到角色 B

1. 当前 _profile 是 A，用户选择 B 的 JSON。load_profile 先验证路径、数值与基础模型 ID。
2. 解析成功后，GUI 将 B 存到 _pending_profile；此时界面不能提前声称模型已经是 B。
3. _changed 硬失效旧结果，_loading_lora=True 阻止中间画布变化提交新推理。
4. worker 从 pending_lora 取 B；保留 A 的适配器，尝试加载并激活 B。
5. 若成功，worker 记 B 为当前，清理 A，发 info。GUI 收到后才提交 _profile=B、更新触发词与输入框。
6. 变量改变触发 trace 时门禁仍关闭；设置结束后解除 _loading_lora，再提交完整 B 配置的预览。
7. 若失败，worker 尝试恢复 A，并发 lora_error；GUI 清掉待提交 B，保留 A。若连恢复也失败，模型错误状态阻止继续生成。

把流程分成“准备”和“成功后生效”，才能避免输入框说 B、模型实际还是 A 的正常切换中间态。第三方操作仍可能失败，所以错误分支必须单独理解。

### 10.3 保存一张固定的旧参考

1. frame 事件带回输出图、当时的线稿快照和 metadata，Reference.capture 将三者绑定。
2. 用户勾选固定参考，然后继续画，sketch_img 与输入参数可以变化，当前 _reference 保持原对象。
3. _save_result 在打开保存对话框前捕获这个 Reference。
4. Reference.save 从捕获对象取 image/sketch/metadata，不去读取现在的画布或输入框。
5. PNG 外观是选中的参考，附加信息也是该参考的生成输入。当前画布的新笔画不会混入这张历史图的配方。

<a id="tests-tools"></a>
## 11. 测试与工具代码怎么学

本卷逐函数展开的是应用核心；测试和实验工具按下面的顺序读，重点理解它们怎样验证业务规则。第三方库的内部算法继续参考技术指南，不把库源码当成项目自行实现的代码。

| 文件 | 阅读顺序与作用 |
|---|---|
| [test_pipeline.py](<D:/code project/AniFaceProject/tests/test_pipeline.py>) | 先看 prepare_control_image 和参数传递断言，再看模拟 GUI 的 setUp；用 __new__ 跳过真实窗口初始化是测试技巧，不是正式启动方式 |
| [test_thread_safety.py](<D:/code project/AniFaceProject/tests/test_thread_safety.py>) | setUp → render → start_preview → 各 test → tearDown；用 Event 门控制“正在计算”与“允许完成” |
| [test_continuous_preview.py](<D:/code project/AniFaceProject/tests/test_continuous_preview.py>) | 推理被门阻塞时连续 soft 更新，放行后应当获得 frame；另测过期、HQ 取消和 LoRA 更新 |
| [test_startup.py](<D:/code project/AniFaceProject/tests/test_startup.py>) | 假工厂制造成功、失败与取消，检查关闭后绝不再交接到画板 |
| [test_reference_workflow.py](<D:/code project/AniFaceProject/tests/test_reference_workflow.py>) | 创建真实隐藏 Tk，模拟推理；测试固定/暂停/历史/保存，以及缩放、橡皮和预设失败恢复 |
| [test_quality.py](<D:/code project/AniFaceProject/tests/test_quality.py>) | 临时文件 + mock GPU/模型，检查评估矩阵数量、文件与拦截标记；不评判图片好不好看 |
| [benchmark.py](<D:/code project/AniFaceProject/tools/benchmark.py>) | argparse → 加载 → 预热 → GPU synchronize → 两档计时 → 写 JSON；verify_lora 验证零增量接口 |
| [runtime_gpu.py](<D:/code project/AniFaceProject/tools/runtime_gpu.py>) | 实际取消、恢复推理、损坏适配器回滚、原地改权重、替换和卸载；assert 让失败真正影响退出码 |
| [continuous_preview.py](<D:/code project/AniFaceProject/tools/continuous_preview.py>) | 持续输入并轮询，记录帧年龄与间隔，期间不故意等用户停笔 |
| [gui_smoke.py](<D:/code project/AniFaceProject/tools/gui_smoke.py>) | 实际 GPU + 隐藏 Tk，驱动画布方法，root.update 泵送事件，再检查固定参考保存 |
| [stress_renderer.py](<D:/code project/AniFaceProject/tools/stress_renderer.py>) | FakePipe 模拟适配器，TrackedPipe 统计模型操作并发；随机操作后验证工作线程退出和结果规则 |
| [evaluate_quality.py](<D:/code project/AniFaceProject/tools/evaluate_quality.py>) | 解析模型/预设/线稿/种子/表情组合，逐次生成并记录；输出对照图用于人工评审，不把黑色占位当作可见生成结果 |

### 11.1 一个小测试逐行读


```python
def test_clear_drops_already_completed_output(self):
    self.start_preview()
    self.release.set()
    self.idle()
    self.manager.invalidate()
    self.assertEqual(self.manager.poll_events(), [])
```

- start_preview 启动第一张图，并等假渲染器确认进入计算。
- release.set 允许假推理完成；idle 等 worker 回到空闲，但尚未消费结果队列。
- invalidate 模拟用户此时清空或更换输入。
- 最后一行要求 poll_events 返回空列表，证明“已完成但还没显示”的旧结果也被过滤。

如果只测正在计算的任务能被取消，就覆盖不到这个完成后到消费前的时间窗口。这个小测试直接对应一条容易遗漏的竞态。

### 11.2 工具中的 contextmanager 和 yield


```python
@contextmanager
def operation(self):
    with self.lock:
        self.concurrent += 1
        self.maximum = max(self.maximum, self.concurrent)
        self.thread_ids.add(threading.get_ident())
    try:
        yield
    finally:
        with self.lock:
            self.concurrent -= 1
```

这个方法配合 `with pipe.operation():` 使用。进入 with 时执行 yield 之前的计数；yield 把控制权交给 with 内部操作；退出时无论成功失败，都在 finally 将并发计数减回去。thread_ids 记录哪些线程操作过模型，maximum 记录峰值并发。

这里的锁只保护计数器，统计的是真实操作执行期间是否重叠。它没有拿统计锁包住整个模型操作，否则会人为串行化被测行为，掩盖被测程序自己的并发问题。

### 11.3 getattr 和函数包装


```python
def __getattr__(self, name):
    method = getattr(self.pipe, name)
    if name in {"load_lora_weights", "set_adapters", "delete_adapters", "unload_lora_weights"}:
        def tracked(*args, **kwargs):
            with self.operation():
                return method(*args, **kwargs)
        return tracked
    return method
```

访问 TrackedPipe 自己没有的属性时才会进入 __getattr__。它先到被包装的 pipe 上取真正属性；如果是需要统计的 LoRA 方法，就返回一个内部 tracked 函数。调用该函数时，先进入统计上下文，再用 *args/**kwargs 原样调用真实方法并返回结果。其他属性直接透传。

这是包装/代理思想：不修改第三方类，也能观测它被哪些线程使用。返回 method 和返回 method() 不同，后者会错误地提前执行。

<a id="practice"></a>
## 12. 动手练习与答案

### 练习 A：为什么鼠标在 x=0 时仍能画？

在 _on_paint 找到 `self.last_x is not None`。若改成 `if self.last_x`，0 会被当成 False，画布左边界就可能无法正常续画。判断“有没有值”与判断“值是否为真”要分开。

### 练习 B：为什么 _redo 不能直接调用 _remember？

_remember 会清空 redo 栈。如果重做时调用它，尚未重做的历史会丢失。因此 _redo 只把当前图压到 undo，并从 redo 弹出一张。

### 练习 C：为什么冻结的 dataclass 还要 image.copy？

frozen 限制给字段重新赋值，不阻止图像对象内部像素改变。GUI 如果继续修改同一张 PIL 图，后台看到的输入也会变；复制才隔离像素数据。

### 练习 D：固定参考后后台为什么还在运行？

查 _can_generate，它检查关闭、暂停、LoRA 加载，没有检查 pinned。再查 _poll_renderer，pinned 只影响是否调用 _show_reference。固定控制展示，暂停控制计算。

### 练习 E：after(50, callback) 是否每 50ms 必定执行？

不会。它把回调排入 Tk 事件循环，主线程忙时仍会延迟。因此要避免主线程长时间计算，并且用实际测量确认界面响应。

### 练习 F：为什么 lambda 里保留的是取消函数而不是布尔值？

worker 要在每个去噪步骤重新读取关闭/版本状态。提前算好的布尔值不会随用户后续操作更新，因此必须传入可调用对象。

### 练习 G：为什么 png 中不能保存现在输入框的参数？

当前显示图可能是固定的历史图。只有 Reference 自带的线稿和 metadata 能描述它当时怎样生成；现在输入框里的参数可能已经属于下一张图。

### 建议的学习节奏

先只读入口、config、prompts 和 references，熟悉函数、字典和对象；第二轮读画笔到请求，再读请求到显示；第三轮读启动、LoRA 失败恢复和关闭。每轮用一个测试验证自己对代码的判断，比一次背完所有函数更容易形成理解。

想检查自己的掌握程度，可以暂时合上文档，画出“用户改变提示词，旧 frame 已在队列里”的处理时序，再回到 poll_events 对照。
