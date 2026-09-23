"""启动加载页。后台只发布事件，所有 Tk 调用均由主线程执行。"""
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


@dataclass(frozen=True)
class LoadEvent:
    kind: str
    value: object = None


class ModelLoader:
    """串行加载/重试，异常事件仅保存文字，不保留模型的异常栈引用。"""

    def __init__(self, factory=create_pipelines):
        self._factory = factory
        self._events = queue.Queue()
        self._cancel = threading.Event()
        self._thread = None

    @property
    def busy(self):
        return self._thread is not None and self._thread.is_alive()

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

    def cancel(self):
        self._cancel.set()

    def join(self, timeout=None):
        if self._thread is not None:
            self._thread.join(timeout)
        return not self.busy

    def poll_events(self):
        events = []
        while True:
            try:
                events.append(self._events.get_nowait())
            except queue.Empty:
                return events

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


class StartupWindow:
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

    def _retry(self):
        if self._closing or self._done or not self.loader.start():
            return
        self._error = None
        self.details.grid_remove()
        self.status.set("正在加载模型…")
        self.retry_button.configure(state="disabled")
        self.progress.start(12)

    def _show_error(self, message):
        self._error = message
        self.progress.stop()
        self.status.set("加载失败。请查看下面的原因，处理后点击重试。")
        self.details.configure(state="normal")
        self.details.delete("1.0", "end")
        self.details.insert("1.0", message)
        self.details.configure(state="disabled")
        self.details.grid()

    def _close(self):
        if self._done or self._closing:
            return
        self._closing = True
        self.loader.cancel()
        self.retry_button.configure(state="disabled")
        self.close_button.configure(state="disabled")
        self.status.set("正在关闭，等待当前加载步骤结束…")

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
