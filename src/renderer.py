"""单工作线程管理模型；GUI 主线程轮询事件，后台不调用 Tkinter。"""
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


@dataclass(frozen=True)
class RenderEvent:
    kind: str
    value: object
    version: int | None = None
    request: RenderRequest | None = None
    metadata: dict | None = None


class LoraSwitchError(Exception):
    """LoRA 切换失败，消息包含恢复结果。"""


class RenderManager:
    """模型仅由 worker 使用；待处理图像和 LoRA 各保留最新一次请求。

    invalidate/request_*/poll_events 由 GUI 主线程调用。epoch 隔离配置，
    version 排序输入；普通笔画允许近期预览完成，HQ 只接受最新输入。
    """

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

    @property
    def is_shutting_down(self):
        return self._shutdown_event.is_set()

    def invalidate(self, *, soft=False):
        """丢弃待处理输入；soft 仅使 HQ 失效，保留同配置的近期预览。"""
        with self._condition:
            self._version += 1
            if not soft:
                self._epoch += 1
                self._render_settings = None
            self._last_input = None
            self._pending_render = None

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

    def request_preview(self, image, prompt, *, control_end=None, metadata=None):
        return self._request("preview", image, prompt, control_end, metadata)

    def request_hq(self, image, prompt, *, control_end=None, metadata=None):
        return self._request("hq", image, prompt, control_end, metadata)

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

    def shutdown(self):
        with self._condition:
            self._shutdown_event.set()
            self._version += 1
            self._pending_render = None
            self._pending_lora = None
            self._condition.notify_all()

    def join(self, timeout=None):
        self._worker.join(timeout)
        return not self._worker.is_alive()

    def _should_cancel(self, version):
        with self._condition:
            return self.is_shutting_down or version != self._version

    def _cancel_request(self, request):
        with self._condition:
            return (self.is_shutting_down or request.epoch != self._epoch or
                    (request.kind == "hq" and request.version != self._version) or
                    (request.kind == "preview" and
                     time.monotonic() - request.submitted_at > config.MAX_PREVIEW_AGE_SECONDS))

    @property
    def active_lora(self):
        with self._condition:
            return {"path": self._lora_path, "weight": self._current_lora[1] if self._current_lora else None}

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
                                    base_revision=config.BASE_MODEL_REVISION,
                                    controlnet=config.CONTROLNET_MODEL_ID,
                                    controlnet_revision=config.CONTROLNET_MODEL_REVISION,
                                    inference_size=list(config.INFERENCE_SIZE),
                                    guidance_scale=config.GUIDANCE_SCALE,
                                    negative_prompt=config.NEGATIVE_PROMPT,
                                    scheduler=type(getattr(self.pipe, "scheduler", None)).__name__,
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
