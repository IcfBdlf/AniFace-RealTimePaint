"""渲染管理器：线程调度、推理执行、状态管理"""
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
        self.pipe = base_pipe
        self.stream = stream
        self._schedule = schedule_fn
        self._on_frame = on_frame_fn
        self._get_sketch = get_sketch_fn

        self.is_rendering = False
        self.need_update_again = False
        self._preview_thread = None
        self._last_prompt = ""

        self.state_lock = threading.Lock()
        self._shutdown_event = threading.Event()

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    @property
    def is_shutting_down(self):
        return self._shutdown_event.is_set()

    def request_preview(self, sketch_snapshot):
        """发起流式预览渲染（后台线程）。"""
        t = threading.Thread(
            target=self._run_preview, args=(sketch_snapshot,), daemon=True
        )
        t.start()
        with self.state_lock:
            self._preview_thread = t

    def request_hq(self, sketch_snapshot, prompt):
        """请求 25 步超清重绘；若正在渲染则排队，完成后自动重试一次。"""
        with self.state_lock:
            if self.is_rendering:
                self.need_update_again = True
                self._last_prompt = prompt
                return
            self.is_rendering = True
            self._last_prompt = prompt

        t = threading.Thread(
            target=self._run_hq, args=(sketch_snapshot, prompt), daemon=True
        )
        t.start()

    def load_lora(self, lora_path, weight, current_prompt):
        """加载/更新 LoRA 权重并重新准备流式管线。"""
        self.pipe.unload_lora_weights()
        self.pipe.load_lora_weights(lora_path, adapter_name="paint_lora")
        self.pipe.set_adapters(["paint_lora"], adapter_weights=[weight])
        self.stream.prepare(
            prompt=current_prompt,
            num_inference_steps=config.PREVIEW_NUM_INFERENCE_STEPS,
        )

    def shutdown(self):
        """发出关闭信号，后台线程检测后自行退出。"""
        self._shutdown_event.set()

    def check_busy(self):
        """返回 (preview_alive, hq_busy)，供优雅退出轮询使用。"""
        with self.state_lock:
            preview_alive = (
                self._preview_thread is not None
                and self._preview_thread.is_alive()
            )
            hq_busy = self.is_rendering
        return preview_alive, hq_busy

    # ------------------------------------------------------------------
    # 后台线程
    # ------------------------------------------------------------------

    def _run_preview(self, img_snapshot):
        try:
            x_output = self.stream(img_snapshot)
            output_image = postprocess_image(x_output, output_type="pil")[0]
            if not self._shutdown_event.is_set():
                self._schedule(lambda img=output_image: self._on_frame(img))
        except Exception as e:
            logger.error(f"⚠️ 预览渲染失败: {e}")
        finally:
            with self.state_lock:
                self._preview_thread = None

    def _run_hq(self, img_snapshot, prompt_snapshot):
        try:
            if self._shutdown_event.is_set():
                return
            result = self.pipe(
                prompt=prompt_snapshot,
                image=img_snapshot,
                num_inference_steps=config.HQ_NUM_INFERENCE_STEPS,
                controlnet_conditioning_scale=config.HQ_CONTROLNET_CONDITIONING_SCALE,
                controlnet_ending_step=config.HQ_CONTROLNET_ENDING_STEP,
            )
            output_image = result.images[0]
            if not self._shutdown_event.is_set():
                self._schedule(lambda img=output_image: self._on_frame(img))
        except Exception as e:
            logger.error(f"⚠️ 超清重绘失败: {e}")
        finally:
            need_again = False
            with self.state_lock:
                self.is_rendering = False
                if self.need_update_again:
                    self.need_update_again = False
                    self.is_rendering = True
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
