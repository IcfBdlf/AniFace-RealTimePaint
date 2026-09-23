"""用同步门控制推理边界，验证连续输入、过期隔离和 LoRA 操作。"""
import threading
import time
import unittest
from unittest.mock import Mock, patch
from PIL import Image

import config
from src.renderer import RenderManager


class ContinuousPreviewTests(unittest.TestCase):
    def test_frames_arrive_while_strokes_continue_and_hard_change_discards(self):
        started = threading.Event()
        release = threading.Event()
        completed = threading.Event()
        cancelled = []

        def render(pipe, image, prompt, steps, stop):
            started.set()
            self.assertTrue(release.wait(2))
            cancelled.append(stop())
            completed.set()
            return image

        manager = RenderManager(Mock(), render_fn=render)
        try:
            for batch in range(3):
                started.clear()
                release.clear()
                completed.clear()
                manager.request_preview(Image.new("RGB", (8, 8), (batch, 0, 0)), "fixed character")
                self.assertTrue(started.wait(2))
                for _ in range(24):
                    manager.invalidate(soft=True)
                # 用户仍在画，并没有停笔；必须能展示当前预览。
                release.set()
                self.assertTrue(completed.wait(2))
                deadline = time.monotonic() + 2
                frames = []
                while not frames and time.monotonic() < deadline:
                    frames.extend(manager.poll_events())
                    time.sleep(.002)
                self.assertEqual([event.kind for event in frames], ["frame"])
                self.assertEqual(frames[0].value.getpixel((0, 0)), (batch, 0, 0))
            self.assertEqual(cancelled, [False] * 3)
            started.clear()
            release.clear()
            manager.request_preview(Image.new("RGB", (8, 8)), "fixed character")
            self.assertTrue(started.wait(2))
            manager.invalidate()
            release.set()
            deadline = time.monotonic() + 2
            while manager.status != "就绪" and time.monotonic() < deadline:
                time.sleep(.002)
            self.assertEqual(manager.poll_events(), [])
            self.assertTrue(cancelled[-1])
        finally:
            release.set()
            manager.shutdown()
            self.assertTrue(manager.join(2))

    def test_expired_preview_and_soft_cancelled_hq_never_display(self):
        for kind in ("preview", "hq"):
            with self.subTest(kind=kind):
                started, release = threading.Event(), threading.Event()
                def render(pipe, image, prompt, steps, stop):
                    started.set()
                    release.wait(2)
                    return image
                manager = RenderManager(Mock(), render_fn=render)
                try:
                    getattr(manager, f"request_{kind}")(Image.new("RGB", (8, 8)), "same")
                    self.assertTrue(started.wait(2))
                    manager.invalidate(soft=True)
                    with patch.object(config, "MAX_PREVIEW_AGE_SECONDS", 0):
                        time.sleep(.02)  # Windows monotonic 时钟可能以约 15ms 为粒度
                        release.set()
                        deadline = time.monotonic() + 2
                        while manager.status != "就绪" and time.monotonic() < deadline:
                            time.sleep(.002)
                        self.assertEqual(manager.poll_events(), [])
                finally:
                    release.set()
                    manager.shutdown()
                    self.assertTrue(manager.join(2))

    def test_reweight_does_not_reload_and_unload_resets_state(self):
        pipe = Mock()
        manager = RenderManager(pipe)
        def idle():
            deadline = time.monotonic() + 2
            while manager.status != "就绪" and time.monotonic() < deadline:
                time.sleep(.002)
            self.assertEqual(manager.status, "就绪")
        try:
            manager.load_lora("same", .5)
            idle()
            manager.load_lora("same", .8)
            idle()
            pipe.load_lora_weights.assert_called_once()
            self.assertEqual(manager.active_lora, {"path": "same", "weight": .8})
            manager.load_lora(None, 0)
            idle()
            pipe.unload_lora_weights.assert_called_once()
            self.assertIsNone(manager.active_lora["path"])
        finally:
            manager.shutdown()
            self.assertTrue(manager.join(2))
