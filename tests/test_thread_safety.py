"""真实 RenderManager 的确定性并发测试：无需 GPU、Tk 窗口或模型。"""
import threading
import time
import unittest
from unittest.mock import Mock
from PIL import Image

import config
from src.renderer import RenderManager
from src.pipeline import RenderBlocked, RenderCancelled


class TestRenderManager(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (16, 16), "white")
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = []
        self.pipe = Mock()
        self.manager = RenderManager(self.pipe, render_fn=self.render)

    def tearDown(self):
        self.release.set()
        self.manager.shutdown()
        self.assertTrue(self.manager.join(3), "worker 必须实际退出")

    def render(self, pipe, image, prompt, steps, should_stop):
        self.calls.append((prompt, steps, image.getpixel((0, 0))))
        if len(self.calls) == 1:
            self.started.set()
            if not self.release.wait(3):
                raise TimeoutError("test did not release inference")
        return image

    def start_preview(self):
        self.manager.request_preview(self.image, "old")
        self.assertTrue(self.started.wait(2))

    def test_control_mode_change_cancels_old_input_but_hq_upgrade_does_not(self):
        options = []
        cancellations = []

        def render(pipe, image, prompt, steps, should_stop, **kwargs):
            options.append(kwargs)
            if len(options) == 1:
                self.started.set()
                self.release.wait(3)
                cancellations.append(should_stop())
            return image

        self.manager._render = render
        self.manager.request_preview(self.image, "same", control_end=1.0)
        self.assertTrue(self.started.wait(2))
        version = self.manager._version
        self.manager.request_hq(self.image, "same", control_end=1.0)
        self.assertEqual(self.manager._version, version)
        self.manager.request_hq(self.image, "same")
        self.assertGreater(self.manager._version, version)
        self.release.set()
        self.idle()
        self.assertEqual(cancellations, [True])
        self.assertEqual(options, [{"control_end": 1.0}, {}])
        self.assertEqual(len([e for e in self.manager.poll_events() if e.kind == "frame"]), 1)

    def idle(self):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if self.manager.status == "就绪":
                return
            time.sleep(0.005)
        self.fail("worker 没有回到空闲状态")

    def test_latest_request_replaces_backlog_and_captures_prompt(self):
        self.start_preview()
        for i in range(30):
            self.manager.request_hq(self.image, f"new-{i}")
        self.release.set()
        self.idle()
        self.assertEqual([c[0] for c in self.calls], ["old", "new-29"])
        frames = self.manager.poll_events()
        self.assertEqual([e.kind for e in frames], ["frame"])
        self.assertEqual(self.calls[-1][1], config.HQ_NUM_INFERENCE_STEPS)

    def test_same_input_preview_can_display_before_hq(self):
        self.start_preview()
        self.manager.request_hq(self.image, "old")
        self.release.set()
        self.idle()
        self.assertEqual([e.kind for e in self.manager.poll_events()], ["frame", "frame"])

    def test_new_prompt_used_for_preview(self):
        self.start_preview()
        self.manager.request_preview(self.image, "new prompt")
        self.release.set()
        self.idle()
        self.assertEqual(self.calls[-1][:2], ("new prompt", config.PREVIEW_NUM_INFERENCE_STEPS))
        self.assertEqual(len(self.manager.poll_events()), 1)

    def test_lora_waits_for_render_and_precedes_next_render(self):
        operations = []
        self.pipe.unload_lora_weights.side_effect = lambda: operations.append("unload")
        self.pipe.load_lora_weights.side_effect = lambda *a, **k: operations.append("load")
        self.pipe.set_adapters.side_effect = lambda *a, **k: operations.append("weight")
        original = self.manager._render
        def render(*args):
            operations.append("render-start")
            result = original(*args)
            operations.append("render-end")
            return result
        self.manager._render = render
        self.start_preview()
        self.manager.load_lora("style", 0.7)
        self.manager.request_hq(self.image, "new")
        self.pipe.load_lora_weights.assert_not_called()
        self.release.set()
        self.idle()
        self.assertEqual(operations, ["render-start", "render-end", "load",
                                      "weight", "render-start", "render-end"])
        self.pipe.set_adapters.assert_called_once_with(["paint_lora"], adapter_weights=[0.7])

    def test_only_latest_pending_lora_is_loaded(self):
        self.start_preview()
        self.manager.load_lora("first", 0.1)
        self.manager.load_lora("last", 0.9)
        self.release.set()
        self.idle()
        self.pipe.load_lora_weights.assert_called_once_with("last", adapter_name="paint_lora")

    def test_clear_drops_active_and_pending_outputs(self):
        self.start_preview()
        self.manager.request_hq(self.image, "pending")
        self.manager.invalidate()
        self.release.set()
        self.idle()
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.manager.poll_events(), [])

    def test_clear_drops_already_completed_output(self):
        self.start_preview()
        self.release.set()
        self.idle()
        self.manager.invalidate()
        self.assertEqual(self.manager.poll_events(), [])

    def test_snapshot_is_not_mutated_by_caller(self):
        self.start_preview()
        self.manager.request_hq(self.image, "new")
        self.image.paste("black", (0, 0, 16, 16))
        self.release.set()
        self.idle()
        self.assertEqual(self.calls[-1][2], (255, 255, 255))

    def test_shutdown_discards_queue_and_rejects_new_work(self):
        self.start_preview()
        self.manager.request_hq(self.image, "pending")
        self.manager.shutdown()
        self.assertFalse(self.manager.request_preview(self.image, "late"))
        self.assertFalse(self.manager.load_lora("late", 1))
        self.release.set()
        self.assertTrue(self.manager.join(2))
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.manager.poll_events(), [])

    def test_render_error_is_reported_and_worker_recovers(self):
        self.start_preview()
        original = self.manager._render
        self.manager._render = Mock(side_effect=ValueError("broken"))
        self.manager.request_hq(self.image, "broken")
        with self.assertLogs("AniFace.renderer", level="ERROR"):
            self.release.set()
            self.idle()
        events = self.manager.poll_events()
        self.assertEqual([e.kind for e in events], ["error"])
        self.assertIn("broken", events[0].value)
        self.manager._render = original
        self.manager.request_hq(self.image, "recovered")
        self.idle()
        self.assertEqual([e.kind for e in self.manager.poll_events()], ["frame"])

    def test_blocked_event_is_versioned_and_worker_recovers(self):
        self.start_preview()
        original = self.manager._render
        self.manager._render = Mock(side_effect=RenderBlocked("blocked"))
        self.manager.request_hq(self.image, "blocked")
        with self.assertLogs("AniFace.renderer", level="WARNING"):
            self.release.set()
            self.idle()
        self.assertEqual([e.kind for e in self.manager.poll_events()], ["blocked"])
        with self.assertLogs("AniFace.renderer", level="WARNING"):
            self.manager.request_hq(self.image, "blocked again")
            self.idle()
        self.manager.invalidate()
        self.assertEqual(self.manager.poll_events(), [])
        self.manager._render = original
        self.manager.request_hq(self.image, "recovered")
        self.idle()
        self.assertEqual([e.kind for e in self.manager.poll_events()], ["frame"])

    def test_lora_failure_cleans_adapter_and_reports_error(self):
        self.start_preview()
        self.pipe.load_lora_weights.side_effect = ValueError("bad lora")
        self.manager.load_lora("bad", 0.5)
        self.manager.request_hq(self.image, "base")
        with self.assertLogs("AniFace.renderer", level="ERROR"):
            self.release.set()
            self.idle()
        self.assertEqual(self.pipe.unload_lora_weights.call_count, 1)
        self.assertEqual([e.kind for e in self.manager.poll_events()], ["lora_error", "frame"])

    def test_stale_render_exits_at_next_step_and_runs_latest(self):
        stopped = threading.Event()
        def render(pipe, image, prompt, steps, should_stop):
            self.calls.append(prompt)
            if prompt == "old":
                self.started.set()
                self.release.wait(2)
                if should_stop():
                    stopped.set()
                    raise RenderCancelled()
                self.fail("旧请求未被取消")
            return image
        self.manager._render = render
        self.start_preview()
        self.manager.request_hq(self.image, "latest")
        self.release.set()
        self.assertTrue(stopped.wait(2))
        self.idle()
        self.assertEqual(self.calls, ["old", "latest"])
        self.assertEqual([e.kind for e in self.manager.poll_events()], ["frame"])

    def test_same_input_upgrade_does_not_cancel_preview(self):
        def render(pipe, image, prompt, steps, should_stop):
            if steps == config.PREVIEW_NUM_INFERENCE_STEPS:
                self.started.set()
                self.release.wait(2)
                self.assertFalse(should_stop())
            return image
        self.manager._render = render
        self.start_preview()
        self.manager.request_hq(self.image, "old")
        self.release.set()
        self.idle()
        self.assertEqual(len(self.manager.poll_events()), 2)

    def test_clear_lora_and_shutdown_cancel_active_request(self):
        for action in (self.manager.invalidate, lambda: self.manager.load_lora("style", 0.7),
                       self.manager.shutdown):
            with self.subTest(action=action):
                version = self.manager._version
                self.assertFalse(self.manager._should_cancel(version))
                action()
                self.assertTrue(self.manager._should_cancel(version))

    def test_failed_lora_restores_previous_adapter_without_reloading_file(self):
        self.manager.load_lora("original", 0.7)
        self.idle()
        self.manager.poll_events()
        self.pipe.load_lora_weights.side_effect = ValueError("broken")
        with self.assertLogs("AniFace.renderer", level="ERROR"):
            self.manager.load_lora("broken", 0.3)
            self.idle()
        self.assertEqual(self.manager._current_lora, ("paint_lora", 0.7))
        self.pipe.set_adapters.assert_called_with(["paint_lora"], adapter_weights=[0.7])
        self.pipe.delete_adapters.assert_called_with("paint_lora_1")
        self.pipe.unload_lora_weights.assert_not_called()
        self.assertEqual(self.pipe.load_lora_weights.call_count, 2)
        self.assertIn("已恢复原来的", self.manager.poll_events()[0].value)

    def test_activation_failure_also_restores_previous_weight(self):
        self.manager.load_lora("original", 0.7)
        self.idle()
        self.pipe.set_adapters.side_effect = [ValueError("activation failed"), None]
        with self.assertLogs("AniFace.renderer", level="ERROR"):
            self.manager.load_lora("other", 0.2)
            self.idle()
        self.assertEqual(self.manager._current_lora, ("paint_lora", 0.7))
        self.pipe.set_adapters.assert_called_with(["paint_lora"], adapter_weights=[0.7])

    def test_successful_switch_releases_previous_adapter(self):
        self.manager.load_lora("original", 0.7)
        self.idle()
        self.manager.load_lora("other", 0.2)
        self.idle()
        self.assertEqual(self.manager._current_lora, ("paint_lora_1", 0.2))
        self.pipe.delete_adapters.assert_called_with("paint_lora")

    def test_failed_rollback_pauses_rendering_until_recovery(self):
        self.manager.load_lora("original", 0.7)
        self.idle()
        self.manager.poll_events()
        self.pipe.load_lora_weights.side_effect = ValueError("broken")
        self.pipe.delete_adapters.side_effect = ValueError("cleanup failed")
        with self.assertLogs("AniFace.renderer", level="ERROR"):
            self.manager.load_lora("broken", 0.3)
            self.manager.request_hq(self.image, "must not render")
            self.idle()
        self.assertEqual(self.calls, [])
        self.assertIsNotNone(self.manager._model_error)
        self.pipe.load_lora_weights.side_effect = None
        self.pipe.delete_adapters.side_effect = None
        self.manager.load_lora("recovered", 0.5)
        self.idle()
        self.assertIsNone(self.manager._model_error)
        self.pipe.unload_lora_weights.assert_called_once()

    def test_retired_cleanup_failure_does_not_undo_successful_switch(self):
        self.manager.load_lora("original", 0.7)
        self.idle()
        self.pipe.delete_adapters.side_effect = ValueError("release failed")
        with self.assertLogs("AniFace.renderer", level="ERROR"):
            self.manager.load_lora("new", 0.2)
            self.idle()
        self.assertEqual(self.manager._current_lora, ("paint_lora_1", 0.2))
        self.assertEqual(self.manager._retired_adapters, {"paint_lora"})
        self.pipe.delete_adapters.side_effect = None
        self.manager.load_lora("next", 0.4)
        self.idle()
        self.assertEqual(self.manager._retired_adapters, set())


if __name__ == "__main__":
    unittest.main(verbosity=2)
