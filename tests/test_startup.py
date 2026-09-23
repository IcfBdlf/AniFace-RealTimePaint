"""启动加载的失败、重试、取消和主线程交接测试，无 GPU/窗口。"""
import sys
import threading
import unittest
from unittest.mock import Mock, patch

from src.pipeline import create_pipelines, LoadCancelled
from src.startup import LoadEvent, ModelLoader, StartupWindow


class TestModelLoader(unittest.TestCase):
    def tearDown(self):
        if hasattr(self, "release"):
            self.release.set()
        if hasattr(self, "loader"):
            self.loader.cancel()
            self.assertTrue(self.loader.join(3))

    def test_success_reports_stages_and_returns_model(self):
        model = object()
        threads = []
        def factory(on_progress, should_stop):
            threads.append(threading.get_ident())
            on_progress("loading")
            return model
        self.loader = ModelLoader(factory)
        gc_threads = []
        with patch("src.startup.gc.collect", side_effect=lambda: gc_threads.append(threading.get_ident())):
            self.assertTrue(self.loader.start())
            self.assertTrue(self.loader.join(3))
        self.assertEqual(gc_threads, [threading.get_ident()])
        events = self.loader.poll_events()
        self.assertEqual([e.kind for e in events], ["progress", "ready"])
        self.assertIs(events[-1].value, model)
        self.assertNotEqual(threads[0], threading.get_ident())

    def test_failure_retries_without_restarting_process(self):
        model = object()
        factory = Mock(side_effect=[OSError("missing model"), model])
        self.loader = ModelLoader(factory)
        with self.assertLogs("AniFace.startup", level="ERROR"):
            self.loader.start()
            self.assertTrue(self.loader.join(3))
        error = self.loader.poll_events()[0]
        self.assertEqual(error.kind, "error")
        self.assertIn("OSError: missing model", error.value)
        self.assertTrue(self.loader.start())
        self.assertTrue(self.loader.join(3))
        self.assertEqual([e.kind for e in self.loader.poll_events()], ["ready"])

    def test_close_during_load_suppresses_ready_and_rejects_duplicate_start(self):
        started = threading.Event()
        self.release = threading.Event()
        def factory(on_progress, should_stop):
            started.set()
            if not self.release.wait(3):
                raise TimeoutError()
            return object()  # 模拟不支持中途取消的第三方加载调用。
        self.loader = ModelLoader(factory)
        self.loader.start()
        self.assertTrue(started.wait(2))
        self.assertFalse(self.loader.start())
        self.loader.cancel()
        self.release.set()
        self.assertTrue(self.loader.join(3))
        self.assertEqual(self.loader.poll_events(), [])

    def test_cooperative_cancellation_is_not_an_error(self):
        self.loader = ModelLoader(Mock(side_effect=LoadCancelled()))
        self.loader.start()
        self.assertTrue(self.loader.join(3))
        self.assertEqual(self.loader.poll_events(), [])

    def test_pipeline_cancelled_before_import_or_loading(self):
        with self.assertRaises(LoadCancelled):
            create_pipelines(should_stop=lambda: True)

    def test_cancel_between_models_skips_base_loading(self):
        stopped = threading.Event()
        torch, diffusers = Mock(), Mock()
        torch.device.return_value = Mock(type="cpu")
        diffusers.ControlNetModel.from_pretrained.side_effect = lambda *a, **k: stopped.set()
        with patch.dict(sys.modules, {"torch": torch, "diffusers": diffusers}):
            with self.assertRaises(LoadCancelled):
                create_pipelines(should_stop=stopped.is_set)
        diffusers.StableDiffusionControlNetPipeline.from_pretrained.assert_not_called()


class TestStartupWindowEvents(unittest.TestCase):
    def setUp(self):
        self.window = StartupWindow.__new__(StartupWindow)
        for name in ("root", "loader", "on_ready", "frame", "progress", "status",
                     "retry_button", "close_button", "details"):
            setattr(self.window, name, Mock())
        self.window._done = False
        self.window._closing = False
        self.window._error = None

    def test_ready_transitions_to_app_on_poll(self):
        window = self.window
        model = object()
        window.loader.poll_events.return_value = [LoadEvent("ready", model)]
        window._poll()
        window.on_ready.assert_called_once_with(window.root, model)
        window.frame.destroy.assert_called_once()
        self.assertTrue(window._done)
        window.root.after.assert_not_called()

    def test_error_shows_details_and_enables_retry_only_after_worker_exits(self):
        window = self.window
        window.loader.busy = True
        window.loader.poll_events.return_value = [LoadEvent("error", "load failed")]
        window._poll()
        window.details.insert.assert_called_with("1.0", "load failed")
        window.retry_button.configure.assert_not_called()
        window.loader.busy = False
        window.loader.poll_events.return_value = []
        window._poll()
        window.retry_button.configure.assert_called_with(state="normal")
        window._retry()
        self.assertIsNone(window._error)
        window.loader.start.assert_called_once()

    def test_close_never_enters_app_even_if_ready_is_already_queued(self):
        window = self.window
        window.loader.join.side_effect = [False, True]
        window.loader.poll_events.return_value = [LoadEvent("ready", object())]
        window._close()
        window._poll()
        window.root.destroy.assert_not_called()
        window.loader.poll_events.return_value = []
        window._poll()
        window.root.destroy.assert_called_once()
        window.on_ready.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
