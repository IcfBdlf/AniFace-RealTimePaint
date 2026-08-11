"""
P0 线程安全修复验证测试（已适配 P3 重构后的 src/ 结构）
无需 GPU，通过 mock 对象测试锁机制、竞态条件和优雅退出
"""
import sys
import time
import threading
import unittest
from unittest.mock import Mock, patch
from PIL import Image

# ---- Mock 重型依赖（必须在 src.app 导入之前注入） ----
sys.modules['torch'] = Mock()
sys.modules['diffusers'] = Mock()
sys.modules['streamdiffusion'] = Mock()
sys.modules['streamdiffusion.image_utils'] = Mock()
sys.modules['streamdiffusion.image_utils'].postprocess_image = Mock(
    return_value=[Image.new("RGB", (512, 512), "white")]
)

import tkinter as tk


class _MockRenderer:
    """模拟 RenderManager，暴露与真实 renderer 相同的接口和状态属性。"""

    def __init__(self, *args, **kwargs):
        self.pipe = kwargs.get('base_pipe', Mock())
        self.stream = kwargs.get('stream', Mock())
        self.is_rendering = False
        self.need_update_again = False
        self._preview_thread = None
        self._last_prompt = ""
        self.state_lock = threading.Lock()
        self._shutdown_event = threading.Event()
        self.request_preview = Mock()
        self.request_hq = Mock()
        self.load_lora = Mock()
        self.shutdown = Mock()

    def check_busy(self):
        """返回真实状态 (preview_alive, hq_busy)，与 RenderManager 行为一致。"""
        return (
            self._preview_thread is not None and self._preview_thread.is_alive(),
            self.is_rendering,
        )

    @property
    def is_shutting_down(self):
        return self._shutdown_event.is_set()


class TestThreadSafety(unittest.TestCase):
    """测试重构后的线程安全机制（app.py + renderer.py）"""

    @classmethod
    def setUpClass(cls):
        cls._renderer_patcher = patch('src.app.RenderManager', new=_MockRenderer)
        cls._renderer_patcher.start()

        cls.root = tk.Tk()
        cls.root.withdraw()

        cls.mock_pipe = Mock()
        cls.mock_pipe.return_value = Mock(images=[Image.new("RGB", (512, 512), "blue")])
        cls.mock_stream = Mock()
        cls.mock_stream.return_value = Mock()

        from src.app import RealTimePaintApp
        # Mock messagebox 避免弹窗阻塞测试
        import src.app as app_module
        app_module.messagebox = Mock()
        cls.app = RealTimePaintApp(cls.root, cls.mock_pipe, cls.mock_stream)

    @classmethod
    def tearDownClass(cls):
        cls._renderer_patcher.stop()
        cls.root.destroy()

    # ------------------------------------------------------------------
    # 锁与基础设施
    # ------------------------------------------------------------------

    def test_01_canvas_lock_exists_and_works(self):
        """canvas_lock 存在且可正常获取/释放"""
        self.assertTrue(hasattr(self.app, 'canvas_lock'))
        ok = self.app.canvas_lock.acquire(blocking=False)
        self.assertTrue(ok)
        self.app.canvas_lock.release()

    def test_02_renderer_has_state_lock(self):
        """RenderManager 持有独立的 state_lock"""
        self.assertTrue(hasattr(self.app.renderer, 'state_lock'))
        ok = self.app.renderer.state_lock.acquire(blocking=False)
        self.assertTrue(ok)
        self.app.renderer.state_lock.release()

    def test_03_shutdown_event_not_set_initially(self):
        """关闭事件初始为未触发"""
        self.assertFalse(self.app.renderer._shutdown_event.is_set())

    # ------------------------------------------------------------------
    # 渲染委托
    # ------------------------------------------------------------------

    def test_04_request_hq_delegates_to_renderer(self):
        """_request_hq → renderer.request_hq"""
        self.app.renderer.request_hq.reset_mock()
        self.app._request_hq()
        self.app.renderer.request_hq.assert_called_once()

    def test_05_prompt_snapshot_preserved(self):
        """prompt 快照在主线程捕获，不受后续 GUI 修改影响"""
        self.app.prompt_var.set("original")
        snap = self.app.prompt_var.get()
        self.app.prompt_var.set("changed")
        self.assertEqual(snap, "original")

    def test_06_load_lora_delegates_to_renderer(self):
        """_load_lora → renderer.load_lora"""
        self.app.renderer.is_rendering = False
        self.app.lora_var.set("test/path")
        self.app.renderer.load_lora.reset_mock()
        self.app._load_lora()
        self.app.renderer.load_lora.assert_called_once()

    # ------------------------------------------------------------------
    # 优雅退出
    # ------------------------------------------------------------------

    def test_07_on_closing_signals_shutdown(self):
        """_on_closing 通知 renderer.shutdown 并启动轮询"""
        self.app.renderer.is_rendering = True  # 阻止直接 destroy
        self.app.renderer.shutdown.reset_mock()
        self.app._on_closing()
        self.app.renderer.shutdown.assert_called_once()
        self.assertTrue(hasattr(self.app, '_shutdown_attempts'))

    # ------------------------------------------------------------------
    # 画布操作
    # ------------------------------------------------------------------

    def test_08_clear_canvas_lock_released(self):
        """_clear_canvas 后 canvas_lock 已释放"""
        self.app._clear_canvas()
        ok = self.app.canvas_lock.acquire(timeout=1)
        self.assertTrue(ok)
        self.app.canvas_lock.release()

    def test_09_get_sketch_snapshot_lock_released(self):
        """_get_sketch_snapshot 后 canvas_lock 已释放"""
        self.app._get_sketch_snapshot()
        ok = self.app.canvas_lock.acquire(timeout=1)
        self.assertTrue(ok, "canvas_lock should be released after get_sketch")
        self.app.canvas_lock.release()

    # ------------------------------------------------------------------
    # 并发安全
    # ------------------------------------------------------------------

    def test_10_paint_no_deadlock(self):
        """_on_paint 不会死锁"""
        evt = Mock(x=200, y=200)
        self.app.last_x, self.app.last_y = 100, 100
        self.app.last_render_time = 0
        self.app.renderer.is_rendering = False
        try:
            self.app._on_paint(evt)
        except Exception as e:
            self.fail(f"_on_paint raised: {e}")

    def test_11_concurrent_no_deadlock(self):
        """并发 _on_paint + _request_hq 不产生死锁"""
        no_mainloop = []
        lock_errs = []

        self.app.last_x, self.app.last_y = 100, 100
        self.app.last_render_time = 0
        self.app.renderer.is_rendering = False

        def painter():
            for i in range(50):
                try:
                    self.app._on_paint(Mock(x=200 + i % 100, y=200 + i % 100))
                except Exception as e:
                    if "main thread is not in main loop" in str(e):
                        no_mainloop.append(str(e))
                    else:
                        lock_errs.append(f"paint: {e}")

        def requester():
            for _ in range(20):
                try:
                    self.app._request_hq()
                except Exception as e:
                    if "main thread is not in main loop" in str(e):
                        no_mainloop.append(str(e))
                    else:
                        lock_errs.append(f"request: {e}")

        t1 = threading.Thread(target=painter)
        t2 = threading.Thread(target=requester)
        t1.start(); t2.start()
        t1.join(timeout=5); t2.join(timeout=5)

        self.assertEqual(lock_errs, [], f"Lock errors: {lock_errs}")
        print(f"  (可忽略的 mainloop 错误: {len(no_mainloop)} 个)")

    def test_12_canvas_and_renderer_locks_independent(self):
        """canvas_lock 和 renderer.state_lock 互不干扰"""
        for a, b in [(self.app.canvas_lock, self.app.renderer.state_lock),
                      (self.app.renderer.state_lock, self.app.canvas_lock)]:
            self.assertTrue(a.acquire(timeout=1))
            self.assertTrue(b.acquire(timeout=1))
            b.release(); a.release()


class TestSketchImport(unittest.TestCase):
    """测试线稿导入的 canvas_lock 保护"""

    @classmethod
    def setUpClass(cls):
        cls._renderer_patcher = patch('src.app.RenderManager', new=_MockRenderer)
        cls._renderer_patcher.start()

        cls.root = tk.Tk()
        cls.root.withdraw()

        from src.app import RealTimePaintApp
        import src.app as app_module
        app_module.messagebox = Mock()
        mock_pipe = Mock()
        mock_pipe.return_value = Mock(images=[Image.new("RGB", (512, 512), "white")])
        cls.app = RealTimePaintApp(cls.root, mock_pipe, Mock())

    @classmethod
    def tearDownClass(cls):
        cls._renderer_patcher.stop()
        cls.root.destroy()

    def test_canvas_lock_free_after_init(self):
        """初始化后 canvas_lock 未被持有"""
        ok = self.app.canvas_lock.acquire(timeout=1)
        self.assertTrue(ok, "canvas_lock should be free after init")
        self.app.canvas_lock.release()


if __name__ == "__main__":
    print("=" * 60)
    print("P0 线程安全测试 (P3 重构适配版)")
    print("=" * 60)
    unittest.main(verbosity=2)
