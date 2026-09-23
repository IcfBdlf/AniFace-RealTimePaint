"""实际 Tk 控件、后台队列、保存与预设的集成回归；使用模拟推理。"""
import base64
from io import BytesIO
import json
from pathlib import Path
import tempfile
import time
import tkinter as tk
import unittest
from unittest.mock import Mock, patch

from PIL import Image
import config
from src.app import RealTimePaintApp
from src.pipeline import RenderCancelled
from src.profiles import load_profile
from src.renderer import RenderManager


class ReferenceWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.pipe = Mock()
        def render(pipe, image, prompt, steps, stop, **options):
            if stop():
                raise RenderCancelled()
            return image.copy()
        with patch("src.app.RenderManager", side_effect=lambda pipe: RenderManager(pipe, render_fn=render)):
            self.app = RealTimePaintApp(self.root, self.pipe)
        self.temp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.app._on_closing()
        self.app.renderer.join(2)
        try:
            for timer in self.root.tk.call("after", "info"):
                self.root.after_cancel(timer)
            self.root.destroy()
        except tk.TclError:
            pass
        self.temp.cleanup()

    def wait(self, predicate):
        deadline = time.monotonic() + 3
        while not predicate() and time.monotonic() < deadline:
            self.root.update()
            time.sleep(.005)
        self.assertTrue(predicate())

    def draw(self, x):
        self.app._on_mouse_down(Mock(x=x, y=50))
        self.app._on_paint(Mock(x=x+20, y=50))
        self.app._on_mouse_up(Mock())

    def profile(self):
        root = Path(self.temp.name)
        (root / "adapter.safetensors").write_bytes(b"mock only")
        path = root / "profile.json"
        path.write_text(json.dumps({"name": "测试角色", "kind": "character", "base_model": config.BASE_MODEL_ID,
                                    "lora_path": "adapter.safetensors", "weight": .7,
                                    "trigger_words": "fixed_character", "default_prompt": "full body"}), encoding="utf-8")
        return path

    def test_pin_pause_history_and_saved_recipe_match_displayed_sketch(self):
        app = self.app
        self.assertFalse(app.auto_hq_var.get())
        self.draw(10)
        self.wait(lambda: app._reference is not None)
        first = app._reference
        app.pinned_var.set(True)
        self.draw(100)
        self.wait(lambda: app._references[-1].metadata["canvas_revision"] == app._canvas_revision)
        self.assertIs(app._reference, first)
        output = Path(self.temp.name) / "saved.png"
        with patch("src.app.filedialog.asksaveasfilename", return_value=str(output)):
            app._save_result()
        with Image.open(output) as result:
            recipe = json.loads(result.info["aniface.recipe"])
            sketch = Image.open(BytesIO(base64.b64decode(result.info["aniface.sketch.png.base64"])))
            self.assertEqual(sketch.tobytes(), first.sketch.tobytes())
            self.assertEqual(recipe["canvas_revision"], first.metadata["canvas_revision"])
        app.pinned_var.set(False)
        app._toggle_pin()
        self.assertIsNot(app._reference, first)
        app._browse_reference(-1)
        self.assertTrue(app.pinned_var.get())
        app.paused_var.set(True)
        app._toggle_pause()
        count = len(app._references)
        self.draw(200)
        self.root.update()
        self.assertEqual(len(app._references), count)
        app.paused_var.set(False)
        app._toggle_pause()
        self.wait(lambda: len(app._references) > count)

    def test_zoom_eraser_and_layout(self):
        app = self.app
        app.paused_var.set(True)
        app.zoom_var.set("200%")
        app._redraw_canvas()
        self.draw(100)
        self.assertEqual(app.sketch_img.getpixel((55, 25)), (0, 0, 0))
        app.eraser_var.set(True)
        self.draw(100)
        self.assertEqual(app.sketch_img.getpixel((55, 25)), (255, 255, 255))
        self.root.update_idletasks()
        self.assertLessEqual(self.root.winfo_reqwidth(), self.root.winfo_screenwidth())
        self.assertLessEqual(self.root.winfo_reqheight(), self.root.winfo_screenheight())

    def test_profile_applies_only_after_success_and_failed_switch_keeps_previous(self):
        app = self.app
        path = self.profile()
        with patch("src.app.filedialog.askopenfilename", return_value=str(path)):
            app._open_profile()
        self.wait(lambda: not app._loading_lora)
        self.assertEqual(app._current_prompt(), "fixed_character, full body")
        self.assertEqual(app._profile.name, "测试角色")
        self.wait(lambda: app._reference is not None)
        self.assertEqual(app._reference.metadata["profile"]["name"], "测试角色")
        app.prompt_var.set("side view")
        app.lora_weight_var.set(.9)
        app._load_lora()
        self.wait(lambda: not app._loading_lora)
        self.assertEqual(app._current_prompt(), "fixed_character, side view")
        self.assertEqual(app._profile.weight, .9)
        self.pipe.load_lora_weights.assert_called_once()
        self.pipe.load_lora_weights.side_effect = ValueError("damaged")
        app.lora_var.set("broken")
        with self.assertLogs("AniFace.renderer", level="ERROR"):
            app._load_lora()
            self.wait(lambda: not app._loading_lora)
        self.assertEqual(app._profile.name, "测试角色")
        self.assertEqual(app._current_prompt(), "fixed_character, side view")
        app._unload_lora()
        self.wait(lambda: not app._loading_lora)
        self.assertIsNone(app._profile)
        self.assertNotIn("fixed_character", app._current_prompt())

    def test_profile_rejects_incompatible_model_before_loading(self):
        path = self.profile()
        data = json.loads(path.read_text(encoding="utf-8"))
        data["base_model"] = "incompatible"
        path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "基础模型"):
            load_profile(path)
        self.pipe.load_lora_weights.assert_not_called()
