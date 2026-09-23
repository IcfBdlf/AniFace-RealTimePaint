"""输入与推理契约测试，无 GPU、无窗口。"""
import contextlib
import sys
import unittest
from unittest.mock import Mock, patch
from PIL import Image

import config
from src.pipeline import render_sketch, prepare_control_image, RenderBlocked, RenderCancelled, control_model_id
from src.app import RealTimePaintApp, fit_sketch
from src.prompts import compose_prompt
from src.renderer import RenderEvent, RenderRequest


class TestPipelineContract(unittest.TestCase):
    def test_anime_model_polarity_and_clip_skip_are_explicit(self):
        with self.assertRaisesRegex(ValueError, "SD1.5"):
            control_model_id("lineart_anime")
        sketch = Image.new("RGB", (16, 16), "white")
        sketch.putpixel((1, 1), (180, 180, 180))
        self.assertEqual(prepare_control_image(sketch, "lineart_anime").tobytes(), sketch.tobytes())
        torch = Mock()
        torch.no_grad.side_effect = contextlib.nullcontext
        pipe = Mock(return_value=Mock(images=[sketch], nsfw_content_detected=None))
        pipe._aniface_control_type = "lineart_anime"
        with patch.dict(sys.modules, {"torch": torch}):
            render_sketch(pipe, sketch, "portrait", 10, clip_skip=1)
            self.assertEqual(pipe.call_args.kwargs["clip_skip"], 1)
            render_sketch(pipe, sketch, "portrait", 10)
            self.assertNotIn("clip_skip", pipe.call_args.kwargs)

    def test_lineart_preserves_white_background_and_faint_lines(self):
        sketch = Image.new("RGB", (16, 16), "white")
        sketch.putpixel((1, 1), (180, 180, 180))
        control = prepare_control_image(sketch, "lineart")
        self.assertEqual(control.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(control.getpixel((1, 1)), (180, 180, 180))
        self.assertEqual(sketch.getpixel((1, 1)), (180, 180, 180))
        with self.assertRaises(ValueError):
            prepare_control_image(sketch, "unknown")

    def test_render_uses_pipeline_control_type(self):
        torch = Mock()
        torch.no_grad.side_effect = contextlib.nullcontext
        pipe = Mock(return_value=Mock(images=[Image.new("RGB", (16, 16))], nsfw_content_detected=None))
        sketch = Image.new("RGB", (16, 16), "white")
        with patch.dict(sys.modules, {"torch": torch}):
            for kind, background in (("lineart", (255, 255, 255)), ("scribble", (0, 0, 0))):
                pipe._aniface_control_type = kind
                render_sketch(pipe, sketch, "portrait", 10)
                self.assertEqual(pipe.call_args.kwargs["image"].getpixel((0, 0)), background)

    def test_control_override_does_not_change_default(self):
        torch = Mock()
        torch.no_grad.side_effect = contextlib.nullcontext
        pipe = Mock(return_value=Mock(images=[Image.new("RGB", (16, 16))], nsfw_content_detected=None))
        original = config.HQ_CONTROLNET_ENDING_STEP
        with patch.dict(sys.modules, {"torch": torch}):
            render_sketch(pipe, Image.new("RGB", (16, 16)), "portrait", 10, control_end=1.0)
            self.assertEqual(pipe.call_args.kwargs["control_guidance_end"], 1.0)
            render_sketch(pipe, Image.new("RGB", (16, 16)), "portrait", 10)
            self.assertEqual(pipe.call_args.kwargs["control_guidance_end"], original)
        self.assertEqual(config.HQ_CONTROLNET_ENDING_STEP, original)

    def test_scribble_polarity(self):
        sketch = Image.new("RGB", (16, 16), "white")
        sketch.putpixel((0, 0), (0, 0, 0))
        control = prepare_control_image(sketch)
        self.assertEqual(control.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(control.getpixel((1, 1)), (0, 0, 0))
        self.assertEqual(sketch.getpixel((0, 0)), (0, 0, 0))

    def test_render_uses_correct_controlnet_keyword_seed_and_cancellation(self):
        torch = Mock()
        torch.no_grad.side_effect = contextlib.nullcontext
        pipe = Mock(return_value=Mock(images=[Image.new("RGB", (16, 16))], nsfw_content_detected=None))
        stop = Mock(return_value=False)
        with patch.dict(sys.modules, {"torch": torch}):
            render_sketch(pipe, Image.new("RGB", (16, 16)), "portrait", 10, stop)
        kwargs = pipe.call_args.kwargs
        self.assertEqual(kwargs["control_guidance_end"], config.HQ_CONTROLNET_ENDING_STEP)
        self.assertNotIn("controlnet_ending_step", kwargs)
        self.assertEqual(kwargs["prompt"], "portrait")
        self.assertEqual(kwargs["image"].size, config.INFERENCE_SIZE)
        self.assertEqual((kwargs["width"], kwargs["height"]), config.INFERENCE_SIZE)
        self.assertEqual(kwargs["guidance_scale"], config.GUIDANCE_SCALE)
        self.assertEqual(kwargs["negative_prompt"], config.NEGATIVE_PROMPT)
        torch.Generator.return_value.manual_seed.assert_called_once_with(config.SEED)
        stop.return_value = True
        with self.assertRaises(RenderCancelled):
            kwargs["callback_on_step_end"](pipe, 1, 1, {})

    def test_checker_flag_blocks_result_even_when_pixels_are_not_black(self):
        torch = Mock()
        torch.no_grad.side_effect = contextlib.nullcontext
        pipe = Mock(return_value=Mock(images=[Image.new("RGB", (16, 16), "blue")],
                                     nsfw_content_detected=[True]))
        with patch.dict(sys.modules, {"torch": torch}), self.assertRaises(RenderBlocked):
            render_sketch(pipe, Image.new("RGB", (16, 16)), "portrait", 10)

    def test_unflagged_black_image_is_not_assumed_to_be_blocked(self):
        torch = Mock()
        torch.no_grad.side_effect = contextlib.nullcontext
        image = Image.new("RGB", (16, 16), "black")
        for flags in (None, [False]):
            with self.subTest(flags=flags), patch.dict(sys.modules, {"torch": torch}):
                pipe = Mock(return_value=Mock(images=[image], nsfw_content_detected=flags))
                self.assertIs(render_sketch(pipe, image, "portrait", 10), image)

    def test_transparent_import_and_aspect_ratio(self):
        image = Image.new("RGBA", (200, 100), (0, 0, 0, 255))
        image.putpixel((100, 50), (0, 0, 0, 0))
        fitted = fit_sketch(image)
        self.assertEqual(fitted.size, config.IMAGE_SIZE)
        self.assertEqual(fitted.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(fitted.getpixel((10, 256)), (0, 0, 0))
        transparent = fit_sketch(Image.new("RGBA", (100, 100)))
        self.assertEqual(transparent.getextrema(), ((255, 255),) * 3)


class TestAppInteractions(unittest.TestCase):
    def setUp(self):
        # 不创建窗口，只替换显示和定时器；事件处理使用真实实现。
        self.app = RealTimePaintApp.__new__(RealTimePaintApp)
        self.app.root = Mock()
        self.app.renderer = Mock()
        self.app.prompt_var = Mock()
        self.app.prompt_var.get.return_value = "current"
        self.app.expression_var = Mock()
        self.app.expression_var.get.return_value = "不附加表情"
        self.app.follow_sketch_var = Mock()
        self.app.follow_sketch_var.get.return_value = False
        self.app.status_var = Mock()
        self.app._profile = None
        self.app._pending_profile = None
        self.app._loading_lora = False
        self.app._reference = None
        self.app._references = []
        self.app._reference_index = -1
        self.app._canvas_revision = 0
        self.app.reference_var = Mock()
        self.app.active_lora_var = Mock()
        self.app.renderer.active_lora = {"path": None, "weight": None}
        self.app.canvas = Mock()
        self.app.canvas.canvasx.side_effect = lambda x: x
        self.app.canvas.canvasy.side_effect = lambda y: y
        for name, value in [("zoom_var", "100%"), ("brush_var", 4), ("eraser_var", False),
                            ("paused_var", False), ("pinned_var", False), ("auto_hq_var", True)]:
            variable = Mock()
            variable.get.return_value = value
            setattr(self.app, name, variable)
        self.app._result = None
        self.app._closing = False
        self.app._hq_timer = None
        self.app._notice = ""
        self.app._lora_notice = ""
        self.app._undo_stack = []
        self.app._redo_stack = []
        self.app._redraw_canvas = Mock()
        self.app._display_image = Mock()
        self.app.last_render_time = 0
        self.app._replace_sketch(Image.new("RGB", config.IMAGE_SIZE, "white"))

    def frame(self, image):
        import time
        return RenderEvent("frame", image, 1,
                           RenderRequest(1, "preview", image, "test", submitted_at=time.monotonic()),
                           {"mode": "preview", "canvas_revision": self.app._canvas_revision})

    def test_origin_stroke_and_undo_redo(self):
        app = self.app
        app._on_mouse_down(Mock(x=0, y=0))
        app._on_paint(Mock(x=30, y=0))
        app._on_mouse_up(Mock())
        self.assertEqual(app.sketch_img.getpixel((15, 0)), (0, 0, 0))
        app._undo()
        self.assertEqual(app.sketch_img.getpixel((15, 0)), (255, 255, 255))
        app._redo()
        self.assertEqual(app.sketch_img.getpixel((15, 0)), (0, 0, 0))

    def test_new_stroke_cancels_idle_hq_and_release_defers(self):
        app = self.app
        app._hq_timer = "old-timer"
        app._on_mouse_down(Mock(x=1, y=1))
        app.root.after_cancel.assert_called_with("old-timer")
        app.renderer.request_hq.assert_not_called()
        app._on_mouse_up(Mock())
        app.root.after.assert_called_with(config.HQ_IDLE_DELAY_MS, app._request_hq)
        self.assertEqual(app.renderer.request_preview.call_args.args, (app.sketch_img, "current"))

    def test_clear_invalidates_and_cancels_timer(self):
        app = self.app
        app._hq_timer = "hq"
        app._clear_canvas()
        app.root.after_cancel.assert_called_with("hq")
        app.renderer.invalidate.assert_called()
        self.assertIsNone(app._result)
        self.assertIsNone(app._hq_timer)

    def test_expression_applies_to_both_modes_without_rewriting_prompt(self):
        app = self.app
        app.expression_var.get.return_value = "闭眼微笑"
        app._on_mouse_down(Mock(x=5, y=5))
        app._on_mouse_up(Mock())
        app._request_hq()
        expected = "current, closed eyes, smiling"
        self.assertEqual(app.renderer.request_preview.call_args.args[1], expected)
        self.assertEqual(app.renderer.request_hq.call_args.args[1], expected)
        app.prompt_var.set.assert_not_called()
        app.expression_var.get.return_value = "不附加表情"
        self.assertEqual(app._current_prompt(), "current")

    def test_expression_change_invalidates_old_results_and_defers_render(self):
        app = self.app
        app.expression_var.get.return_value = "惊讶"
        app._on_prompt_changed()
        app.renderer.invalidate.assert_called()
        app.root.after.assert_called_with(config.HQ_IDLE_DELAY_MS, app._request_hq)

    def test_follow_mode_applies_to_both_modes_and_can_be_disabled(self):
        app = self.app
        app.follow_sketch_var.get.return_value = True
        app._on_prompt_changed()
        app.renderer.invalidate.assert_called()
        app._on_mouse_down(Mock(x=5, y=5))
        app._on_mouse_up(Mock())
        app._request_hq()
        self.assertEqual(app.renderer.request_preview.call_args.kwargs["control_end"], 1.0)
        self.assertEqual(app.renderer.request_hq.call_args.kwargs["control_end"], 1.0)
        app.follow_sketch_var.get.return_value = False
        app._request_hq()
        self.assertIsNone(app.renderer.request_hq.call_args.kwargs["control_end"])
        app.prompt_var.set.assert_not_called()

    def test_blocked_keeps_previous_result_and_success_clears_notice(self):
        app = self.app
        previous = Image.new("RGB", (16, 16), "blue")
        app._result = previous
        app.renderer.status = "就绪"
        app.renderer.poll_events.return_value = [RenderEvent("blocked", "被模型检查拦截")]
        app._poll_renderer()
        self.assertIs(app._result, previous)
        app._display_image.assert_not_called()
        self.assertIn("保留上一张", app._notice)
        new = Image.new("RGB", (16, 16), "green")
        app.renderer.poll_events.return_value = [self.frame(new)]
        app._poll_renderer()
        self.assertEqual(app._result.tobytes(), new.tobytes())
        self.assertEqual(app._notice, "")

    def test_first_blocked_result_does_not_become_saveable(self):
        app = self.app
        app.renderer.status = "就绪"
        app.renderer.poll_events.return_value = [RenderEvent("blocked", "被模型检查拦截")]
        app._poll_renderer()
        self.assertIsNone(app._result)
        self.assertIn("尚无可显示", app._notice)
        app._display_image.assert_not_called()

    def test_lora_failure_notice_survives_a_successful_frame(self):
        app = self.app
        app.renderer.status = "就绪"
        app.renderer.poll_events.return_value = [
            RenderEvent("lora_error", "已恢复原来的 LoRA"),
            self.frame(Image.new("RGB", (16, 16))),
        ]
        app._poll_renderer()
        self.assertIn("已恢复原来的", app.status_var.set.call_args.args[0])


class TestExpressionPrompts(unittest.TestCase):
    def test_switching_does_not_accumulate_presets(self):
        base = "portrait"
        self.assertEqual(compose_prompt(base, "闭眼微笑"), "portrait, closed eyes, smiling")
        self.assertEqual(compose_prompt(base, "惊讶"), "portrait, surprised, open mouth")
        self.assertEqual(compose_prompt(base, "不附加表情"), base)

    def test_manual_tags_not_duplicated_and_empty_prompt_supported(self):
        self.assertEqual(compose_prompt("portrait, SMILING", "闭眼微笑"),
                         "portrait, SMILING, closed eyes")
        self.assertEqual(compose_prompt("", "微笑"), "smiling")


if __name__ == "__main__":
    unittest.main(verbosity=2)
