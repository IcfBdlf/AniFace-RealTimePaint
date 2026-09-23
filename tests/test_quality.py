"""评估矩阵的组合、外部输入和结果记录，无需 GPU。"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch
import config
from tools import evaluate_quality as quality


class QualityEvaluationTests(unittest.TestCase):
    def test_matrix_external_input_and_blocked_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "drawing.png"
            Image.new("RGBA", (100, 50), (0, 0, 0, 0)).save(source)
            output = root / "results"
            calls = []

            def render(pipe, image, prompt, steps):
                self.assertEqual(image.size, config.IMAGE_SIZE)
                self.assertEqual(image.getpixel((0, 0)), (255, 255, 255))
                calls.append((config.SEED, prompt, steps))
                if config.SEED == 123 and "smiling" in prompt:
                    raise quality.RenderBlocked()
                return image.copy()

            argv = ["evaluate_quality.py", "--offline", "--input", str(source),
                    "--seed", "42", "--seed", "123", "--expression", "不附加表情",
                    "--expression", "微笑", "--output", str(output)]
            with patch("sys.argv", argv), patch.object(config, "SEED", config.SEED), \
                    patch.object(config, "HQ_CONTROLNET_ENDING_STEP", config.HQ_CONTROLNET_ENDING_STEP), \
                    patch.object(quality, "create_pipelines", return_value=object()) as factory, \
                    patch.object(quality, "render_sketch", side_effect=render), \
                    patch.object(torch.cuda, "is_available", return_value=True), \
                    patch.object(torch.cuda, "get_device_name", return_value="mock"), \
                    patch.object(torch.cuda, "synchronize"), \
                    patch.object(torch.cuda, "reset_peak_memory_stats"), \
                    patch.object(torch.cuda, "max_memory_allocated", return_value=0), \
                    patch("builtins.print"):
                quality.main()
            factory.assert_called_once_with(local_files_only=True)
            report = json.loads((output / "metrics.json").read_text())
            self.assertEqual(len(calls), 9)  # 一次预热、八次正式生成
            self.assertEqual(len(report["runs"]), 8)
            self.assertEqual(sum(run["blocked"] for run in report["runs"]), 2)
            self.assertEqual({run["case"] for run in report["runs"]}, {"external-01"})
            self.assertEqual(len({run["image"] for run in report["runs"]}), 8)
            self.assertTrue(all((output / run["image"]).exists() for run in report["runs"]))
            self.assertEqual(len(list(output.glob("*/comparison.png"))), 4)
            for run in report["runs"]:
                self.assertEqual("smiling" in run["prompt"], run["expression"] == "微笑")
                if run["blocked"]:
                    self.assertEqual(run["image_role"], "diagnostic_placeholder")


if __name__ == "__main__":
    unittest.main()
