"""真实 Tk + GPU 画板回归（隐藏窗口）：python -m tools.gui_smoke --offline。"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time
import tkinter as tk
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image
import config
from src.app import RealTimePaintApp
from src.pipeline import create_pipelines, render_sketch
from tools.benchmark import create_mock_sketch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--seconds", type=float, default=12)
    parser.add_argument("--output", type=Path, default=Path("artifacts/checks") / ("gui-" + datetime.now().strftime("%Y%m%d-%H%M%S")))
    args = parser.parse_args()
    if args.seconds < 3:
        parser.error("至少运行 3 秒")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("输出目录必须为空")
    import torch
    if not torch.cuda.is_available():
        parser.error("此回归需要 CUDA；模拟窗口测试请运行 tests/test_reference_workflow.py")
    pipe = create_pipelines(local_files_only=args.offline)
    sketch = create_mock_sketch()
    prompt = "1girl, anime portrait, closed eyes, smiling"
    render_sketch(pipe, sketch, prompt, config.PREVIEW_NUM_INFERENCE_STEPS)
    root = tk.Tk()
    root.withdraw()
    app = RealTimePaintApp(root, pipe)
    frames = []
    last_reference = None
    input_events = 0
    loop_times = []
    try:
        app._replace_sketch(sketch)
        app.prompt_var.set(prompt)
        app._on_mouse_down(SimpleNamespace(x=50, y=450))
        started = time.monotonic()
        next_input = started
        while time.monotonic() - started < args.seconds:
            now = time.monotonic()
            if now >= next_input:
                app._on_paint(SimpleNamespace(x=50 + input_events % 400, y=450))
                input_events += 1
                next_input = now + .04
            root.update()
            updated = time.monotonic()
            loop_times.append(updated - started)
            if app._reference is not None and app._reference is not last_reference:
                last_reference = app._reference
                frames.append({"tk_updated_at_seconds": updated - started,
                               "snapshot_to_tk_update_seconds": updated - last_reference.metadata["submitted_at"]})
            time.sleep(.005)
        # 固定图后更改输入，保存必须对应固定图的快照。
        if last_reference is None:
            raise AssertionError("Tk 未显示参考图")
        app.pinned_var.set(True)
        app.paused_var.set(True)
        app._toggle_pause()
        app._on_paint(SimpleNamespace(x=400, y=50))
        app._on_mouse_up(SimpleNamespace())
        root.update()
        assert app._reference is last_reference
        args.output.mkdir(parents=True, exist_ok=True)
        with patch("src.app.filedialog.asksaveasfilename", return_value=str(args.output / "pinned.png")):
            app._save_result()
        with Image.open(args.output / "pinned.png") as saved:
            assert saved.tobytes() == last_reference.image.tobytes()
            recipe = json.loads(saved.info["aniface.recipe"])
            assert recipe["canvas_revision"] == last_reference.metadata["canvas_revision"]
            assert recipe["canvas_revision"] != app._canvas_revision
        timestamps = [0] + [frame["tk_updated_at_seconds"] for frame in frames] + [args.seconds]
        max_gap = max(b-a for a,b in zip(timestamps, timestamps[1:]))
        report = {"gpu": torch.cuda.get_device_name(), "seconds": args.seconds,
                  "input_events": input_events, "frames_during_drawing": len(frames),
                  "max_display_update_gap_seconds": max_gap,
                  "max_event_loop_gap_seconds": max(b-a for a,b in zip([0]+loop_times,loop_times)),
                  "pinned_save_matches": True, "frames": frames,
                  "scope": "Synthetic canvas events, actual Tk PhotoImage/update and GPU; hidden window, excludes physical pointer/display latency",
                  "passed": len(frames) >= 2 and max_gap <= config.MAX_PREVIEW_AGE_SECONDS + 1}
        (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({k:v for k,v in report.items() if k != "frames"}), flush=True)
        assert report["passed"], "绘画期间 Tk 更新不足"
    finally:
        app._on_closing()
        if not app.renderer.join(30):
            raise RuntimeError("worker 未退出")
        try:
            for timer in root.tk.call("after", "info"):
                root.after_cancel(timer)
            root.destroy()
        except tk.TclError:
            pass


if __name__ == "__main__":
    main()
