"""不停笔预览验收：python -m tools.continuous_preview [--gpu --offline]。"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import config
from src.pipeline import create_pipelines, render_sketch, RenderCancelled
from src.renderer import RenderManager
from tools.benchmark import create_mock_sketch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--seconds", type=float, default=12)
    parser.add_argument("--output", type=Path, default=Path("artifacts/checks") / ("continuous-" + datetime.now().strftime("%Y%m%d-%H%M%S")))
    args = parser.parse_args()
    if args.seconds < 3:
        parser.error("至少运行 3 秒")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("输出目录必须为空")
    sketch = create_mock_sketch()
    prompt = "1girl, anime portrait, closed eyes, smiling"
    if args.gpu:
        import torch
        if not torch.cuda.is_available():
            parser.error("--gpu 需要 CUDA")
        pipe = create_pipelines(local_files_only=args.offline)
        render_sketch(pipe, sketch, prompt, config.PREVIEW_NUM_INFERENCE_STEPS)
        render = render_sketch
    else:
        # worker 的元数据读取使用 __dict__。
        pipe = type("MockPipe", (), {})()
        def render(pipe, image, prompt, steps, stop):
            for _ in range(25):
                time.sleep(.01)
                if stop():
                    raise RenderCancelled()
            return image.copy()
    manager = RenderManager(pipe, render_fn=render)
    start = time.monotonic()
    next_submit = start
    frames, blocked = [], 0
    requests = 0
    last_frame = None
    try:
        while time.monotonic() - start < args.seconds:
            now = time.monotonic()
            if now >= next_submit:
                manager.invalidate(soft=True)
                sketch.putpixel((requests % 400 + 50, 450), (0, 0, 0))
                manager.request_preview(sketch, prompt)
                requests += 1
                next_submit = now + .04
            for event in manager.poll_events():
                if event.kind == "frame":
                    displayed = time.monotonic()
                    frames.append({"displayed_at_seconds": displayed - start,
                                   "snapshot_to_display_seconds": displayed - event.request.submitted_at})
                    last_frame = event.value
                elif event.kind == "blocked":
                    blocked += 1
                elif event.kind == "error":
                    raise RuntimeError(event.value)
            time.sleep(.005)
    finally:
        manager.shutdown()
        if not manager.join(30):
            raise RuntimeError("worker 未退出")
    ages = sorted(frame["snapshot_to_display_seconds"] for frame in frames)
    timestamps = [0] + [frame["displayed_at_seconds"] for frame in frames] + [args.seconds]
    gaps = [b-a for a,b in zip(timestamps, timestamps[1:])]
    passed = len(frames) >= 2 and max(gaps) <= config.MAX_PREVIEW_AGE_SECONDS + 1
    report = {"gpu": args.gpu, "seconds": args.seconds, "requests": requests,
              "frames_while_drawing": len(frames), "blocked": blocked,
              "p95_snapshot_to_display_seconds": ages[min(len(ages)-1, int(len(ages)*.95))] if ages else None,
              "max_display_gap_seconds": max(gaps), "frames": frames, "passed": passed,
              "scope": "RenderManager + polling; excludes Tk upload and user input dispatch"}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if last_frame is not None:
        last_frame.save(args.output / "last-frame.png")
    print(json.dumps({k:v for k,v in report.items() if k != "frames"}), flush=True)
    if not passed:
        raise AssertionError("持续输入期间没有足够稳定的可见反馈；详见指标")


if __name__ == "__main__":
    main()
