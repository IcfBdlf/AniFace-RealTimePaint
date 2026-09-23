"""持续调度测试：真实 RenderManager，可选择模拟模型或真实 GPU。

python -m tools.stress_renderer --seconds 180
python -m tools.stress_renderer --gpu --seconds 120 --lora artifacts/checks/runtime-20260912/lora-fixture
"""
import argparse
from contextlib import contextmanager
from datetime import datetime
import json
import logging
from pathlib import Path
import random
import threading
import time
import tracemalloc

from src.pipeline import create_pipelines, render_sketch, RenderCancelled
from src.renderer import RenderManager
from tools.benchmark import create_mock_sketch


class FakePipe:
    def __init__(self):
        self.adapters = {}

    def load_lora_weights(self, path, adapter_name):
        if "broken" in path:
            raise ValueError("deliberate damaged adapter")
        self.adapters[adapter_name] = 1.0

    def set_adapters(self, names, adapter_weights):
        for name, weight in zip(names, adapter_weights):
            if name not in self.adapters:
                raise ValueError("adapter missing")
            self.adapters[name] = weight

    def delete_adapters(self, name):
        self.adapters.pop(name, None)

    def unload_lora_weights(self):
        self.adapters.clear()


class TrackedPipe:
    def __init__(self, pipe):
        self.pipe = pipe
        self.concurrent = 0
        self.maximum = 0
        self.thread_ids = set()
        self.started = self.completed = self.cancelled = 0
        self.lock = threading.Lock()

    @contextmanager
    def operation(self):
        with self.lock:
            self.concurrent += 1
            self.maximum = max(self.maximum, self.concurrent)
            self.thread_ids.add(threading.get_ident())
        try:
            yield
        finally:
            with self.lock:
                self.concurrent -= 1

    def __getattr__(self, name):
        method = getattr(self.pipe, name)
        if name in {"load_lora_weights", "set_adapters", "delete_adapters", "unload_lora_weights"}:
            def tracked(*args, **kwargs):
                with self.operation():
                    return method(*args, **kwargs)
            return tracked
        return method

    def __call__(self, *args, **kwargs):
        return self.pipe(*args, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=180)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--lora", type=Path, help="GPU 模式可选的本地适配器")
    parser.add_argument("--output", type=Path, default=Path("artifacts/checks") /
                        ("stress-" + datetime.now().strftime("%Y%m%d-%H%M%S")))
    args = parser.parse_args()
    if not 0 < args.seconds <= 3600:
        parser.error("seconds 必须在 0 到 3600 之间")
    if args.lora and not args.lora.exists():
        parser.error("LoRA 路径不存在")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("输出目录必须为空")
    args.output.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("AniFace.renderer")
    logger.addHandler(logging.FileHandler(args.output / "expected-errors.log", encoding="utf-8"))
    logger.propagate = False
    pipe = TrackedPipe(create_pipelines(local_files_only=True) if args.gpu else FakePipe())
    broken = args.output / "broken.safetensors"
    broken.write_bytes(b"deliberately damaged test adapter")
    valid_lora = str(args.lora) if args.lora else "fixture"
    use_lora = not args.gpu or args.lora is not None
    def render(pipe, image, prompt, steps, should_stop):
        with pipe.operation():
            pipe.started += 1
            try:
                if args.gpu:
                    result = render_sketch(pipe, image, prompt, steps, should_stop)
                else:
                    for _ in range(8):
                        time.sleep(0.002)
                        if should_stop():
                            raise RenderCancelled()
                    result = image.copy()
                result.info["stress_tag"] = image.info["stress_tag"]
                pipe.completed += 1
                return result
            except RenderCancelled:
                pipe.cancelled += 1
                raise

    manager = RenderManager(pipe, render_fn=render)
    randomizer = random.Random(42)
    counters = {"requests": 0, "clear": 0, "lora_requests": 0, "frames": 0,
                "lora_errors": 0, "blocked": 0}
    last_tag = None
    sketch = create_mock_sketch()
    tracemalloc.start()
    started = time.monotonic()
    samples = []
    next_sample = started + 15
    def consume():
        for event in manager.poll_events():
            if event.kind == "frame":
                if (event.value.info["stress_tag"] != event.request.image.info["stress_tag"] or
                        last_tag is None or event.value.info["stress_tag"] > last_tag):
                    raise AssertionError("stale frame displayed")
                counters["frames"] += 1
            elif event.kind in {"lora_error", "blocked"}:
                counters["lora_errors" if event.kind == "lora_error" else "blocked"] += 1
            elif event.kind == "error":
                raise AssertionError(event.value)

    try:
        while time.monotonic() - started < args.seconds:
            consume()
            action = randomizer.randrange(12)
            if action == 0:
                manager.invalidate()
                last_tag = None
                counters["clear"] += 1
            elif action in (1, 2) and use_lora:
                manager.load_lora(str(broken) if action == 1 else valid_lora, 0.5)
                last_tag = None
                counters["lora_requests"] += 1
            else:
                counters["requests"] += 1
                last_tag = counters["requests"]
                sketch.putpixel((0, 0), ((last_tag >> 16) & 255, (last_tag >> 8) & 255, last_tag & 255))
                sketch.info["stress_tag"] = last_tag
                if action % 2:
                    manager.request_hq(sketch, "1girl, anime portrait, closed eyes, smiling")
                else:
                    manager.request_preview(sketch, "1girl, anime portrait, closed eyes, smiling")
            # 周期性停笔给有效结果显示机会，其余时间频繁更新以触发取消。
            pause = (3.0 if args.gpu else 0.05) if action == 11 else (0.10 if args.gpu else 0.003)
            until = min(time.monotonic() + pause, started + args.seconds)
            while time.monotonic() < until:
                consume()
                time.sleep(0.01 if args.gpu else 0.002)
            if time.monotonic() >= next_sample:
                current, peak = tracemalloc.get_traced_memory()
                sample = {"seconds": time.monotonic() - started, "python_bytes": current,
                          "python_peak_bytes": peak, "threads": threading.active_count(),
                          "queued_events": manager._events.qsize()}
                if args.gpu:
                    import torch
                    sample["cuda_allocated_mib"] = torch.cuda.memory_allocated() / 2**20
                samples.append(sample)
                print(json.dumps({**sample, **counters}), flush=True)
                next_sample = time.monotonic() + 15
    finally:
        shutdown_started = time.monotonic()
        manager.shutdown()
        if not manager.join(30):
            raise AssertionError("worker did not terminate")
        shutdown_seconds = time.monotonic() - shutdown_started
        tracemalloc.stop()
    assert pipe.maximum == 1 and len(pipe.thread_ids) == 1, "model operations overlapped"
    assert counters["frames"] > 0 and pipe.cancelled > 0, "workload did not exercise both completion and cancellation"
    assert not any(t.name == "AniFace-render" for t in threading.enumerate()), "worker leaked"
    report = {"gpu": args.gpu, "seconds": time.monotonic() - started, **counters,
              "started": pipe.started, "completed": pipe.completed, "cancelled": pipe.cancelled,
              "max_model_concurrency": pipe.maximum, "model_threads": len(pipe.thread_ids),
              "shutdown_seconds": shutdown_seconds, "samples": samples, "passed": True}
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "samples"}), flush=True)


if __name__ == "__main__":
    main()
