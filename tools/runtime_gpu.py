"""显式运行的 GPU 回归：取消后恢复推理、LoRA 回滚及旧适配器释放。

python -m tools.runtime_gpu --offline
使用本地生成的零增量 LoRA，不下载社区风格模型。
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time
import shutil

import config
from src.pipeline import create_pipelines, render_sketch, RenderCancelled
from src.renderer import RenderManager
from tools.benchmark import create_mock_sketch, verify_lora


def wait_frame(manager, timeout=45):
    deadline = time.monotonic() + timeout
    events = []
    while time.monotonic() < deadline:
        for event in manager.poll_events():
            events.append(event)
            if event.kind in {"error", "blocked"}:
                raise AssertionError(event.value)
            if event.kind == "frame":
                return event.value, events
        time.sleep(0.01)
    raise TimeoutError("未收到渲染结果")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/checks") /
                        ("runtime-" + datetime.now().strftime("%Y%m%d-%H%M%S")))
    args = parser.parse_args()
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("输出目录必须为空")
    args.output.mkdir(parents=True, exist_ok=True)
    pipe = create_pipelines(local_files_only=args.offline)
    sketch = create_mock_sketch()
    warmup_started = time.perf_counter()
    render_sketch(pipe, sketch, config.DEFAULT_PROMPT, config.PREVIEW_NUM_INFERENCE_STEPS)
    warmup_seconds = time.perf_counter() - warmup_started
    report = {"fixture_validation": verify_lora(pipe, sketch, args.output)}
    report["first_render_seconds"] = warmup_seconds
    checks = 0
    def should_stop():
        nonlocal checks
        checks += 1
        return checks >= 3  # 入口检查后，在第二个去噪步骤结束处取消。
    started = time.perf_counter()
    try:
        render_sketch(pipe, sketch, config.DEFAULT_PROMPT, config.HQ_NUM_INFERENCE_STEPS, should_stop)
    except RenderCancelled:
        report["cancelled_at_second_step"] = True
        report["cancel_seconds"] = time.perf_counter() - started
    else:
        raise AssertionError("取消请求未生效")
    fixture = str(args.output / "lora-fixture")
    # 一个已存在但格式损坏的文件，保证不触发远端模型查找。
    broken = args.output / "broken.safetensors"
    broken.write_bytes(b"invalid adapter fixture")
    manager = RenderManager(pipe)
    try:
        manager.load_lora(fixture, 0.5)
        manager.request_hq(sketch, config.DEFAULT_PROMPT)
        original, events = wait_frame(manager)
        assert not any(e.kind == "lora_error" for e in events)
        original.save(args.output / "original.png")
        manager.load_lora(str(broken), 0.2)
        manager.request_hq(sketch, config.DEFAULT_PROMPT)
        restored, events = wait_frame(manager)
        assert any(e.kind == "lora_error" and "已恢复原来的" in e.value for e in events)
        assert pipe.get_active_adapters() == ["paint_lora"]
        scales = [m.scaling["paint_lora"] for m in pipe.unet.modules()
                  if hasattr(m, "scaling") and "paint_lora" in m.scaling]
        assert scales and all(value == 0.5 for value in scales)
        restored.save(args.output / "restored.png")
        report["rollback_adapter_and_weight"] = True
        report["rollback_pixels_match"] = original.tobytes() == restored.tobytes()
        assert report["rollback_pixels_match"], "LoRA 回滚后图像与原结果不一致"
        manager.load_lora(fixture, 0.8)
        manager.request_preview(sketch, config.DEFAULT_PROMPT)
        replacement, events = wait_frame(manager)
        assert not any(e.kind == "lora_error" for e in events)
        assert pipe.get_active_adapters() == ["paint_lora"]
        scales = [m.scaling["paint_lora"] for m in pipe.unet.modules()
                  if hasattr(m, "scaling") and "paint_lora" in m.scaling]
        assert scales and all(value == 0.8 for value in scales)
        report["same_file_updates_weight_without_replacement"] = True
        alternate = args.output / "replacement-fixture"
        shutil.copytree(fixture, alternate)
        manager.load_lora(alternate, 0.8)
        manager.request_preview(sketch, config.DEFAULT_PROMPT)
        replacement, events = wait_frame(manager)
        assert not any(e.kind == "lora_error" for e in events)
        assert pipe.get_active_adapters() == ["paint_lora_2"]
        assert pipe.get_list_adapters()["unet"] == ["paint_lora_2"]
        replacement.save(args.output / "replacement.png")
        report["replacement_releases_old_adapter"] = True
        manager.load_lora(None, 0)
        manager.request_preview(sketch, config.DEFAULT_PROMPT)
        base, events = wait_frame(manager)
        assert not any(e.kind == "lora_error" for e in events)
        assert manager.active_lora["path"] is None
        base.save(args.output / "unloaded.png")
        report["unload_restores_rendering"] = True
    finally:
        manager.shutdown()
        if not manager.join(30):
            raise TimeoutError("工作线程未退出")
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
