"""复用应用管线的 GPU 集成基准，输出图像和 JSON 指标。

python -m tools.benchmark --offline --output artifacts/checks/benchmark
"""
import argparse
import json
import platform
import time
from pathlib import Path
from PIL import Image, ImageDraw

import config
from src.pipeline import create_pipelines, render_sketch


def create_mock_sketch():
    image = Image.new("RGB", config.IMAGE_SIZE, "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse([120, 100, 390, 420], outline="black", width=3)
    draw.ellipse([180, 210, 230, 250], outline="black", width=4)
    draw.ellipse([280, 210, 330, 250], outline="black", width=4)
    draw.arc([210, 310, 300, 350], 0, 180, fill="black", width=4)
    return image


def verify_lora(pipe, sketch, output_dir):
    """用本地生成的零增量 LoRA 验证真实加载/权重/推理接口，不下载风格模型。"""
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
    from diffusers.utils import convert_state_dict_to_diffusers
    from src.renderer import RenderManager

    pipe.unet.add_adapter(LoraConfig(r=2, lora_alpha=2, target_modules=["to_q", "to_v"]))
    state = convert_state_dict_to_diffusers(get_peft_model_state_dict(pipe.unet))
    fixture = output_dir / "lora-fixture"
    pipe.save_lora_weights(str(fixture), unet_lora_layers=state)
    pipe.unload_lora_weights()
    manager = RenderManager(pipe)
    try:
        manager.load_lora(str(fixture), 0.5)
        manager.request_preview(sketch, config.DEFAULT_PROMPT)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            for event in manager.poll_events():
                if event.kind in {"error", "blocked", "lora_error"}:
                    raise AssertionError(event.value)
                if event.kind == "frame":
                    if pipe.get_active_adapters() != ["paint_lora"]:
                        raise AssertionError("LoRA 未激活")
                    scales = [module.scaling["paint_lora"] for module in pipe.unet.modules()
                              if hasattr(module, "scaling") and "paint_lora" in module.scaling]
                    if not scales or any(value != 0.5 for value in scales):
                        raise AssertionError("LoRA 权重未应用")
                    event.value.save(output_dir / "lora-smoke.png")
                    return {"passed": True, "fixture": "zero-delta LoRA", "weight": 0.5}
            time.sleep(0.01)
        raise TimeoutError("LoRA 验证超时")
    finally:
        manager.shutdown()
        if not manager.join(30):
            raise TimeoutError("LoRA 验证工作线程未退出")
        pipe.unload_lora_weights()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="仅使用本机模型缓存")
    parser.add_argument("--output", type=Path, default=Path("artifacts/checks/benchmark"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--verify-lora", action="store_true", help="验证真实 LoRA 加载及推理")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats 必须至少为 1")
    import torch
    import diffusers
    if not torch.cuda.is_available():
        parser.error("性能基准需要 CUDA GPU；无 GPU 测试请运行 unittest")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    pipe = create_pipelines(local_files_only=args.offline)
    load_seconds = time.perf_counter() - started
    sketch = create_mock_sketch()
    sketch.save(args.output / "sketch.png")
    # 真实推理预热，排除首帧初始化耗时。
    render_sketch(pipe, sketch, config.DEFAULT_PROMPT, config.PREVIEW_NUM_INFERENCE_STEPS)
    torch.cuda.synchronize()
    records = []
    for name, steps in [("preview", config.PREVIEW_NUM_INFERENCE_STEPS),
                        ("hq", config.HQ_NUM_INFERENCE_STEPS)]:
        for i in range(args.repeats):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            image = render_sketch(pipe, sketch, config.DEFAULT_PROMPT, steps)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if image.size != config.INFERENCE_SIZE:
                raise AssertionError(f"意外输出尺寸: {image.size}")
            output = args.output / f"{name}-{i + 1}.png"
            image.save(output)
            record = {"mode": name, "steps": steps, "seconds": elapsed,
                      "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                      "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                      "image": output.name}
            records.append(record)
            print(json.dumps(record), flush=True)
    report = {"gpu": torch.cuda.get_device_name(), "python": platform.python_version(),
              "torch": torch.__version__, "diffusers": diffusers.__version__,
              "model": config.BASE_MODEL_ID, "controlnet": config.CONTROLNET_MODEL_ID,
              "model_revision": config.BASE_MODEL_REVISION,
              "controlnet_revision": config.CONTROLNET_MODEL_REVISION,
              "inference_size": config.INFERENCE_SIZE,
              "guidance_scale": config.GUIDANCE_SCALE,
              "negative_prompt": config.NEGATIVE_PROMPT,
              "scheduler": type(pipe.scheduler).__name__,
              "cpu_offload": config.ENABLE_MODEL_CPU_OFFLOAD,
              "seed": config.SEED, "prompt": config.DEFAULT_PROMPT,
              "control_guidance_end": config.HQ_CONTROLNET_ENDING_STEP,
              "load_seconds": load_seconds, "runs": records}
    if args.verify_lora:
        report["lora_test"] = verify_lora(pipe, sketch, args.output)
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"基准完成: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
