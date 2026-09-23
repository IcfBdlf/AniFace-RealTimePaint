"""可复现的线稿评估；比较多种种子、表情预设的预览与精细重绘。

python -m tools.evaluate_quality --offline
输出图像供人工评价，不把像素差异当作画质评分。
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import platform
import time

from PIL import Image, ImageDraw, ImageFont
import config
from src.pipeline import RenderBlocked, create_pipelines, render_sketch, control_model_id
from src.prompts import EXPRESSION_PRESETS, compose_prompt
from src.app import fit_sketch
from tools.benchmark import create_mock_sketch
from src.profiles import load_profile


def evaluation_sketches():
    cases = [("01-front", create_mock_sketch())]
    narrow = Image.new("RGB", config.IMAGE_SIZE, "white")
    draw = ImageDraw.Draw(narrow)
    draw.ellipse((240, 80, 430, 420), outline="black", width=4)
    draw.ellipse((270, 205, 305, 238), outline="black", width=4)
    draw.ellipse((350, 205, 385, 238), outline="black", width=4)
    draw.arc((300, 315, 365, 350), 0, 180, fill="black", width=4)
    cases.append(("02-narrow-right", narrow))
    closed = Image.new("RGB", config.IMAGE_SIZE, "white")
    draw = ImageDraw.Draw(closed)
    draw.ellipse((120, 100, 390, 420), outline="black", width=4)
    draw.arc((175, 200, 235, 245), 0, 180, fill="black", width=5)
    draw.arc((280, 200, 340, 245), 0, 180, fill="black", width=5)
    draw.arc((210, 300, 300, 355), 0, 180, fill="black", width=5)
    cases.append(("03-closed-eyes", closed))
    rough = create_mock_sketch()
    draw = ImageDraw.Draw(rough)
    draw.arc((115, 96, 395, 422), 30, 240, fill="black", width=3)
    draw.line([(150, 140), (175, 95), (250, 74), (328, 105), (380, 158)],
              fill="black", width=5)
    draw.line([(105, 430), (120, 375), (139, 347), (112, 350)], fill="black", width=3)
    cases.append(("04-rough-lines", rough))
    return cases


def make_contact_sheet(rows, path):
    size, heading = 384, 34
    sheet = Image.new("RGB", (size * 3, heading + (size + heading) * len(rows)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=18)
    for column, title in enumerate(("Input sketch", f"Preview / {config.PREVIEW_NUM_INFERENCE_STEPS} steps",
                                   f"Refined / {config.HQ_NUM_INFERENCE_STEPS} steps")):
        draw.text((column * size + 10, 7), title, fill="black", font=font)
    for row, (name, images) in enumerate(rows):
        y = heading + row * (size + heading)
        draw.text((10, y + 5), name, fill="black", font=font)
        for column, image in enumerate(images):
            sheet.paste(image.resize((size, size), Image.Resampling.LANCZOS),
                        (column * size, y + heading))
    sheet.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--control-type", choices=["scribble"], default="scribble",
                        help="实验控制模型；模型与输入极性一同切换，不修改 GUI 默认")
    parser.add_argument("--control-cache-dir", type=Path, help="仅控制模型的缓存目录")
    parser.add_argument("--clip-skip", type=int, choices=[1], help="实验：使用倒数第二层正向文本表示")
    parser.add_argument("--follow-sketch", action="store_true", help="启用应用的尽量跟随线稿模式")
    parser.add_argument("--control-end", type=float, default=config.HQ_CONTROLNET_ENDING_STEP,
                        help="仅覆盖本次评估进程的控制结束比例，大于 0 且不超过 1")
    parser.add_argument("--case", action="append", choices=[name for name, _ in evaluation_sketches()],
                        help="仅评估指定线稿；可重复传入，默认全部")
    parser.add_argument("--profile", type=Path, help="使用与 GUI 相同的本地角色/画风预设")
    parser.add_argument("--prompt", default=None,
                        help="仅用于本次评估的提示词")
    parser.add_argument("--seed", type=int, action="append", help="随机种子，可重复；默认配置种子")
    parser.add_argument("--expression", choices=list(EXPRESSION_PRESETS), action="append",
                        help="使用应用表情预设，可重复；默认不附加表情")
    parser.add_argument("--input", type=Path, action="append",
                        help="外部线稿，可重复；单独使用时替代合成样本，配合 --case 可同时评估")
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/evaluations") / ("quality-" + datetime.now().strftime("%Y%m%d-%H%M%S")))
    args = parser.parse_args()
    try:
        profile = load_profile(args.profile) if args.profile else None
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    args.prompt = args.prompt if args.prompt is not None else (profile.default_prompt if profile else config.DEFAULT_PROMPT)
    if profile:
        args.prompt = f"{profile.trigger_words}, {args.prompt}"
    render_options = {"control_end": 1.0} if args.follow_sketch else {}
    if profile and not args.follow_sketch:
        render_options["control_end"] = profile.control_end
    if args.clip_skip is not None:
        render_options["clip_skip"] = args.clip_skip
    if not 0 < args.control_end <= 1:
        parser.error("--control-end 必须大于 0 且不超过 1")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("输出目录非空，请选择新目录以保留已有评估")
    seeds = list(dict.fromkeys(args.seed or [config.SEED]))
    expressions = list(dict.fromkeys(args.expression or ["不附加表情"]))
    if any(seed < 0 or seed >= 2**63 for seed in seeds):
        parser.error("种子须在 0 到 2**63-1 之间")
    external_cases = []
    for index, path in enumerate(args.input or []):
        try:
            with Image.open(path) as source:
                external_cases.append((f"external-{index+1:02d}", fit_sketch(source)))
        except (OSError, ValueError) as exc:
            parser.error(f"无法读取线稿 {path}: {exc}")
    import torch
    import diffusers
    if not torch.cuda.is_available():
        parser.error("本评估使用 CUDA GPU")
    args.output.mkdir(parents=True, exist_ok=True)
    # 独立 CLI 进程中的实验值，不写回 config.py 或改变正在运行的应用。
    config.HQ_CONTROLNET_ENDING_STEP = args.control_end
    cases = evaluation_sketches()
    if args.case:
        cases = [(name, image) for name, image in cases if name in args.case]
    elif args.input:
        cases = []
    cases.extend(external_cases)
    config.SEED = seeds[0]
    model_options = {"control_type": args.control_type} if args.control_type != "scribble" else {}
    if args.control_cache_dir is not None:
        model_options["control_cache_dir"] = str(args.control_cache_dir)
    pipe = create_pipelines(local_files_only=args.offline, **model_options)
    if profile:
        pipe.load_lora_weights(profile.lora_path, adapter_name="evaluation_profile")
        pipe.set_adapters(["evaluation_profile"], adapter_weights=[profile.weight])
    try:
        render_sketch(pipe, cases[0][1], args.prompt, config.PREVIEW_NUM_INFERENCE_STEPS, **render_options)
    except RenderBlocked:
        pass  # 预热仍完成；正式测量单独记录拦截状态。
    records = []
    report = {
        "gpu": torch.cuda.get_device_name(), "python": platform.python_version(),
        "torch": torch.__version__, "diffusers": diffusers.__version__,
        "model": config.BASE_MODEL_ID,
        "model_revision": config.BASE_MODEL_REVISION,
        "controlnet_revision": config.CONTROLNET_MODEL_REVISION,
        "inference_size": config.INFERENCE_SIZE,
        "guidance_scale": config.GUIDANCE_SCALE,
        "negative_prompt": config.NEGATIVE_PROMPT,
        "scheduler": type(pipe.scheduler).__name__,
        "profile": profile.metadata() if profile else None,
        "controlnet": control_model_id(args.control_type),
        "clip_skip": args.clip_skip,
        "control_cache_dir": str(args.control_cache_dir.resolve()) if args.control_cache_dir else None,
        "control_type": args.control_type,
        "control_polarity": "black_background" if args.control_type == "scribble" else "white_background",
        "seeds": seeds, "expressions": expressions, "prompt": args.prompt,
        "seed": seeds[0] if len(seeds) == 1 else None,
        "external_inputs": [str(path.resolve()) for path in args.input or []],
        "control_scale": config.HQ_CONTROLNET_CONDITIONING_SCALE,
        "control_end": render_options.get("control_end", config.HQ_CONTROLNET_ENDING_STEP),
        "follow_sketch": args.follow_sketch,
        "scope": "Synthetic and/or supplied sketches; manual review, not a quality score",
        "runs": records,
    }
    matrix = len(seeds) * len(expressions) > 1
    for seed in seeds:
        config.SEED = seed
        for expression in expressions:
            prompt = compose_prompt(args.prompt, expression)
            expression_index = list(EXPRESSION_PRESETS).index(expression)
            group = args.output / f"seed-{seed}-expression-{expression_index}" if matrix else args.output
            group.mkdir(parents=True, exist_ok=True)
            rows = []
            for name, sketch in cases:
                sketch.save(group / f"{name}-input.png")
                images = [sketch]
                for mode, steps in (("preview", config.PREVIEW_NUM_INFERENCE_STEPS),
                                    ("hq", config.HQ_NUM_INFERENCE_STEPS)):
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    started = time.perf_counter()
                    blocked = False
                    try:
                        image = render_sketch(pipe, sketch, prompt, steps, **render_options)
                    except RenderBlocked:
                        blocked = True
                        # 仅为评估总图创建占位；与实际返回图像明确区分。
                        image = Image.new("RGB", sketch.size, "black")
                        ImageDraw.Draw(image).text((20, 20), "BLOCKED by model checker", fill="white",
                                                  font=ImageFont.load_default(size=20))
                    torch.cuda.synchronize()
                    seconds = time.perf_counter() - started
                    filename = f"{name}-{mode}.png"
                    image.save(group / filename)
                    images.append(image)
                    record = {"case": name, "seed": seed, "expression": expression, "prompt": prompt, "mode": mode, "steps": steps, "seconds": seconds,
                              "manual_review": {"identity_or_style": None, "pose_and_count": None,
                                                "temporal_stability": None, "reference_usefulness": None},
                              "blocked": blocked,
                              "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                              "black_output": None if blocked else image.convert("RGB").getbbox() is None,
                              "image_role": "diagnostic_placeholder" if blocked else "generated",
                              "image": str((group / filename).relative_to(args.output))}
                    records.append(record)
                    print(json.dumps(record), flush=True)
                    # 逐次保存，意外失败也保留已完成的结果。
                    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
                rows.append((name, images))
            make_contact_sheet(rows, group / "comparison.png")
    print(f"评估完成: {args.output.resolve()}")


if __name__ == "__main__":
    main()
