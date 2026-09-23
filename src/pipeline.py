"""ControlNet 基线：预览和精细重绘共享管线、输入处理和随机种子。"""
import logging
from PIL import ImageOps
import config

logger = logging.getLogger("AniFace.pipeline")

LINEART_MODEL_ID = "lllyasviel/control_v11p_sd15_lineart"
ANIME_LINEART_MODEL_ID = "lllyasviel/control_v11p_sd15s2_lineart_anime"


def control_model_id(control_type):
    return {"scribble": config.CONTROLNET_MODEL_ID, "lineart": LINEART_MODEL_ID,
            "lineart_anime": ANIME_LINEART_MODEL_ID}[control_type]


class RenderCancelled(Exception):
    """关闭应用或输入过时时在去噪步骤边界退出。"""


class RenderBlocked(Exception):
    """模型检查器未放行结果；不将占位黑图作为正常图像显示。"""


class LoadCancelled(Exception):
    """在模型加载阶段边界取消启动。"""


def create_pipelines(device=None, local_files_only=False, *, on_progress=None,
                     should_stop=lambda: False, control_type="scribble", control_cache_dir=None):
    """返回唯一的 ControlNet 管线。延迟导入重型依赖以支持无 GPU 测试。"""
    if control_type not in {"scribble", "lineart", "lineart_anime"}:
        raise ValueError("不支持的线稿控制类型")
    def stage(message):
        if should_stop():
            raise LoadCancelled()
        if on_progress is not None:
            on_progress(message)

    stage("正在初始化推理环境…")
    import torch
    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    logger.info("加载 ControlNet 管线，设备=%s，精度=%s", device, dtype)
    stage("正在加载线稿控制模型；首次使用可能需要下载…")
    controlnet = ControlNetModel.from_pretrained(
        control_model_id(control_type),
        torch_dtype=dtype, local_files_only=local_files_only,
        **({"cache_dir": control_cache_dir} if control_cache_dir is not None else {})
    )
    stage("正在加载基础绘画模型；首次使用可能需要下载…")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        config.BASE_MODEL_ID, controlnet=controlnet, torch_dtype=dtype,
        local_files_only=local_files_only,
    )
    stage(f"正在将模型准备到 {device} 设备…")
    pipe = pipe.to(device)
    stage("正在配置推理管线…")
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    pipe._aniface_control_type = control_type
    stage("模型已就绪")
    return pipe


def prepare_control_image(sketch, control_type="scribble"):
    """Scribble 使用黑底白线；Lineart 保留白底黑线。"""
    gray = sketch.convert("L")
    if control_type == "scribble":
        gray = ImageOps.invert(gray)
    elif control_type not in {"lineart", "lineart_anime"}:
        raise ValueError("不支持的线稿控制类型")
    return gray.convert("RGB")


def render_sketch(pipe, sketch, prompt, steps, should_stop=lambda: False, *, control_end=None, clip_skip=None):
    control_end = config.HQ_CONTROLNET_ENDING_STEP if control_end is None else control_end
    if not 0 < control_end <= 1:
        raise ValueError("线稿控制结束比例必须大于 0 且不超过 1")
    import torch

    def on_step_end(_pipe, _step, _timestep, callback_kwargs):
        if should_stop():
            raise RenderCancelled()
        return callback_kwargs

    if should_stop():
        raise RenderCancelled()
    # 两个档位从相同噪声开始；步数差异仍可能导致图像差异。
    generator = torch.Generator(device="cpu").manual_seed(config.SEED)
    with torch.inference_mode():
        result = pipe(
            prompt=prompt, image=prepare_control_image(
                sketch, pipe.__dict__.get("_aniface_control_type", "scribble")),
            width=sketch.width, height=sketch.height,
            num_inference_steps=steps, generator=generator,
            controlnet_conditioning_scale=config.HQ_CONTROLNET_CONDITIONING_SCALE,
            control_guidance_end=control_end,
            callback_on_step_end=on_step_end,
            **({"clip_skip": clip_skip} if clip_skip is not None else {}),
        )
    if should_stop():
        raise RenderCancelled()
    flags = getattr(result, "nsfw_content_detected", None)
    if flags is not None and any(flags):
        raise RenderBlocked("本次结果被模型安全检查拦截，请调整提示词或线稿后重试。")
    return result.images[0]
