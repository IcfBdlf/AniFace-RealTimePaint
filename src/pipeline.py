"""模型加载与管线初始化"""
import logging
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from streamdiffusion import StreamDiffusion
import config

logger = logging.getLogger("AniFace.pipeline")


def create_pipelines(device=None):
    """加载模型，返回 (base_pipe, stream)。两个管线共享底层 UNet/VAE/TextEncoder。"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"正在配置高性能设备: {device}...")

    # 1. ControlNet（只加载一次）
    controlnet = ControlNetModel.from_pretrained(
        config.CONTROLNET_MODEL_ID, torch_dtype=torch.float16
    )

    # 2. 基础 Pipeline（UNet / VAE / TextEncoder 唯此一份）
    base_pipe = StableDiffusionControlNetPipeline.from_pretrained(
        config.BASE_MODEL_ID,
        controlnet=controlnet,
        torch_dtype=torch.float16,
    ).to(device)
    base_pipe.safety_checker = None

    # 3. 共享组件，创建 StreamDiffusion 专用 Pipeline（节省 ~5GB 显存）
    stream_pipe = StableDiffusionControlNetPipeline(
        vae=base_pipe.vae,
        text_encoder=base_pipe.text_encoder,
        tokenizer=base_pipe.tokenizer,
        unet=base_pipe.unet,
        controlnet=base_pipe.controlnet,
        scheduler=base_pipe.scheduler,
        safety_checker=None,
        feature_extractor=base_pipe.feature_extractor,
        requires_safety_checker=False,
    )

    stream = StreamDiffusion(
        stream_pipe,
        t_index_list=config.PREVIEW_T_INDEX_LIST,
        torch_dtype=torch.float16,
    )
    stream.prepare(
        prompt=config.DEFAULT_PROMPT,
        num_inference_steps=config.PREVIEW_NUM_INFERENCE_STEPS,
    )

    return base_pipe, stream
