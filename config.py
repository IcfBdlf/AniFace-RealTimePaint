"""
AniFaceProject 全局配置文件
将所有硬编码参数集中管理，方便调参与部署
"""

# =============================================================================
# 模型路径（HuggingFace model IDs）
# =============================================================================

from pathlib import Path

CONTROLNET_MODEL_ID = "xinsir/controlnet-scribble-sdxl-1.0"
BASE_MODEL_ID = "cagliostrolab/animagine-xl-3.1"
BASE_MODEL_REVISION = "483f0c322568ed13697ed01dd0be07204746d12b"
CONTROLNET_MODEL_REVISION = "0bed4973b80c2329f76e0abf4196c51dfbf9108c"
MODEL_CACHE_DIR = str(Path(__file__).resolve().parent / "artifacts" / "model-cache")
ENABLE_MODEL_CPU_OFFLOAD = True  # 12 GB 显卡优先保证 SDXL + ControlNet 显存可用
GUIDANCE_SCALE = 6.0
NEGATIVE_PROMPT = "nsfw, lowres, worst quality, low quality, text, watermark, extra digits"

# =============================================================================
# 推理参数
# =============================================================================

# ControlNet 预览基线；尚非毫秒级流式推理，速度和质量需实测。
PREVIEW_NUM_INFERENCE_STEPS = 10
SEED = 42
HQ_IDLE_DELAY_MS = 800
UI_POLL_INTERVAL_MS = 50
HISTORY_LIMIT = 30

# 精细重绘（25 步，分辨率不变）
HQ_NUM_INFERENCE_STEPS = 25
HQ_CONTROLNET_CONDITIONING_SCALE = 1.0
HQ_CONTROLNET_ENDING_STEP = 1.0  # 默认全程跟随线稿；预设可覆盖

# 渲染间隔（秒），防止预览请求过于密集
RENDER_INTERVAL = 0.15
MAX_PREVIEW_AGE_SECONDS = 30.0  # SDXL 含 CPU offload 的延迟预算，不代表实时性能承诺
REFERENCE_HISTORY_LIMIT = 12

# =============================================================================
# 画布与图像参数
# =============================================================================

IMAGE_SIZE = (512, 512)  # 交互画布宽高；与模型生成尺寸分开
INFERENCE_SIZE = (1024, 1024)  # SDXL 生成尺寸，预览和精细档共用

# 手绘笔刷参数
PEN_WIDTH = 4
PEN_COLOR = "black"

# =============================================================================
# LoRA 默认值
# =============================================================================

DEFAULT_LORA_WEIGHT = 0.8
LORA_WEIGHT_MIN = 0.0
LORA_WEIGHT_MAX = 1.5

# =============================================================================
# 默认提示词
# =============================================================================

DEFAULT_PROMPT = (
    "anime illustration, masterpiece, best quality, very aesthetic, general"
)

# =============================================================================
# 优雅退出参数
# =============================================================================

SHUTDOWN_CHECK_INTERVAL_MS = 200  # 轮询间隔（毫秒）

# =============================================================================
# UI 文本
# =============================================================================

WINDOW_TITLE = "AniFace — 线稿转动漫画板"
