"""角色/画风预设：验证兼容性，将固定描述与自由构图描述分离。"""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import config


@dataclass(frozen=True)
class ReferenceProfile:
    name: str
    kind: str
    base_model: str
    lora_path: str
    weight: float
    trigger_words: str
    default_prompt: str = "anime illustration, high quality"
    control_end: float = 1.0

    def metadata(self):
        return asdict(self)


def load_profile(path):
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    profile = ReferenceProfile(**data)
    for key in ("name", "base_model", "lora_path", "trigger_words", "default_prompt"):
        if not isinstance(getattr(profile, key), str) or not getattr(profile, key).strip():
            raise ValueError(f"预设 {key} 不能为空")
    if profile.kind not in {"character", "style"}:
        raise ValueError("kind 必须是 character 或 style")
    if profile.base_model != config.BASE_MODEL_ID:
        raise ValueError("预设的基础模型与当前管线不一致；请使用匹配的预设")
    if not math.isfinite(profile.weight) or not config.LORA_WEIGHT_MIN <= profile.weight <= config.LORA_WEIGHT_MAX:
        raise ValueError("LoRA 权重超出范围")
    if not math.isfinite(profile.control_end) or not 0 < profile.control_end <= 1:
        raise ValueError("control_end 必须在 (0, 1] 范围内")
    # 预设使用本地适配器，以免拼写错误触发隐式下载。
    lora = Path(profile.lora_path)
    if not lora.is_absolute():
        lora = path.parent / lora
    if not lora.exists():
        raise ValueError(f"找不到预设 LoRA：{lora}")
    return ReferenceProfile(**{**data, "lora_path": str(lora.resolve())})
