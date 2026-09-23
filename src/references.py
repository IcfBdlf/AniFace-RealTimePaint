"""参考图快照与自包含 PNG 保存；元数据始终来自实际生成请求。"""
from copy import deepcopy
from dataclasses import dataclass
import base64
from io import BytesIO
import json

from PIL.PngImagePlugin import PngInfo


@dataclass
class Reference:
    image: object
    sketch: object
    metadata: dict

    @classmethod
    def capture(cls, image, sketch, metadata):
        return cls(image.copy(), sketch.copy(), deepcopy(metadata))

    def save(self, path):
        buffer = BytesIO()
        self.sketch.save(buffer, format="PNG")
        info = PngInfo()
        info.add_itxt("aniface.recipe", json.dumps(self.metadata, ensure_ascii=False))
        info.add_itxt("aniface.sketch.png.base64", base64.b64encode(buffer.getvalue()).decode("ascii"))
        self.image.save(path, format="PNG", pnginfo=info)
