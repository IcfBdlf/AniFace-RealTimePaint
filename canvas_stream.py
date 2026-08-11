"""AniFaceProject 入口 — 实时线稿转动漫交互画板

用法:
    PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 python canvas_stream.py
"""
import logging
import sys
import tkinter as tk

from src.pipeline import create_pipelines
from src.app import RealTimePaintApp

# ---- 日志 ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("aniface.log", encoding="utf-8"),
    ],
)


def main():
    base_pipe, stream = create_pipelines()
    root = tk.Tk()
    RealTimePaintApp(root, base_pipe, stream)
    root.mainloop()


if __name__ == "__main__":
    main()
