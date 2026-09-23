"""AniFaceProject 入口 — 实时线稿转动漫交互画板

用法:
    .venv/Scripts/python.exe -X utf8 canvas_stream.py
"""
import logging
import sys
import tkinter as tk
from pathlib import Path

from src.app import RealTimePaintApp
from src.startup import StartupWindow

# ---- 日志 ----
LOG_DIR = Path(__file__).resolve().parent / "artifacts" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "aniface.log", encoding="utf-8"),
    ],
)


def main():
    root = tk.Tk()
    StartupWindow(root, RealTimePaintApp)
    root.mainloop()


if __name__ == "__main__":
    main()
