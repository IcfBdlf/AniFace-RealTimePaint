"""AniFace GUI 应用 — 画布交互、事件处理、界面布局"""
import logging
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageDraw, ImageTk

import config
from src.renderer import RenderManager

logger = logging.getLogger("AniFace.app")


class RealTimePaintApp:
    """实时线稿转动漫交互画板。"""

    def __init__(self, root, base_pipe, stream):
        self.root = root
        self.root.title(config.WINDOW_TITLE)

        # ---- 渲染管理器 ----
        self.renderer = RenderManager(
            base_pipe=base_pipe,
            stream=stream,
            schedule_fn=self.root.after,
            on_frame_fn=self._on_frame_ready,
            get_sketch_fn=self._get_sketch_snapshot,
        )

        # ---- 画布状态 ----
        self.sketch_img = Image.new("RGB", config.IMAGE_SIZE, "white")
        self.draw_buffer = ImageDraw.Draw(self.sketch_img)
        self.sketch_tk_img = None
        self.canvas_lock = threading.Lock()

        # ---- 渲染节流 ----
        self.last_render_time = 0
        self.render_interval = config.RENDER_INTERVAL

        # ---- 画笔状态 ----
        self.last_x, self.last_y = None, None

        # ---- 构建界面 ----
        self._build_ui()

        # ---- 优雅退出 ----
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        # ---- 初始画面 ----
        self._display_image(Image.new("RGB", config.IMAGE_SIZE, "white"))
        logger.info("🚀 带有【本地线稿导入】功能的综合画板系统全线就绪！")

    # ==================================================================
    # 界面构建
    # ==================================================================

    def _build_ui(self):
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # -- 左右双屏 --
        self.canvas = tk.Canvas(
            self.main_frame,
            width=config.IMAGE_SIZE[0],
            height=config.IMAGE_SIZE[1],
            bg="white",
            cursor="pencil",
        )
        self.canvas.grid(row=0, column=0, padx=5, pady=5)

        self.output_label = ttk.Label(self.main_frame)
        self.output_label.grid(row=0, column=1, padx=5, pady=5)

        # -- 底部控制面板 --
        self.control_frame = ttk.LabelFrame(
            self.main_frame, text=" 🎛️ AI 实时引导控制面板 ", padding="10"
        )
        self.control_frame.grid(
            row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10
        )

        # 第一行：提示词
        ttk.Label(self.control_frame, text="提示词 (Prompt):").grid(
            row=0, column=0, sticky=tk.W, padx=5
        )
        self.prompt_var = tk.StringVar(value=config.DEFAULT_PROMPT)
        self.entry_prompt = ttk.Entry(
            self.control_frame, textvariable=self.prompt_var, width=55
        )
        self.entry_prompt.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        self.entry_prompt.bind("<Return>", lambda e: self._request_hq())

        ttk.Button(
            self.control_frame, text="💎 25步超清重绘", command=self._request_hq
        ).grid(row=0, column=2, padx=5)

        ttk.Button(
            self.control_frame, text="📥 导入线稿图片", command=self._import_sketch
        ).grid(row=0, column=3, padx=5)

        ttk.Button(
            self.control_frame, text="🧹 擦除画布", command=self._clear_canvas
        ).grid(row=0, column=4, padx=5)

        # 第二行：LoRA
        ttk.Label(self.control_frame, text="LoRA 路径/ID:").grid(
            row=1, column=0, sticky=tk.W, padx=5
        )
        self.lora_var = tk.StringVar(value="")
        self.entry_lora = ttk.Entry(
            self.control_frame, textvariable=self.lora_var, width=50
        )
        self.entry_lora.grid(
            row=1, column=1, columnspan=1, padx=5, pady=5, sticky=tk.W
        )

        ttk.Label(self.control_frame, text="权重(Strength):").grid(
            row=1, column=1, sticky=tk.E, padx=120
        )
        self.lora_weight_var = tk.DoubleVar(value=config.DEFAULT_LORA_WEIGHT)
        self.scale_lora = ttk.Scale(
            self.control_frame,
            from_=config.LORA_WEIGHT_MIN,
            to=config.LORA_WEIGHT_MAX,
            variable=self.lora_weight_var,
            orient=tk.HORIZONTAL,
            length=100,
        )
        self.scale_lora.grid(row=1, column=1, sticky=tk.E, padx=10)

        ttk.Button(
            self.control_frame,
            text="📦 挂载/更新 LoRA",
            command=self._load_lora,
        ).grid(row=1, column=2, columnspan=3, padx=5, sticky=(tk.W, tk.E))

        # -- 画笔事件 --
        self.canvas.bind("<B1-Motion>", self._on_paint)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)

    # ==================================================================
    # 画布绘制
    # ==================================================================

    def _on_paint(self, event):
        x, y = event.x, event.y
        if self.last_x and self.last_y:
            self.canvas.create_line(
                self.last_x, self.last_y, x, y,
                width=config.PEN_WIDTH, fill=config.PEN_COLOR,
                capstyle=tk.ROUND, smooth=True,
            )
            with self.canvas_lock:
                self.draw_buffer.line(
                    [self.last_x, self.last_y, x, y],
                    fill=config.PEN_COLOR, width=config.PEN_WIDTH,
                )

            # 节流预览渲染
            now = time.time()
            if now - self.last_render_time > self.render_interval:
                preview_busy = self.renderer.check_busy()[0]
                if not preview_busy and not self.renderer.is_rendering:
                    self.last_render_time = now
                    with self.canvas_lock:
                        snapshot = self.sketch_img.copy()
                    self.renderer.request_preview(snapshot)

        self.last_x, self.last_y = x, y

    def _on_mouse_up(self, event):
        self.last_x, self.last_y = None, None
        self._request_hq()

    # ==================================================================
    # 渲染请求
    # ==================================================================

    def _request_hq(self):
        prompt = self.prompt_var.get()
        with self.canvas_lock:
            snapshot = self.sketch_img.copy()
        self.renderer.request_hq(snapshot, prompt)

    # ==================================================================
    # 线稿导入
    # ==================================================================

    def _import_sketch(self):
        file_path = filedialog.askopenfilename(
            title="选择本地线稿图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp *.webp")],
        )
        if not file_path:
            return

        try:
            logger.info(f"📥 正在读取线稿: {file_path}")
            imported = Image.open(file_path).convert("RGB").resize(config.IMAGE_SIZE)

            with self.canvas_lock:
                self.sketch_img = imported
                self.draw_buffer = ImageDraw.Draw(self.sketch_img)

            self.sketch_tk_img = ImageTk.PhotoImage(self.sketch_img)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.sketch_tk_img)

            logger.info("💎 线稿载入成功，正在全力触发 25 步超清画质对齐...")
            self._request_hq()
        except Exception as e:
            logger.error(f"❌ 读取线稿失败: {e}")
            messagebox.showerror("读取错误", f"无法解析该图片文件。\n错误: {e}")

    # ==================================================================
    # LoRA 加载
    # ==================================================================

    def _load_lora(self):
        lora_path = self.lora_var.get().strip()
        weight = self.lora_weight_var.get()

        if not lora_path:
            messagebox.showwarning("提示", "请先填入 LoRA 文件路径！")
            return

        if self.renderer.is_rendering:
            messagebox.showwarning("提示", "当前正在渲染中，请稍后再加载 LoRA")
            return

        try:
            self.renderer.load_lora(lora_path, weight, self.prompt_var.get())
            messagebox.showinfo("成功", "LoRA 补丁挂载成功！")
        except Exception as e:
            messagebox.showerror("加载失败", f"错误信息: {e}")
        finally:
            self._request_hq()

    # ==================================================================
    # 显示 & 画布清理
    # ==================================================================

    def _on_frame_ready(self, pil_img):
        """渲染完成回调 — 在右侧输出区域显示结果。"""
        self._display_image(pil_img)

    def _display_image(self, pil_img):
        if hasattr(self, "tk_img"):
            del self.tk_img
        self.tk_img = ImageTk.PhotoImage(pil_img)
        self.output_label.configure(image=self.tk_img)

    def _clear_canvas(self):
        self.canvas.delete("all")
        with self.canvas_lock:
            self.sketch_img = Image.new("RGB", config.IMAGE_SIZE, "white")
            self.draw_buffer = ImageDraw.Draw(self.sketch_img)
        self._display_image(Image.new("RGB", config.IMAGE_SIZE, "white"))

    # ==================================================================
    # 线程安全回调
    # ==================================================================

    def _get_sketch_snapshot(self):
        """渲染器通过此回调获取画布副本（线程安全）。"""
        with self.canvas_lock:
            return self.sketch_img.copy()

    # ==================================================================
    # 优雅退出
    # ==================================================================

    def _on_closing(self):
        logger.info("🛑 正在安全关闭应用...")
        self.renderer.shutdown()
        self._shutdown_attempts = 0
        self._shutdown_max_attempts = config.SHUTDOWN_MAX_ATTEMPTS
        self._check_shutdown()

    def _check_shutdown(self):
        self._shutdown_attempts += 1
        preview_alive, hq_busy = self.renderer.check_busy()

        if (preview_alive or hq_busy) and self._shutdown_attempts < self._shutdown_max_attempts:
            self.root.after(config.SHUTDOWN_CHECK_INTERVAL_MS, self._check_shutdown)
        else:
            self.root.destroy()
