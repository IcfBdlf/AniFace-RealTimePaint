"""AniFace GUI。所有 Tkinter 调用只发生在主线程。"""
import logging
import time
import tkinter as tk
from dataclasses import replace
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageDraw, ImageTk, ImageOps

import config
from src.renderer import RenderManager
from src.prompts import EXPRESSION_PRESETS, compose_prompt
from src.profiles import load_profile
from src.references import Reference

logger = logging.getLogger("AniFace.app")


def fit_sketch(image):
    """保留宽高比、EXIF 方向；透明区域合成到白底。"""
    image = ImageOps.exif_transpose(image).convert("RGBA")
    fitted = ImageOps.contain(image, config.IMAGE_SIZE, Image.Resampling.LANCZOS)
    result = Image.new("RGB", config.IMAGE_SIZE, "white")
    result.paste(fitted, ((result.width - fitted.width) // 2,
                          (result.height - fitted.height) // 2), fitted)
    return result


class RealTimePaintApp:
    def __init__(self, root, pipe):
        self.root = root
        root.title(config.WINDOW_TITLE)
        self.sketch_img = Image.new("RGB", config.IMAGE_SIZE, "white")
        self.draw_buffer = ImageDraw.Draw(self.sketch_img)
        self.last_x = self.last_y = None
        self.last_render_time = 0
        self._hq_timer = None
        self._poll_timer = None
        self._closing = False
        self._undo_stack = []
        self._redo_stack = []
        self._result = None
        self._notice = ""
        self._lora_notice = ""
        self._profile = None
        self._pending_profile = None
        self._loading_lora = False
        self._reset_profile_prompt = False
        self._references = []
        self._reference_index = -1
        self._reference = None
        self._canvas_revision = 0
        self._build_ui()
        self.renderer = RenderManager(pipe)
        self.prompt_var.trace_add("write", self._on_prompt_changed)
        self.expression_var.trace_add("write", self._on_prompt_changed)
        self.follow_sketch_var.trace_add("write", self._on_prompt_changed)
        root.protocol("WM_DELETE_WINDOW", self._on_closing)
        self._display_image(Image.new("RGB", config.IMAGE_SIZE, "white"))
        self._poll_renderer()
        logger.info("AniFace 画板已就绪")

    def _build_ui(self):
        frame = ttk.Frame(self.root, padding=10)
        frame.grid(sticky="nsew")
        drawing = ttk.Frame(frame)
        drawing.grid(row=0, column=0, padx=5, pady=5)
        self.canvas = tk.Canvas(drawing, width=config.IMAGE_SIZE[0],
                                height=config.IMAGE_SIZE[1], bg="white", cursor="pencil",
                                highlightthickness=0)
        self.canvas.grid(row=0, column=0)
        xbar = ttk.Scrollbar(drawing, orient="horizontal", command=self.canvas.xview)
        ybar = ttk.Scrollbar(drawing, orient="vertical", command=self.canvas.yview)
        xbar.grid(row=1, column=0, sticky="ew")
        ybar.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(xscrollcommand=xbar.set, yscrollcommand=ybar.set)
        self.output_label = ttk.Label(frame)
        self.output_label.grid(row=0, column=1, padx=5, pady=5)
        controls = ttk.LabelFrame(frame, text="生成设置", padding=10)
        controls.grid(row=1, column=0, columnspan=2, sticky="ew")
        controls.columnconfigure(1, weight=1)
        ttk.Label(controls, text="提示词").grid(row=0, column=0, padx=5)
        self.prompt_var = tk.StringVar(value=config.DEFAULT_PROMPT)
        entry = ttk.Entry(controls, textvariable=self.prompt_var, width=65)
        entry.grid(row=0, column=1, columnspan=3, sticky="ew", pady=5)
        entry.bind("<Return>", lambda _e: self._request_hq())
        ttk.Label(controls, text="LoRA 路径/ID").grid(row=1, column=0, padx=5)
        self.lora_var = tk.StringVar()
        ttk.Entry(controls, textvariable=self.lora_var, width=45).grid(row=1, column=1, sticky="ew")
        self.lora_weight_var = tk.DoubleVar(value=config.DEFAULT_LORA_WEIGHT)
        ttk.Scale(controls, from_=config.LORA_WEIGHT_MIN, to=config.LORA_WEIGHT_MAX,
                  variable=self.lora_weight_var, length=110).grid(row=1, column=2, padx=10)
        ttk.Button(controls, text="加载 LoRA", command=self._load_lora).grid(row=1, column=3)
        ttk.Label(controls, text="表情").grid(row=2, column=0, padx=5, pady=5)
        self.expression_var = tk.StringVar(value="不附加表情")
        ttk.Combobox(controls, textvariable=self.expression_var,
                     values=list(EXPRESSION_PRESETS), state="readonly", width=18).grid(
                         row=2, column=1, sticky="w", pady=5)
        ttk.Label(controls, text="附加到提示词；自定义表情可直接写入提示词").grid(
            row=2, column=2, columnspan=2, sticky="w")
        self.follow_sketch_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(controls, text="尽量跟随线稿", variable=self.follow_sketch_var).grid(
            row=3, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        ttk.Label(controls, text="更重视轮廓和姿势；细节仍可能改变").grid(
            row=3, column=2, columnspan=2, sticky="w")
        self.profile_var = tk.StringVar(value="未使用角色/画风预设")
        ttk.Button(controls, text="打开角色/画风预设", command=self._open_profile).grid(row=4, column=0)
        ttk.Label(controls, textvariable=self.profile_var, wraplength=620).grid(row=4, column=1, columnspan=2, sticky="w")
        ttk.Button(controls, text="卸载 LoRA / 预设", command=self._unload_lora).grid(row=4, column=3)
        self.active_lora_var = tk.StringVar(value="实际生效：基础模型")
        ttk.Label(controls, textvariable=self.active_lora_var, wraplength=950).grid(row=5, column=0, columnspan=4, sticky="w")
        toolbar = ttk.Frame(frame, padding=(0, 8))
        toolbar.grid(row=2, column=0, columnspan=2, sticky="w")
        for title, command in [("精细重绘", self._request_hq), ("导入线稿", self._import_sketch),
                               ("清空", self._clear_canvas), ("撤销", self._undo),
                               ("重做", self._redo), ("保存结果", self._save_result)]:
            ttk.Button(toolbar, text=title, command=command).pack(side=tk.LEFT, padx=4)
        self.status_var = tk.StringVar(value="就绪")
        options = ttk.Frame(frame)
        options.grid(row=3, column=0, columnspan=2, sticky="w")
        self.eraser_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options, text="橡皮擦", variable=self.eraser_var).pack(side=tk.LEFT)
        ttk.Label(options, text="笔刷大小").pack(side=tk.LEFT)
        self.brush_var = tk.IntVar(value=config.PEN_WIDTH)
        ttk.Spinbox(options, from_=1, to=64, textvariable=self.brush_var, width=4).pack(side=tk.LEFT)
        self.zoom_var = tk.StringVar(value="100%")
        zoom = ttk.Combobox(options, textvariable=self.zoom_var, values=["100%", "150%", "200%", "300%"], state="readonly", width=6)
        zoom.pack(side=tk.LEFT, padx=8)
        zoom.bind("<<ComboboxSelected>>", lambda _e: self._redraw_canvas())
        ttk.Label(options, text="放大后用滚动条或中键平移").pack(side=tk.LEFT)
        self.paused_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options, text="暂停生成", variable=self.paused_var, command=self._toggle_pause).pack(side=tk.LEFT, padx=8)
        self.pinned_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options, text="固定参考", variable=self.pinned_var, command=self._toggle_pin).pack(side=tk.LEFT)
        self.auto_hq_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options, text="停笔后自动精细重绘", variable=self.auto_hq_var).pack(side=tk.LEFT)
        history = ttk.Frame(frame)
        history.grid(row=4, column=0, columnspan=2, sticky="w", pady=5)
        ttk.Button(history, text="上一张", command=lambda: self._browse_reference(-1)).pack(side=tk.LEFT)
        ttk.Button(history, text="下一张", command=lambda: self._browse_reference(1)).pack(side=tk.LEFT)
        self.reference_var = tk.StringVar(value="尚无参考图")
        ttk.Label(history, textvariable=self.reference_var, wraplength=820).pack(side=tk.LEFT, padx=8)
        ttk.Label(frame, textvariable=self.status_var, wraplength=1000).grid(
            row=5, column=0, columnspan=2, sticky="w")
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_paint)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.canvas.bind("<ButtonPress-2>", lambda e: self.canvas.scan_mark(e.x, e.y))
        self.canvas.bind("<B2-Motion>", lambda e: self.canvas.scan_dragto(e.x, e.y, gain=1))

    def _remember(self):
        self._undo_stack.append(self.sketch_img.copy())
        del self._undo_stack[:-config.HISTORY_LIMIT]
        self._redo_stack.clear()

    def _cancel_hq_timer(self):
        if self._hq_timer is not None:
            self.root.after_cancel(self._hq_timer)
            self._hq_timer = None

    def _changed(self, *, soft=False):
        self._cancel_hq_timer()
        self.renderer.invalidate(soft=soft)
        self._canvas_revision += 1
        self._notice = ""
        if self._reference is not None:
            self.reference_var.set("保留的参考图对应较早输入；新结果生成后更新（固定时保持）")

    def _defer_hq(self):
        self._cancel_hq_timer()
        if self._can_generate() and self.auto_hq_var.get():
            self._hq_timer = self.root.after(config.HQ_IDLE_DELAY_MS, self._request_hq)

    def _on_prompt_changed(self, *_args):
        if not self._closing:
            self._changed()
            self._request_preview()
            self._defer_hq()

    def _point(self, event):
        zoom = int(self.zoom_var.get().rstrip("%")) / 100
        return (max(0, min(self.canvas.canvasx(event.x) / zoom, self.sketch_img.width - 1)),
                max(0, min(self.canvas.canvasy(event.y) / zoom, self.sketch_img.height - 1)))

    def _current_prompt(self):
        prompt = compose_prompt(self.prompt_var.get(), self.expression_var.get())
        return f"{self._profile.trigger_words}, {prompt}" if self._profile else prompt

    def _render_options(self):
        end = 1.0 if self.follow_sketch_var.get() else (self._profile.control_end if self._profile else None)
        return {"control_end": end, "metadata": {"profile": self._profile.metadata() if self._profile else None,
                                                 "canvas_revision": self._canvas_revision}}

    def _can_generate(self):
        return not self._closing and not self.paused_var.get() and not self._loading_lora

    def _request_preview(self):
        if self._can_generate():
            self.renderer.request_preview(self.sketch_img, self._current_prompt(), **self._render_options())

    def _on_mouse_down(self, event):
        if self._closing:
            return
        self._remember()
        self.last_x, self.last_y = self._point(event)
        self._draw_to(self.last_x, self.last_y)

    def _on_paint(self, event):
        if not self._closing and self.last_x is not None and self.last_y is not None:
            self._draw_to(*self._point(event))

    def _draw_to(self, x, y):
        self._changed(soft=True)
        try:
            width = max(1, min(64, self.brush_var.get()))
        except (tk.TclError, ValueError):
            width = config.PEN_WIDTH
        color = "white" if self.eraser_var.get() else config.PEN_COLOR
        self.draw_buffer.line((self.last_x, self.last_y, x, y),
                              fill=color, width=width)
        radius = width / 2
        self.draw_buffer.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)
        self.last_x, self.last_y = x, y
        # 同一 PIL 图像既用于显示也用于推理，避免两套笔刷边缘不一致。
        self._redraw_canvas()
        now = time.monotonic()
        if now - self.last_render_time >= config.RENDER_INTERVAL:
            self.last_render_time = now
            self._request_preview()

    def _on_mouse_up(self, _event):
        if self._closing or self.last_x is None:
            return
        self.last_x = self.last_y = None
        self._request_preview()
        self._defer_hq()

    def _request_hq(self):
        self._cancel_hq_timer()
        if self._can_generate():
            self._notice = ""
            self.renderer.request_hq(self.sketch_img, self._current_prompt(), **self._render_options())

    def _redraw_canvas(self):
        zoom = int(self.zoom_var.get().rstrip("%")) / 100
        size = tuple(int(x * zoom) for x in self.sketch_img.size)
        self.sketch_tk_img = ImageTk.PhotoImage(self.sketch_img.resize(size))
        self.canvas.configure(scrollregion=(0, 0, *size))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.sketch_tk_img)

    def _replace_sketch(self, image):
        self._changed()
        self.sketch_img = image
        self.draw_buffer = ImageDraw.Draw(image)
        self.last_x = self.last_y = None
        self._redraw_canvas()

    def _clear_canvas(self):
        if self._closing:
            return
        self._remember()
        self._replace_sketch(Image.new("RGB", config.IMAGE_SIZE, "white"))
        self._result = None
        if self.pinned_var.get():
            self._result = self._reference.image if self._reference else None
            return
        self._reference = None
        self.reference_var.set("尚无参考图；历史记录仍可浏览")
        self._display_image(Image.new("RGB", config.IMAGE_SIZE, "white"))

    def _undo(self):
        if not self._closing and self._undo_stack:
            self._redo_stack.append(self.sketch_img.copy())
            self._replace_sketch(self._undo_stack.pop())
            self._request_preview()
            self._defer_hq()

    def _redo(self):
        if not self._closing and self._redo_stack:
            self._undo_stack.append(self.sketch_img.copy())
            self._replace_sketch(self._redo_stack.pop())
            self._request_preview()
            self._defer_hq()

    def _import_sketch(self):
        if self._closing:
            return
        path = filedialog.askopenfilename(title="选择白底黑线的线稿",
                    filetypes=[("图片", "*.jpg *.jpeg *.png *.bmp *.webp")])
        if not path:
            return
        try:
            with Image.open(path) as image:
                imported = fit_sketch(image)
            self._remember()
            self._replace_sketch(imported)
            self._request_hq()
        except Exception as exc:
            messagebox.showerror("导入失败", str(exc))

    def _load_lora(self):
        if self._closing or self._loading_lora:
            return
        path = self.lora_var.get().strip()
        if not path:
            messagebox.showwarning("提示", "请填写 LoRA 文件路径或模型 ID")
            return
        self._changed()
        self._loading_lora = True
        self._reset_profile_prompt = False
        self._pending_profile = (replace(self._profile, weight=self.lora_weight_var.get())
                                 if self._profile and path == self._profile.lora_path else None)
        self.renderer.load_lora(path, self.lora_weight_var.get())

    def _open_profile(self):
        if self._closing or self._loading_lora:
            return
        path = filedialog.askopenfilename(title="选择角色/画风预设", filetypes=[("预设", "*.json")])
        if not path:
            return
        try:
            profile = load_profile(path)
        except Exception as exc:
            messagebox.showerror("预设无效", str(exc))
            return
        self._changed()
        self._pending_profile = profile
        self._reset_profile_prompt = True
        self._loading_lora = True
        self.renderer.load_lora(profile.lora_path, profile.weight)

    def _unload_lora(self):
        if self._closing or self._loading_lora:
            return
        self._changed()
        self._loading_lora = True
        self._pending_profile = None
        self._reset_profile_prompt = False
        self.renderer.load_lora(None, 0)

    def _toggle_pause(self):
        self._changed()
        if not self.paused_var.get():
            self._request_preview()

    def _toggle_pin(self):
        if not self.pinned_var.get() and self._references:
            self._show_reference(len(self._references) - 1)

    def _browse_reference(self, direction):
        if not self._references:
            return
        self.pinned_var.set(True)
        self._show_reference(max(0, min(len(self._references) - 1, self._reference_index + direction)))

    def _show_reference(self, index):
        self._reference_index = index
        self._reference = self._references[index]
        self._result = self._reference.image
        self._display_image(self._result)
        meta = self._reference.metadata
        profile = meta.get("profile")
        name = profile["name"] if profile else "自定义配置"
        age = "当前输入" if meta.get("canvas_revision") == self._canvas_revision else "较早输入"
        self.reference_var.set(f"参考 {index + 1}/{len(self._references)} · {name} · {meta['mode']} · {age} · "
                               f"快照到显示 {meta['input_to_display_seconds']:.2f}s")

    def _poll_renderer(self):
        if self._closing:
            return
        for event in self.renderer.poll_events():
            if event.kind == "frame":
                metadata = dict(event.metadata)
                metadata["input_to_display_seconds"] = time.monotonic() - event.request.submitted_at
                self._references.append(Reference.capture(event.value, event.request.image, metadata))
                if len(self._references) > config.REFERENCE_HISTORY_LIMIT:
                    self._references.pop(0)
                    self._reference_index = max(-1, self._reference_index - 1)
                if not self.pinned_var.get():
                    self._show_reference(len(self._references) - 1)
                self._notice = ""
            elif event.kind == "blocked":
                retained = "已保留上一张有效结果。" if self._result is not None else "尚无可显示的结果。"
                self._notice = f"{event.value} {retained}"
            elif event.kind in {"lora_error", "info"}:
                self._lora_notice = str(event.value)
                if self._loading_lora:
                    if event.kind == "info":
                        self._profile = self._pending_profile
                        self.profile_var.set(self._profile.name if self._profile else "未使用角色/画风预设")
                        if self._profile:
                            if self._reset_profile_prompt:
                                self.prompt_var.set(self._profile.default_prompt)
                                self.follow_sketch_var.set(False)
                                self.expression_var.set("不附加表情")
                            self.lora_var.set(self._profile.lora_path)
                            self.lora_weight_var.set(self._profile.weight)
                    self._pending_profile = None
                    self._loading_lora = False
                    if event.kind == "info":
                        self._request_preview()
                active = self.renderer.active_lora
                self.active_lora_var.set(f"实际生效：{active['path']} · 权重 {active['weight']}" if active["path"] else "实际生效：基础模型")
            else:
                # 非模态提示，不在处理结果过程中进入嵌套 Tk 事件循环。
                self._notice = str(event.value)
        status = "已暂停生成" if self.paused_var.get() else self.renderer.status
        if self.pinned_var.get():
            status += " · 当前参考已固定"
        self.status_var.set("  |  ".join(part for part in (status, self._lora_notice, self._notice) if part))
        self._poll_timer = self.root.after(config.UI_POLL_INTERVAL_MS, self._poll_renderer)

    def _display_image(self, image):
        self.tk_img = ImageTk.PhotoImage(image)
        self.output_label.configure(image=self.tk_img)

    def _save_result(self):
        if self._closing:
            return
        if self._result is None:
            messagebox.showinfo("提示", "还没有生成结果")
            return
        reference = self._reference
        path = filedialog.asksaveasfilename(title="保存当前显示结果", defaultextension=".png",
                                           filetypes=[("PNG", "*.png")])
        if path:
            try:
                reference.save(path)
            except Exception as exc:
                messagebox.showerror("保存失败", str(exc))

    def _on_closing(self):
        if self._closing:
            return
        self._closing = True
        self._cancel_hq_timer()
        if self._poll_timer is not None:
            self.root.after_cancel(self._poll_timer)
        self.renderer.shutdown()
        self._check_shutdown()

    def _check_shutdown(self):
        if self.renderer.join(timeout=0):
            self.root.destroy()
        else:
            self.status_var.set(self.renderer.status)
            self.root.after(config.SHUTDOWN_CHECK_INTERVAL_MS, self._check_shutdown)
