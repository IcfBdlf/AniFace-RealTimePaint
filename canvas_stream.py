import torch
import time
import threading
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog  # 🔥 新增：用于打开本地文件选择弹窗
from PIL import Image, ImageDraw, ImageTk

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image

class RealTimePaintApp:
    def __init__(self, root, base_pipe, stream_pipeline):
        self.root = root
        self.root.title("🎨 AniFace 交互完全体 (集成线稿导入功能)")
        self.pipe = base_pipe         
        self.stream = stream_pipeline 
        
        # 初始化画布
        self.sketch_img = Image.new("RGB", (512, 512), "white")
        self.draw_buffer = ImageDraw.Draw(self.sketch_img)
        self.sketch_tk_img = None  # 🔥 用于保存左侧导入图片的引用，防止垃圾回收
        
        # 1. 界面总布局
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 左右双屏
        self.canvas = tk.Canvas(self.main_frame, width=512, height=512, bg="white", cursor="pencil")
        self.canvas.grid(row=0, column=0, padx=5, pady=5)
        
        self.output_label = ttk.Label(self.main_frame)
        self.output_label.grid(row=0, column=1, padx=5, pady=5)
        
        # 2. 底部综合控制面板
        self.control_frame = ttk.LabelFrame(self.main_frame, text=" 🎛️ AI 实时引导控制面板 ", padding="10")
        self.control_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        # --- 第一行：提示词与基础功能控制 ---
        ttk.Label(self.control_frame, text="提示词 (Prompt):").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.prompt_var = tk.StringVar(value="1girl, masterpiece, hyper detailed, anime portrait, high quality, sharp focus")
        self.entry_prompt = ttk.Entry(self.control_frame, textvariable=self.prompt_var, width=55)
        self.entry_prompt.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        self.entry_prompt.bind("<Return>", lambda event: self.trigger_high_quality_render())
        
        self.btn_update = ttk.Button(self.control_frame, text="💎 25步超清重绘", command=self.trigger_high_quality_render)
        self.btn_update.grid(row=0, column=2, padx=5)
        
        # 🔥 新增：导入线稿按钮
        self.btn_import = ttk.Button(self.control_frame, text="📥 导入线稿图片", command=self.import_sketch_image)
        self.btn_import.grid(row=0, column=3, padx=5)
        
        self.btn_clear = ttk.Button(self.control_frame, text="🧹 擦除画布", command=self.clear_canvas)
        self.btn_clear.grid(row=0, column=4, padx=5)
        
        # --- 第二行：LoRA 动态挂载区 ---
        ttk.Label(self.control_frame, text="LoRA 路径/ID:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.lora_var = tk.StringVar(value="") 
        self.entry_lora = ttk.Entry(self.control_frame, textvariable=self.lora_var, width=50)
        self.entry_lora.grid(row=1, column=1, columnspan=1, padx=5, pady=5, sticky=tk.W)
        self.entry_lora.insert(0, "")  # 用户自行填入 LoRA 路径 
        
        ttk.Label(self.control_frame, text="权重(Strength):").grid(row=1, column=1, sticky=tk.E, padx=120)
        self.lora_weight_var = tk.DoubleVar(value=0.8)
        self.scale_lora = ttk.Scale(self.control_frame, from_=0.0, to=1.5, variable=self.lora_weight_var, orient=tk.HORIZONTAL, length=100)
        self.scale_lora.grid(row=1, column=1, sticky=tk.E, padx=10)
        
        self.btn_lora = ttk.Button(self.control_frame, text="📦 挂载/更新 LoRA", command=self.load_lora_weights_action)
        self.btn_lora.grid(row=1, column=2, columnspan=3, padx=5, sticky=(tk.W, tk.E))
        
        # 3. 画笔事件绑定（用户依然可以在导入的图片上继续手绘叠加代码）
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)
        
        self.last_x, self.last_y = None, None
        
        # 4. 线程锁与状态变量
        self.last_render_time = 0
        self.render_interval = 0.15
        self.is_rendering = False
        self.need_update_again = False
        self._preview_thread = None  # 防止预览线程堆积
        self.state_lock = threading.Lock()  # 保护 is_rendering / need_update_again
        
        self.update_right_display(Image.new("RGB", (512, 512), "white"))
        print("🚀 带有【本地线稿导入】功能的综合画板系统全线就绪！")

    def import_sketch_image(self):
        """🔥 核心新增：打开本地弹窗，直接读取用户的 JPG/PNG 线稿图片"""
        file_path = filedialog.askopenfilename(
            title="选择本地线稿图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        
        if file_path:
            try:
                print(f"📥 正在读取线稿: {file_path}")
                # 1. 读取并规范化图片（转换成RGB，强制缩放到512x512）
                imported_img = Image.open(file_path).convert("RGB").resize((512, 512))
                
                # 2. 覆盖后台内存里的画布，并重置画笔，让用户可以在导入的图上接着画
                self.sketch_img = imported_img
                self.draw_buffer = ImageDraw.Draw(self.sketch_img)
                
                # 3. 将图片同步渲染刷新到左边的 Tkinter Canvas 屏幕上
                self.sketch_tk_img = ImageTk.PhotoImage(self.sketch_img)
                self.canvas.delete("all")  # 擦掉以前手绘的线条痕迹
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.sketch_tk_img)
                
                # 4. 瞬间激活 4080 显卡，利用刚刚导入的线稿直接出超清成品大图！
                print("💎 线稿载入成功，正在全力触发 25 步超清画质对齐...")
                self.trigger_high_quality_render()
                
            except Exception as e:
                print(f"❌ 读取线稿失败: {e}")
                messagebox.showerror("读取错误", f"无法解析该图片文件，请确保它未损坏。\n错误报告: {e}")

    def paint(self, event):
        x, y = event.x, event.y
        if self.last_x and self.last_y:
            self.canvas.create_line(self.last_x, self.last_y, x, y, width=4, fill="black", capstyle=tk.ROUND, smooth=True)
            self.draw_buffer.line([self.last_x, self.last_y, x, y], fill="black", width=4)

            current_time = time.time()
            if (current_time - self.last_render_time > self.render_interval):
                # 检查是否有预览线程仍在运行，避免堆积
                preview_busy = self._preview_thread and self._preview_thread.is_alive()
                with self.state_lock:
                    hq_busy = self.is_rendering

                if not preview_busy and not hq_busy:
                    self.last_render_time = current_time
                    sketch_snapshot = self.sketch_img.copy()
                    t = threading.Thread(target=self.async_preview_render, args=(sketch_snapshot,), daemon=True)
                    t.start()
                    self._preview_thread = t

        self.last_x = x
        self.last_y = y

    def on_mouse_release(self, event):
        self.last_x, self.last_y = None, None
        self.trigger_high_quality_render()

    def trigger_high_quality_render(self):
        with self.state_lock:
            if not self.is_rendering:
                self.is_rendering = True
                need_skip = False
            else:
                self.need_update_again = True
                need_skip = True

        if need_skip:
            return

        sketch_snapshot = self.sketch_img.copy()
        t = threading.Thread(target=self.async_hq_render, args=(sketch_snapshot,), daemon=True)
        t.start()

    def load_lora_weights_action(self):
        lora_path = self.lora_var.get().strip()
        weight = self.lora_weight_var.get()

        if not lora_path:
            messagebox.showwarning("提示", "请先在输入框中填入正确的 LoRA 文件绝对路径！")
            return

        with self.state_lock:
            if self.is_rendering:
                messagebox.showwarning("提示", "当前正在渲染中，请稍后再加载 LoRA")
                return
            self.is_rendering = True

        try:
            self.pipe.unload_lora_weights()
            self.stream.pipe.unload_lora_weights()

            self.pipe.load_lora_weights(lora_path, adapter_name="paint_lora")
            self.stream.pipe.load_lora_weights(lora_path, adapter_name="paint_lora")

            self.pipe.set_adapters(["paint_lora"], adapter_weights=[weight])
            self.stream.pipe.set_adapters(["paint_lora"], adapter_weights=[weight])

            current_prompt = self.prompt_var.get()
            self.stream.prepare(prompt=current_prompt, num_inference_steps=2)

            messagebox.showinfo("成功", f"LoRA 补丁挂载成功！")
        except Exception as e:
            messagebox.showerror("加载失败", f"错误信息: {e}")
        finally:
            with self.state_lock:
                self.is_rendering = False
            self.trigger_high_quality_render()

    def async_preview_render(self, img_snapshot):
        try:
            x_output = self.stream(img_snapshot)
            output_image = postprocess_image(x_output, output_type="pil")[0]
            self.root.after(0, self.update_right_display, output_image)
        except Exception as e:
            print(f"⚠️ 预览渲染失败: {e}")
        finally:
            self._preview_thread = None

    def async_hq_render(self, img_snapshot):
        try:
            current_prompt = self.prompt_var.get()
            result = self.pipe(
                prompt=current_prompt,
                image=img_snapshot,
                num_inference_steps=25,
                controlnet_conditioning_scale=1.1,
                controlnet_ending_step=0.35  # 高情商纠错，防止导入的线稿有细微瑕疵
            )
            output_image = result.images[0]
            self.root.after(0, self.update_right_display, output_image)
        except Exception as e:
            print(f"⚠️ 超清重绘失败: {e}")
        finally:
            need_again = False
            with self.state_lock:
                self.is_rendering = False
                if self.need_update_again:
                    self.need_update_again = False
                    need_again = True

            if need_again:
                with self.state_lock:
                    self.is_rendering = True
                latest_snapshot = self.sketch_img.copy()
                t = threading.Thread(target=self.async_hq_render, args=(latest_snapshot,), daemon=True)
                t.start()

    def update_right_display(self, pil_img):
        # 先删除旧的 PhotoImage 引用，避免 Tcl 层面内存泄漏
        if hasattr(self, 'tk_img'):
            del self.tk_img
        self.tk_img = ImageTk.PhotoImage(pil_img)
        self.output_label.configure(image=self.tk_img)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.sketch_img = Image.new("RGB", (512, 512), "white")
        self.draw_buffer = ImageDraw.Draw(self.sketch_img)
        self.update_right_display(Image.new("RGB", (512, 512), "white"))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在配置高性能设备: {device}...")

    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-scribble", torch_dtype=torch.float16
    )
    base_pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "stablediffusionapi/counterfeit-v30", 
        controlnet=controlnet, 
        torch_dtype=torch.float16
    ).to(device)
    base_pipe.safety_checker = None

    stream_pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "stablediffusionapi/counterfeit-v30", 
        controlnet=controlnet, 
        torch_dtype=torch.float16
    ).to(device)
    
    stream = StreamDiffusion(stream_pipe, t_index_list=[0, 1], torch_dtype=torch.float16)
    initial_prompt = "1girl, masterpiece, hyper detailed, anime portrait, high quality, sharp focus"
    stream.prepare(prompt=initial_prompt, num_inference_steps=2)

    root = tk.Tk()
    app = RealTimePaintApp(root, base_pipe, stream)
    root.mainloop()

if __name__ == "__main__":
    main()