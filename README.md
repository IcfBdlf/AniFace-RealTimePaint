# 🎨 AniFace — 实时线稿交互画板

将手绘线稿实时转换为高画质动漫风格头像。基于 **Stable Diffusion + ControlNet Scribble + StreamDiffusion**。

## 功能

| 功能 | 说明 |
|------|------|
| 🖊️ 手绘输入 | 在左侧画布自由涂鸦，右侧实时预览动漫风格渲染结果 |
| ⚡ 实时预览 | StreamDiffusion 超低延迟流式渲染，笔触即出效果 |
| 💎 超清重绘 | 停笔自动触发 25 步 ControlNet 高清渲染 |
| 📥 导入线稿 | 支持 JPG/PNG/BMP/WebP 本地线稿图片导入 |
| 📦 LoRA 挂载 | 动态加载/切换 LoRA 模型，自定义风格 |
| 🧹 一键擦除 | 清除画布重新开始 |

## 环境要求

- Python 3.10+
- NVIDIA GPU（至少 8GB 显存，推荐 RTX 4080）
- CUDA 12.1+

## 安装

```bash
# 克隆仓库
git clone https://github.com/IcfBdlf/AniFace-RealTimePaint.git
cd AniFace-RealTimePaint

# 安装依赖
pip install torch diffusers streamdiffusion transformers accelerate
pip install pillow tkinter
```

首次运行时会自动从 Hugging Face 下载模型（共约 10GB），请保持网络畅通。

## 使用

```bash
python canvas_stream.py
```

### 界面操作

1. **左侧画布**：鼠标拖动画线稿
2. **右侧面板**：实时预览渲染结果
3. **提示词输入框**：修改生成的动漫风格描述，按 Enter 触发重绘
4. **💎 25步超清重绘**：手动触发高质量渲染
5. **📥 导入线稿图片**：选择本地线稿文件
6. **🧹 擦除画布**：清空画板
7. **LoRA 路径/ID**：填入 LoRA 文件路径，调整权重，点击挂载

### 提示词示例

```
1girl, masterpiece, hyper detailed, anime portrait, high quality, sharp focus
```

## 项目结构

```
├── canvas_stream.py       # 主应用（GUI + 渲染管线）
├── test_stream.py         # 集成测试（模拟草图 → 渲染输出）
├── test_thread_safety.py  # 线程安全单元测试（17 项，无需 GPU）
├── saber/                 # 示例素材
│   └── fgfdgs1.png        # 测试用线稿图片
└── CHANGELOG.md           # 工作日志
```

## 技术栈

| 组件 | 说明 |
|------|------|
| **GUI** | Tkinter |
| **基础模型** | Counterfeit-V3.0（动漫风格 Stable Diffusion） |
| **ControlNet** | lllyasviel/sd-controlnet-scribble（线稿控制） |
| **流式推理** | StreamDiffusion（低延迟实时渲染） |
| **LoRA** | 动态风格挂载 |

## 测试

```bash
# 线程安全测试（mock pipeline，无需 GPU）
python test_thread_safety.py

# 集成测试（需要 GPU）
python test_stream.py
```

## 许可证

MIT
