# AniFace — 手绘驱动的动漫参考画板

目标：在选定角色或画风的情况下，由用户线稿控制姿势、构图和表情，持续生成参考图片，辅助同人创作与绘画学习。

当前 SDXL 分支使用 Tkinter + Animagine XL 3.1 + Xinsir SDXL Scribble。操作画布 512×512，生成与保存 1024×1024；预览 10 步，精细重绘 25 步，均重新采样。真实角色一致性尚待匹配 LoRA 验证。参数、兼容性与 Git 回退见 [SDXL 迁移说明](docs/SDXL_MIGRATION.md)，实测状态见 [验证记录](docs/reports/SDXL_VALIDATION.md)。

以下两份长篇手册正文保留为迁移前 SD1.5 学习快照；当前模型代码与参数先读 [SDXL 迁移说明](docs/SDXL_MIGRATION.md)。

准备面试或系统学习代码，请读 [项目完整技术说明与面试准备](docs/INTERVIEW_GUIDE.md)：包含技术原理、源码流程、调度与失败恢复、测试证据及 25 个面试追问。旧版实现指南已删除。

想逐段学习原版代码，请读 [核心源码逐函数学习手册](docs/CODE_WALKTHROUGH.md)：覆盖 76 个核心函数、全部配置项，并附原代码、执行说明、三个完整调用过程、测试例子和练习答案。

## 运行

在项目根目录运行（Windows PowerShell，推荐 Python 3.10/3.11 和 NVIDIA CUDA GPU）：

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pip check
.venv/Scripts/python.exe -X utf8 canvas_stream.py
```

已有本机环境可直接执行最后一条。它通过系统包复用 `aniface_diff` 的 PyTorch；新机器请完整安装。Tkinter 由 Python 安装程序提供。CPU 使用 float32，但不适合当前交互速度目标。

仅使用已有缓存：先设置 `$env:HF_HUB_OFFLINE = "1"`。首次启动可能下载模型；关闭加载窗口在第三方加载调用结束后的阶段边界生效。日志位于 `artifacts/logs/aniface.log`。

## 使用

SDXL 当前默认在前 60% 去噪步骤施加线稿引导，后期放松约束以补充细节。想试这个效果，请不要勾选“尽量跟随线稿”（勾选会恢复全程引导）；角色预设可覆盖默认比例。改配置后需重启应用。

- 在左侧绘制或导入白底黑线线稿；支持橡皮擦、1–64 像素笔刷、撤销/重做、100%–300% 缩放、滚动条与中键平移。缩放仅影响显示，不改变推理分辨率。
- 连续绘画时提交最新快照，允许同一配置的近期预览完成；清空、提示词或 LoRA 切换立即淘汰旧配置结果。预览最大年龄由 `MAX_PREVIEW_AGE_SECONDS` 控制，当前 SDXL 默认 30 秒；这是过期预算，不是速度承诺。
- 默认手动“精细重绘”；可以勾选停笔后 800ms 自动精细重绘。固定随机种子有助于减少随机变化，但不保证角色或细节一致。
- “暂停生成”停止新推理并取消当前任务；恢复时重新提交当前画布。“固定参考”保留屏幕上的图，同时允许后台结果进入历史。上一张/下一张会自动固定参考，取消固定回到最新结果。最多保留 12 张历史。
- “保存结果”保存当前显示图片为 PNG，并内嵌它对应的线稿和参数。字段为 `aniface.recipe`（JSON）、`aniface.sketch.png.base64`（线稿 PNG 的 Base64）。Pillow 的 `Image.open(path).info` 可读取；图片编辑器重新保存时可能删除这些字段。
- 加载 LoRA 后才应用权重；同一路径只更新权重，不重新加载。界面分别显示编辑值和实际生效状态。可以卸载并恢复基础模型。
- “打开角色/画风预设”读取本地 JSON；基础模型不匹配时拒绝应用，LoRA 加载成功后才切换预设。固定触发词与可编辑构图提示词分开。格式见 [预设说明](profiles/README.md)。
- 当前 SDXL 管线没有旧版 SD1.5 的内置输出检查器；负面提示词不等于内容检测。

## 目录

```text
canvas_stream.py       桌面启动入口
config.py              模型与推理默认设置
src/                   应用、后台调度、管线、预设与参考图保存
tests/                 自动回归测试，包含隐藏 Tk 窗口测试
tools/                 基准、连续输入验收、GPU 回归、画质评估
profiles/              角色/画风预设说明与模板（不包含模型）
docs/                  当前设计与验收说明
  reports/             评估报告
  INTERVIEW_GUIDE.md    当前实现的完整技术与面试指南
  history/             历史交接记录
assets/                小型样例资源
artifacts/             本地产物（Git 忽略）
  checks/              GPU/窗口/压力测试结果及测试适配器
  evaluations/         画质评估图片与指标
  environment/         环境报告、版本快照与清理清单
  model-cache/         已下载的模型权重
  logs/                运行日志
SESSION_HANDOFF.md     最新交接入口
CHANGELOG.md           改动历史
```

旧根目录脚本已迁入 `tools`，用 `python -m tools.模块名` 从根目录启动。当前 `.venv`、模型权重、素材和用户工具配置保留；一次性安装验证环境、下载缓存和过时脚本已清理，释放约 6.84 GiB。目录迁移与删除说明见 [docs/README.md](docs/README.md)。

## 验证与评估

```powershell
# 不需要 GPU 或下载模型；其中 4 项测试会创建隐藏 Tk 窗口，需要图形环境
.venv/Scripts/python.exe -X utf8 -m unittest discover -s tests -v
# 连续输入验收：不插入停笔等待
.venv/Scripts/python.exe -X utf8 -m tools.continuous_preview
.venv/Scripts/python.exe -X utf8 -m tools.continuous_preview --gpu --offline
# 实际 Tk 图片更新 + GPU（隐藏窗口，连续模拟落笔）
.venv/Scripts/python.exe -X utf8 -m tools.gui_smoke --offline
# GPU 性能与适配器接口回归（零增量 LoRA 不代表角色一致性）
.venv/Scripts/python.exe -X utf8 -m tools.benchmark --offline --verify-lora
.venv/Scripts/python.exe -X utf8 -m tools.runtime_gpu --offline
# 综合调度压力测试
.venv/Scripts/python.exe -X utf8 -m tools.stress_renderer --seconds 180
# 真实手绘/连续线稿快照评估；--input 可重复并按传入顺序记录
.venv/Scripts/python.exe -X utf8 -m tools.evaluate_quality --offline --input "实际线稿路径.png"
# 模板须先填写有效本地 LoRA 路径
.venv/Scripts/python.exe -X utf8 -m tools.evaluate_quality --offline --profile "profiles/my-character.json" --input "实际线稿路径.png"
```

以下为迁移前 SD1.5 的历史成绩，不代表 SDXL 性能。

2026-09-19 连续输入 GPU 验收：12 秒、256 次快照更新、11 张可见预览，最大间隔 1.125 秒，无拦截。这是已预热模型的调度器加轮询测试，不包含完整鼠标/Tk 显示耗时，不代表任何机器都达到此速度。

追加实际 Tk + GPU 验收：12 秒、255 次模拟落笔、11 次图片更新，最大更新间隔 1.297 秒，固定参考保存检查通过。此测试使用隐藏窗口，包含 Tk 图片对象更新与事件循环，不含鼠标硬件和屏幕物理延迟。

以角色/画风一致性、人数/姿态响应、连续稳定性、参考价值和反馈延迟为验收标准，详见 [验收方案](docs/ACCEPTANCE.md)。历史 Lineart/Scribble 实验保留在 [报告目录](docs/README.md)，其中发现的构图失控和细节变化仍未证明解决；此轮调度和界面改进不等于模型画质提升。
