# AniFace 最新交接 — 2026-09-23

## 当前任务：旧版已备份，SDXL 迁移完成并验证

用户授权先上传现版本，再换 SDXL。旧版已提交并推送至 GitHub `IcfBdlf/AniFace-RealTimePaint` 的 main：`780f1cb`；永久标签 `backup/pre-sdxl-20260923` 同步成功。当前迁移分支 `codex/sdxl-migration`。不删除旧模型缓存、不升级依赖，main 保留回退版本。

已下载固定 revision 的 Animagine XL 3.1 和 Xinsir SDXL Scribble，切换 XL 管线/Euler a/CFG 6，操作画布 512、生成保存 1024，默认 CPU offload + VAE tiling。旧 SD1.5 控制模型拒绝混用，示例角色预设更新。新参数写入 PNG recipe。详见 docs/SDXL_MIGRATION.md。

初步 GPU 输出通过，原 inference_mode + CPU offload 组合使 PEFT 回滚参数状态失败，已改用 no_grad，真实 GPU 回滚、取消恢复、调权重、替换与卸载全部通过。首次基准约 8.4 秒预览、15.4 秒 HQ，冷启动约 17.4 秒，不能沿用旧版一秒成绩。预览年龄预算设为 30 秒。隐藏 Tk 连续 40 秒、841 次输入、3 次图像更新、固定图保存通过。61 项测试包含 1024 保存/512 显示验证。真实“陈”线稿单种子生成通过：8.285 秒预览、15.735 秒精细，目视姿势/人物构图保留；不代表角色一致性验证。最终状态见 docs/reports/SDXL_VALIDATION.md。

旧面试手册和代码摘录已在顶部标注 SD1.5 历史快照，当前模型变化查 SDXL 迁移文档。下文是此前记录，其中“未 commit/push”已被本节备份事实取代。

## 最新：逐函数代码学习手册

用户希望具体学习每段代码。新增 docs/CODE_WALKTHROUGH.md，直接从当前源码摘录并讲解 76 个核心函数/方法（含 2 个内部回调）、全部 config 设置、模块初始化与数据类；附画一笔/切换 LoRA/保存参考的完整流程、代表性测试与工具解释、练习答案。源码定位链接和 SHA-256 用于检查摘录是否过时，后续业务代码改变时需同步。

当前 README、docs 导航与面试指南均已链接此手册。此轮仅修改学习文档，不改变业务逻辑；并非重新运行 GPU 或宣称新增模型效果。

## 最新：按当前源码重写面试指南

用户不再需要旧版实现说明。已删除 docs/legacy/PROJECT_GUIDE.md 和 docs/history/README_20260913.md，移除空 legacy 目录；实验报告和历史交接作为事实记录保留。

新文档 docs/INTERVIEW_GUIDE.md 共 22 章，基于当前源码与固定依赖讲解产品边界、Python/Tkinter/Pillow、PyTorch/CUDA、Stable Diffusion/CLIP/U-Net/VAE、Diffusers/ControlNet/UniPC、LoRA/PEFT、线程与 epoch/version、错误恢复、可追溯 PNG、测试指标、环境和面试问答。包含 30 秒/2 分钟介绍、简历模板和 25 个追问；明确区分接口验证与真实角色一致性，不使用旧 StreamDiffusion 架构冒充当前实现。README/docs 导航已更新。此次未改应用逻辑。

## 最新：文件清理与产物归档

用户随后明确授权整理目录并删除无用内容。已删除 artifacts/pip-cache、用于独立安装验证的 artifacts/clean-env、7 个过时一次性脚本，以及根目录/src/tests/tools 的 __pycache__；共释放约 6.835 GiB。当前 .venv 使用 conda 基础解释器，不依赖被删环境。保留模型权重、素材、用户配置、Git 和所有有效评估/回归结果。

artifacts 下产物归为 checks、evaluations、environment、model-cache、logs；工具默认路径及文档链接已更新。旧文档提到的临时环境和一次性脚本不再存在；历史指标中的原始绝对路径仅表示实验当时的位置。删除/移动明细在 artifacts/environment/cleanup-20260919.json、moves-20260919.json。下文原先“不搬动产物”的描述被本次明确清理要求取代。

清理后重新运行 58 项测试全部通过，6 个工具入口 --help、文档本地链接及迁移后的测试预设依赖检查通过；artifacts 从约 8.31 GiB 降到 1.473 GiB。

## 先读目标

用户明确：固定一个角色或一种画风，由用户手绘线稿控制姿势、构图、表情，实时生成参考，辅助同人创作和绘画初学者。不是以逐像素忠实上色为主要目标。

本轮用户授权按审查顺序修改并整理文件夹。已完成可独立推进的调度、交互、预设流程、保存和目录整理；尚未得到真实角色/画风及 LoRA 路径的回答，不能声称完成角色一致性验证。没有 commit/push，原有未提交修改保留。

## 本轮代码

- 普通落笔 `invalidate(soft=True)`：取消过期 HQ、替换待处理输入，允许同 epoch 的近期预览显示；配置变化/清空/LoRA/退出使旧 epoch 失效。预览最大年龄默认 5 秒，结果版本单调显示。控制参数和提示词改变也会自动分离 epoch。
- GUI：默认手动 HQ，可开启停笔自动 HQ；暂停生成、固定参考、12 张历史、橡皮擦/笔刷大小、缩放/平移。
- `src/profiles.py`：本地 JSON 预设，匹配基础模型，固定触发词与自由提示词分开。LoRA 成功后才提交 GUI 预设；失败保留旧预设。同文件调权重不重新加载，也不覆盖用户当前构图提示词。
- LoRA 卸载、实际生效状态；GPU 仍由单工作线程串行访问。
- `src/references.py`：当前显示图片保存为 PNG，内嵌该张图对应的线稿和参数；不是用保存时的当前输入冒充生成输入。历史及固定参考也可保存。
- 评估工具支持 `--profile`，输出人工评价字段，不自动给画质打分。
- GPU 回滚像素一致性从仅记录布尔值改成断言；扩展真实同文件调权重/卸载回归。

## 整理后的入口

主程序 `canvas_stream.py`，默认配置 `config.py`。源码 `src`，测试 `tests`，CLI `tools`，预设 `profiles`，文档/报告 `docs`。所有 CLI 从根目录以 `python -m tools.xxx` 运行，详见 README 和 docs/README.md。

旧实现指南与旧 README 已按用户要求删除，旧交接保留在 docs/history。artifacts 的最新归档方式见上方清理记录。旧根日志移至 artifacts/logs/aniface-before-20260919.log，新日志写 artifacts/logs/aniface.log。用户 .claude/.vscode 配置未改。

## 验证

- `.venv/Scripts/python.exe -X utf8 -m unittest discover -s tests -q`：58 项通过，含 4 项实际隐藏 Tk 窗口测试。
- `tools.continuous_preview --gpu --offline --seconds 12`：256 次输入更新、11 张绘画期间预览、最大间隔 1.125 秒，无拦截。指标 artifacts/checks/continuous-gpu-20260919/metrics.json；范围是预热模型 + 调度器 + 轮询，不含完整 Tk 显示链路。
- `tools.runtime_gpu --offline`：最终通过，结果 artifacts/checks/runtime-20260919-final/metrics.json。首次运行发现旧测试假定同文件重载产生新适配器；测试已改为分别验证权重更新和真正换文件，首次产物保留在 artifacts/checks/runtime-20260919 以供追溯。
- 模拟综合压力 15 秒通过：572 次请求，21 张可见结果，最大模型并发 1；artifacts/checks/stress-20260919/metrics.json。这是短时回归，不是长时间稳定性证明。
- `tools.evaluate_quality --profile` 真实 GPU 接口冒烟通过：零增量测试预设生成预览/精细各一张，最终提示词、预设及权重均写入 artifacts/checks/profile-evaluation-20260919/metrics.json；不是角色一致性验证。
- 用户途中要求“继续”，追加 `tools.gui_smoke --offline --seconds 12` 通过：真实 Tk + GPU，255 次模拟落笔、11 次图片更新、最大间隔 1.297 秒、事件循环最大间隔 0.032 秒，固定参考保存与其原参数一致。隐藏窗口，不含硬件输入/屏幕物理延迟；artifacts/checks/gui-gpu-20260919/metrics.json。6 个 tools 入口 --help 与所有文档链接检查通过。

## 下一步

等待用户指定角色或画风及兼容 LoRA；先用 profiles/example.json 填写真实预设，再按 docs/ACCEPTANCE.md 做同角色连续线稿、小幅改笔、表情和构图验收。模板不能直接作为真实角色模型。不要再把零增量 LoRA 接口通过写成角色一致性通过。

本轮没有从算法上解决跨帧漂移或历史巨脸/构图失控。固定参考提供交互上的稳定使用方式。模型文件版本与 LoRA 内容尚未完全锁定，保存参数不保证跨机器像素复现。

三张用户原图和历史实验详情见 [历史交接](docs/history/SESSION_HANDOFF_20260913.md)，原文件未改，不重复索要。历史文档中的“下一步换模型”等建议已被本文件和新验收目标取代。
