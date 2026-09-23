# AniFace 会话交接 — 2026-09-11

## 最新：动漫专用 Lineart 已验证，未接入GUI（2026-09-13）

用户要求下一步，已从作者HF仓库下载 control_v11p_sd15s2_lineart_anime（约1.45GB）到 artifacts/model-cache，snapshot 9c03a1c4f055643faa78f5b6869189642b5b8f66。无需再次下载。src.pipeline新增ANIME_LINEART_MODEL_ID/control_model_id、control_type=lineart_anime、control_cache_dir、render_sketch可选clip_skip；evaluate_quality支持对应参数并记录。默认GUI/Scribble未改。

持剑原图三种子两档，原文本编码6次（拦截1）+clip_skip=1六次（拦截2），目录artifacts/evaluations/quality-anime-cheng-20260913及quality-anime-cheng-clip1-20260913，external-01为陈。9张可见均已查看，仍巨脸/纹路装饰化，因此未扩展另外两张。51项unittest通过。详见 ANIME_LINEART_REVIEW.md，包含官方方案与本轮clip_skip非完全等价的限制；未commit/push，原图未改。

下一步应检查共同的基础模型、实际条件图、推理配置，避免无目标继续换模型。三种控制模型有相似巨脸不能证明单一原因；尤其当前是Counterfeit+默认单人头像提示词+512缩放，尚未复现官方完整基础模型方案。后文“未下载动漫版”已过时。

## 2026-09-13 最新进度（先读）

用户要求下一步，已比较本机缓存通用 Lineart：lllyasviel/control_v11p_sd15_lineart，非 lineart_anime。作者模型卡确认白底黑线。src.pipeline 新增 LINEART_MODEL_ID、create_pipelines(control_type=...)、prepare_control_image第二参数；类型绑定pipe._aniface_control_type，render_sketch读取，默认Scribble保持反色。evaluate_quality.py支持 --control-type lineart，指标记录类型/极性。GUI尚未加入模型选项。

实际GPU24次：artifacts/evaluations/quality-real-lineart-20260913（三原稿×三种子×两档，全程1.0）18次拦截4；artifacts/evaluations/quality-real-lineart-short-20260913（三原稿×seed42×两档，0.35）6次拦截1。19张可见图全部查看；全程有浮雕纹理和错脸，短时回头像并丢失双人。因此没有替换GUI当前模型。50项unittest通过，详见LINEART_REVIEW.md。原图未改，未commit/push。

下一步可验证作者动漫专用 control_v11p_sd15s2_lineart_anime（本机未缓存，本轮未下载），先持剑图小对照再扩展，不要把通用版本结果冒充动漫版本。也需考虑现有基础模型/提示词/512缩放的影响，尚不能归因单一原因。用户三原稿已提供，路径见后文，不要重复索要。

## 2026-09-12 续接进度（优先阅读）

### 最新：尽量跟随线稿模式完成

用户要求下一步，已实现 src.app 的 follow_sketch_var 开关，默认 False，开启通过 control_end=1.0 传到 RenderManager/RenderRequest/render_sketch；关闭不覆盖原默认0.35。模式纳入输入identity和版本取消，切换trace复用_on_prompt_changed。旧五参数render_fn默认路径兼容，仅有覆盖值时传新增关键字。evaluate_quality.py 新增 --follow-sketch（优先于 --control-end）。

48项测试通过；artifacts/follow_gui_smoke.py 实际隐藏Tk开关、真实worker+模拟推理验证通过。真实GPU评估 artifacts/evaluations/quality-real-follow-mode-20260912：三原稿×种子42/123/2026×两档18张均可见，已全部查看。双人六张均保留两人；阿福seed123/2026及陈全部种子仍有巨脸错位。详见 REAL_SKETCH_REVIEW.md 最新段落。不能宣称忠实上色问题解决。

下一步优先受控比较细线稿控制模型/输入处理和分辨率，以持剑图为困难样本；不要重复索要三张原稿，原路径见下段。未commit/push，用户源图未修改。

### 最新：三张用户真实线稿评估完成

用户已提供 C:/Users/wyf15/Desktop/板绘/阿福.jpg、板绘/陈/cheng线稿.jpg、板绘/6月新番/超跑T（草稿）.jpg，已读取并实际评估。不要再说没有真实手绘或要求重复提供。前两张 3507×4960、第三张 1280×1024，原图只读未改。

详见 REAL_SKETCH_REVIEW.md。默认三种子×三输入×两档 18 次；seed42追加“仅控制延长到1”“仅中性插画提示词”“两项结合”各6次，共36次。31张可见输出均查看，5次拦截。artifacts/evaluations/quality-real-default-20260912、quality-real-control-full-20260912、quality-real-neutral-0.35-20260912、quality-real-neutral-1-20260912 保存指标原图和对比图，源文件尺寸/hash在默认目录source-manifest.json。

默认配置把全身变头像，双人可见结果都只剩单人；控制延长可恢复阿福轮廓及双人人数动作，但持剑图错位巨大脸；去掉单人头像约束单独无效，两项结合仍有错脸/姿势和拦截。未改应用默认参数，未commit/push。

建议下一步做可选“尽量跟随线稿”设置，参数随请求传递、沿用串行取消，三种子回归后再决定默认；进一步验证细线控制模型/分辨率，但尚未下载其他模型。真实需求更接近原稿保留/上色，应优先构图、人数、姿势，不再只调表情。后文“无真实线稿”均为历史状态。

### 最新：惊讶提示词对照完成，未替换预设

用户要求下一步，针对惊讶做三候选×两线稿×三种子×两档共 36 次追加 GPU 生成。详见 SURPRISE_REVIEW.md；artifacts/evaluations/surprise-surprised-only-20260912、surprise-parted-lips-20260912、surprise-facial-cues-20260912 保留 metrics、原图和总览。分别拦截 9/12、6/12、4/12，17 张可见结果均已查看。

surprised 单标签可用性差；surprised, parted lips 较温和且保持彩色画风，但默认 seed=42 两类线稿预览均拦截；raised eyebrows, wide eyes, parted lips 的可见结果常偏生气。没有稳妥候选，因此 src/prompts.py 仍是 surprised, open mouth，不能声称已优化成功。只更新报告/文档，不重复跑应用单元测试，前轮 45 项通过。本轮 GPU 实验均退出码 0。未 commit/push。

下一步宜补真实使用线稿并明确温和惊讶还是夸张漫画表情，避免仅针对两类合成脸反复调参。项目目前没有可用真实手绘；assets/sample_sketch.png 是彩色成图。也可转向有证据支持的其他功能改进，不要自动重跑本轮实验。

### 最新：多种子表情画质评估完成

用户要求下一步。扩展 evaluate_quality.py 支持 --seed/--expression/--input（均可重复），共用一次模型加载，分组对比图，外部图片复用 src.app.fit_sketch。新增 test_quality.py，全套 45 项无 GPU 测试通过。

已实际运行两类合成线稿（01-front、03-closed-eyes）、种子 42/123/2026、四种现有预设、两档共 48 次。artifacts/evaluations/quality-expression-matrix-20260912/ 保存逐条 metrics、12 组 comparison 和三张 overview。31 张可见结果均已查看；17 次拦截：不附加 3、微笑 2、闭眼微笑 6、惊讶 6。详见 EXPRESSION_REVIEW.md。未改应用推理默认值/表情标签。

关键发现：无表情时闭眼线稿依然睁眼；微笑有效，闭眼微笑可见结果有效但不能忽略被拦截的一半；种子 2026 闭眼微笑出现镜框式构图；惊讶可见结果偏夸张惊恐和漫画画风。下一步建议针对惊讶标签做受控对比，验证减轻表情夸张和画风漂移后再替换预设。真实手绘需要实际样本，当前资产 sample_sketch.png 是彩色图，未算手绘。不要声称验证了真实手绘或拦截误判；未 commit/push。

### 最新：独立安装与连续操作验证完成

用户要求继续下一步，已完成同机独立安装和短时连续操作验证，详见 ENVIRONMENT_STRESS_REPORT.md。artifacts/clean-env 的 include-system-site-packages=false，33 个安装包均在环境内；原 .venv 和 conda 未修改。旧 pip 23.0.1 安装 PyTorch 索引依赖失败，升级新环境 pip 至 26.2.1 后成功，README 已补升级步骤。

独立环境 pip check、44 项 unittest、test_runtime_gpu.py --offline、artifacts/gui_smoke.py 均通过；GPU 第二步取消约 0.196 秒，损坏 LoRA 回滚原权重和图像一致、成功替换释放旧适配器通过。GUI 包含模拟拦截恢复、启动失败重试、加载中关闭、真实模型启动后进入画板并生成（主线程心跳 317 次）。故意触发的错误日志符合预期。

新增 stress_renderer.py。模拟 180 秒：6899 渲染请求、1544 LoRA 请求、6126 取消；GPU 120 秒：252 渲染请求、60 LoRA 请求、192 取消。均未发现消费旧帧、模型操作重叠或关闭后渲染线程残留。证据路径见报告；artifacts 被忽略，未 commit/push。

下一步建议真实手绘、多种子和表情预设画质评估；社区 LoRA 需实际样本。数小时稳定性、另一台机器安装仍未验证，内存采样不能证明无泄漏。以下保留历史状态，独立环境验证已完成，不要重复安装。

### 最新：过时推理取消与 LoRA 回滚

用户要求继续改进，已完成两项。renderer 的取消回调比较请求版本与关闭状态，pipeline 在去噪步骤边界及返回后检查；相同输入预览升级精细重绘保留原预览。LoRA 改为保留旧适配器、独立命名加载新的、成功启用后删除旧的；失败从内存恢复旧适配器和权重。恢复失败暂停推理，重试加载会先清理模型；旧适配器清理失败单独提示并延后重试。GUI 持久显示 LoRA 失败/恢复通知，不被普通帧清除。

44 项单元测试；新增 test_runtime_gpu.py --offline。真实 GPU 结果在 artifacts/checks/runtime-20260912/metrics.json：第二步取消成功（约 0.217 秒，仅该次调用）、取消后正常生成、损坏适配器回滚原权重 0.5、回滚前后像素一致、成功替换后旧适配器删除均通过。使用本地零增量 LoRA，社区风格验证、独立环境复现、长期操作压力测试仍未完成。后面的交接段落保留历史状态。

### 最新：启动加载页已完成

用户同意优先改进启动体验。新增 src/startup.py（ModelLoader + StartupWindow），入口先创建 Tk 窗口再后台加载模型，提供阶段提示、错误详情、重试和关闭。create_pipelines 新增可选 on_progress/should_stop；关闭在第三方加载调用结束后的阶段边界生效，不能立即中断下载。后台不主动 gc.collect，避免跨线程销毁 Tk 对象；重试前在主线程回收。35 项单元测试通过；扩展 artifacts/gui_smoke.py 验证失败重试、加载中关闭、真实缓存 GPU 模型加载后进入画板并成功生成图，事件循环保持响应。本轮只处理启动体验，过时推理提前取消、LoRA 原风格回滚和独立环境验证仍待后续。

### 最新：表情预设与拦截提示已实现

验证补充：真实 GPU 结果在 artifacts/evaluations/quality-20260912-blocked-status/，粗糙样本预览 blocked=true、精细图正常。已扩展并通过 artifacts/gui_smoke.py，验证表情组合、被拦截时保留结果、正常帧清除提示和安全关闭。尚未对微笑/惊讶预设做多样本画质验证。

用户接着要求“下一步”，已落实表情预设和检查器结果状态处理。src/prompts.py 提供不附加表情、微笑、闭眼微笑、惊讶；GUI 下拉框通过提交时组合提示词同步两档，不改写手动提示词。pipeline 根据 nsfw_content_detected 抛出 RenderBlocked，renderer 转为带版本号的 blocked 事件，GUI 提示并保留上一有效图；成功结果清除旧提示。evaluate_quality.py 支持记录 blocked 与生成诊断占位图。26 项无 GPU 测试通过。下述“未修复”是前一阶段历史记录，现已处理。

- 用户已选择先做“多种线稿画质评估”。已完成四类合成线稿、三组实验，共 18 张输出，详见 QUALITY_REVIEW.md。
- 新增 evaluate_quality.py：复用应用管线，支持 --offline、--control-end、--case、--prompt，时间戳目录输出且拒绝覆盖非空目录。
- 产物：artifacts/evaluations/quality-20260912/（默认 0.35）、artifacts/evaluations/quality-20260912-control-full/（1.0）、artifacts/evaluations/quality-20260912-closed-prompt/（0.35 + closed eyes, smiling）。每处均有 comparison.png 和 metrics.json。
- 默认配置能生成头像但闭眼线稿仍生成睁眼；全程控制也未解决，并出现更多线条残留。加表情提示词后两档均呈现闭眼笑容，但发饰等细节仍变化。
- 粗糙样本：默认组预览、全程组精细图分别被安全检查器替换为黑图；日志明确报告替换，不能判断未显示原图是否应拦截。应用目前只返回 images[0]，未向 GUI 透传检查状态，尚未修复。
- 本轮未改 config.py 或应用核心代码，只新增评估脚本与报告，更新 README、CHANGELOG、本交接记录。
- 建议后续：表情预设验证、GUI 正确处理被替换结果、多种子/真实手绘扩展。不要声称已解决人物一致性或已完成大样本画质评测。

以下为昨天的交接记录，保留作为历史背景。

## 用户目标与当前暂停点

用户先要求阅读项目并提出意见，随后授权按建议修改。第一批核心问题已全部处理，第二、三批大部分也已完成。用户因今天 token 不足暂停，计划明天继续；未要求定时提醒或后台继续工作。

这是会话摘要，不是逐字对话备份。恢复时先读本文件、README.md、git status/diff，避免重复修改或测试。下一步尚未获指定具体任务，可先与用户确认想验证的方向。

## 已完成

- 移除没有实际调用 ControlNet 的 StreamDiffusion 预览路径。预览 10 步、精细重绘 25 步共用唯一 ControlNet 管线、UniPC 调度器、固定种子 42。
- 修正 control_guidance_end 参数；白底黑线在推理前转换为模型要求的黑底白线。精细重绘仍为 512×512，非超分辨率。
- RenderManager 单工作线程串行执行推理及 LoRA；各保留最新待处理图像/LoRA，结果消费时按版本过滤。相同输入升级高清允许预览先显示。
- GUI 主线程每 50ms 轮询结果，后台不调用 Tkinter；同步提示词，清空使旧结果失效。
- 停笔 800ms 后重绘、新笔取消计时；异步 LoRA、状态/错误提示、PNG 保存、撤销/重做、等比例/透明背景导入、坐标 0 绘画修复。
- 关闭时停止接收请求、在去噪步骤边界退出，等待 worker 结束后销毁窗口。
- 更新 README、CHANGELOG、requirements；PROJECT_GUIDE 加了历史实现提示，旧版学习内容保留。

## 验证证据

- 17 项无 GPU 单元测试通过：真实 RenderManager + 模拟模型，另有输入/推理契约/交互测试。
- GPU 基准复用 create_pipelines/render_sketch，离线缓存运行成功。
- RTX 4080 Laptop、Python 3.10.20、torch 2.5.1+cu121：预热后三轮预览 0.94–0.99 秒，精细 2.13–2.15 秒。峰值已分配约 3996 MiB，保留约 4556 MiB。不是端到端交互延迟。
- artifacts/checks/benchmark/metrics.json 和 sketch.png、preview-*.png、hq-*.png 已保存；实际查看两档图像，构图基本一致，细节不同。
- artifacts/checks/lora-validation/metrics.json：本地生成零增量 LoRA 的真实加载、0.5 权重设置、GPU 推理通过；未验证社区风格质量。
- artifacts/gui_smoke.py：真实隐藏 Tk 窗口 + 真实渲染器/模拟推理，绘画、提示词、显示、清空、撤销/重做、关闭通过。
- .venv 的 pip check 通过；语法解析通过。git diff --check 排除 PROJECT_GUIDE 后通过，该文件原有用户尾部空格未擅自清除。

## 环境与命令

项目：D:\code project\AniFaceProject。

项目内 .venv 用 C:/ANACONDA/envs/aniface_diff/python.exe 的 --system-site-packages 创建，复用原环境 PyTorch。经用户批准联网，将 numpy 1.26.4、accelerate 0.25.0、peft 0.7.1 安装在项目 .venv 中，原 conda 环境未修改。原环境有 NumPy 2/ONNX 警告、Accelerate/Hub 版本冲突且缺少 PEFT，因此继续工作使用 .venv。

```powershell
.\.venv\Scripts\python.exe -X utf8 canvas_stream.py
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -v
.\.venv\Scripts\python.exe -X utf8 test_stream.py --offline
.\.venv\Scripts\python.exe -X utf8 test_stream.py --offline --verify-lora
.\.venv\Scripts\python.exe -m pip check
```

.venv/ 和 artifacts/ 被 gitignore 排除；基准脚本默认写 artifacts/checks/benchmark，会覆盖同名产物。当前未 commit 或 push。

## 后续可选方向与限制

1. 全新独立环境/新机器安装复现（当前仅验证了复用本机环境的 .venv）。
2. 多种线稿的质量评估，比较控制强度、控制结束比例、步数；尚无系统画质评估集。
3. 用户实际社区 LoRA 的兼容性与风格表现；已有测试仅验证零增量适配器接口。
4. 真正的低延迟预览优化。目前明确为秒级 ControlNet 基线，不能声称毫秒级实时效果。

开始本轮工作前已有用户改动：src/renderer.py 两处 root.after 调用补了 0 参数（现由主线程轮询替代）；PROJECT_GUIDE.md 一处尾部空格；未跟踪的 .claude/。不要删除或回滚用户文件。
