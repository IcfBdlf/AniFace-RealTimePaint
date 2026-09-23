# 多线稿画质评估 — 2026-09-12

新增三种种子、四种表情预设的 48 次生成评估，见 [EXPRESSION_REVIEW.md](EXPRESSION_REVIEW.md)。下文保留首轮单种子实验。

> 后续实现：已加入表情预设，并透传模型检查器的拦截状态；GUI 保留上一张有效图且显示提示。下文实验观察保留原样，其中“未透传状态”描述的是评估时版本。新的评估脚本会记录 blocked 状态并生成标注占位图，不再把拦截结果作为正常图片。

## 结论

当前管线能将简笔头像转换为动漫头像，但不能可靠地仅凭线稿还原表情，也不能保证预览与精细结果的人物细节一致。本轮不更改应用默认参数。

四类合成线稿、固定 seed=42，进行了三组实验，共保存 18 张输出，其中 2 张被模型自带安全检查器替换为黑图。其余输出均已查看。以下为人工观察，不是量化质量评分，不能推断所有线稿、种子或 LoRA 的表现。

## 实验设置

- RTX 4080 Laptop GPU，Python 3.10.20，PyTorch 2.5.1+cu121，Diffusers 0.24.0。
- 与应用共用 create_pipelines/render_sketch，512×512，ControlNet 强度 1.1，预览 10 步、精细重绘 25 步。
- 基础提示词为 config.DEFAULT_PROMPT；仅第三组附加 `closed eyes, smiling`。
- 每组先预热一次；各线稿/档位只测一次，耗时仅描述本次运行，不作为稳定性能统计。
- 全部使用缓存模型离线执行。样本由代码绘制，未使用素材目录的 sample_sketch.png：该文件实际是彩色成图，不是线稿。

## 第一组：当前默认配置，控制结束比例 0.35

[对比图](../../artifacts/evaluations/quality-20260912/comparison.png) · [原始指标](../../artifacts/evaluations/quality-20260912/metrics.json)

| 线稿 | 实际观察 | 判断 |
| --- | --- | --- |
| 正脸 | 两档均为正面头像；瞳色、脸部比例、光影和嘴形有变化，线稿的微笑未明显保留 | 大构图可用，表情与细节一致性不足 |
| 偏右窄脸 | 输出脸形和构图发生变化，但原本正脸草图被解释为偏侧角度；两档发饰、眼睛与脸形不同 | 能影响生成，但不是严格几何跟随 |
| 闭眼笑脸 | 两档均画成睁眼，精细结果还改变了发型装饰和服装 | 仅靠简笔闭眼线条不足以表达目标表情 |
| 粗糙重复线条 | 预览返回全黑，日志同时报告安全检查器替换输出；精细图正常，但头顶出现横向条纹 | 不能把黑图算作质量通过；粗糙线条下还存在图像瑕疵 |

预览 0.935–0.958 秒；精细 2.143–2.184 秒；峰值已分配约 3983 MiB。

## 第二组：仅将控制结束比例提高到 1.0

[对比图](../../artifacts/evaluations/quality-20260912-control-full/comparison.png) · [原始指标](../../artifacts/evaluations/quality-20260912-control-full/metrics.json)

相同四类输入、提示词、种子和强度，只有控制结束比例改变。值仅覆盖评估进程内的配置，不写回 config.py。

- 闭眼样本仍为睁眼，未解决表情问题。
- 正脸图的口鼻附近出现明显白色线条，精细图头部两侧出现横向条纹。
- 粗糙线条样本的预览可以显示，但精细结果被检查器替换为黑图；因此不能说延长控制解决了黑图问题。
- 本轮没有证据支持把默认比例直接从 0.35 改为 1.0。

预览 0.944–0.962 秒；精细 2.137–2.297 秒；峰值已分配约 3994 MiB。

## 第三组：闭眼样本配合明确表情提示词

[对比图](../../artifacts/evaluations/quality-20260912-closed-prompt/comparison.png) · [原始指标](../../artifacts/evaluations/quality-20260912-closed-prompt/metrics.json)

恢复控制结束比例 0.35，只使用闭眼线稿，在原提示词后附加 `closed eyes, smiling`。

- 预览和精细结果均呈现闭眼、露齿笑容，较第一组更符合目标表情。
- 露齿笑比线稿的简单微笑更强烈；精细图新增花饰，人物细节仍会变化。
- 这验证了这一输入/种子上显式提示词的作用，尚不能认为系统已经自动识别闭眼表情。

预览 1.018 秒；精细 2.241 秒。两张均正常返回。

## 后续建议

1. 优先验证表情提示词预设，如闭眼、微笑、惊讶；让用户明确选择，不能宣称从线稿自动识别。
2. 将管线返回的安全检查状态传到 GUI，保留上一张有效图并解释未产生可显示结果，避免把黑图当成功。当前 render_sketch 只返回 images[0]，未透传该状态；本轮未修改这段行为。
3. 在多个种子和真实手绘样本上重复表情实验，再决定预设和默认值；扩展到侧脸、遮挡等输入。
4. 人物一致性需单独评估；固定种子和同一模型并不保证两种步数的输出是同一个人物。

检查器日志与黑图对应，但不可见的原图内容未被检查，因此不判断其拦截是否准确。统计字段 black_output 仅表示全黑像素，不单独用于推断拦截原因。

## 复现

```powershell
.\.venv\Scripts\python.exe -X utf8 evaluate_quality.py --offline
.\.venv\Scripts\python.exe -X utf8 evaluate_quality.py --offline --control-end 1.0
.\.venv\Scripts\python.exe -X utf8 evaluate_quality.py --offline --case 03-closed-eyes --prompt '1girl, masterpiece, hyper detailed, anime portrait, high quality, sharp focus, closed eyes, smiling'
```

默认创建带时间戳的输出目录；--output 可指定目录，但拒绝覆盖非空目录。输出含原始线稿、各档图像、comparison.png 和 metrics.json。artifacts 被 Git 忽略，本报告的图像链接指向本机产物；新机器需重跑脚本生成。
