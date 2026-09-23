# 惊讶预设受控对照 — 2026-09-12

结论：三种候选都未同时改善表情、画风和预览可用性，本轮不替换现有 `surprised, open mouth`。这不是确认原预设理想，而是尚无足够证据支持改成某个候选。

## 设置

新增三组、每组 12 次真实 GPU 生成：正脸/闭眼两类合成线稿，种子 42/123/2026，预览 10 步及精细 25 步。基础提示词、模型、控制参数和检查器均沿用前轮。原预设的 12 次结果复用前轮，未重新生成；本轮新增 36 次。所有可见候选结果已查看。

| 提示词后缀 | 可见/总数 | 被拦截 | 可见结果观察 |
| --- | ---: | ---: | --- |
| 原预设：surprised, open mouth | 6/12 | 6 | 夸张张嘴、惊恐感、低饱和漫画画风 |
| surprised | 3/12 | 9 | 种子 123 的两张可见结果较温和；种子 42 的一张仍明显张嘴，种子 2026 无可见结果 |
| surprised, parted lips | 6/12 | 6 | 嘴部和表情更收敛，彩色头像画风保留较好，但惊讶较弱，部分更像疑惑 |
| raised eyebrows, wide eyes, parted lips | 8/12 | 4 | 嘴部收敛、彩色画风保留，但多张眉头紧蹙，更像生气或质疑，不够符合惊讶 |

选择依据不能只看拦截数。眉眼候选虽然可见数量较多，表情语义却偏离目标。“嘴唇微张”候选更接近温和惊讶，但默认种子 42 的两类输入均出现预览被拦截、精细可见；原预设同种子的正脸预览可见。因此当前不直接替换。

17/36 次候选结果可见，19 次被拦截。不可见原图未判断，不能认定检查器误判，也未停用或调整检查器。数据是小规模固定组合，不是统计成功率；没有真实手绘或社区 LoRA 样本。

## 本地证据

- [仅 surprised 总览](../../artifacts/evaluations/surprise-surprised-only-20260912/overview.png) · [指标](../../artifacts/evaluations/surprise-surprised-only-20260912/metrics.json)
- [嘴唇微张总览](../../artifacts/evaluations/surprise-parted-lips-20260912/overview.png) · [指标](../../artifacts/evaluations/surprise-parted-lips-20260912/metrics.json)
- [眉眼描述总览](../../artifacts/evaluations/surprise-facial-cues-20260912/overview.png) · [指标](../../artifacts/evaluations/surprise-facial-cues-20260912/metrics.json)
- [原预设报告](EXPRESSION_REVIEW.md)

总览行依次为三个种子，列依次为正脸预览、正脸精细、闭眼线稿预览、闭眼线稿精细。每组还保留原尺寸图片、输入和 comparison.png。指标逐条保存最终完整提示词，文件位于被 Git 忽略的 artifacts 目录。

可使用评估脚本的现有参数复现，例如嘴唇微张候选（不额外叠加应用预设）：

```powershell
.\.venv\Scripts\python.exe -X utf8 evaluate_quality.py --offline --case 01-front --case 03-closed-eyes --seed 42 --seed 123 --seed 2026 --prompt "1girl, masterpiece, hyper detailed, anime portrait, high quality, sharp focus, surprised, parted lips"
```

原实验在独立评估进程内临时替换表情映射，未写回应用文件；上述命令的最终提示词和推理设置相同，只是指标的表情标签显示“不附加表情”。其余候选替换提示词末尾即可，输出默认创建新目录。

## 后续

先保留现有预设及其已知限制。进一步决定默认表情前，优先补实际使用线稿，并确认用户需要的是温和惊讶还是夸张漫画表情；不宜继续仅凭这两类合成脸反复调参。嘴唇微张可作为后续候选，但当前不宣称已经修复惊讶表现。
