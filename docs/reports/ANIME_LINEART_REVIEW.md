# 动漫专用 Lineart 对照 — 2026-09-13

本轮下载并验证 `lllyasviel/control_v11p_sd15s2_lineart_anime`，仅以用户持剑线稿为困难样本。12次正式生成中9张可见、3次拦截；可见结果均已查看，仍有巨脸、位置错乱和线条装饰化，未达到接入GUI的标准，因此没有扩大到另外两张图。

## 设置与边界

模型来自[作者仓库](https://huggingface.co/lllyasviel/control_v11p_sd15s2_lineart_anime)，缓存于 `artifacts/model-cache`，快照 `9c03a1c4f055643faa78f5b6869189642b5b8f66`。下载约1.45GB权重及配置；之后离线推理，用户图片未上传，原图未修改。

基础模型保持 Counterfeit，输入等比例居中512×512、白底黑线，不额外提线；控制强度1.1、结束比例1.0、默认提示词、无LoRA。种子42/123/2026，预览10步与精细25步。

作者示例还将文本编码器截为11层。本轮先不改变现有文本编码，再测试 Diffusers 的 `clip_skip=1`，读取倒数第二层正向文本表示。后者不是作者示例的完整复现：本机实现只对正向编码应用 clip_skip，负向编码路径未同样截层，且基础模型也不同。因此结果限于以下两套具体配置，不宣称官方方案已被完整复现。

| 组别 | 正式次数 | 可见 | 拦截 | 观察 |
| --- | ---: | ---: | ---: | --- |
| 动漫Lineart，原文本编码 | 6 | 5 | 1 | 人物轮廓与剑成为装饰纹路，原位置的脸未正确生成，背景出现巨脸 |
| 动漫Lineart，clip_skip=1 | 6 | 4 | 2 | 颜色/脸形有所变化，但巨脸和线条装饰化仍在，没有稳定改善 |

拦截仅作为不可见状态记录，不判断原图或误判。两组只有一张真实输入，不能推断所有动漫线稿或其他基础模型的效果。

## 代码与验证

- 管线支持 `control_type="lineart_anime"`，绑定白底输入；默认仍为 Scribble。
- `control_cache_dir` 仅影响控制模型缓存，不要求重新下载基础模型。
- 评估脚本新增动漫模型选择、`--control-cache-dir` 和实验 `--clip-skip 1`；指标保存类型、完整模型ID、缓存路径及编码选项。
- GUI没有更换模型，默认文本编码未修改；51项单元测试通过，新增模型映射、输入保留及单次clip参数不残留的验证。

## 证据与复现

原文本编码：[seed42](../../artifacts/evaluations/quality-anime-cheng-20260913/seed-42-expression-0/comparison.png)、[seed123](../../artifacts/evaluations/quality-anime-cheng-20260913/seed-123-expression-0/comparison.png)、[seed2026](../../artifacts/evaluations/quality-anime-cheng-20260913/seed-2026-expression-0/comparison.png)。

clip_skip=1：[seed42](../../artifacts/evaluations/quality-anime-cheng-clip1-20260913/seed-42-expression-0/comparison.png)、[seed123](../../artifacts/evaluations/quality-anime-cheng-clip1-20260913/seed-123-expression-0/comparison.png)、[seed2026](../../artifacts/evaluations/quality-anime-cheng-clip1-20260913/seed-2026-expression-0/comparison.png)。各目录保留metrics.json及原尺寸图。这里 external-01 就是持剑图，不是阿福。

```powershell
.\.venv\Scripts\python.exe -X utf8 evaluate_quality.py --offline --control-type lineart_anime --control-cache-dir artifacts/model-cache --follow-sketch --input "C:/Users/wyf15/Desktop/板绘/陈/cheng线稿.jpg" --seed 42 --seed 123 --seed 2026
# 第二组另加 --clip-skip 1
```

下一步建议检查共同的基础模型、实际条件图和推理配置，再决定更换方案。Scribble、通用Lineart和动漫Lineart有相似巨脸现象，提示共同环节值得检查，但还不能据此断言基础模型或某处代码有错。不继续无目标地更换控制模型或将失败候选加入GUI。
