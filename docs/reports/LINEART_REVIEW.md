# Lineart 控制模型对照 — 2026-09-13

后续已完成动漫专用版本的持剑图验证，见 [ANIME_LINEART_REVIEW.md](ANIME_LINEART_REVIEW.md)。下文“未验证动漫版”为本轮历史状态。

结论：本机缓存的通用 Lineart 模型在当前基础模型、提示词和输入尺寸下，没有解决真实细线稿的脸部错位问题；不替换 GUI 当前 Scribble 模型。结果不代表所有 Lineart 模型或配置都无效。

## 模型与输入

对照模型为 `lllyasviel/control_v11p_sd15_lineart`，与专用动漫版本 `control_v11p_sd15s2_lineart_anime` 不同。本次使用已有完整缓存，未下载新模型。

根据[作者模型卡](https://huggingface.co/lllyasviel/control_v11p_sd15_lineart)，Lineart 条件通常是白底黑线；因此切换模型时同时改用白底灰度线稿，不沿用 Scribble 的反色。输入是用户已有手绘线稿，未额外运行照片提线器。模型卡推荐与 SD1.5 搭配；本轮固定现有 Counterfeit 基础模型，结论限于这一组合。

新增 `create_pipelines(control_type="lineart")` 和模型绑定的输入处理；默认仍为 scribble。评估脚本支持 `--control-type lineart`，指标记录模型 ID、控制类型及白底/黑底约定。没有增加 GUI 模型切换，也没有修改用户原图、默认提示词、分辨率或检查器。

## 实验与观察

使用阿福、cheng线稿、超跑T草稿，缩放方式同前轮；512×512、控制强度 1.1、默认提示词、预览10步及精细25步、无LoRA。

| 组别 | 种子 | 次数 | 可见 | 拦截 |
| --- | --- | ---: | ---: | ---: |
| Lineart 全程控制，结束比例1.0 | 42/123/2026 | 18 | 14 | 4 |
| Lineart 短时控制，结束比例0.35 | 42 | 6 | 5 | 1 |

19张可见结果均查看。全程控制能留下部分原稿轮廓，但大量线条呈凸起装饰/浮雕纹理，持剑人物脸部仍不正常，背景还有巨大眼睛或脸；双人部分保留人数，却不能视为正常上色。缩短控制后，这类强纹理减少，但又回到头像近景，双人两档均只剩一人，原姿势丢失。

因此两种结束比例均未达到目标。检查器未放行的5次只记录拦截，不判断不可见原图或误判。全程组与上一轮 Scribble 跟随模式的三个种子、两档设置相同，但更换模型必然同时更换适配输入；不能将全部差异单独归因于某一个网络参数。

## 证据与复现

- 全程组：[seed42](../../artifacts/evaluations/quality-real-lineart-20260913/seed-42-expression-0/comparison.png)、[seed123](../../artifacts/evaluations/quality-real-lineart-20260913/seed-123-expression-0/comparison.png)、[seed2026](../../artifacts/evaluations/quality-real-lineart-20260913/seed-2026-expression-0/comparison.png)。
- [短时控制组](../../artifacts/evaluations/quality-real-lineart-short-20260913/comparison.png)。
- 各目录 `metrics.json` 保存完整设置和逐张结果；默认 Scribble 对照见 [真实线稿报告](REAL_SKETCH_REVIEW.md)。

```powershell
.\.venv\Scripts\python.exe -X utf8 evaluate_quality.py --offline --control-type lineart --follow-sketch --input "C:/Users/wyf15/Desktop/板绘/陈/cheng线稿.jpg" --seed 42 --seed 123 --seed 2026
```

50项单元测试通过，新增验证 Lineart 白底、浅线保留、不修改源图、未知类型拒绝，以及渲染按管线类型选择输入极性。24次实际GPU生成均执行完成。

## 下一步

尚未验证作者的动漫专用 Lineart 模型；下一轮可将它作为独立候选，先用持剑图做小规模对照，再决定是否扩展三张原稿。也需保留基础模型、提示词和512尺寸对结果的影响这一可能，不能因通用 Lineart 失败就断言是用户线稿问题。当前不把未经验证的候选接入 GUI 默认流程。
