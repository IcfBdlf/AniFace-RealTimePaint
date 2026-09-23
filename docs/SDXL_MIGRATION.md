# SDXL 迁移与回退

2026-09-23。此文件记录当前分支与 SD1.5 基线的区别；历史画质、性能报告不作为 SDXL 成绩。

## 已上传的回退点

- 仓库：https://github.com/IcfBdlf/AniFace-RealTimePaint
- SD1.5 提交：`780f1cb`，在 `main`。
- 标签：`backup/pre-sdxl-20260923`（已推送）。
- SDXL 开发分支：`codex/sdxl-migration`。
- 备份前 58 项测试通过。Git 备份包含源码、文档、测试、样例；不包含虚拟环境、模型缓存、生成图或本机工具权限配置，这些文件保留在本机。

工作区没有未提交修改时，回到旧版：

```powershell
git switch main
```

或从永久标签创建独立回退分支（即使以后 main 已更新也有效）：

```powershell
git switch -c restore/pre-sdxl backup/pre-sdxl-20260923
```

回到新版：`git switch codex/sdxl-migration`。切换前如有新改动，先提交或暂存；不使用 `reset --hard` 丢弃工作。

## 模型与参数

| 项目 | 当前 SDXL 配置 |
|---|---|
| 动漫底模 | `cagliostrolab/animagine-xl-3.1` |
| 线稿控制 | `xinsir/controlnet-scribble-sdxl-1.0` |
| Diffusers 管线 | `StableDiffusionXLControlNetPipeline` |
| 采样器 | `EulerAncestralDiscreteScheduler` |
| 画布 / 生成 | 512×512 / 1024×1024 |
| 预览 / 精细 | 10 / 25 步，均重新采样 |
| CFG / ControlNet 强度 / 结束比例 | 6.0 / 1.0 / 1.0 |
| 显存策略 | CUDA float16，默认 model CPU offload，VAE tiling |
| 模型缓存 | 项目 `artifacts/model-cache` |

两个模型的提交 SHA 已固定在 [config.py](../config.py)。保留当前依赖版本，避免迁移模型同时更改整套运行环境。旧模型缓存不删除。

选择 Animagine 是为了先建立具有明确文档和 Diffusers 支持的 SDXL 动漫基线，不声称它是最新或最优模型。官方建议使用标签式提示词、Euler a、CFG 5–7；具体组合效果仍需本项目实测。

来源：[Animagine 模型说明](https://huggingface.co/cagliostrolab/animagine-xl-3.1)、[Xinsir 控制模型说明](https://huggingface.co/xinsir/controlnet-scribble-sdxl-1.0)。

## 代码变化与学习重点

- [src/pipeline.py](../src/pipeline.py)：加载 SDXL 的双文本编码器管线及匹配 ControlNet，固定版本和缓存。CPU offload 按需移动组件，降低显存压力，但增加传输耗时；CPU 路径仍使用 float32。VAE tiling 降低大图解码峰值。
- 使用 `torch.no_grad()` 禁用梯度图；不使用 `inference_mode`，因为 CPU offload 在前向过程中创建的 inference tensor 会与 PEFT 后续切换/回滚时设置梯度标记冲突。真实 GPU 回归发现并覆盖了这个问题。
- `render_sketch`：线稿反色为黑底白线，再放大为模型控制图；显式传入 CFG 和负面提示词。512 输入放大不会恢复导入时已丢失的细节。回调取消、固定种子和共享管线机制仍适用。
- [src/app.py](../src/app.py)：仅显示时缩小生成图；历史与保存使用原始 1024 图像，避免窗口尺寸膨胀或保存缩水。
- [src/renderer.py](../src/renderer.py)：PNG 参数增加模型版本、控制模型、生成尺寸、CFG、负面提示词和采样器；未包含完整环境哈希或 LoRA 文件哈希，不保证跨设备像素一致。
- 旧 SD1.5 Lineart 实验入口明确拒绝，避免混合不兼容权重。纯图像极性辅助函数保留以支持历史测试。
- `profiles/example.json` 改为新底模；旧角色预设会被基础模型检查拒绝。直接加载 LoRA 时仍依赖 Diffusers 检查结构并由 worker 回滚；不要把 SD1.5 LoRA 改名当成 SDXL LoRA。
- 默认提示词移除强制单人和肖像构图；人物数量、角色名称、全身/半身等由用户或预设表达。

原 SD1.5 管线带有检查器输出字段；当前 SDXL 管线没有同样的内置输出检查器。负面提示词不是内容检测器，保留的 `RenderBlocked` 兼容逻辑不会凭空提供 SDXL 检测能力。

## 验证边界

验证结果见 [SDXL 验证记录](reports/SDXL_VALIDATION.md)。既有约 1 秒预览成绩属于 SD1.5，不得沿用。当前预览结果最大年龄预算为 30 秒，以免较慢的有效结果永远被淘汰；这只是过期策略，不意味着已达到实时目标。角色一致性需另选真实匹配 LoRA 评估。
