# 文档导航与目录迁移

当前 SDXL 参数、代码变化、回退步骤见 [迁移说明](SDXL_MIGRATION.md)；[验证记录](reports/SDXL_VALIDATION.md) 区分新旧性能。原面试指南与逐函数手册正文保留为 SD1.5 学习快照。

当前使用说明见 [项目首页](../README.md)，恢复开发先读 [最新交接](../SESSION_HANDOFF.md)，产品评价标准见 [验收方案](ACCEPTANCE.md)。

系统学习与面试准备请读 [项目完整技术说明与面试准备](INTERVIEW_GUIDE.md)。以当前源码重写，共 22 章，覆盖每项技术的用途、调用细节、设计取舍、实测数据和常见追问。

逐段读代码请读 [核心源码逐函数学习手册](CODE_WALKTHROUGH.md)：按当前源码展开 76 个核心函数和配置，提供实际代码、源码定位、状态变化及调用衔接；附三个执行过程、代表性测试和工具解析、练习答案。源码摘录带文件哈希，后续修改代码时需同步更新。

| 原根目录文件 | 新位置/运行入口 |
|---|---|
| test_pipeline.py、test_quality.py、test_startup.py、test_thread_safety.py | tests/，运行 `python -m unittest discover -s tests -v` |
| test_stream.py | `python -m tools.benchmark` |
| test_runtime_gpu.py | `python -m tools.runtime_gpu` |
| evaluate_quality.py | `python -m tools.evaluate_quality` |
| stress_renderer.py | `python -m tools.stress_renderer` |
| 旧 PROJECT_GUIDE.md | 已删除；使用当前 INTERVIEW_GUIDE.md |
| 各类 *_REVIEW.md 和环境报告 | reports/ |
| 旧 SESSION_HANDOFF.md | [2026-09-13 及以前交接](history/SESSION_HANDOFF_20260913.md) |
| 旧 README.md 备份 | 已删除；当前运行方法以根 README.md 为准 |
| aniface.log | artifacts/logs/aniface-before-20260919.log；新日志为 artifacts/logs/aniface.log |

所有命令从项目根目录运行。历史文档的旧命令仅用于追溯，请按此表换成当前入口。`artifacts` 内过时的一次性脚本已清理；当前可重复运行的工具均在 `tools`。

## 2026-09-19 清理记录

用户授权删除无用文件。清理了 `artifacts/pip-cache`（依赖安装缓存）、`artifacts/clean-env`（已完成独立安装验证的临时环境）、7 个已被现有工具替代的一次性脚本，以及项目源码/测试/工具字节码缓存。清理前确认当前 `.venv` 使用 conda 基础解释器，不依赖临时环境，且没有进程使用临时环境。

共释放约 **6.835 GiB**。保留 `.venv`、`.git`、用户工具配置、模型权重、原始素材、有效图片/指标与环境版本报告。历史文档中关于 clean-env 和旧脚本的描述是当时的验证记录，路径不再可运行；复测请使用当前 README 的入口。

`artifacts` 现在分为 `checks`（运行回归）、`evaluations`（画质评估）、`environment`（环境报告）、`model-cache`（权重）和 `logs`（日志）。工具的默认输出路径和文档链接同步更新，后续运行也按新目录归档。

逐项记录：[删除清单](../artifacts/environment/cleanup-20260919.json)、[移动映射](../artifacts/environment/moves-20260919.json)。字节码是自动生成的，正常启动后可能再次出现。

评估报告：

- [本轮改进验证](reports/WORKFLOW_20260919.md)
- [真实手绘与全程控制](reports/REAL_SKETCH_REVIEW.md)
- [通用 Lineart](reports/LINEART_REVIEW.md)
- [动漫 Lineart](reports/ANIME_LINEART_REVIEW.md)
- [合成线稿](reports/QUALITY_REVIEW.md)
- [表情](reports/EXPRESSION_REVIEW.md)、[惊讶提示词](reports/SURPRISE_REVIEW.md)
- [历史环境与压力测试](reports/ENVIRONMENT_STRESS_REPORT.md)

开发约定：业务代码放 `src`，自动测试放 `tests`，显式运行的实验脚本放 `tools`；结果、日志和大模型缓存放 `artifacts`；手工评估结论写入 `docs/reports`。保留当前使用的 `.venv` 和用户工具配置，不将模型权重纳入源码。
