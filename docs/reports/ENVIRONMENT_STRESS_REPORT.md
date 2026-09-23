# 独立环境与连续操作验证 — 2026-09-12

完成同一台 Windows / RTX 4080 Laptop 机器上的隔离安装、实际 GPU 和窗口验证，以及两组短时连续操作测试。未发现需要新增应用代码修复的问题；新增可复用压力测试脚本，并补齐安装说明。

## 独立安装

`artifacts/clean-env` 使用 Python 3.10.20，`include-system-site-packages = false`。33 个已安装发行包全部位于新环境内；Python 基础解释器仍来自本机 conda。原 `.venv` 和 conda 未改动。

旧 pip 23.0.1 在 PyTorch CUDA 索引安装时遇到 typing_extensions 名称解析及后续构建依赖失败。仅升级新环境 pip 到 26.2.1 后，安装 torch 2.5.1+cu121 和 requirements.txt 成功；README 已加入先升级 pip。

- `pip check`：通过。
- `unittest discover -q`：44 项通过。
- 实际 GPU：第二步取消成功（本次约 0.196 秒），取消后恢复生成；损坏 LoRA 回滚原适配器和 0.5 权重，前后图像像素一致；成功替换后释放旧适配器。
- 隐藏 Tk 窗口：表情、拦截结果保留及恢复、绘画、撤销/重做、关闭通过；启动失败重试、加载中关闭通过；实际缓存模型启动后进入画板并生成，主线程心跳执行 317 次。

本地证据（artifacts 被 Git 忽略）：

- `artifacts/environment/clean-environment.json`：解释器、包路径及版本。
- `artifacts/environment/clean-environment-freeze.txt`：完整版本快照。
- `artifacts/environment/clean-install-report.json`：依赖安装报告。
- `artifacts/checks/clean-runtime-20260912/metrics.json`：GPU 回归结果。
- 窗口验证执行 `artifacts/gui_smoke.py`，退出码 0；模拟缓存缺失与拦截提示为预期分支。

## 连续操作

新增 `stress_renderer.py` 使用真实 RenderManager，交替提交预览/精细请求、清空及 LoRA 切换，并故意引入损坏适配器。检查消费结果输入标记、模型操作并发数、关闭后的工作线程，保存线程数、事件队列和内存采样。随机种子 42；调度和次数仍随实际耗时变化。

| 指标 | 模拟模型，180 秒 | 实际 GPU，120 秒 |
| --- | ---: | ---: |
| 渲染请求 | 6899 | 252 |
| 清空请求 | 706 | 26 |
| LoRA 请求 | 1544 | 60 |
| 取消推理 | 6126 | 192 |
| 消费有效帧 | 710 | 28 |
| 预期 LoRA 错误 | 768 | 36 |
| 最大模型操作并发数 | 1 | 1 |
| 模型工作线程数 | 1 | 1 |

两组通过旧帧过滤、串行执行及关闭断言；采样时事件队列均为 0。GPU 关闭等待约 0.078 秒。报告在 `artifacts/checks/stress-mock-20260912/metrics.json` 和 `artifacts/checks/stress-gpu-20260912/metrics.json`，预期错误另记于各自 `expected-errors.log`。

模拟测试 Python 当前分配多数约 1.51–1.54 MB；GPU 测试约 3.00→3.59 MB，已分配显存约 3354–3419 MiB。采样阶段不同，Python 跟踪不覆盖全部原生分配，不能据此断言无泄漏。

复现命令：

```powershell
.\.venv\Scripts\python.exe -X utf8 stress_renderer.py --seconds 180
.\.venv\Scripts\python.exe -X utf8 stress_renderer.py --gpu --seconds 120 --lora artifacts/checks/runtime-20260912/lora-fixture
.\artifacts\clean-env\Scripts\python.exe -X utf8 -m unittest discover -q
.\artifacts\clean-env\Scripts\python.exe -X utf8 test_runtime_gpu.py --offline
```

GPU 压测需要模型缓存；上述零增量 LoRA 由此前 GPU 回归生成。输出默认新建时间戳目录，拒绝覆盖非空目录。

## 范围与后续

本次是同机独立第三方依赖验证，不等于另一台机器或首次联网下载模型已验证。两组压力测试使用原 `.venv`，新环境另做上述 GPU 和窗口回归。LoRA 为零增量测试适配器，不代表社区风格兼容性。

180/120 秒测试不能替代数小时使用。建议接下来补真实手绘、多种子和表情预设的画质评估，再根据使用需求安排更长时间运行及实际社区 LoRA 验证。
