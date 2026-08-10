# 工作日志

## 2026-08-10 — P0 线程安全修复

### 背景

`canvas_stream.py` 是 Tkinter + Stable Diffusion ControlNet 实时线稿转动漫的交互画板。主线程处理 GUI 事件，后台线程执行模型推理。经审查发现多处线程安全隐患。

### 修复内容

#### 🔴 P0：消除竞态条件

| 问题 | 位置 | 修复 |
|------|------|------|
| `_preview_thread` 无锁访问 | `paint()` | 纳入 `state_lock` 保护 |
| `last_render_time` 无锁访问 | `paint()` | 纳入 `state_lock` 保护 |
| 后台线程读取 Tkinter `StringVar` | `async_hq_render()` | 主线程捕获 prompt 快照 → 存入 `_last_prompt` → 作为参数传入 |
| `sketch_img` / `draw_buffer` 并发读写 | 多处 | 新增 `canvas_lock` 统一保护 |
| `need_update_again` 处理存在 TOCTOU 窗口 | `async_hq_render()` | `is_rendering=False` + 清 flag + `is_rendering=True` 合并为一次原子操作 |
| MessageBox 在 `state_lock` 内调用 | `load_lora_weights_action()` | 条件检查在锁内 → 释放锁 → 再弹窗 |

#### 🔴 P0：优雅退出

- 新增 `_shutdown_event`（`threading.Event`）
- 新增 `on_closing()` — 设置退出事件，启动轮询
- 新增 `_check_shutdown_complete()` — 每 200ms 轮询，最多等 5 秒让 CUDA 线程安全结束
- 所有 `root.after()` 调用前检查退出信号，防止窗口销毁后崩溃

#### 🧪 测试

新增 `test_thread_safety.py`：17 项线程安全单元测试（mock pipeline，无需 GPU），全部通过。

### 文件变更

| 文件 | 变更 |
|------|------|
| `canvas_stream.py` | +116 / -70 行 |
| `test_thread_safety.py` | 新增 402 行 |
| `CHANGELOG.md` | 新增（本文件） |

### 提交

`2814fa1` — fix: P0 线程安全修复 — 消除竞态条件与优雅退出
