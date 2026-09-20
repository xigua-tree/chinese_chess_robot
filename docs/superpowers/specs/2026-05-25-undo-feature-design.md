# 悔棋功能 设计文档

> 日期: 2026-05-25 | 状态: 待实现

## 问题

当前悔棋只有预留接口（`State.UNDO`、`_handle_undo`、`request_undo`），未完整实现：
- 无法在「请玩家落子」阻塞等待期间接收悔棋指令
- `request_undo` 未恢复 `_last_board`、`_use_chess_count`
- `MoveRecord` 未记录吃子信息，无法恢复被吃棋子

## 方案

### 触发条件

仅在 `_handle_human_turn` 的「请玩家落子」等待阶段可触发。服务端发送 `悔棋` 消息时，`wait_for` 改为双消息轮询。

### 撤销范围

撤销最近两步（引擎上一步 + 人类上一步），局面回到**上一轮人类走完之后**，保持在 `HUMAN_TURN` 状态。

### 机械臂动作

按序发送坐标指令：
1. 人类棋子从 dst 移回 src（反向 UCI）
2. 引擎棋子从 dst 移回 src（反向 UCI）
3. 如有吃子，从废棋区取回被吃棋子放回原位（即引擎走法 dst 位置）

### 局面恢复

1. 从 `_move_history` 弹出两条 MoveRecord
2. 对 `_last_board` 依次反向应用两步 UCI 走法
3. 从 `_situation` 移除最后两个 UCI
4. 如有吃子，恢复 `_use_chess_count`
5. 用剩余走法重建 `_cchess`
6. 引擎 restart 并同步剩余走法

## 改动范围

全部在 `pikafish_auto/Pikafish_auto4.py` 内：

| 位置 | 改动 |
|------|------|
| `MoveRecord` | 新增 `is_capture: bool`、`captured_type: str \| None` 字段 |
| `_handle_human_turn` | `wait_for("玩家落子完成")` 改为轮询 `悔棋` 或 `玩家落子完成`；收到悔棋则切到 UNDO 状态 |
| `_handle_human_turn` | 人类走法确认后记录吃子信息到 MoveRecord |
| `_handle_engine_turn` | `MoveRecord` 构造增加吃子字段 |
| `request_undo()` | 重写：基于 `_last_board` 反推 + MoveRecord 坐标 + 吃子恢复 |
| `_handle_undo()` | 执行完后回到 HUMAN_TURN，发送「请玩家落子」 |

## 不变的部分

- 状态机结构（UNDO 已存在）
- `capture_stable_board`、`get_move`、引擎通信 — 不动
- 开局/残局初始化 — 不动
- 后验校验 — 不动
