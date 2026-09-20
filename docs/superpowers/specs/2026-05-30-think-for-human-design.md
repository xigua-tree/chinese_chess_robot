# 替人类思考 设计文档

> 日期: 2026-05-30 | 状态: 待实现

## 问题

当前人类方走棋必须由人手动完成。需要支持服务端发送"替我思考"后，机器人自动替人类方计算并执行一步走法。

## 触发条件

在 `HUMAN_TURN` 状态的 `wait_for_any` 阻塞等待中收到"替我思考"。

## 流程

```
HUMAN_TURN → wait_for_any("玩家落子完成", "悔棋", "替我思考")
  → 收到"替我思考"
  → 引擎计算人类方最佳走法（get_best_move，当前局面就是人类方走）
  → 合法性校验（_is_move_legal）
  → 检测吃子（_check_capture_before_push）
  → push 到 cchess，更新 situation
  → 如有吃子：发送去除被吃棋子指令
  → 发送主移动指令 + wait_for("运动完成")
  → 记录 MoveRecord（支持悔棋）
  → 更新 _fen_before_human、_last_board
  → 检查将军/游戏结束
  → 转到 ENGINE_TURN
```

## 改动范围

全部在 `pikafish_auto/Pikafish_auto4.py`，`_handle_human_turn` 方法内：

| 改动 | 说明 |
|------|------|
| `wait_for_any` 加参数 | `"玩家落子完成", "悔棋", "替我思考"` |
| 新增 `elif msg == "替我思考"` 分支 | 引擎计算 + 机械臂执行，处理吃子/将军/游戏结束 |

## 复用

- `_engine.get_best_move` — 计算人类方最佳走法
- `_is_move_legal` — 走法合法性校验
- `_check_capture_before_push` — 吃子检测
- `_push_engine_move` — 更新 cchess
- `detect_piece_at` + `uci_to_pixel_public` — 机械臂坐标
- `_get_waste_xy` — 废棋区坐标
- `_symbol_to_label` — 棋子符号转标签
- `apply_uci_move` — 更新 _last_board
- `_check_game_over` — 游戏结束检查
- `MoveRecord` — 走法记录（支持悔棋）

## 不变的部分

- `_handle_engine_turn` — 不动
- `request_undo` / `_handle_undo` — 不动（MoveRecord 格式一致，悔棋自动兼容）
- 开局/残局初始化 — 不动
