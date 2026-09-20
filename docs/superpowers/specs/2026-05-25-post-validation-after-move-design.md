# 人类走法后验校验 设计文档

> 日期: 2026-05-25 | 状态: 待实现 | 父文档: 2026-05-25-board-recognition-voting-design.md

## 问题

`_handle_human_turn` 在调用 `capture_stable_board` 时传入 `expected_red/expected_black/expected_rc/expected_bc`，
这些预期值来自 `_last_board`（人类走子前的棋盘）。当人类吃子时，对手棋子永久少一个，预期值与实际永
不匹配，导致 `_validate_piece_count` 始终返回 False，`capture_stable_board` 无限重试。

## 方案

**后验校验**：将精确棋子数校验从「识别阶段」移到「走法确认阶段」。

### 新流程

```
capture_stable_board(只做绝对上限+将帅检查, 不传expected)
  → get_move() 算出候选走法
  → _is_move_legal() 检查 cchess 合法性（不 push）
  → _check_capture_before_push() 判断是否吃子
  → 用「_last_board 棋子数 + 吃子调整」计算预期值
  → 与 board_now 实际棋子类型数量比对
  → 一致 → push 到 cchess，接受此棋盘
  → 不一致 → human_move = None，下一轮重新识别
```

### 核心逻辑

知道走法后就能精确推算预期棋子数：
- 无吃子：红黑总数和各类型数量与 `_last_board` 完全一致
- 吃子（如吃黑车）：黑方总数 -1，黑方 `Car` -1，其余不变

## 新增方法

### `GameController._compute_expected_counts_after_move(board, uci_move, is_capture, captured_symbol) -> Tuple[Dict, Dict]`

根据走法前的棋盘和走法信息，计算走法后的预期棋子类型数量。

内部逻辑：
1. 调用 `count_pieces_by_type(board)` 得到走前数量
2. 如果 `is_capture` 为 True，将 `captured_symbol` 对应的类型数量 -1
3. 返回调整后的 `(rc_dict, bc_dict)`

需要新增一个 cchess 符号 → 棋子类型的映射表（`k→Kin, r→Car, n→Hor, c→Can, b→Ele, a→Shi, p→Paw`）。

## 改动范围

全部在 `pikafish_auto/Pikafish_auto4.py` 内：

| 位置 | 改动 |
|------|------|
| `GameController` 类 | 新增 `_CChess_SYMBOL_TO_TYPE` 映射表 |
| `GameController._compute_expected_counts_after_move()` | 新增方法 |
| `GameController._handle_human_turn()` | 移除 `expected_*` 传参；在 `get_move`+`is_legal` 之后插入后验校验块 |
| `VisionController.capture_stable_board()` | 不变 |
| `VisionController._validate_piece_count()` | 不变 |

## 不变的部分

- `recognize_board`、`_vote_board`、`capture_stable_board`、`_validate_piece_count` — 不动
- `_handle_engine_turn` — 不调用 `capture_stable_board`，不受影响
- `_handle_wait_start`、`_resync_from_camera` — 不传 expected，不受影响
- `_handle_move_recognition_failure`、`_handle_illegal_move` — 恢复路径，不受影响
