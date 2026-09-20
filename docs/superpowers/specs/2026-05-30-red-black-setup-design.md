# 一键摆盘红/黑 设计文档

> 日期: 2026-05-30 | 状态: 待实现

## 问题

当前一键摆盘只支持标准开局 FEN（黑方在摄像头顶部=人类执黑）。需要支持人类选择执红或执黑。

## 触发条件

服务端发送"一键摆盘红"或"一键摆盘黑"（替代原来的"一键摆盘"），在 `WAIT_START` 状态下接收。

## 方案

新增 `RED_STANDARD_OPENING_FEN` 常量（红方在摄像头顶部）。根据收到的指令选择 FEN。开局校验用 `detect_side` 自动匹配对应 FEN。

### 两套 FEN

**人类执黑（不变）**：`STANDARD_OPENING_FEN`，黑在上红在下
```
rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1
```

**人类执红（新增）**：`RED_STANDARD_OPENING_FEN`，红在上黑在下
```
RNBAKABNR/9/1C5C1/P1P1P1P1P/9/9/p1p1p1p1p/1c5c1/9/rnbakabnr w - - 0 1
```

## 改动范围

全部在 `pikafish_auto/Pikafish_auto4.py` 内：

| 位置 | 改动 |
|------|------|
| `Config` 类 | 新增 `RED_STANDARD_OPENING_FEN` 常量 |
| `_handle_wait_start` | 解析"一键摆盘红"/"一键摆盘黑"，设置 `self._setup_fen` |
| `_handle_wait_start`（开局校验） | 用 `detect_side(board)` 选对应 FEN 做 `_validate_against_expected` |
| `_handle_auto_setup` | 全部 `Config.STANDARD_OPENING_FEN` → `self._setup_fen`（共4处） |

## 不变的部分

- `detect_side` — 直接复用
- `fen_to_uci_pieces` — 直接复用
- `_validate_against_expected` — 直接复用
- 摆盘分批/校验/重试逻辑 — 不动
- 引擎通信、悔棋、对弈逻辑 — 不动
