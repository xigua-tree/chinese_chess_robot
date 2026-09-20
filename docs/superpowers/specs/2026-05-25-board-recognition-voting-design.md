# 棋盘识别帧间投票 + 棋子数校验 设计文档

> 日期: 2026-05-25 | 状态: 待实现

## 问题

`capture_stable_board()` 当前要求连续 3 帧全盘布尔掩码完全一致才通过。YOLO 分类器在部分棋子上置信度波动（接近 0.70 阈值上下），导致单帧之间棋子"闪现"，任何一格波动都重置计数器。结果：开局识别需要 28+ 次尝试，且漏检棋子导致 FEN 不完整（如 `1NBAKABN1` 缺两个车），后续 `get_move()` 产生虚假移动。

## 方案

**帧间投票 + 棋子数校验**，分两阶段：

### 阶段 1：帧间投票

采集 M 帧（默认 15），对每个棋盘格位 `(r, c)` 独立统计棋子类型出现次数。一个格子有标签 label 且置信度 > CONF_THRESH 则 `count[label] += 1`，为空则 `count[None] += 1`。M 帧后对每个格子独立裁决：

- 某个 label 票数 ≥ K（默认 8，即 > M/2）→ 确认该棋子
- 无 label 达到 K 票 → 该格为空
- 两个 label 都 ≥ K（极罕见）→ 取票数多的；票数相同取累计置信度高的

关键参数（`Config` 新增）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `VOTE_FRAMES` | 15 | 投票帧数 |
| `VOTE_THRESHOLD` | 8 | 最少确认票数 |
| `VOTE_INTERVAL` | 0.06 | 帧间间隔（秒） |

### 阶段 2：棋子数校验

对投票产出的候选棋盘做三级校验：

1. **绝对上限**：红方 ≤ 16，黑方 ≤ 16，0 < 总数 ≤ 32
2. **将帅必须存在**：`R_Kin` 和 `B_Kin` 至少各出现一次
3. **棋子数变化一致性**（当提供 expected 参数时）：
   - 已知上一步准确的红/黑棋子数
   - 本次结果必须与预期一致（无吃子时不变，吃子时对方减 1）

校验失败 → 追加 5 帧继续投票，最多重试 3 次。3 次后取棋子数最接近预期的那次结果。

## 接口变化

### `capture_stable_board()` 签名

```python
def capture_stable_board(self, required_stable: int = None,
                         expected_red: int = None,
                         expected_black: int = None) -> Optional[List[List[str]]]:
```

新增可选参数 `expected_red`、`expected_black`，不传则只做绝对上限和将帅校验。

### 调用方调整

- `_handle_wait_start()`：首次识别，不传 expected
- `_handle_human_turn()`：传入上一步 `count_pieces(_last_board)` 的结果，根据 cchess 走法判断是否吃子来预期棋子数
- `_handle_engine_turn()`：同上

## 新增方法

### `_vote_board(frames: int, threshold: int) -> List[List[str]]`

采集 frames 帧，每格独立投票，返回候选棋盘。

内部逻辑：
1. 初始化 `votes[r][c] = {}`（每个格子的标签→票数映射）
2. 循环 frames 次：获取帧 → `recognize_board()` → 对每个有标签的格子 `votes[r][c][label] += 1`，空则 `votes[r][c][None] += 1`
3. 循环后对每个格子裁决，产出 final_board

### `_validate_piece_count(board, exp_red, exp_black) -> Tuple[bool, str]`

返回（是否通过，失败原因）。校验棋盘棋子数合法性。

## 改动范围

全部在 `pikafish_auto/Pikafish_auto4.py` 内：

| 位置 | 改动 |
|------|------|
| `Config` (L44-47) | 新增 `VOTE_FRAMES=15`、`VOTE_THRESHOLD=8`、`VOTE_INTERVAL=0.06`，保留 `STABLE_FRAMES` 备查 |
| `VisionController.capture_stable_board()` (L502-549) | 重写：调用 `_vote_board` → `_validate_piece_count` → 失败追加帧重试 |
| 新增 `VisionController._vote_board()` | 帧间投票核心逻辑 |
| 新增 `VisionController._validate_piece_count()` | 棋子数校验（静态方法） |
| `GameController._handle_wait_start()` | 首次识别不传 expected |
| `GameController._handle_human_turn()` | 传入 expected 棋子数 |
| `GameController._handle_engine_turn()` | 传入 expected 棋子数 |
| `Config.MAX_CAPTURE_ATTEMPTS` | 从 40 降到 15（每次 15 帧 + 最多 3 次重试 = 45 帧上限） |

## 不变的部分

- `recognize_board()` — 单帧识别逻辑不变
- `get_move()` — 走法差分逻辑不变
- `board_from_fen()` / `construct_fen()` — FEN 转换逻辑不变
- `_to_uci()` / `_uci_to_pixel()` / `apply_uci_move()` — 坐标转换逻辑不变（上一轮已修正）
- `_board_mask()` — 保留，但 `capture_stable_board` 不再使用它
