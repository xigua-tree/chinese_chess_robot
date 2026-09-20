# 悔棋功能重构 设计文档

> 日期: 2026-05-27 | 状态: 待实现 | 替代: 2026-05-25-undo-feature-design.md

## 问题

当前悔棋有以下不足：
1. 指令发送后每条都等待"运动完成"，效率低
2. 人类吃子后被吃棋子进入人类废棋区，无法从 MoveRecord 坐标恢复
3. 引擎吃子恢复的废棋坐标依赖 MoveRecord 存储，不够鲁棒

## 核心原则

- **撤销一整轮**：撤销引擎上一步 + 人类上一步，共 2 个 UCI 走法，局面回到上一轮人类落子前
- **所有机械臂指令连续发送，不等待"运动完成"**
- **引擎废棋区（右侧）**继续用 `_get_waste_xy()` 计算坐标（位置可控）
- **人类废棋区（左侧）**用 HoughCircles+YOLO 实时识别找棋子

## 废除的假设

原设计中：
- `_handle_undo` 逐条发送指令并等待"运动完成" — 废除，改为连续发送
- `request_undo` 负责生成机械臂指令 — 废除，改为只负责局面恢复

## 新增数据结构

### 废棋区 ROI 常量（VisionController）

```python
# 人类废棋区（棋盘左侧，warped 图坐标）
HUMAN_WASTE_ROI = (10, 0, 103, 625)  # (x1, y1, x2, y2)
```

## 新增方法

### `VisionController.recognize_waste_area(roi_rect, warped_frame) -> List[Tuple[str, int, int]]`

在指定 ROI 内做 HoughCircles 寻圆 + YOLO 分类，返回 `[(label, x, y), ...]`。

**多帧投票**：内部连续取 N=5 帧，每帧独立 HoughCircles+YOLO。同一棋子类型在相近位置（距离 < 阈值）出现超过半数帧才确认。返回的是确认的棋子标签和像素坐标。

逻辑复用现有 `recognize_board` 中的寻圆+YOLO 流程，去掉网格映射步骤。

### `GameController._find_piece_in_human_waste(warped_frame, piece_label) -> Tuple[int, int]`

```python
def _find_piece_in_human_waste(self, warped_frame, piece_label):
    pieces = self._vis.recognize_waste_area(HUMAN_WASTE_ROI, warped_frame)
    for label, x, y in pieces:
        if label == piece_label:
            return (x, y)
    # 找不到则返回废棋区中心默认坐标
    return (52, 312)
```

调用多帧识别，在结果中找第一个匹配 `piece_label` 的棋子，返回其 warped 坐标。找不到返回默认坐标。

## 修改的方法

### `GameController.request_undo()` 简化

**只负责局面恢复**，不再生成机械臂指令：

1. 检查 `len(_move_history) >= 2`
2. `m_engine = _move_history.pop()`，`m_human = _move_history.pop()`
3. 对 `_last_board` 依次反向应用两步 UCI 走法
4. 更新 `_situation`（移除最后两个 UCI）
5. 恢复 `_use_chess_count`
6. 重建 `_cchess` 并重启引擎
7. 返回 `(m_engine, m_human)` 或 `None`

### `GameController._handle_undo()` 重写

```python
def _handle_undo(self):
    frame = self._vis._stream.get_latest_warped()
    result = self.request_undo()
    if result is None:
        self._p.send("请玩家落子")
        self._p.send("文字:请落子")
        self._state = self.State.HUMAN_TURN
        return

    m_engine, m_human = result

    human_dst = self._vis.uci_to_pixel_public(m_human.uci[2:])
    human_src = self._vis.uci_to_pixel_public(m_human.uci[:2])

    cmds = []

    # ① 引擎棋子移回：dst → src
    cmds.append(f"棋子移动:{m_engine.dst_xy},{m_engine.src_xy}")

    # ② 若引擎吃子：从机器人废棋区 → 引擎 dst
    #    必须在步骤③之前！被吃的棋子正是人类的棋子，
    #    必须先放回棋盘 dst 位，步骤③才能从 dst 移回 src
    if m_engine.is_capture:
        cmds.append(f"棋子移动:{m_engine.waste_xy},{m_engine.dst_xy}")

    # ③ 人类棋子移回：dst → src
    #    此时人类棋子一定在棋盘上（从未被吃 or 已被步骤②恢复）
    cmds.append(f"棋子移动:{human_dst},{human_src}")

    # ④ 若人类吃子：从人类废棋区 → 人类 dst
    if m_human.is_capture:
        xy = self._find_piece_in_human_waste(frame, m_human.captured_piece)
        cmds.append(f"棋子移动:{xy},{human_dst}")

    # 连续发送，不等待
    for cmd in cmds:
        self._p.send(cmd)

    self._p.send("请玩家落子")
    self._p.send("文字:请落子")
    self._state = self.State.HUMAN_TURN
```

## 四场景自然适配

四种 `(m_engine.is_capture, m_human.is_capture)` 组合：

| 场景 | m_engine.is_capture | m_human.is_capture | 指令条数 |
|------|---------------------|--------------------|----------|
| 未吃子 | F | F | 2 |
| 机器人吃子 | T | F | 3 |
| 人类吃子 | F | T | 3 |
| 两者都吃子 | T | T | 4 |

以"人类炮吃马 → 机器人车吃炮"为例（两者都吃子）：
- `m_engine`=车，吃炮 → ② 触发：从机器人废棋区取炮放回车走法 dst
- ③：炮从 dst 移回 src
- `m_human`=炮，吃马 → ④ 触发：从人类废棋区找到马放回炮走法 dst

## 触发方式不变

`_handle_human_turn` 中 `wait_for_any("玩家落子完成", "悔棋")` 收到"悔棋"时切换到 UNDO 状态，其他逻辑不动。

## 改动范围

全部在 `pikafish_auto/Pikafish_auto4.py` 内：

| 位置 | 改动 |
|------|------|
| `VisionController` 类 | 新增 `HUMAN_WASTE_ROI` 常量 |
| `VisionController.recognize_waste_area()` | 新增方法（多帧投票 HoughCircles+YOLO） |
| `GameController._find_piece_in_human_waste()` | 新增方法 |
| `GameController.request_undo()` | 简化：移除指令生成，只负责局面恢复，返回 `(m_engine, m_human)` |
| `GameController._handle_undo()` | 重写：获取 frame → 调用 request_undo → 生成四场景指令 → 连续发送 → 回到 HUMAN_TURN |

## 不变的部分

- `MoveRecord` 类 — 不动
- `_handle_human_turn` 悔棋触发逻辑 — 不动
- `_CChess_SYMBOL_TO_TYPE` — 不动
- `_get_waste_xy()` — 不动（引擎废棋区仍用计算坐标）
- `capture_stable_board`、`get_move`、引擎通信 — 不动
