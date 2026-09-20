# 一键摆盘功能 设计文档

> 日期: 2026-05-27 | 状态: 待实现

## 问题

当前没有自动摆盘功能。用户需要手动把棋子放到正确位置，效率低、易出错。

## 方案：废棋区摆盘（方案一）

用户把 32 枚棋子分放到左右两个废棋区（不需分类），机器人在两个废棋区内识别所有棋子，然后按标准开局 FEN 从左上角开始逐一摆放到棋盘上。

## 触发条件

服务端发送"一键摆盘"消息（与"开始对弈"同级），在 `WAIT_START` 状态下接收。

## 核心流程

```
收到"一键摆盘" → AUTO_SETUP 状态
  → 识别左废棋区（HUMAN_WASTE_ROI）+ 右废棋区（ROBOT_WASTE_ROI）
  → 解析标准开局 FEN → [(uci, label), ...] 按 a9→i0 排列
  → 对每个 (uci, label)：从废棋识别池中匹配 → 生成移动指令
  → 连续发送全部指令（不等待"运动完成"）
  → 识别棋盘校验 → READY
```

## 新增数据结构

### 机器人废棋区 ROI 常量

```python
# 机器人废棋区（棋盘右侧，warped 图坐标）
ROBOT_WASTE_ROI = (609, 2, 705, 623)  # (x1, y1, x2, y2)
```

### 状态枚举

```python
class State(Enum):
    ...
    AUTO_SETUP = "auto_setup"  # 新增
```

## 新增/修改方法

### `Config.fen_to_uci_pieces(fen: str) -> List[Tuple[str, str]]`（静态方法，新增）

解析 FEN 行棋部分（空格前的第一段），返回按 a9→i0 排列的 `[(uci, label), ...]` 列表。空格和数字跳过。

```python
@staticmethod
def fen_to_uci_pieces(fen: str) -> List[Tuple[str, str]]:
    rows = fen.split()[0].split("/")
    pieces = []
    for rank_idx, row in enumerate(rows):
        col = 0
        for ch in row:
            if ch.isdigit():
                col += int(ch)
            else:
                uci = f"{chr(ord('a') + col)}{9 - rank_idx}"
                label = Config.INV_FEN_MAP.get(ch)
                if label:
                    pieces.append((uci, label))
                col += 1
    return pieces
```

### `ProtocolClient.wait_for_any_prefix(*prefixes: str) -> Optional[str]`（新增）

当前 `wait_for_prefix` 只监听单个前缀，不能同时监听多个。需要新增：

```python
def wait_for_any_prefix(self, *prefixes: str) -> Optional[str]:
    """阻塞等待任意一个前缀匹配的消息，返回完整消息。"""
    print(f"[等待前缀] {list(prefixes)} ...")
    while self._running:
        try:
            msg = self._queue.get(timeout=1.0)
            if msg == "退出":
                return None
            for p in prefixes:
                if msg.startswith(p):
                    print(f"[匹配] {msg}")
                    return msg
        except queue.Empty:
            pass
    return None
```

### `GameController._handle_wait_start()`（修改）

用 `wait_for_any_prefix` 替代链式 `or`，同时监听三个前缀：

```python
msg = self._p.wait_for_any_prefix("开始对弈", "残局对弈", "一键摆盘")
if msg is None:
    return
if msg.startswith("一键摆盘"):
    self._state = self.State.AUTO_SETUP
    return
```

### `GameController._handle_auto_setup()`（新增）

```python
def _handle_auto_setup(self):
    # 1. 识别两个废棋区
    human_pieces = self._vis.recognize_waste_area(VisionController.HUMAN_WASTE_ROI)
    robot_pieces = self._vis.recognize_waste_area(VisionController.ROBOT_WASTE_ROI)
    all_waste = human_pieces + robot_pieces

    # 2. 解析目标 FEN
    target_pieces = Config.fen_to_uci_pieces(Config.STANDARD_OPENING_FEN)

    # 3. 逐格匹配 + 生成指令
    used = [False] * len(all_waste)
    cmds = []
    for uci, label in target_pieces:
        target_pixel = self._vis.uci_to_pixel_public(uci)
        found = None
        for i, (w_label, wx, wy) in enumerate(all_waste):
            if not used[i] and w_label == label:
                found = (wx, wy)
                used[i] = True
                break
        if found:
            cmds.append(f"棋子移动:{found},{target_pixel}")
        else:
            print(f"[摆盘] 未找到棋子 {label} @ {uci}")

    # 4. 连续发送
    for cmd in cmds:
        self._p.send(cmd)

    # 5. 校验
    self._p.send("文字:摆盘完成，正在校验...")
    board = self._vis.capture_stable_board()
    if board and self._validate_against_expected(board, Config.STANDARD_OPENING_FEN):
        self._p.send("文字:摆盘校验通过")
    else:
        self._p.send("文字:摆盘校验未通过，请检查")

    self._state = self.State.READY
```

### 主循环 `run()`（修改）

增加 AUTO_SETUP 分支：

```python
elif self._state == self.State.AUTO_SETUP:
    self._handle_auto_setup()
```

## 改动范围

全部在 `pikafish_auto/Pikafish_auto4.py` 内：

| 位置 | 改动 |
|------|------|
| `ProtocolClient` 类 | 新增 `wait_for_any_prefix()` 方法 |
| `VisionController` 类 | 新增 `ROBOT_WASTE_ROI` 类常量 |
| `Config` 类 | 新增 `fen_to_uci_pieces()` 静态方法 |
| `GameController.State` | 新增 `AUTO_SETUP` |
| `GameController.run()` | 新增 `AUTO_SETUP` 分支 |
| `GameController._handle_wait_start()` | 用 `wait_for_any_prefix` 同时监听三个前缀 |
| `GameController._handle_auto_setup()` | 新增方法 |

## 不变的部分

- `recognize_waste_area` — 直接复用
- `uci_to_pixel_public` — 直接复用
- `_validate_against_expected` — 直接复用
- `capture_stable_board` — 直接复用
- 引擎通信、悔棋、对弈逻辑 — 不动
