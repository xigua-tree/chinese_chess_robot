# 一键摆盘功能 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现一键摆盘功能：服务端发送"一键摆盘"后，机器人识别两个废棋区的棋子并按标准开局 FEN 摆放到棋盘上。

**Architecture:** ProtocolClient 新增 `wait_for_any_prefix` 支持多前缀监听；Config 新增 FEN 解析器；GameController 新增 AUTO_SETUP 状态和处理器，复用 `recognize_waste_area` + `uci_to_pixel_public`。

**Tech Stack:** Python, cchess, YOLO/OpenCV (复用)

---

### Task 1: 新增 `wait_for_any_prefix` 到 ProtocolClient

**Files:**
- Modify: `pikafish_auto/Pikafish_auto4.py` — 在 `wait_for_any` 之后（line 247）插入新方法

- [ ] **Step 1: 插入 `wait_for_any_prefix` 方法**

在 `wait_for_any` 方法的 `return None`（line 247）和 `wait_for_prefix` 方法（line 249）之间插入：

```python
    def wait_for_any_prefix(self, *prefixes: str) -> Optional[str]:
        """阻塞等待任意一个前缀匹配的消息，返回完整消息。"""
        print(f"[等待前缀] {list(prefixes)} ...")
        while self._running:
            try:
                msg = self._queue.get(timeout=1.0)
                if msg == "退出":
                    print("[协议] 收到退出，中断等待")
                    return None
                for p in prefixes:
                    if msg.startswith(p):
                        print(f"[匹配] {msg}")
                        return msg
            except queue.Empty:
                pass
        return None
```

- [ ] **Step 2: 验证语法**

```bash
cd "e:/chess_robot/project/oo/2026---" && python -c "import ast; ast.parse(open('pikafish_auto/Pikafish_auto4.py', encoding='utf-8').read()); print('语法 OK')"
```

Expected: `语法 OK`

- [ ] **Step 3: Commit**

```bash
git add pikafish_auto/Pikafish_auto4.py
git commit -m "feat: add wait_for_any_prefix to ProtocolClient"
```

---

### Task 2: 新增 `ROBOT_WASTE_ROI` 和 `fen_to_uci_pieces`

**Files:**
- Modify: `pikafish_auto/Pikafish_auto4.py` — VisionController 类（加 ROBOT_WASTE_ROI）和 Config 类（加 fen_to_uci_pieces）

- [ ] **Step 1: 添加 `ROBOT_WASTE_ROI` 类常量到 VisionController**

在 `HUMAN_WASTE_ROI` 之后（约 line 476）加一行：

```python
    # 人类废棋区 ROI（warped 图坐标，棋盘左侧）
    HUMAN_WASTE_ROI = (10, 0, 103, 625)  # (x1, y1, x2, y2)
    # 机器人废棋区 ROI（warped 图坐标，棋盘右侧）
    ROBOT_WASTE_ROI = (609, 2, 705, 623)  # (x1, y1, x2, y2)
```

- [ ] **Step 2: 添加 `fen_to_uci_pieces` 静态方法到 Config**

在 `Config.STANDARD_OPENING_FEN` 定义之后（约 line 94）、`ENDGAME_FEN_DB` 之前（约 line 96），插入：

```python
    @staticmethod
    def fen_to_uci_pieces(fen: str) -> List[Tuple[str, str]]:
        """解析 FEN 行棋部分，返回 [(uci, label), ...]，按 a9→i0 排列。"""
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

- [ ] **Step 3: 验证语法**

```bash
cd "e:/chess_robot/project/oo/2026---" && python -c "import ast; ast.parse(open('pikafish_auto/Pikafish_auto4.py', encoding='utf-8').read()); print('语法 OK')"
```

Expected: `语法 OK`

- [ ] **Step 4: Commit**

```bash
git add pikafish_auto/Pikafish_auto4.py
git commit -m "feat: add ROBOT_WASTE_ROI and fen_to_uci_pieces"
```

---

### Task 3: 新增 AUTO_SETUP 状态 + `_handle_auto_setup` + 接通消息

**Files:**
- Modify: `pikafish_auto/Pikafish_auto4.py` — State 枚举、run()、_handle_wait_start()、_handle_auto_setup()

- [ ] **Step 1: 添加 `AUTO_SETUP` 到 State 枚举**

在 `State` 枚举中（约 line 1306），`UNDO = "undo"` 之后添加：

```python
        AUTO_SETUP = "auto_setup"
```

- [ ] **Step 2: 添加 `AUTO_SETUP` 分支到 `run()`**

在 `run()` 方法中（约 line 1353-1354），`UNDO` 分支之后添加：

```python
            elif self._state == self.State.AUTO_SETUP:
                self._handle_auto_setup()
```

- [ ] **Step 3: 修改 `_handle_wait_start` 使用 `wait_for_any_prefix`**

替换 `_handle_wait_start` 方法的第一行（约 line 1368）：

旧：
```python
        msg = self._p.wait_for_prefix("开始对弈") or self._p.wait_for_prefix("残局对弈")
        if msg is None:
            return
```

新：
```python
        msg = self._p.wait_for_any_prefix("开始对弈", "残局对弈", "一键摆盘")
        if msg is None:
            return
```

并在 `is_endgame = msg.startswith("残局对弈")` 之前（约 line 1372）插入一键摆盘判断：

```python
        if msg.startswith("一键摆盘"):
            self._state = self.State.AUTO_SETUP
            return
```

注意：`is_endgame = msg.startswith("残局对弈")` 仍然需要保留，放在一键摆盘判断之后。

- [ ] **Step 4: 添加 `_handle_auto_setup` 方法**

在 `_handle_game_over` 方法之后（约 line 1606，`_handle_undo` 之前），插入：

```python
    def _handle_auto_setup(self):
        """一键摆盘：从两个废棋区识别全部棋子，按标准开局 FEN 摆放。"""
        print("[摆盘] 开始一键摆盘...")

        # 1. 识别两个废棋区
        human_pieces = self._vis.recognize_waste_area(VisionController.HUMAN_WASTE_ROI)
        robot_pieces = self._vis.recognize_waste_area(VisionController.ROBOT_WASTE_ROI)
        all_waste = human_pieces + robot_pieces
        print(f"[摆盘] 识别到 {len(human_pieces)} 枚(人类废棋区) + "
              f"{len(robot_pieces)} 枚(机器人废棋区) = {len(all_waste)} 枚棋子")

        # 2. 解析目标 FEN
        target_pieces = Config.fen_to_uci_pieces(Config.STANDARD_OPENING_FEN)

        # 3. 逐格匹配 + 生成指令
        used = [False] * len(all_waste)
        cmds = []
        missing = 0
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
                missing += 1

        if missing:
            print(f"[摆盘] 警告：{missing} 枚棋子未找到")

        # 4. 连续发送，不等待
        print(f"[摆盘] 发送 {len(cmds)} 条指令...")
        for cmd in cmds:
            self._p.send(cmd)

        # 5. 校验
        self._p.send("文字:摆盘完成，正在校验...")
        time.sleep(2.0)  # 等待机械臂完成所有动作
        board = self._vis.capture_stable_board()
        if board and self._validate_against_expected(board, Config.STANDARD_OPENING_FEN):
            print("[摆盘] 校验通过！")
            self._p.send("文字:摆盘校验通过")
        else:
            print("[摆盘] 校验未通过")
            self._p.send("文字:摆盘校验未通过，请检查")

        self._state = self.State.READY
```

- [ ] **Step 5: 验证语法**

```bash
cd "e:/chess_robot/project/oo/2026---" && python -c "import ast; ast.parse(open('pikafish_auto/Pikafish_auto4.py', encoding='utf-8').read()); print('语法 OK')"
```

Expected: `语法 OK`

- [ ] **Step 6: Commit**

```bash
git add pikafish_auto/Pikafish_auto4.py
git commit -m "feat: add AUTO_SETUP state and _handle_auto_setup with multi-prefix wait"
```

---

### Task 4: 端到端验证

- [ ] **Step 1: 确认所有关键定义存在**

```bash
cd "e:/chess_robot/project/oo/2026---" && python -c "
import ast
tree = ast.parse(open('pikafish_auto/Pikafish_auto4.py', encoding='utf-8').read())
funcs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
classes = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}

assert 'GameController' in classes, '缺少 GameController'
assert 'wait_for_any_prefix' in funcs, '缺少 wait_for_any_prefix'
assert 'fen_to_uci_pieces' in funcs, '缺少 fen_to_uci_pieces'
assert '_handle_auto_setup' in funcs, '缺少 _handle_auto_setup'

# 检查 State.AUTO_SETUP
for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef) and node.name == 'GameController':
        for inner in ast.walk(node):
            if isinstance(inner, ast.ClassDef) and inner.name == 'State':
                for stmt in inner.body:
                    if isinstance(stmt, ast.Assign):
                        if hasattr(stmt.targets[0], 'attr') and stmt.targets[0].attr == 'AUTO_SETUP':
                            print('State.AUTO_SETUP 已定义')
                            break

# 检查 run() 中有 AUTO_SETUP 分支
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == 'run':
        # find 'AUTO_SETUP' string in run body
        for sub in ast.walk(node):
            if isinstance(sub, ast.Attribute) and sub.attr == 'AUTO_SETUP':
                print('run() 中有 AUTO_SETUP 分支')
                break

print('全量验证通过')
"
```

Expected: 所有检查通过

- [ ] **Step 2: Commit**

```bash
git add pikafish_auto/Pikafish_auto4.py
git commit -m "chore: final verification of auto board setup"
```
