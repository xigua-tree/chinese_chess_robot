# 一键摆盘红/黑 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现一键摆盘红/黑：服务端发送"一键摆盘红"或"一键摆盘黑"后机器人按对应 FEN 摆放，开局校验自动匹配执方。

**Architecture:** Config 新增 `RED_STANDARD_OPENING_FEN`；`_handle_wait_start` 解析指令设置 `self._setup_fen`，开局校验用 `detect_side` 选 FEN；`_handle_auto_setup` 用 `self._setup_fen` 替代硬编码。

**Tech Stack:** Python

---

### Task 1: 新增 `RED_STANDARD_OPENING_FEN` 到 Config

**Files:**
- Modify: `pikafish_auto/Pikafish_auto4.py` — Config 类，`STANDARD_OPENING_FEN` 之后

- [ ] **Step 1: 添加 FEN 常量**

在 `STANDARD_OPENING_FEN` 定义之后（line 94）、`fen_to_uci_pieces` 之前（line 96）插入：

```python
    # 人类执红时的标准开局：红方在上（摄像头顶部=人类侧）
    RED_STANDARD_OPENING_FEN = (
        "RNBAKABNR/9/1C5C1/P1P1P1P1P/9/9/"
        "p1p1p1p1p/1c5c1/9/rnbakabnr w - - 0 1"
    )
```

- [ ] **Step 2: 验证语法**

```bash
cd "e:/chess_robot/project/oo/2026---" && python -c "import ast; ast.parse(open('pikafish_auto/Pikafish_auto4.py', encoding='utf-8').read()); print('语法 OK')"
```

Expected: `语法 OK`

- [ ] **Step 3: Commit**

```bash
git add pikafish_auto/Pikafish_auto4.py
git commit -m "feat: add RED_STANDARD_OPENING_FEN for human-red setup"
```

---

### Task 2: 解析"一键摆盘红/黑"并设置 `self._setup_fen`

**Files:**
- Modify: `pikafish_auto/Pikafish_auto4.py` — `_handle_wait_start` 方法

- [ ] **Step 1: 修改一键摆盘判断逻辑**

替换 `_handle_wait_start` 中 line 1413-1415：

旧：
```python
        if msg.startswith("一键摆盘"):
            self._state = self.State.AUTO_SETUP
            return
```

新：
```python
        if msg.startswith("一键摆盘"):
            if "红" in msg:
                self._setup_fen = Config.RED_STANDARD_OPENING_FEN
            else:
                self._setup_fen = Config.STANDARD_OPENING_FEN
            self._state = self.State.AUTO_SETUP
            return
```

- [ ] **Step 2: 验证语法**

```bash
cd "e:/chess_robot/project/oo/2026---" && python -c "import ast; ast.parse(open('pikafish_auto/Pikafish_auto4.py', encoding='utf-8').read()); print('语法 OK')"
```

Expected: `语法 OK`

- [ ] **Step 3: Commit**

```bash
git add pikafish_auto/Pikafish_auto4.py
git commit -m "feat: parse red/black setup command and set _setup_fen"
```

---

### Task 3: 开局校验用 `detect_side` 自动选 FEN

**Files:**
- Modify: `pikafish_auto/Pikafish_auto4.py` — `_handle_wait_start` 方法中的开局校验循环

- [ ] **Step 1: 修改开局校验逻辑**

替换 line 1446-1448 和 line 1471-1474：

旧：
```python
        else:
            expected_fen = Config.STANDARD_OPENING_FEN
            self._initial_fen = expected_fen

        # 识别 + 校验循环，直到通过为止
        board = None
        while True:
            ...
            if is_endgame:
                ...
            else:
                if self._validate_against_expected(board, Config.STANDARD_OPENING_FEN):
                    break
                print("[校验] 开局校验未通过，重新识别...")
```

注意 `expected_fen` 赋值移到循环内部，每次识别后根据 `detect_side` 动态选择。

完整替换为：

```python
        else:
            expected_fen = Config.STANDARD_OPENING_FEN  # 循环内会被覆盖
            self._initial_fen = expected_fen  # 循环内会被覆盖

        # 识别 + 校验循环，直到通过为止
        board = None
        while True:
            if self._p.shutdown_requested():
                return
            board = self._vis.capture_stable_board()
            if board is None:
                print("初始棋盘识别失败，重试...")
                time.sleep(0.5)
                continue

            recognized_fen = self._vis.construct_fen(board, side_to_move="w")

            if is_endgame:
                ...
            else:
                # 根据棋盘上人类执方选择对应 FEN 校验
                human_side = self._vis.detect_side(board)
                if human_side == "red":
                    expected_fen = Config.RED_STANDARD_OPENING_FEN
                else:
                    expected_fen = Config.STANDARD_OPENING_FEN
                self._initial_fen = expected_fen

                if self._validate_against_expected(board, expected_fen):
                    break
                print("[校验] 开局校验未通过，重新识别...")
```

- [ ] **Step 2: 验证语法**

```bash
cd "e:/chess_robot/project/oo/2026---" && python -c "import ast; ast.parse(open('pikafish_auto/Pikafish_auto4.py', encoding='utf-8').read()); print('语法 OK')"
```

Expected: `语法 OK`

- [ ] **Step 3: Commit**

```bash
git add pikafish_auto/Pikafish_auto4.py
git commit -m "feat: use detect_side to auto-select FEN for game start validation"
```

---

### Task 4: `_handle_auto_setup` 用 `self._setup_fen` 替代硬编码

**Files:**
- Modify: `pikafish_auto/Pikafish_auto4.py` — `_handle_auto_setup` 方法

- [ ] **Step 1: 替换 4 处硬编码引用**

在 `_handle_auto_setup` 中，将所有 `Config.STANDARD_OPENING_FEN` 替换为 `self._setup_fen`（共 3 处）：

**第一处**（line 1739，`fen_to_uci_pieces`）：

旧：
```python
        target_pieces = Config.fen_to_uci_pieces(Config.STANDARD_OPENING_FEN)
```

新：
```python
        target_pieces = Config.fen_to_uci_pieces(self._setup_fen)
```

**第二处**（line ~1752，外层循环整盘校验）：

旧：
```python
            if board and self._validate_against_expected(board, Config.STANDARD_OPENING_FEN):
```

新：
```python
            if board and self._validate_against_expected(board, self._setup_fen):
```

**第三处**（line ~1850，最终校验）：

旧：
```python
        if final_board and self._validate_against_expected(final_board, Config.STANDARD_OPENING_FEN):
```

新：
```python
        if final_board and self._validate_against_expected(final_board, self._setup_fen):
```

- [ ] **Step 2: 验证替换结果**

```bash
cd "e:/chess_robot/project/oo/2026---" && grep -n "STANDARD_OPENING_FEN" pikafish_auto/Pikafish_auto4.py
```

`_handle_auto_setup` 方法内不应再有 `Config.STANDARD_OPENING_FEN`，只有 `self._setup_fen`。

- [ ] **Step 3: 验证语法**

```bash
cd "e:/chess_robot/project/oo/2026---" && python -c "import ast; ast.parse(open('pikafish_auto/Pikafish_auto4.py', encoding='utf-8').read()); print('语法 OK')"
```

Expected: `语法 OK`

- [ ] **Step 4: Commit**

```bash
git add pikafish_auto/Pikafish_auto4.py
git commit -m "feat: use self._setup_fen in _handle_auto_setup instead of hardcoded FEN"
```

---

### Task 5: 端到端验证

- [ ] **Step 1: 确认关键定义和引用一致性**

```bash
cd "e:/chess_robot/project/oo/2026---" && python -c "
import ast
tree = ast.parse(open('pikafish_auto/Pikafish_auto4.py', encoding='utf-8').read())

# 1. Config.RED_STANDARD_OPENING_FEN 存在
classes = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
config = classes.get('Config')
red_fen_found = False
for node in ast.walk(config):
    if isinstance(node, ast.Assign):
        for t in node.targets:
            if isinstance(t, ast.Attribute) and t.attr == 'RED_STANDARD_OPENING_FEN':
                red_fen_found = True
                break
print(f'RED_STANDARD_OPENING_FEN: {\"OK\" if red_fen_found else \"MISSING\"}')

# 2. _setup_fen 引用存在
setup_fen_refs = 0
for node in ast.walk(tree):
    if isinstance(node, ast.Attribute) and node.attr == '_setup_fen':
        setup_fen_refs += 1
print(f'self._setup_fen 引用数: {setup_fen_refs}')

# 3. _handle_auto_setup 内没有 Config.STANDARD_OPENING_FEN
funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
auto_setup = funcs.get('_handle_auto_setup')
if auto_setup:
    hardcoded = 0
    for node in ast.walk(auto_setup):
        if isinstance(node, ast.Attribute) and node.attr == 'STANDARD_OPENING_FEN':
            if isinstance(node.value, ast.Attribute) and node.value.attr == 'Config':
                hardcoded += 1
    print(f'_handle_auto_setup 内 Config.STANDARD_OPENING_FEN: {hardcoded} (期望 0)')

# 4. detect_side 在 _handle_wait_start 的校验循环中被调用
wait_start = funcs.get('_handle_wait_start')
if wait_start:
    detect_calls = 0
    for node in ast.walk(wait_start):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == 'detect_side':
                detect_calls += 1
    print(f'_handle_wait_start 内 detect_side 调用: {detect_calls} (期望 ≥1)')

print('验证完成')
"
```

Expected: 所有检查通过

- [ ] **Step 2: Commit**

```bash
git add pikafish_auto/Pikafish_auto4.py
git commit -m "chore: final verification of red-black auto setup"
```
