# 悔棋功能重构 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 重构悔棋功能：新增人类废棋区多帧投票识别，简化 request_undo 只负责局面恢复，重写 _handle_undo 实现四场景指令连续发送。

**Architecture:** VisionController 新增 `recognize_waste_area` 做多帧 HoughCircles+YOLO 识别，GameController 新增 `_find_piece_in_human_waste` 封装查找逻辑，`request_undo` 去掉指令生成只返回 MoveRecord 对，`_handle_undo` 获取帧、调局面恢复、生成四场景指令、连续发送。

**Tech Stack:** Python, OpenCV (HoughCircles), YOLO (ultralytics), cchess

---

### Task 1: 新增 `HUMAN_WASTE_ROI` 常量到 VisionController

**Files:**
- Modify: `pikafish_auto/Pikafish_auto4.py:471-475`

- [ ] **Step 1: 添加 HUMAN_WASTE_ROI 类属性**

在 `VisionController` 类的 `__init__` 之前添加常量。当前 line 472-475：

```python
class VisionController:
    """视觉识别 —— YOLO + HoughCircles 棋盘识别"""

    def __init__(self, yolo_model, video_stream):
        self._model = yolo_model
```

改为：

```python
class VisionController:
    """视觉识别 —— YOLO + HoughCircles 棋盘识别"""

    # 人类废棋区 ROI（warped 图坐标，棋盘左侧）
    HUMAN_WASTE_ROI = (10, 0, 103, 625)  # (x1, y1, x2, y2)

    def __init__(self, yolo_model, video_stream):
        self._model = yolo_model
```

- [ ] **Step 2: 验证语法**

```bash
cd "e:/chess_robot/project/oo/2026---" && python -c "import ast; ast.parse(open('pikafish_auto/Pikafish_auto4.py', encoding='utf-8').read()); print('语法 OK')"
```

Expected: `语法 OK`

- [ ] **Step 3: Commit**

```bash
git add pikafish_auto/Pikafish_auto4.py
git commit -m "feat: add HUMAN_WASTE_ROI constant to VisionController"
```

---

### Task 2: 新增 `recognize_waste_area` 方法到 VisionController

**Files:**
- Modify: `pikafish_auto/Pikafish_auto4.py` — 在 `detect_piece_at` 方法之后（约 line 889）插入新方法

- [ ] **Step 1: 实现 `recognize_waste_area` 多帧投票方法**

在 `detect_piece_at` 的 `return piece_type, detected_center`（line 889）之后、`_uci_to_pixel` 方法（line 891）之前，插入：

```python
    def recognize_waste_area(self, roi_rect: Tuple[int, int, int, int]
                             ) -> List[Tuple[str, int, int]]:
        """在指定 ROI 内多帧投票识别棋子，返回 [(label, x, y), ...]。

        连续取 N=5 帧，每帧 HoughCircles 寻圆 + YOLO 分类。
        同一 (label, 位置) 出现在 >=3 帧才确认，返回 warped 图坐标。
        """
        N_FRAMES = 5
        VOTE_MIN = 3
        PROXIMITY = 30  # 相近位置阈值（像素）

        x1, y1, x2, y2 = roi_rect
        all_detections: List[List[Tuple[str, int, int]]] = []

        for _ in range(N_FRAMES):
            frame = self._stream.get_latest_warped()
            if frame is None:
                continue

            roi = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 5)

            circles = cv2.HoughCircles(
                gray, cv2.HOUGH_GRADIENT, dp=1,
                minDist=32, param1=49, param2=30,
                minRadius=15, maxRadius=25
            )

            frame_dets = []
            if circles is not None:
                circles = np.round(circles[0, :]).astype(int)
                for cx, cy, r in circles:
                    rx1 = max(0, cx - 24)
                    ry1 = max(0, cy - 24)
                    rx2 = min(roi.shape[1], cx + 24)
                    ry2 = min(roi.shape[0], cy + 24)
                    piece_roi = roi[ry1:ry2, rx1:rx2]
                    if piece_roi.size == 0:
                        continue

                    results = self._model.predict(piece_roi, verbose=False)
                    if results and results[0].probs is not None:
                        conf = results[0].probs.top1conf
                        if conf > Config.CONF_THRESH:
                            label = results[0].names[results[0].probs.top1]
                            frame_dets.append((label, x1 + cx, y1 + cy))

            all_detections.append(frame_dets)
            time.sleep(0.06)

        # 多帧投票：按 (label, 近似位置) 分组，>=3 帧确认
        groups: Dict[Tuple[str, int, int], Dict] = {}
        for fi, dets in enumerate(all_detections):
            for label, wx, wy in dets:
                key = (label, round(wx / PROXIMITY) * PROXIMITY,
                       round(wy / PROXIMITY) * PROXIMITY)
                if key not in groups:
                    groups[key] = {'frames': set(), 'xs': [], 'ys': []}
                groups[key]['frames'].add(fi)
                groups[key]['xs'].append(wx)
                groups[key]['ys'].append(wy)

        confirmed = []
        for (label, _, _), data in groups.items():
            if len(data['frames']) >= VOTE_MIN:
                avg_x = int(np.mean(data['xs']))
                avg_y = int(np.mean(data['ys']))
                confirmed.append((label, avg_x, avg_y))

        return confirmed
```

- [ ] **Step 2: 验证语法**

```bash
cd "e:/chess_robot/project/oo/2026---" && python -c "import ast; ast.parse(open('pikafish_auto/Pikafish_auto4.py', encoding='utf-8').read()); print('语法 OK')"
```

Expected: `语法 OK`

- [ ] **Step 3: Commit**

```bash
git add pikafish_auto/Pikafish_auto4.py
git commit -m "feat: add recognize_waste_area with multi-frame voting to VisionController"
```

---

### Task 3: 简化 `request_undo` —— 移除指令生成

**Files:**
- Modify: `pikafish_auto/Pikafish_auto4.py:1772-1834`

- [ ] **Step 1: 重写 `request_undo` 方法体**

当前代码（line 1772-1834）：

```python
    def request_undo(self) -> Optional[List[str]]:
        """撤销两步（人+引擎），返回机械臂复位坐标指令列表"""
        if len(self._move_history) < 2:
            print("[悔棋] 走法数不足，无法悔棋")
            return None

        m_engine = self._move_history.pop()
        m_human = self._move_history.pop()

        # --- 1. 反向更新 _last_board ---
        # 先撤销引擎走法
        rev_eng = m_engine.uci[2:] + m_engine.uci[:2]
        self._last_board = self._vis.apply_uci_move(self._last_board, rev_eng)
        if m_engine.is_capture and m_engine.captured_piece:
            col = ord(m_engine.uci[2]) - ord('a')
            row = 9 - int(m_engine.uci[3])
            self._last_board[row][col] = m_engine.captured_piece

        # 再撤销人类走法
        rev_human = m_human.uci[2:] + m_human.uci[:2]
        self._last_board = self._vis.apply_uci_move(self._last_board, rev_human)
        if m_human.is_capture and m_human.captured_piece:
            col = ord(m_human.uci[2]) - ord('a')
            row = 9 - int(m_human.uci[3])
            self._last_board[row][col] = m_human.captured_piece

        # --- 2. 更新 _situation ---
        parts = self._situation.strip().split()
        moves_before = " ".join(parts[:-2]) if len(parts) >= 2 else ""
        self._situation = moves_before

        # --- 3. 恢复 _use_chess_count ---
        if m_engine.is_capture:
            self._use_chess_count += 1
        if m_human.is_capture:
            self._use_chess_count += 1

        # --- 4. 重建 cchess 局面 ---
        if self._initial_fen:
            self._cchess = cchess.Board(self._initial_fen)
        else:
            self._cchess = cchess.Board()
        if self._situation.strip():
            for m in self._situation.strip().split():
                self._cchess.push(cchess.Move.from_uci(m))

        # --- 5. 重启引擎 ---
        self._engine.restart(self._situation.strip(), self._initial_fen)

        # --- 6. 机械臂指令：人类棋子复位 → 引擎棋子复位 → 吃子恢复 ---
        cmds = []
        human_dst = self._vis.uci_to_pixel_public(m_human.uci[2:])
        human_src = self._vis.uci_to_pixel_public(m_human.uci[:2])
        cmds.append(f"棋子移动:{human_dst},{human_src}")

        if m_engine.src_xy and m_engine.dst_xy:
            cmds.append(f"棋子移动:{m_engine.dst_xy},{m_engine.src_xy}")

        if m_engine.is_capture and m_engine.waste_xy:
            cmds.append(f"棋子移动:{m_engine.waste_xy},{m_engine.dst_xy}")

        print(f"[悔棋] 已撤销引擎 {m_engine.uci} 和人类 {m_human.uci}")
        return cmds
```

替换为（保留步骤 1-5，替换步骤 6 + 返回值）：

```python
    def request_undo(self) -> Optional[Tuple[MoveRecord, MoveRecord]]:
        """撤销两步（人+引擎），只负责局面恢复，返回弹出的一对 MoveRecord。"""
        if len(self._move_history) < 2:
            print("[悔棋] 走法数不足，无法悔棋")
            return None

        m_engine = self._move_history.pop()
        m_human = self._move_history.pop()

        # --- 1. 反向更新 _last_board ---
        # 先撤销引擎走法
        rev_eng = m_engine.uci[2:] + m_engine.uci[:2]
        self._last_board = self._vis.apply_uci_move(self._last_board, rev_eng)
        if m_engine.is_capture and m_engine.captured_piece:
            col = ord(m_engine.uci[2]) - ord('a')
            row = 9 - int(m_engine.uci[3])
            self._last_board[row][col] = m_engine.captured_piece

        # 再撤销人类走法
        rev_human = m_human.uci[2:] + m_human.uci[:2]
        self._last_board = self._vis.apply_uci_move(self._last_board, rev_human)
        if m_human.is_capture and m_human.captured_piece:
            col = ord(m_human.uci[2]) - ord('a')
            row = 9 - int(m_human.uci[3])
            self._last_board[row][col] = m_human.captured_piece

        # --- 2. 更新 _situation ---
        parts = self._situation.strip().split()
        moves_before = " ".join(parts[:-2]) if len(parts) >= 2 else ""
        self._situation = moves_before

        # --- 3. 恢复 _use_chess_count ---
        if m_engine.is_capture:
            self._use_chess_count += 1
        if m_human.is_capture:
            self._use_chess_count += 1

        # --- 4. 重建 cchess 局面 ---
        if self._initial_fen:
            self._cchess = cchess.Board(self._initial_fen)
        else:
            self._cchess = cchess.Board()
        if self._situation.strip():
            for m in self._situation.strip().split():
                self._cchess.push(cchess.Move.from_uci(m))

        # --- 5. 重启引擎 ---
        self._engine.restart(self._situation.strip(), self._initial_fen)

        print(f"[悔棋] 已撤销引擎 {m_engine.uci} 和人类 {m_human.uci}")
        return (m_engine, m_human)
```

注意：还需更新文件顶部的 `Tuple` import（line 11 已有 `from typing import List, Tuple, Dict, Optional`，已包含 `Tuple`）。

- [ ] **Step 2: 验证语法**

```bash
cd "e:/chess_robot/project/oo/2026---" && python -c "import ast; ast.parse(open('pikafish_auto/Pikafish_auto4.py', encoding='utf-8').read()); print('语法 OK')"
```

Expected: `语法 OK`

- [ ] **Step 3: Commit**

```bash
git add pikafish_auto/Pikafish_auto4.py
git commit -m "refactor: simplify request_undo to only restore board state"
```

---

### Task 4: 新增 `_find_piece_in_human_waste` 方法到 GameController

**Files:**
- Modify: `pikafish_auto/Pikafish_auto4.py` — 在 `request_undo` 之前（约 line 1770）插入

- [ ] **Step 1: 实现 `_find_piece_in_human_waste`**

在 `# 悔棋接口` 注释块内，`request_undo` 之前，插入：

```python
    def _find_piece_in_human_waste(self, piece_label: str) -> Tuple[int, int]:
        """从人类废棋区（左侧）通过视觉识别找到指定棋子，返回 warped 坐标。"""
        pieces = self._vis.recognize_waste_area(VisionController.HUMAN_WASTE_ROI)
        for label, x, y in pieces:
            if label == piece_label:
                print(f"[悔棋] 在人类废棋区找到 {piece_label} @ ({x}, {y})")
                return (x, y)
        # 找不到则返回废棋区中心默认坐标
        print(f"[悔棋] 未在人类废棋区找到 {piece_label}，使用默认坐标")
        return (52, 312)
```

- [ ] **Step 2: 验证语法**

```bash
cd "e:/chess_robot/project/oo/2026---" && python -c "import ast; ast.parse(open('pikafish_auto/Pikafish_auto4.py', encoding='utf-8').read()); print('语法 OK')"
```

Expected: `语法 OK`

- [ ] **Step 3: Commit**

```bash
git add pikafish_auto/Pikafish_auto4.py
git commit -m "feat: add _find_piece_in_human_waste method to GameController"
```

---

### Task 5: 重写 `_handle_undo`

**Files:**
- Modify: `pikafish_auto/Pikafish_auto4.py:1610-1618`

- [ ] **Step 1: 替换 `_handle_undo` 方法体**

当前代码：

```python
    def _handle_undo(self):
        cmds = self.request_undo()
        if cmds:
            for cmd in cmds:
                self._p.send(cmd)
                self._p.wait_for("运动完成")
        self._p.send("请玩家落子")
        self._p.send("文字:请落子")
        self._state = self.State.HUMAN_TURN
```

替换为：

```python
    def _handle_undo(self):
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
        #    必须在前！被吃棋子是人类棋子，先放回棋盘，步骤③才能移走
        if m_engine.is_capture and m_engine.waste_xy:
            cmds.append(f"棋子移动:{m_engine.waste_xy},{m_engine.dst_xy}")

        # ③ 人类棋子移回：dst → src
        #    此时人类棋子一定在棋盘上（从未被吃 or 已被步骤②恢复）
        cmds.append(f"棋子移动:{human_dst},{human_src}")

        # ④ 若人类吃子：从人类废棋区 → 人类 dst
        if m_human.is_capture and m_human.captured_piece:
            xy = self._find_piece_in_human_waste(m_human.captured_piece)
            cmds.append(f"棋子移动:{xy},{human_dst}")

        # 连续发送，不等待
        for cmd in cmds:
            self._p.send(cmd)

        self._p.send("请玩家落子")
        self._p.send("文字:请落子")
        self._state = self.State.HUMAN_TURN
```

- [ ] **Step 2: 验证语法**

```bash
cd "e:/chess_robot/project/oo/2026---" && python -c "import ast; ast.parse(open('pikafish_auto/Pikafish_auto4.py', encoding='utf-8').read()); print('语法 OK')"
```

Expected: `语法 OK`

- [ ] **Step 3: Commit**

```bash
git add pikafish_auto/Pikafish_auto4.py
git commit -m "feat: rewrite _handle_undo with 4-scenario command logic"
```

---

### Task 6: 端到端逻辑验证

- [ ] **Step 1: 确认所有 import 和类型注解一致**

检查文件顶部 import 行（line 11）：

```python
from typing import List, Tuple, Dict, Optional
```

确认 `Tuple` 已 import（是，line 11 已有）。

确认 `time` 已 import（是，line 9 `import time`）。

确认 `np` 已 import（是，line 14 `import numpy as np`）。

- [ ] **Step 2: 确认 MoveRecord 字段完整性**

检查 `MoveRecord.__slots__`（line 1201）：

```python
__slots__ = ('uci', 'src_xy', 'dst_xy', 'is_capture', 'captured_piece', 'waste_xy')
```

确认所有 undo 需要访问的字段都在：`uci`（人类坐标推算），`src_xy`/`dst_xy`（引擎坐标），`is_capture`/`captured_piece`/`waste_xy`（吃子恢复）。全部到位。

- [ ] **Step 3: 全文件语法检查 + import 验证**

```bash
cd "e:/chess_robot/project/oo/2026---" && python -c "
import ast
tree = ast.parse(open('pikafish_auto/Pikafish_auto4.py', encoding='utf-8').read())
print('语法检查通过')
# 检查关键方法和类是否存在
classes = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
funcs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
assert 'VisionController' in classes, '缺少 VisionController'
assert 'GameController' in classes, '缺少 GameController'
assert 'recognize_waste_area' in funcs, '缺少 recognize_waste_area'
assert '_find_piece_in_human_waste' in funcs, '缺少 _find_piece_in_human_waste'
assert 'request_undo' in funcs, '缺少 request_undo'
assert '_handle_undo' in funcs, '缺少 _handle_undo'
print('所有关键方法/类存在')
print('全量验证通过')
"
```

Expected: `语法检查通过` + `所有关键方法/类存在` + `全量验证通过`

- [ ] **Step 4: Commit**

```bash
git add pikafish_auto/Pikafish_auto4.py
git commit -m "chore: final verification of undo refactor"
```
