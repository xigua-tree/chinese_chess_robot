# Pikafish_auto4.py 重构 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 Pikafish_auto4.py 从 1058 行单文件重构为 5 类 + 1 入口的类化架构，修复已知 bug，增加引擎难度设置、协议更新、引擎崩溃恢复和悔棋预留。

**Architecture:** 单文件内拆分为 Config / ProtocolClient / EngineController / VisionController / GameController 五个类，通过 main() 组装。GameController 使用显式状态枚举消除人类先行/引擎先行两套重复循环。

**Tech Stack:** Python 3, OpenCV (cv2), NumPy, ultralytics YOLO, cchess, threading, socket, subprocess

---

## File Structure

- **Rewrite:** `pikafish_auto/Pikafish_auto4.py` — 完整重写，保留原文件名方便部署
- 不创建其他文件（单文件部署策略）

---

### Task 1: File header, imports, and Config class

**Files:**
- Rewrite: `pikafish_auto/Pikafish_auto4.py`

- [ ] **Step 1: Write file skeleton with imports and Config class**

```python
# -*- coding: utf-8 -*-
"""象棋机器人自动化对弈系统 —— Pikafish_auto4.py 重构版"""

import os
import sys
import subprocess
import threading
import queue
import time
from enum import Enum
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
import socket
import cchess
from ultralytics import YOLO


class Config:
    """集中配置 —— 所有可调参数"""

    # --- 引擎 ---
    PIKAFISH_PATH = "pikafish\\pikafish-avx2.exe"
    LEVEL_MOVETIME = {1: 500, 2: 1500, 3: 3000, 4: 6000}

    # --- 模型 ---
    MODEL_PATH = "model_train/runs/classify/chess_r_test/weights/best.pt"
    CONF_THRESH = 0.70

    # --- 棋盘定位 ---
    FIXED_PTS = np.float32([
        [310, 75], [935, 90], [258, 670], [965, 685]
    ])
    WARP_W, WARP_H = 707, 630
    DST_PTS = np.float32([[0, 0], [WARP_W, 0], [0, WARP_H], [WARP_W, WARP_H]])
    STATIC_MATRIX = cv2.getPerspectiveTransform(FIXED_PTS, DST_PTS)

    GRID_LT = (120, 45)
    GRID_RT = (580, 36)
    GRID_LB = (130, 600)
    GRID_RB = (582, 602)

    # --- 视觉 ---
    STABLE_FRAMES = 3
    EDGE_PADDING = 3
    MAX_CAPTURE_ATTEMPTS = 40

    # --- 网络 ---
    SERVER_HOST = "127.0.0.1"
    SERVER_PORT = 8888

    # --- 棋子映射表 ---
    FEN_MAP = {
        'R_Kin': 'K', 'R_Car': 'R', 'R_Hor': 'N', 'R_Can': 'C',
        'R_Ele': 'B', 'R_Shi': 'A', 'R_Paw': 'P',
        'B_Kin': 'k', 'B_Car': 'r', 'B_Hor': 'n', 'B_Can': 'c',
        'B_Ele': 'b', 'B_Shi': 'a', 'B_Paw': 'p'
    }
    CHINESE_MAP = {
        'R_Kin': '帅', 'R_Car': '俥', 'R_Hor': '傌', 'R_Can': '炮',
        'R_Ele': '相', 'R_Shi': '仕', 'R_Paw': '兵',
        'B_Kin': '将', 'B_Car': '車', 'B_Hor': '馬', 'B_Can': '砲',
        'B_Ele': '象', 'B_Shi': '士', 'B_Paw': '卒'
    }
    INV_FEN_MAP = {
        'K': 'R_Kin', 'R': 'R_Car', 'N': 'R_Hor', 'C': 'R_Can',
        'B': 'R_Ele', 'A': 'R_Shi', 'P': 'R_Paw',
        'k': 'B_Kin', 'r': 'B_Car', 'n': 'B_Hor', 'c': 'B_Can',
        'b': 'B_Ele', 'a': 'B_Shi', 'p': 'B_Paw'
    }
```

- [ ] **Step 2: Verify syntax**

Run: `python -m py_compile pikafish_auto/Pikafish_auto4.py`
Expected: No output (compiles successfully)

- [ ] **Step 3: Commit**

```bash
git add pikafish_auto/Pikafish_auto4.py
git commit -m "refactor: add Config class with centralized settings"
```

---

### Task 2: ProtocolClient class

**Files:**
- Modify: `pikafish_auto/Pikafish_auto4.py` — append ProtocolClient class

- [ ] **Step 1: Append ProtocolClient after Config**

Insert the following class definition after the Config class (before any other code at file end):

```python
class ProtocolClient:
    """TCP 通信封装 —— 与 Qt 服务端交互"""

    def __init__(self):
        self._sock: Optional[socket.socket] = None
        self._queue: queue.Queue = queue.Queue()
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None

    def connect(self, host: str = None, port: int = None):
        """创建 socket 连接并启动后台接收线程"""
        host = host or Config.SERVER_HOST
        port = port or Config.SERVER_PORT
        self._sock = socket.socket()
        self._sock.connect((host, port))
        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()
        print(f"[协议] 已连接 {host}:{port}")

    def _recv_loop(self):
        """后台线程：持续监听服务端消息"""
        while self._running:
            try:
                data = self._sock.recv(1024)
                if not data:
                    print("[协议] 服务器断开连接")
                    self._running = False
                    break
                msg = data.decode().strip()
                print(f"[收到] {msg}")
                self._queue.put(msg)
            except Exception as e:
                print(f"[协议] 接收异常: {e}")
                self._running = False
                break

    def send(self, msg: str):
        """发送消息到服务端"""
        if not self._sock:
            print("[协议] 未连接，无法发送")
            return
        try:
            self._sock.send((msg + "\n").encode())
            print(f"[发送] {msg}")
        except Exception as e:
            print(f"[协议] 发送异常: {e}")
            self._running = False

    def wait_for(self, expected: str, timeout: float = None) -> bool:
        """阻塞等待精确匹配的消息"""
        print(f"[等待] {expected} ...")
        deadline = time.time() + (timeout if timeout else 999999)
        while time.time() < deadline:
            remaining = max(0.1, deadline - time.time())
            try:
                msg = self._queue.get(timeout=min(1.0, remaining))
                if msg == expected:
                    print(f"[匹配] {expected}")
                    return True
            except queue.Empty:
                pass
        print(f"[超时] 等待 {expected} 超时")
        return False

    def wait_for_prefix(self, prefix: str, timeout: float = None) -> Optional[str]:
        """阻塞等待前缀匹配的消息，返回完整消息"""
        print(f"[等待前缀] {prefix} ...")
        deadline = time.time() + (timeout if timeout else 999999)
        while time.time() < deadline:
            remaining = max(0.1, deadline - time.time())
            try:
                msg = self._queue.get(timeout=min(1.0, remaining))
                if msg.startswith(prefix):
                    print(f"[匹配] {msg}")
                    return msg
            except queue.Empty:
                pass
        print(f"[超时] 等待前缀 '{prefix}' 超时")
        return None

    def peek(self) -> Optional[str]:
        """非阻塞读取一条消息（用于检测悔棋等中断信号）"""
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def is_connected(self) -> bool:
        return self._running

    def disconnect(self):
        """断开连接并清理资源"""
        self._running = False
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        print("[协议] 已断开连接")
```

- [ ] **Step 2: Verify syntax**

Run: `python -m py_compile pikafish_auto/Pikafish_auto4.py`
Expected: No output

- [ ] **Step 3: Commit**

```bash
git add pikafish_auto/Pikafish_auto4.py
git commit -m "refactor: add ProtocolClient class with new message format support"
```

---

### Task 3: EngineController class

**Files:**
- Modify: `pikafish_auto/Pikafish_auto4.py` — append EngineController class

- [ ] **Step 1: Append EngineController after ProtocolClient**

```python
class EngineController:
    """UCI 象棋引擎进程封装 —— 含崩溃自动恢复"""

    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.alive: bool = False
        self._line_queue: queue.Queue = queue.Queue()
        self._reader_thread: Optional[threading.Thread] = None
        self._level: int = 1

    # ------------------------------------------------------------------
    def start(self):
        """启动引擎子进程并创建持久化读取线程"""
        try:
            self.process = subprocess.Popen(
                [Config.PIKAFISH_PATH],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            self.alive = True
            self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
            self._reader_thread.start()
            print("[引擎] 进程已启动")
        except Exception as e:
            raise RuntimeError(f"[引擎] CreateProcess 失败: {e}")

    def _reader_loop(self):
        """持久化线程：持续读取引擎 stdout 行并放入队列"""
        while self.alive:
            try:
                line = self.process.stdout.readline()
                if line:
                    self._line_queue.put(line.strip())
                else:
                    self.alive = False
                    break
            except Exception:
                self.alive = False
                break

    # ------------------------------------------------------------------
    def write(self, line: str):
        """向引擎发送命令"""
        if not self.alive:
            return
        try:
            self.process.stdin.write(line + "\n")
            self.process.stdin.flush()
            print(f"[→Engine] {line}")
        except (BrokenPipeError, OSError):
            self.alive = False
            print("[引擎] 写入失败，进程可能已退出")

    def readline(self, timeout_ms: int = 10000) -> str:
        """从队列读取一行（带超时），统一超时处理"""
        try:
            return self._line_queue.get(timeout=timeout_ms / 1000.0)
        except queue.Empty:
            return ""

    # ------------------------------------------------------------------
    def uci_handshake(self, level: int = 1) -> bool:
        """UCI 协议握手并设置难度"""
        self._level = level
        self.write("uci")
        for _ in range(50):
            line = self.readline(3000)
            if line == "uciok":
                break
            if not self.alive:
                return False
        self.write("isready")
        for _ in range(50):
            line = self.readline(3000)
            if line == "readyok":
                break
            if not self.alive:
                return False
        print(f"[引擎] 握手完成，难度={level} (movetime={Config.LEVEL_MOVETIME.get(level, 3000)}ms)")
        return True

    def set_level(self, level: int):
        """动态修改引擎难度"""
        self._level = max(1, min(4, level))
        print(f"[引擎] 难度已设置为 {self._level}")

    # ------------------------------------------------------------------
    def get_best_move(self, fen_moves: str) -> str:
        """发送当前局面并获取最佳走法"""
        if not self.alive:
            raise RuntimeError("[引擎] 引擎不可用")

        if fen_moves.strip():
            self.write(f"position startpos moves {fen_moves}")
        else:
            self.write("position startpos")

        movetime = Config.LEVEL_MOVETIME.get(self._level, 3000)
        self.write(f"go movetime {movetime}")

        deadline = time.time() + (movetime + 5000) / 1000.0
        while time.time() < deadline:
            remaining_ms = max(100, int((deadline - time.time()) * 1000))
            line = self.readline(remaining_ms)
            if line.startswith("bestmove"):
                parts = line.split()
                if len(parts) >= 2:
                    return parts[1]
            if not self.alive:
                raise RuntimeError("[引擎] 搜索期间引擎崩溃")
        return ""

    # ------------------------------------------------------------------
    def restart(self, fen_moves: str) -> bool:
        """崩溃恢复：杀掉旧进程、启动新进程、恢复局面"""
        print("[引擎] 正在重启...")
        self.stop()
        try:
            self.start()
        except Exception as e:
            print(f"[引擎] 重启失败: {e}")
            return False
        if not self.uci_handshake(self._level):
            return False
        if fen_moves.strip():
            self.write(f"position startpos moves {fen_moves}")
        print("[引擎] 重启完成，局面已同步")
        return True

    def is_alive(self) -> bool:
        """检测引擎进程是否存活"""
        if not self.alive:
            return False
        if self.process and self.process.poll() is not None:
            self.alive = False
        return self.alive

    # ------------------------------------------------------------------
    def stop(self):
        """优雅停止引擎进程"""
        if not self.alive:
            return
        try:
            self.write("quit")
        except Exception:
            pass
        time.sleep(0.2)
        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
                time.sleep(0.1)
                self.process.kill()
                self.process.wait()
            except Exception:
                pass
        self.alive = False
        print("[引擎] 已停止")
```

- [ ] **Step 2: Verify syntax**

Run: `python -m py_compile pikafish_auto/Pikafish_auto4.py`
Expected: No output

- [ ] **Step 3: Commit**

```bash
git add pikafish_auto/Pikafish_auto4.py
git commit -m "refactor: add EngineController with persistent reader thread and crash recovery"
```

---

### Task 4: VisionController class

**Files:**
- Modify: `pikafish_auto/Pikafish_auto4.py` — append VisionController class

- [ ] **Step 1: Append VisionController after EngineController**

```python
class VisionController:
    """视觉识别 —— YOLO + HoughCircles 棋盘识别"""

    def __init__(self, yolo_model, video_stream):
        self._model = yolo_model
        self._stream = video_stream
        self._red_perspective: bool = True   # True=红方在下(row9)

    # ------------------------------------------------------------------
    # 棋盘识别
    # ------------------------------------------------------------------
    def recognize_board(self, warped_img: np.ndarray) -> List[List[str]]:
        """单帧识别：HoughCircles 寻圆 + YOLO 分类 → 10x9 矩阵"""
        board = [[None for _ in range(9)] for _ in range(10)]

        gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)

        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=32,
            param1=49, param2=31, minRadius=15, maxRadius=25
        )

        if circles is None:
            return board

        circles = np.round(circles[0, :]).astype(int)
        for cx, cy, r in circles:
            roi_size = 24 + Config.EDGE_PADDING
            x1, y1 = cx - roi_size, cy - roi_size
            x2, y2 = cx + roi_size, cy + roi_size

            # 边界棋子处理：用 BORDER_REPLICATE 扩展而非 clamp
            if x1 < 0 or y1 < 0 or x2 > Config.WARP_W or y2 > Config.WARP_H:
                x1c, y1c = max(0, x1), max(0, y1)
                x2c, y2c = min(Config.WARP_W, x2), min(Config.WARP_H, y2)
                roi = warped_img[y1c:y2c, x1c:x2c]
                roi = cv2.copyMakeBorder(
                    roi,
                    top=max(0, -y1), bottom=max(0, y2 - Config.WARP_H),
                    left=max(0, -x1), right=max(0, x2 - Config.WARP_W),
                    borderType=cv2.BORDER_REPLICATE
                )
            else:
                roi = warped_img[y1:y2, x1:x2]

            results = self._model.predict(roi, verbose=False)

            if results and results[0].probs is not None:
                conf = results[0].probs.top1conf
                if conf > Config.CONF_THRESH:
                    label = results[0].names[results[0].probs.top1]

                    # 可视化标注
                    color = (0, 0, 255) if label.startswith("R_") else (0, 0, 0)
                    cv2.circle(warped_img, (cx, cy), r, color, 2)
                    cv2.circle(warped_img, (cx, cy), 2, (0, 255, 0), -1)
                    display = label.split('_')[-1]
                    cv2.putText(warped_img, display, (cx - 15, cy - r - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                    # 映射到 10x9 网格
                    rel_x = (cx - Config.GRID_LT[0]) / (Config.GRID_RT[0] - Config.GRID_LT[0])
                    rel_y = (cy - Config.GRID_LT[1]) / (Config.GRID_LB[1] - Config.GRID_LT[1])
                    grid_c = min(8, max(0, int(round(rel_x * 8))))
                    grid_r = min(9, max(0, int(round(rel_y * 9))))

                    board[grid_r][grid_c] = label

        return board

    # ------------------------------------------------------------------
    def capture_stable_board(self, required_stable: int = None) -> Optional[List[List[str]]]:
        """多帧稳定捕获，附带棋子数合法性校验"""
        frames = required_stable or Config.STABLE_FRAMES
        last_board = None
        stable_count = 0

        print(f"[视觉] 多帧校验中 (目标连续一致: {frames} 帧)...")

        for i in range(Config.MAX_CAPTURE_ATTEMPTS):
            warped = self._stream.get_latest_warped()
            if warped is None:
                time.sleep(0.1)
                continue

            current = self.recognize_board(warped.copy())

            if last_board is None:
                last_board = current
                stable_count = 1
                continue

            if current == last_board:
                stable_count += 1
            else:
                if stable_count > 0:
                    print(f"[警告] 局面波动，重新校验... (尝试: {i})")
                stable_count = 1
                last_board = current

            if stable_count >= frames:
                total = sum(1 for row in current for p in row if p)
                if 0 < total <= 32:
                    return current
                else:
                    print(f"[警告] 棋子数异常 ({total})，重新采样")
                    stable_count = 0
                    last_board = None

            time.sleep(0.1)

        print("[错误] 无法获得稳定局面，请检查遮挡或光照")
        return None

    def capture_any_board(self) -> Optional[List[List[str]]]:
        """单帧快速识别（用于失败重试）"""
        warped = self._stream.get_latest_warped()
        if warped is None:
            return None
        return self.recognize_board(warped.copy())

    # ------------------------------------------------------------------
    # 走法计算
    # ------------------------------------------------------------------
    def get_move(self, prev_board: List[List[str]],
                 curr_board: List[List[str]]) -> Optional[str]:
        """差分两个局面，计算人类走法的 UCI 表示"""
        removed, added, changed = [], [], []

        for r in range(10):
            for c in range(9):
                had = bool(prev_board[r][c])
                now = bool(curr_board[r][c])
                if had and not now:
                    removed.append((r, c))
                elif not had and now:
                    added.append((r, c))
                elif had and now and prev_board[r][c] != curr_board[r][c]:
                    changed.append((r, c))

        if not removed and not added and not changed:
            print("[getMove] 未检测到变化")
            return None

        src, dst = (-1, -1), (-1, -1)

        if len(removed) == 1 and len(added) == 1 and not changed:
            src, dst = removed[0], added[0]
        elif len(removed) == 1 and len(changed) == 1 and not added:
            src, dst = removed[0], changed[0]
        else:
            dst_candidates = added + changed
            found = False
            for rmv in removed:
                piece_type = prev_board[rmv[0]][rmv[1]]
                for cand in dst_candidates:
                    if curr_board[cand[0]][cand[1]] == piece_type:
                        src, dst = rmv, cand
                        found = True
                        break
                if found:
                    break
            if not found:
                if removed:
                    src = removed[0]
                if changed:
                    dst = changed[0]
                elif added:
                    dst = added[0]

        if src[0] < 0 or dst[0] < 0:
            print(f"[getMove] 无法确定走法: removed={len(removed)} "
                  f"added={len(added)} changed={len(changed)}")
            return None

        return self._to_uci(src[1], src[0]) + self._to_uci(dst[1], dst[0])

    def _to_uci(self, col: int, row: int) -> str:
        """棋盘坐标 → UCI 坐标"""
        file_char = chr(ord('a') + col)
        rank = (9 - row) if self._red_perspective else row
        return f"{file_char}{rank}"

    # ------------------------------------------------------------------
    # 执子方检测
    # ------------------------------------------------------------------
    def detect_side(self, board: List[List[str]]) -> str:
        """通过上方三行检测人类执子方，返回 'red' 或 'black'"""
        for r in range(3):
            for c in range(9):
                piece = board[r][c]
                if piece == "R_Kin":
                    print("[识别] 棋盘上方发现红方帅 → 人类执红")
                    return "red"
                elif piece == "B_Kin":
                    print("[识别] 棋盘上方发现黑方将 → 人类执黑")
                    return "black"
        print("[警告] 未在上方三行找到将/帅，默认人类执红")
        return "red"

    # ------------------------------------------------------------------
    # 棋子定位（给机械臂）
    # ------------------------------------------------------------------
    def detect_piece_at(self, uci_pos: str,
                        current_frame: np.ndarray) -> Tuple[Optional[str], Tuple[int, int]]:
        """在指定 UCI 坐标附近定位棋子并识别类型"""
        if current_frame is None:
            return None, (0, 0)

        expected_x, expected_y = self._uci_to_pixel(uci_pos)

        padding = 24
        x1 = max(0, expected_x - padding)
        y1 = max(0, expected_y - padding)
        x2 = min(Config.WARP_W, expected_x + padding)
        y2 = min(Config.WARP_H, expected_y + padding)

        roi_img = current_frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)

        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
            param1=50, param2=30, minRadius=15, maxRadius=26
        )

        detected_center = (expected_x, expected_y)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            c = circles[0][0]
            detected_center = (x1 + c[0], y1 + c[1])
            cx, cy = detected_center
            rx1, ry1 = max(0, cx - 24), max(0, cy - 24)
            rx2, ry2 = min(Config.WARP_W, cx + 24), min(Config.WARP_H, cy + 24)
            roi_img = current_frame[int(ry1):int(ry2), int(rx1):int(rx2)]

        results = self._model.predict(roi_img, verbose=False)
        piece_type = None
        if results and results[0].probs is not None:
            if results[0].probs.top1conf > Config.CONF_THRESH:
                piece_type = results[0].names[results[0].probs.top1]

        return piece_type, detected_center

    def _uci_to_pixel(self, uci_pos: str) -> Tuple[int, int]:
        """UCI 坐标 → 透视图像素坐标"""
        if not uci_pos or len(uci_pos) < 2:
            return (0, 0)

        col_idx = ord(uci_pos[0].lower()) - ord('a')
        rank = int(uci_pos[1:])

        if self._red_perspective:
            row_idx = 9 - rank
        else:
            row_idx = rank

        rel_x = col_idx / 8.0
        rel_y = row_idx / 9.0

        top_x = Config.GRID_LT[0] + rel_x * (Config.GRID_RT[0] - Config.GRID_LT[0])
        top_y = Config.GRID_LT[1] + rel_x * (Config.GRID_RT[1] - Config.GRID_LT[1])
        bot_x = Config.GRID_LB[0] + rel_x * (Config.GRID_RB[0] - Config.GRID_LB[0])
        bot_y = Config.GRID_LB[1] + rel_x * (Config.GRID_RB[1] - Config.GRID_LB[1])

        target_x = int(top_x + rel_y * (bot_x - top_x))
        target_y = int(top_y + rel_y * (bot_y - top_y))
        return (target_x, target_y)

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------
    @property
    def red_perspective(self) -> bool:
        return self._red_perspective

    @red_perspective.setter
    def red_perspective(self, value: bool):
        self._red_perspective = value

    @staticmethod
    def count_pieces(board: List[List[str]]) -> Tuple[int, int]:
        """统计红黑棋子数"""
        red = black = 0
        for row in board:
            for p in row:
                if p and p.startswith("R_"):
                    red += 1
                elif p and p.startswith("B_"):
                    black += 1
        return red, black

    @staticmethod
    def board_to_fen(board: List[List[str]]) -> str:
        """10x9 矩阵 → FEN 字符串"""
        fen_rows = []
        for r in range(10):
            empty = 0
            row_str = ""
            for c in range(9):
                piece = board[r][c]
                if piece is None or piece == "":
                    empty += 1
                else:
                    if empty > 0:
                        row_str += str(empty)
                        empty = 0
                    row_str += Config.FEN_MAP.get(piece, '?')
            if empty > 0:
                row_str += str(empty)
            fen_rows.append(row_str)
        return "/".join(fen_rows)

    @staticmethod
    def board_from_fen(fen: str) -> List[List[str]]:
        """FEN → 10x9 矩阵"""
        board_part = fen.split(' ')[0]
        rows = board_part.split('/')
        board = [[None for _ in range(9)] for _ in range(10)]
        for r, row_str in enumerate(rows):
            c = 0
            for ch in row_str:
                if ch.isdigit():
                    c += int(ch)
                else:
                    board[r][c] = Config.INV_FEN_MAP.get(ch)
                    c += 1
        return board

    @staticmethod
    def print_board(board: List[List[str]]):
        """控制台打印中文可视化棋盘"""
        print("\n" + "=" * 25)
        for r in range(10):
            cells = []
            for c in range(9):
                piece = board[r][c]
                cells.append(Config.CHINESE_MAP.get(piece, "十") if piece else "十")
            print(" ".join(cells))
        print("=" * 25)

    @staticmethod
    def move_looks_valid(move: str) -> bool:
        """简单校验 UCI 走法格式"""
        if not move or len(move) < 4:
            return False
        return (ord('a') <= ord(move[0]) <= ord('i') and
                ord('0') <= ord(move[1]) <= ord('9') and
                ord('a') <= ord(move[2]) <= ord('i') and
                ord('0') <= ord(move[3]) <= ord('9'))
```

- [ ] **Step 2: Verify syntax**

Run: `python -m py_compile pikafish_auto/Pikafish_auto4.py`
Expected: No output

- [ ] **Step 3: Commit**

```bash
git add pikafish_auto/Pikafish_auto4.py
git commit -m "refactor: add VisionController with edge-piece BORDER_REPLICATE fix"
```

---

### Task 5: VideoStreamThread and helpers

**Files:**
- Modify: `pikafish_auto/Pikafish_auto4.py` — append VideoStreamThread and draw function

- [ ] **Step 1: Append VideoStreamThread and draw_board_grid function**

```python
# ============================================================================
# VideoStreamThread
# ============================================================================

def draw_board_grid(warped: np.ndarray) -> np.ndarray:
    """在透视图上绘制 10×9 网格和交叉点"""
    res = warped.copy()
    x_lt, y_lt = Config.GRID_LT
    x_rt, y_rt = Config.GRID_RT
    x_lb, y_lb = Config.GRID_LB
    x_rb, y_rb = Config.GRID_RB

    for r in range(10):
        frac_r = r / 9.0
        p_left = (int(x_lt + frac_r * (x_lb - x_lt)),
                  int(y_lt + frac_r * (y_lb - y_lt)))
        p_right = (int(x_rt + frac_r * (x_rb - x_rt)),
                   int(y_rt + frac_r * (y_rb - y_rt)))
        cv2.line(res, p_left, p_right, (0, 255, 0), 1)

        for c in range(9):
            frac_c = c / 8.0
            top_x = x_lt + frac_c * (x_rt - x_lt)
            top_y = y_lt + frac_c * (y_rt - y_lt)
            bot_x = x_lb + frac_c * (x_rb - x_lb)
            bot_y = y_lb + frac_c * (y_rb - y_lb)
            tx = int(top_x + frac_r * (bot_x - top_x))
            ty = int(top_y + frac_r * (bot_y - top_y))
            cv2.circle(res, (tx, ty), 3, (255, 255, 0), -1)

    for c in range(9):
        frac = c / 8.0
        p_top = (int(x_lt + frac * (x_rt - x_lt)),
                 int(y_lt + frac * (y_rt - y_lt)))
        p_bot = (int(x_lb + frac * (x_rb - x_lb)),
                 int(y_lb + frac * (y_rb - y_lb)))
        cv2.line(res, p_top, p_bot, (0, 255, 0), 1)

    return res


def _on_mouse_click(event, x, y, flags, param):
    """鼠标点击回调 —— 打印像素坐标"""
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"\n[点击坐标] X: {x}, Y: {y}")


class VideoStreamThread(threading.Thread):
    """独立线程：实时采集摄像头、显示画面，为识别逻辑提供最新帧"""

    def __init__(self, camera_idx: int = 0):
        super().__init__()
        self.cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -6)
        self.ret = False
        self.frame = None
        self.warped = None
        self.stopped = False
        self.daemon = True

    def run(self):
        print("[系统] 视频监控已启动。按下 'q' 键退出。")
        cv2.namedWindow("Warped View")
        cv2.setMouseCallback("Warped View", _on_mouse_click)

        while not self.stopped:
            self.ret, frame = self.cap.read()
            if not self.ret:
                continue

            warped = cv2.warpPerspective(frame, Config.STATIC_MATRIX,
                                         (Config.WARP_W, Config.WARP_H))
            grid_img = draw_board_grid(warped)

            cv2.imshow("Original Monitor", frame)
            cv2.imshow("Warped View", grid_img)
            self.warped = warped

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[系统] 检测到键盘退出...")
                self.stopped = True
                os._exit(0)

    def get_latest_warped(self) -> Optional[np.ndarray]:
        return self.warped

    def stop(self):
        self.stopped = True
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
```

- [ ] **Step 2: Verify syntax**

Run: `python -m py_compile pikafish_auto/Pikafish_auto4.py`
Expected: No output

- [ ] **Step 3: Commit**

```bash
git add pikafish_auto/Pikafish_auto4.py
git commit -m "refactor: add VideoStreamThread and draw_board_grid helper"
```

---

### Task 6: GameController class

**Files:**
- Modify: `pikafish_auto/Pikafish_auto4.py` — append MoveRecord and GameController

- [ ] **Step 1: Append MoveRecord and GameController**

```python
# ============================================================================
# GameController
# ============================================================================

class MoveRecord:
    """单步走法记录"""
    __slots__ = ('uci', 'src_xy', 'dst_xy')

    def __init__(self, uci: str, src_xy: Tuple[int, int], dst_xy: Tuple[int, int]):
        self.uci = uci
        self.src_xy = src_xy
        self.dst_xy = dst_xy


class GameController:
    """对弈状态机 —— 消除人类先行/引擎先行两套重复代码"""

    class State(Enum):
        INIT = "init"
        READY = "ready"
        WAIT_START = "wait_start"
        HUMAN_TURN = "human_turn"
        ENGINE_TURN = "engine_turn"
        GAME_OVER = "game_over"
        UNDO = "undo"

    # ------------------------------------------------------------------
    def __init__(self, proto: ProtocolClient, engine: EngineController,
                 vision: VisionController):
        self._p = proto
        self._engine = engine
        self._vis = vision

        self._state = self.State.INIT
        self._cchess: Optional[cchess.Board] = None
        self._situation: str = "position startpos moves"
        self._human_is_red: bool = True
        self._level: int = 1
        self._move_history: List[MoveRecord] = []
        self._fen_before_human: str = ""
        self._use_chess_count: int = 16
        self._last_board: Optional[List[List[str]]] = None

    # ==================================================================
    # 主循环
    # ==================================================================
    def run(self):
        while True:
            if self._state == self.State.INIT:
                self._handle_init()
            elif self._state == self.State.READY:
                self._handle_ready()
            elif self._state == self.State.WAIT_START:
                self._handle_wait_start()
            elif self._state == self.State.HUMAN_TURN:
                self._handle_human_turn()
            elif self._state == self.State.ENGINE_TURN:
                self._handle_engine_turn()
            elif self._state == self.State.GAME_OVER:
                self._handle_game_over()
            elif self._state == self.State.UNDO:
                self._handle_undo()

    # ==================================================================
    # 状态处理
    # ==================================================================
    def _handle_init(self):
        """等待进入 READY 的信号"""
        print("[状态] INIT → READY")
        self._state = self.State.READY

    def _handle_ready(self):
        """发送准备完毕，等待对弈开始"""
        self._p.send("准备完毕")
        self._state = self.State.WAIT_START

    def _handle_wait_start(self):
        """等待服务器 '开始对弈N' 指令"""
        msg = self._p.wait_for_prefix("开始对弈")
        if msg is None:
            return

        try:
            self._level = int(msg.replace("开始对弈", ""))
        except ValueError:
            self._level = 1
        self._level = max(1, min(4, self._level))
        self._engine.set_level(self._level)
        print(f"[系统] 对局开始，难度={self._level}")

        self._reset_game_state()

        # 识别初始局面，判断执子方
        board = None
        while board is None:
            board = self._vis.capture_stable_board()
            if board is None:
                print("初始棋盘识别失败，重试...")
                time.sleep(0.5)

        human_color = self._vis.detect_side(board)
        self._human_is_red = (human_color == "red")
        self._vis.red_perspective = not self._human_is_red
        self._last_board = board

        side_name = "红" if self._human_is_red else "黑"
        print(f"[判断] 人类执{side_name}方")

        self._state = (self.State.HUMAN_TURN if self._human_is_red
                       else self.State.ENGINE_TURN)

    # ------------------------------------------------------------------
    def _handle_human_turn(self):
        """人类行棋：请求落子 → 等待完成 → 识别走法 → 校验 → 推送引擎"""
        self._p.send("请玩家落子")
        if not self._p.wait_for("玩家落子完成"):
            return

        human_move = None
        while human_move is None:
            board_now = None
            while board_now is None:
                board_now = self._vis.capture_stable_board()
                if board_now is None:
                    print("棋盘识别失败，重试...")
                    time.sleep(0.5)

            human_move = self._vis.get_move(self._last_board, board_now)

            if human_move is None or not self._vis.move_looks_valid(human_move):
                print(f"[警告] 未识别到有效走法: {human_move!r}")
                # 检查是否是识别错误（和引擎上一步相同）
                if self._move_history and human_move == self._move_history[-1].uci:
                    print("走法与引擎上一步相同，需要拿回棋子重新拍摄")
                    target_square = human_move[2:] + human_move[:2]
                    self._p.send(f"棋子移动:{self._vis._uci_to_pixel(human_move[2:])},"
                                 f"{self._vis._uci_to_pixel(human_move[:2])}")
                    self._p.wait_for("玩家落子完成")
                    # 重新识别 boardA
                    self._last_board = None
                    while self._last_board is None:
                        self._last_board = self._vis.capture_stable_board()
                        if self._last_board is None:
                            time.sleep(0.5)
                    human_move = None
                    self._p.send("请玩家落子")
                    self._p.wait_for("玩家落子完成")
                    continue

                self._p.send("走法无效:请重新落子")
                self._p.wait_for("玩家落子完成")
                human_move = None
                continue

            # 校验合法性
            if not self._validate_and_push(human_move):
                print(f"[警告] 非法走法: {human_move}")
                self._vis.print_board(self._last_board)
                self._vis.print_board(board_now)
                # 检查是否与引擎上一步相同
                if self._move_history and human_move == self._move_history[-1].uci:
                    print("走法与引擎上一步相同，拿回棋子重新拍摄")
                    target = human_move[2:] + human_move[:2]
                    self._p.send(f"棋子移动:"
                                 f"{self._vis._uci_to_pixel(human_move[2:])},"
                                 f"{self._vis._uci_to_pixel(human_move[:2])}")
                    self._p.wait_for("玩家落子完成")
                    self._last_board = None
                    while self._last_board is None:
                        self._last_board = self._vis.capture_stable_board()
                        if self._last_board is None:
                            time.sleep(0.5)
                    human_move = None
                    self._p.send("请玩家落子")
                    self._p.wait_for("玩家落子完成")
                    continue
                # 真的走错了，让机器人拿回
                target = human_move[2:] + human_move[:2]
                self._p.send(f"棋子移动:{self._vis._uci_to_pixel(human_move[2:])},"
                             f"{self._vis._uci_to_pixel(human_move[:2])}")
                self._p.wait_for("玩家落子完成")
                human_move = None
                continue

            self._fen_before_human = self._cchess.fen()
            # 记录 UCI
            self._situation += f" {human_move}"
            chinese = self._cchess.move_to_notation(cchess.Move.from_uci(human_move))
            print(f"[人类] {chinese} ({human_move})")

        # 游戏结束检测
        if self._check_game_over():
            return

        # 更新局面
        self._last_board = board_now
        red, black = self._vis.count_pieces(board_now)
        print(f"[棋子数] 红:{red} 黑:{black}")

        self._state = self.State.ENGINE_TURN

    # ------------------------------------------------------------------
    def _handle_engine_turn(self):
        """引擎行棋：获取走法 → 检测吃子 → 发送坐标 → 等待运动完成"""
        board_before_fen = self._cchess.fen()

        try:
            engine_move = self._engine.get_best_move(
                self._situation.replace("position startpos moves", "").strip()
            )
        except RuntimeError as e:
            print(f"[错误] {e}，尝试恢复引擎...")
            moves = self._situation.replace("position startpos moves", "").strip()
            if self._engine.restart(moves):
                engine_move = self._engine.get_best_move(moves)
            else:
                print("[严重] 引擎恢复失败")
                self._state = self.State.GAME_OVER
                return

        if not engine_move:
            print("[警告] 引擎返回空走法")
            self._state = self.State.GAME_OVER
            return

        # 检测吃子（push 之前）
        is_capture, captured_symbol = self._check_capture_before_push(
            self._cchess, engine_move)

        # 更新引擎走法到 cchess
        self._push_engine_move(engine_move)
        self._situation += f" {engine_move}"
        chinese = self._cchess.move_to_notation(cchess.Move.from_uci(engine_move))
        print(f"[引擎] {chinese} ({engine_move})")

        # 获取当前帧用于棋子定位
        current_frame = self._vis._stream.get_latest_warped()

        # 处理吃子（机械臂先移除被吃棋子）
        if is_capture:
            dst_uci = engine_move[2:]
            _, dst_xy = self._vis.detect_piece_at(dst_uci, current_frame)
            # 计算废棋区坐标
            red_cnt, black_cnt = self._vis.count_pieces(
                self._vis.board_from_fen(board_before_fen))
            if self._human_is_red:
                captured_color_offset = 0 if black_cnt % 2 == 0 else 1
            else:
                captured_color_offset = 0 if red_cnt % 2 == 0 else 1
            waste_x = 635 if captured_color_offset == 0 else 685
            waste_y = 50 + (16 - self._use_chess_count) * 60
            waste_xy = (waste_x, waste_y)
            self._use_chess_count -= 1

            self._p.send(f"棋子移动:{dst_xy},{waste_xy}")
            self._p.wait_for("运动完成")
            print(f"[提示] 引擎吃子！({captured_symbol})")

        # 发送引擎走法（从起点到终点）
        src_uci = engine_move[:2]
        dst_uci = engine_move[2:]
        _, src_xy = self._vis.detect_piece_at(src_uci, current_frame)
        dst_pixel = self._vis._uci_to_pixel(dst_uci)

        self._p.send(f"棋子移动:{src_xy},{dst_pixel}")
        self._p.wait_for("运动完成")

        # 记录走法历史（用于悔棋）
        self._move_history.append(MoveRecord(engine_move, src_xy, dst_pixel))

        # 游戏结束检测
        if self._check_game_over():
            return

        # 更新 last_board（从 cchess fen 重建）
        self._last_board = self._vis.board_from_fen(self._cchess.fen())
        self._state = self.State.HUMAN_TURN

    # ------------------------------------------------------------------
    def _handle_game_over(self):
        """游戏结束：等待重新开始或退出"""
        print("[状态] 等待新对局...")
        self._state = self.State.READY

    # ------------------------------------------------------------------
    def _handle_undo(self):
        """悔棋处理（预留接口）"""
        cmd = self.request_undo()
        if cmd:
            self._p.send(cmd)
        self._state = self.State.HUMAN_TURN

    # ==================================================================
    # 辅助方法
    # ==================================================================
    def _reset_game_state(self):
        """重置单局状态"""
        self._cchess = cchess.Board()
        self._situation = "position startpos moves"
        self._move_history.clear()
        self._fen_before_human = ""
        self._use_chess_count = 16
        self._last_board = None

    def _validate_and_push(self, uci_move: str) -> bool:
        """用 cchess 校验人类走法合法性"""
        try:
            move = cchess.Move.from_uci(uci_move)
        except Exception:
            return False
        if not self._cchess.is_legal(move):
            legal = [m.uci() for m in self._cchess.legal_moves]
            print(f"[cchess] 非法走法: {uci_move}，合法走法: {legal}")
            return False
        self._cchess.push(move)
        return True

    def _push_engine_move(self, uci_move: str):
        """将引擎走法同步到 cchess board"""
        try:
            self._cchess.push(cchess.Move.from_uci(uci_move))
        except Exception as e:
            print(f"[cchess] push 引擎走法失败: {e}")

    def _check_game_over(self) -> bool:
        """检测将死，返回 True 表示游戏已结束"""
        if self._cchess.is_game_over():
            # turn 是下一步该走的一方，刚刚被将死的一方是已走完的一方
            winner = "人类" if self._cchess.turn == (not self._human_is_red) else "机器人"
            self._p.send(f"{winner}获胜")
            print("\n=============================")
            print(f"  游戏结束！{winner}获胜。")
            print("=============================")
            self._state = self.State.GAME_OVER
            return True
        return False

    @staticmethod
    def _check_capture_before_push(board: cchess.Board,
                                   uci_move: str) -> Tuple[bool, Optional[str]]:
        """走法执行前检测是否吃子，返回 (是否吃子, 被吃子符号)"""
        try:
            move = cchess.Move.from_uci(uci_move)
        except Exception:
            return False, None
        target = board.piece_at(move.to_square)
        if target is not None:
            return True, target.symbol()
        return False, None

    # ==================================================================
    # 悔棋接口（预留）
    # ==================================================================
    def request_undo(self) -> Optional[str]:
        """撤销两步（人+引擎），返回机械臂复位坐标指令"""
        if len(self._move_history) < 2:
            print("[悔棋] 走法数不足，无法悔棋")
            return None

        m_engine = self._move_history.pop()
        m_human = self._move_history.pop()

        # 重建 cchess 棋盘到悔棋前的状态
        moves = self._situation.replace("position startpos moves", "").strip()
        # 去掉最后两步
        moves_parts = moves.split()
        moves_before = " ".join(moves_parts[:-2]) if len(moves_parts) >= 2 else ""
        self._cchess = cchess.Board()
        if moves_before:
            for m in moves_before.split():
                self._cchess.push(cchess.Move.from_uci(m))
        self._situation = f"position startpos moves{' ' + moves_before if moves_before else ''}"

        # 恢复引擎
        self._engine.restart(moves_before)

        print(f"[悔棋] 已撤销引擎走法 {m_engine.uci} 和人类走法 {m_human.uci}")
        return (f"棋子移动:{m_engine.dst_xy},{m_engine.src_xy};"
                f"棋子移动:{m_human.dst_xy},{m_human.src_xy}")
```

- [ ] **Step 2: Verify syntax**

Run: `python -m py_compile pikafish_auto/Pikafish_auto4.py`
Expected: No output

- [ ] **Step 3: Commit**

```bash
git add pikafish_auto/Pikafish_auto4.py
git commit -m "refactor: add GameController with state machine and undo reservation"
```

---

### Task 7: main() function

**Files:**
- Modify: `pikafish_auto/Pikafish_auto4.py` — append main() function

- [ ] **Step 1: Append main() entry point**

```python
# ============================================================================
# 入口
# ============================================================================

def main():
    """主函数 —— 组装所有组件并启动对弈循环"""
    print("=" * 60)
    print("  象棋机器人自动化对弈系统 v5.0")
    print("=" * 60)

    # --- 1. 环境初始化 ---
    print("\n[系统] 正在初始化...")

    # 加载 YOLO 模型
    try:
        yolo_model = YOLO(Config.MODEL_PATH)
        print("[系统] YOLO 模型加载完成")
    except Exception as e:
        print(f"[严重] 模型加载失败: {e}")
        return

    # 启动引擎
    engine = EngineController()
    try:
        engine.start()
    except Exception as e:
        print(f"[严重] 引擎启动失败: {e}")
        return

    # 启动摄像头
    video_stream = VideoStreamThread(0)
    video_stream.start()
    time.sleep(0.5)

    # 创建各组件实例
    proto = ProtocolClient()
    vision = VisionController(yolo_model, video_stream)
    game = GameController(proto, engine, vision)

    # --- 2. 连接服务器 ---
    try:
        proto.connect()
        proto.send("象棋引擎")
        print("[系统] 已向服务器注册")
    except Exception as e:
        print(f"[严重] 服务器连接失败: {e}")
        engine.stop()
        video_stream.stop()
        return

    # --- 3. 引擎握手 ---
    if not engine.uci_handshake():
        print("[严重] 引擎握手失败")
        engine.stop()
        video_stream.stop()
        return

    print("[系统] 初始化完成，进入对弈循环。")

    # --- 4. 主循环 ---
    try:
        game.run()
    except KeyboardInterrupt:
        print("\n[系统] 用户中断")
    except Exception as e:
        print(f"[错误] 未捕获异常: {e}")
        import traceback
        traceback.print_exc()

    # --- 5. 资源释放 ---
    print("[清理] 正在释放资源...")
    engine.stop()
    proto.disconnect()
    video_stream.stop()
    print("[系统] 已退出。")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify syntax**

Run: `python -m py_compile pikafish_auto/Pikafish_auto4.py`
Expected: No output (compiles clean)

- [ ] **Step 3: Commit**

```bash
git add pikafish_auto/Pikafish_auto4.py
git commit -m "refactor: add main() entry point with clean init/cleanup flow"
```

---

### Task 8: Integration verification

**Files:**
- Existing: `pikafish_auto/Pikafish_auto4.py`

- [ ] **Step 1: Run Python syntax and import check**

```bash
python -c "import py_compile; py_compile.compile('pikafish_auto/Pikafish_auto4.py', doraise=True); print('OK')"
```
Expected: prints "OK"

- [ ] **Step 2: Verify module can be parsed as AST**

```bash
python -c "import ast; ast.parse(open('pikafish_auto/Pikafish_auto4.py', encoding='utf-8').read()); print('AST OK')"
```
Expected: prints "AST OK"

- [ ] **Step 3: Verify all classes and methods are defined**

```bash
python -c "
import ast, sys
tree = ast.parse(open('pikafish_auto/Pikafish_auto4.py', encoding='utf-8').read())
classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
required = ['Config', 'ProtocolClient', 'EngineController', 'VisionController', 'VideoStreamThread', 'GameController', 'MoveRecord']
for c in required:
    assert c in classes, f'MISSING: {c}'
print('All classes present:', classes)
"
```
Expected: prints "All classes present: [list of classes]"

- [ ] **Step 4: Count lines and verify class structure**

```bash
wc -l pikafish_auto/Pikafish_auto4.py
```
Expected: ~800-1000 lines (down from 1058 original, despite adding features, due to deduplication)

- [ ] **Step 5: Final commit**

```bash
git add pikafish_auto/Pikafish_auto4.py
git commit -m "refactor: complete Pikafish_auto4.py restructuring with integration verification"
```

---

### Task 9: Spec coverage checklist

Run through each spec requirement and confirm it's addressed:

- [ ] 棋盘边缘识别改进 — `VisionController.recognize_board` 使用 `BORDER_REPLICATE` 扩展 + `EDGE_PADDING`
- [ ] 引擎难度设置 — `Config.LEVEL_MOVETIME` + `EngineController.set_level` + `GameController._handle_wait_start` 解析难度
- [ ] 通信协议重构 — `ProtocolClient` 支持新格式 `棋子移动:(x,y),(x,y)` + `wait_for("运动完成")` + `wait_for_prefix("开始对弈")`
- [ ] 引擎崩溃自动恢复 — `EngineController.restart(fen_moves)` 含完整 kill→spawn→handshake→position 流程
- [ ] 悔棋接口预留 — `GameController.request_undo()` + `MoveRecord` 走法历史 + `UNDO` 状态枚举
- [ ] Bug: `daemon` 拼写 — 修正（`VideoStreamThread.daemon = True`, `ProtocolClient._recv_loop` daemon thread）
- [ ] Bug: `engine_move` 作用域 — 状态机保证顺序（HUMAN_TURN → ENGINE_TURN）
- [ ] Bug: `os._exit(0)` — 仅保留在 `VideoStreamThread.run` 的键盘退出，其余走 `GAME_OVER` 清理路径
- [ ] Bug: `check_game_over` flag 参数 — 用 `self._cchess.turn` 和 `self._human_is_red` 正确判断胜者
- [ ] Bug: `use_chess_count` 不重置 — `_reset_game_state()` 中重置
- [ ] Bug: `readline_timeout` 每次新建线程 — `EngineController._reader_loop` 持久化线程 + `_line_queue`
