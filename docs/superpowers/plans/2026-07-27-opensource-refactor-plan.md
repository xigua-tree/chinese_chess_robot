# 象棋机器人开源重构实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将项目重构为适合开源学习的扁平模块化结构，删除冗余文件，完善中文文档。

**Architecture:** 扁平模块结构——vision/, chess_engine/, robot_arm/, voice/, gomoku/, gui/, model_train/, models/, calibration_data/, engines/, tools/。所有目录英文命名，文档全部中文。

**Tech Stack:** Python 3.13, OpenCV, YOLO (ultralytics), cchess, sherpa-onnx, numpy, pyserial

---

### Task 1: 创建新目录结构

**Files:**
- Create: `vision/`, `chess_engine/`, `robot_arm/`, `robot_arm/tests/`, `robot_arm/calibration/`
- Create: `voice/`, `gomoku/`, `gomoku/utils/`, `gomoku/gomoku_engine_binary/`
- Create: `gui/`, `model_train/` (已存在), `models/`, `calibration_data/`
- Create: `engines/pikafish/`, `engines/pbrain-rapfi/`, `tools/`

- [ ] **Step 1: 创建所有新目录**

```bash
mkdir -p vision chess_engine robot_arm/tests robot_arm/calibration
mkdir -p voice gomoku/utils gomoku/gomoku_engine_binary
mkdir -p gui models calibration_data
mkdir -p engines/pikafish engines/pbrain-rapfi tools
```

---

### Task 2: 拆分并迁移 vision/ 模块

**Files:**
- Create: `vision/__init__.py`
- Create: `vision/chess_vision.py`
- Create: `vision/video_stream.py`
- Create: `vision/board_utils.py`
- Source: `pikafish_auto/Pikafish_auto4.py`

- [ ] **Step 1: 创建 `vision/chess_vision.py`**

从 `pikafish_auto/Pikafish_auto4.py` 提取 VisionController 类（第 521-1198 行）。

```python
# -*- coding: utf-8 -*-
"""象棋棋盘视觉识别 —— YOLO + HoughCircles"""

import time
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np

from chess_engine.config import Config
from vision.board_utils import draw_board_grid


class VisionController:
    """视觉识别 —— YOLO + HoughCircles 棋盘识别"""

    # 人类废棋区 ROI（warped 图坐标，棋盘左侧）
    HUMAN_WASTE_ROI = (10, 0, 103, 625)
    # 机器人废棋区 ROI（warped 图坐标，棋盘右侧）
    ROBOT_WASTE_ROI = (609, 2, 705, 623)

    # 每种棋子类型的最大数量（开局初始值）
    _PIECE_TYPE_MAX = {
        'Kin': 1, 'Shi': 2, 'Ele': 2, 'Hor': 2, 'Car': 2, 'Can': 2, 'Paw': 5,
    }

    def __init__(self, yolo_model, video_stream):
        self._model = yolo_model
        self._stream = video_stream
        self._human_is_red: bool = False

    def set_human_is_red(self, is_red: bool):
        self._human_is_red = is_red

    # ------------------------------------------------------------------
    # 棋盘识别
    # ------------------------------------------------------------------
    def recognize_board(self, warped_img: np.ndarray) -> List[List[str]]:
        """单帧识别：HoughCircles 寻圆 + YOLO 分类 → 10x9 矩阵"""
        board = [[None for _ in range(9)] for _ in range(10)]

        p1 = self._stream.hough_param1
        p2 = self._stream.hough_param2
        md = self._stream.hough_minDist
        minR = self._stream.hough_minRadius
        maxR = self._stream.hough_maxRadius

        gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)

        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1,
            minDist=md, param1=p1, param2=p2,
            minRadius=minR, maxRadius=maxR
        )

        if circles is None:
            return board

        circles = np.round(circles[0, :]).astype(int)
        for cx, cy, r in circles:
            roi_size = 24 + Config.EDGE_PADDING
            x1, y1 = cx - roi_size, cy - roi_size
            x2, y2 = cx + roi_size, cy + roi_size

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
                label = results[0].names[results[0].probs.top1]

                display = label.split('_')[-1]
                cv2.circle(warped_img, (cx, cy), r, (255, 0, 0), 2)
                if cy < Config.WARP_H // 2:
                    text_y = cy + r + 14
                else:
                    text_y = cy - r - 5
                cv2.putText(warped_img, f"{display} {conf:.2f}",
                            (cx - 18, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)

                thresh = Config.CLASS_CONF_OVERRIDE.get(label, Config.CONF_THRESH)
                if conf > thresh:
                    cv2.circle(warped_img, (cx, cy), 3, (0, 255, 0), -1)

                    if (cx < Config.BOARD_LT[0] or cx > Config.BOARD_RT[0] or
                        cy < Config.BOARD_LT[1] or cy > Config.BOARD_LB[1]):
                        cv2.circle(warped_img, (cx, cy), r, (0, 0, 255), 1)
                        continue

                    rel_x = (cx - Config.GRID_LT[0]) / max(1, (Config.GRID_RT[0] - Config.GRID_LT[0]))
                    rel_y = (cy - Config.GRID_LT[1]) / max(1, (Config.GRID_LB[1] - Config.GRID_LT[1]))

                    grid_c = min(8, max(0, int(round(rel_x * 8))))
                    grid_r = min(9, max(0, int(round(rel_y * 9))))

                    board[grid_r][grid_c] = label
                else:
                    cv2.circle(warped_img, (cx, cy), 3, (0, 0, 255), -1)

        if hasattr(self._stream, 'annotated_warped'):
            self._stream.annotated_warped = draw_board_grid(warped_img)

        return board

    # ------------------------------------------------------------------
    # 帧间投票
    # ------------------------------------------------------------------
    def _vote_board(self, frames: int, threshold: int,
                    existing_votes: List[List[Dict]] = None
                    ) -> Tuple[List[List[str]], List[List[Dict]]]:
        if existing_votes is not None:
            votes = existing_votes
        else:
            votes = [[{} for _ in range(9)] for _ in range(10)]

        for _ in range(frames):
            warped = self._stream.get_latest_warped()
            if warped is None:
                time.sleep(Config.VOTE_INTERVAL)
                continue

            current = self.recognize_board(warped.copy())
            for r in range(10):
                for c in range(9):
                    label = current[r][c]
                    key = label if label else None
                    votes[r][c][key] = votes[r][c].get(key, 0) + 1

            time.sleep(Config.VOTE_INTERVAL)

        board = [[None for _ in range(9)] for _ in range(10)]
        for r in range(10):
            for c in range(9):
                v = votes[r][c]
                if not v:
                    continue
                best_label = None
                best_count = 0
                for label, count in v.items():
                    if label is not None and count > best_count:
                        best_count = count
                        best_label = label
                if best_count >= threshold:
                    board[r][c] = best_label

        return board, votes

    @staticmethod
    def _validate_piece_count(board: List[List[str]],
                               expected_red: int = None,
                               expected_black: int = None,
                               expected_rc: Dict[str, int] = None,
                               expected_bc: Dict[str, int] = None) -> Tuple[bool, str]:
        rc: Dict[str, int] = {}
        bc: Dict[str, int] = {}
        for row in board:
            for p in row:
                if not p:
                    continue
                ptype = p[2:]
                if p.startswith('R_'):
                    rc[ptype] = rc.get(ptype, 0) + 1
                elif p.startswith('B_'):
                    bc[ptype] = bc.get(ptype, 0) + 1

        if rc.get('Kin', 0) != 1:
            return False, f"红帅数量={rc.get('Kin', 0)}，期望 1"
        if bc.get('Kin', 0) != 1:
            return False, f"黑将数量={bc.get('Kin', 0)}，期望 1"

        for ptype, limit in VisionController._PIECE_TYPE_MAX.items():
            if rc.get(ptype, 0) > limit:
                return False, f"红{ptype}数量={rc[ptype]} > {limit}"
            if bc.get(ptype, 0) > limit:
                return False, f"黑{ptype}数量={bc[ptype]} > {limit}"

        red = sum(rc.values())
        black = sum(bc.values())
        total = red + black
        if total == 0:
            return False, "棋子总数为 0"
        if total > 32:
            return False, f"棋子总数 {total} > 32"
        if red > 16:
            return False, f"红方棋子 {red} > 16"
        if black > 16:
            return False, f"黑方棋子 {black} > 16"

        if expected_red is not None and red != expected_red:
            return False, f"红方总数预期 {expected_red}，实际 {red}"
        if expected_black is not None and black != expected_black:
            return False, f"黑方总数预期 {expected_black}，实际 {black}"

        for ptype in VisionController._PIECE_TYPE_MAX:
            if expected_rc is not None and rc.get(ptype, 0) != expected_rc.get(ptype, 0):
                return False, (f"红{ptype}预期 {expected_rc.get(ptype, 0)}，"
                               f"实际 {rc.get(ptype, 0)}")
            if expected_bc is not None and bc.get(ptype, 0) != expected_bc.get(ptype, 0):
                return False, (f"黑{ptype}预期 {expected_bc.get(ptype, 0)}，"
                               f"实际 {bc.get(ptype, 0)}")

        return True, ""

    def capture_stable_board(self, required_stable: int = None,
                             expected_red: int = None,
                             expected_black: int = None,
                             expected_rc: Dict[str, int] = None,
                             expected_bc: Dict[str, int] = None) -> Optional[List[List[str]]]:
        votes = [[{} for _ in range(9)] for _ in range(10)]
        threshold = Config.VOTE_THRESHOLD
        total_frames = 0

        print(f"[视觉] 帧间投票中 (初始 {Config.VOTE_FRAMES} 帧, 阈值 {threshold})...")

        board, votes = self._vote_board(Config.VOTE_FRAMES, threshold)
        total_frames = Config.VOTE_FRAMES

        while True:
            red, black = VisionController.count_pieces(board)
            valid, reason = self._validate_piece_count(
                board, expected_red, expected_black, expected_rc, expected_bc)

            if valid:
                print(f"[视觉] 投票通过 (总帧={total_frames}, 红={red} 黑={black})")
                VisionController.print_board(board)
                return board

            print(f"[视觉] 校验失败: {reason} (已采 {total_frames} 帧)，追加 {Config.VOTE_RETRY_EXTRA} 帧...")
            board, votes = self._vote_board(Config.VOTE_RETRY_EXTRA, threshold, votes)
            total_frames += Config.VOTE_RETRY_EXTRA

    def capture_any_board(self) -> Optional[List[List[str]]]:
        """单帧快速识别（用于失败重试）"""
        warped = self._stream.get_latest_warped()
        if warped is None:
            return None
        board = self.recognize_board(warped.copy())
        VisionController.print_board(board)
        return board

    # ------------------------------------------------------------------
    # 走法计算
    # ------------------------------------------------------------------
    def get_move(self, prev_board: List[List[str]],
                 curr_board: List[List[str]]) -> Optional[str]:
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

        print(f"[getMove] 差分: removed={len(removed)} added={len(added)} changed={len(changed)}")
        for r, c in removed:
            print(f"  removed ({r},{c}): 之前={prev_board[r][c]}")
        for r, c in added:
            print(f"  added   ({r},{c}): 当前={curr_board[r][c]}")
        for r, c in changed:
            print(f"  changed ({r},{c}): {prev_board[r][c]} -> {curr_board[r][c]}")

        if not removed and not added and not changed:
            print("[getMove] 未检测到变化")
            return None

        src, dst = (-1, -1), (-1, -1)

        if len(removed) == 1 and len(added) == 1 and not changed:
            src, dst = removed[0], added[0]
            print(f"[getMove] 简单移动: {src} -> {dst}")
        elif len(removed) == 1 and len(changed) == 1:
            src, dst = removed[0], changed[0]
            print(f"[getMove] 变更移动: {src} -> {dst}")
        elif len(removed) == 1 and len(added) == 0 and len(changed) == 0:
            print("[getMove] 仅检测到移除，忽略")
            return None
        else:
            dst_candidates = added + changed
            found = False
            for rmv in removed:
                piece_type = prev_board[rmv[0]][rmv[1]]
                for cand in dst_candidates:
                    if curr_board[cand[0]][cand[1]] == piece_type:
                        src, dst = rmv, cand
                        found = True
                        print(f"[getMove] 类型匹配: {src}({piece_type}) -> {dst}")
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
                print(f"[getMove] 模糊匹配: src={src} dst={dst}")

        if src[0] < 0 or dst[0] < 0:
            print(f"[getMove] 无法确定走法: removed={len(removed)} "
                  f"added={len(added)} changed={len(changed)}")
            return None

        uci = self._to_uci(src[1], src[0]) + self._to_uci(dst[1], dst[0])
        print(f"[getMove] 结果: {uci}")
        return uci

    def _to_uci(self, col: int, row: int) -> str:
        file_char = chr(ord('a') + col)
        rank = row if self._human_is_red else 9 - row
        return f"{file_char}{rank}"

    # ------------------------------------------------------------------
    # 执子方检测
    # ------------------------------------------------------------------
    def detect_side(self, board: List[List[str]]) -> str:
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
    # 棋子定位
    # ------------------------------------------------------------------
    def detect_piece_at(self, uci_pos: str,
                        current_frame: np.ndarray) -> Tuple[Optional[str], Tuple[int, int]]:
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
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=32,
            param1=49, param2=30, minRadius=15, maxRadius=25
        )

        detected_center = (expected_x, expected_y)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            c = circles[0][0]
            detected_center = (int(x1 + c[0]), int(y1 + c[1]))
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

    def recognize_waste_area(self, roi_rect: Tuple[int, int, int, int]
                             ) -> List[Tuple[str, int, int]]:
        N_FRAMES = 5
        VOTE_MIN = 3
        PROXIMITY = 15

        x1, y1, x2, y2 = roi_rect
        all_detections: List[List[Tuple[str, int, int]]] = []

        valid_frames = 0
        while valid_frames < N_FRAMES:
            frame = self._stream.get_latest_warped()
            if frame is None:
                time.sleep(0.02)
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
                for cx, cy, _ in circles:
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
                        label = results[0].names[results[0].probs.top1]
                        thresh = Config.CLASS_CONF_OVERRIDE.get(label, Config.CONF_THRESH)
                        if conf > thresh:
                            frame_dets.append((label, x1 + cx, y1 + cy))

            all_detections.append(frame_dets)
            valid_frames += 1
            if valid_frames < N_FRAMES:
                time.sleep(Config.VOTE_INTERVAL)

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

    def _uci_to_pixel(self, uci_pos: str) -> Tuple[int, int]:
        if not uci_pos or len(uci_pos) < 2:
            return (0, 0)

        col_idx = ord(uci_pos[0].lower()) - ord('a')
        rank = int(uci_pos[1:])
        row_idx = rank if self._human_is_red else 9 - rank

        rel_x = col_idx / 8.0
        rel_y = row_idx / 9.0

        top_x = Config.GRID_LT[0] + rel_x * (Config.GRID_RT[0] - Config.GRID_LT[0])
        top_y = Config.GRID_LT[1] + rel_x * (Config.GRID_RT[1] - Config.GRID_LT[1])
        bot_x = Config.GRID_LB[0] + rel_x * (Config.GRID_RB[0] - Config.GRID_LB[0])
        bot_y = Config.GRID_LB[1] + rel_x * (Config.GRID_RB[1] - Config.GRID_LB[1])

        target_x = int(top_x + rel_y * (bot_x - top_x))
        target_y = int(top_y + rel_y * (bot_y - top_y))
        return (target_x, target_y)

    def uci_to_pixel_public(self, uci_pos: str) -> Tuple[int, int]:
        return self._uci_to_pixel(uci_pos)

    def apply_uci_move(self, board: List[List[str]], uci_move: str) -> List[List[str]]:
        new_board = [row[:] for row in board]

        src_col = ord(uci_move[0]) - ord('a')
        src_rank = int(uci_move[1])
        dst_col = ord(uci_move[2]) - ord('a')
        dst_rank = int(uci_move[3])

        if self._human_is_red:
            src_row = src_rank
            dst_row = dst_rank
        else:
            src_row = 9 - src_rank
            dst_row = 9 - dst_rank

        new_board[dst_row][dst_col] = new_board[src_row][src_col]
        new_board[src_row][src_col] = None
        return new_board

    # ------------------------------------------------------------------
    # 静态工具方法
    # ------------------------------------------------------------------
    @staticmethod
    def count_pieces(board: List[List[str]]) -> Tuple[int, int]:
        red = black = 0
        for row in board:
            for p in row:
                if p and p.startswith("R_"):
                    red += 1
                elif p and p.startswith("B_"):
                    black += 1
        return red, black

    @staticmethod
    def count_pieces_by_type(board: List[List[str]]) -> Tuple[Dict[str, int], Dict[str, int]]:
        rc: Dict[str, int] = {}
        bc: Dict[str, int] = {}
        for row in board:
            for p in row:
                if not p:
                    continue
                ptype = p[2:]
                if p.startswith('R_'):
                    rc[ptype] = rc.get(ptype, 0) + 1
                elif p.startswith('B_'):
                    bc[ptype] = bc.get(ptype, 0) + 1
        return rc, bc

    @staticmethod
    def board_to_fen(board: List[List[str]]) -> str:
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
    def construct_fen(board: List[List[str]], side_to_move: str = "w") -> str:
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
        placement = "/".join(fen_rows)
        return f"{placement} {side_to_move} - - 0 1"

    @staticmethod
    def print_board(board: List[List[str]]):
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
        if not move or len(move) < 4:
            return False
        return (ord('a') <= ord(move[0]) <= ord('i') and
                ord('0') <= ord(move[1]) <= ord('9') and
                ord('a') <= ord(move[2]) <= ord('i') and
                ord('0') <= ord(move[3]) <= ord('9'))
```

- [ ] **Step 2: 创建 `vision/video_stream.py`**

从 `Pikafish_auto4.py` 提取 VideoStreamThread 类和 `_on_mouse_click` 回调。

```python
# -*- coding: utf-8 -*-
"""摄像头视频流采集线程"""

import os
import threading
import cv2
import numpy as np
from chess_engine.config import Config
from vision.board_utils import draw_board_grid


def _hough_trackbar_noop(_val):
    """Hough Controls trackbar 回调（空函数，参数通过 getTrackbarPos 实时读取）"""
    pass


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
        self.cap.set(cv2.CAP_PROP_SETTINGS, 0)
        self.ret = False
        self.frame = None
        self.warped = None
        self.annotated_warped = None
        self.stopped = False
        self.daemon = True
        self.hough_param1 = 49
        self.hough_param2 = 30
        self.hough_minDist = 32
        self.hough_minRadius = 15
        self.hough_maxRadius = 25

    def run(self):
        print("[系统] 视频监控已启动。按下 'q' 键退出。")
        cv2.namedWindow("Warped View")
        cv2.setMouseCallback("Warped View", _on_mouse_click)

        cv2.namedWindow("Hough Controls", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Hough Controls", 420, 280)
        cv2.createTrackbar("param1", "Hough Controls", 49, 200, _hough_trackbar_noop)
        cv2.createTrackbar("param2", "Hough Controls", 22, 100, _hough_trackbar_noop)
        cv2.createTrackbar("minDist", "Hough Controls", 32, 80, _hough_trackbar_noop)
        cv2.createTrackbar("minRadius", "Hough Controls", 15, 80, _hough_trackbar_noop)
        cv2.createTrackbar("maxRadius", "Hough Controls", 25, 80, _hough_trackbar_noop)
        print("[系统] 霍夫圆控制面板已创建 (窗口: Hough Controls)")

        while not self.stopped:
            self.ret, frame = self.cap.read()
            if not self.ret:
                continue

            warped = cv2.warpPerspective(frame, Config.STATIC_MATRIX,
                                         (Config.WARP_W, Config.WARP_H))
            grid_img = draw_board_grid(warped)

            cv2.imshow("Original Monitor", frame)
            if self.annotated_warped is not None:
                cv2.imshow("Warped View", self.annotated_warped)
            else:
                cv2.imshow("Warped View", grid_img)
            self.warped = warped

            key = cv2.waitKey(1) & 0xFF

            try:
                self.hough_param1 = max(1, cv2.getTrackbarPos("param1", "Hough Controls"))
                self.hough_param2 = max(1, cv2.getTrackbarPos("param2", "Hough Controls"))
                self.hough_minDist = max(1, cv2.getTrackbarPos("minDist", "Hough Controls"))
                self.hough_minRadius = cv2.getTrackbarPos("minRadius", "Hough Controls")
                self.hough_maxRadius = max(self.hough_minRadius + 1,
                                           cv2.getTrackbarPos("maxRadius", "Hough Controls"))
            except Exception:
                pass

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
        self.join(timeout=3.0)
        cv2.destroyAllWindows()
```

- [ ] **Step 3: 创建 `vision/board_utils.py`**

```python
# -*- coding: utf-8 -*-
"""棋盘透视变换、网格绘制等工具函数"""

import cv2
import numpy as np
from chess_engine.config import Config


def draw_board_grid(warped: np.ndarray) -> np.ndarray:
    """在透视图上绘制 10x9 网格和交叉点"""
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
```

- [ ] **Step 4: 创建 `vision/__init__.py`**

```python
# -*- coding: utf-8 -*-
"""视觉识别模块 —— 摄像头采集、透视变换、棋子识别"""
```

- [ ] **Step 5: 提交**

```bash
git add vision/
git commit -m "feat: 添加 vision/ 视觉识别模块"
```

---

### Task 3: 拆分并迁移 chess_engine/ 模块

**Files:**
- Create: `chess_engine/__init__.py`
- Create: `chess_engine/config.py`
- Create: `chess_engine/protocol_client.py`
- Create: `chess_engine/engine_controller.py`
- Create: `chess_engine/game_controller.py`
- Source: `pikafish_auto/Pikafish_auto4.py`

- [ ] **Step 1: 创建 `chess_engine/config.py`**

从 `Pikafish_auto4.py` 提取 Config 类（第 20-184 行）。

```python
# -*- coding: utf-8 -*-
"""集中配置 —— 所有可调参数"""

import os
import numpy as np
from typing import List, Tuple, Dict


class Config:
    """集中配置 —— 所有可调参数"""

    # --- 引擎 ---
    PIKAFISH_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "engines", "pikafish", "pikafish-avx2.exe")
    LEVEL_MOVETIME = {1: 100, 2: 200, 3: 500, 4: 1500}
    THINK_FOR_HUMAN_MOVETIME = 1500

    # --- 模型 ---
    MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "models", "chess_classifier.pt")
    CONF_THRESH = 0.70
    CLASS_CONF_OVERRIDE = {'R_Ele': 0.6, 'B_Shi': 0.6}

    # --- 棋盘定位 ---
    FIXED_PTS = np.float32([
        [310, 102], [931, 95], [279, 679], [983, 664]
    ])
    WARP_W, WARP_H = 707, 630
    DST_PTS = np.float32([[0, 0], [WARP_W, 0], [0, WARP_H], [WARP_W, WARP_H]])
    STATIC_MATRIX = cv2.getPerspectiveTransform(FIXED_PTS, DST_PTS)

    GRID_LT = (127, 30)
    GRID_RT = (576, 30)
    GRID_LB = (127, 598)
    GRID_RB = (579, 595)

    BOARD_LT = (106, 2)
    BOARD_RT = (604, 2)
    BOARD_LB = (105, 624)
    BOARD_RB = (607, 624)

    # --- 视觉 ---
    STABLE_FRAMES = 3
    EDGE_PADDING = 3
    MAX_CAPTURE_ATTEMPTS = 40
    VOTE_FRAMES = 15
    VOTE_THRESHOLD = 8
    VOTE_INTERVAL = 0.06
    MAX_VOTE_RETRIES = 3
    VOTE_RETRY_EXTRA = 5

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

    # --- FEN 校验 ---
    STANDARD_OPENING_FEN = (
        "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/"
        "P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"
    )
    RED_STANDARD_OPENING_FEN = (
        "RNBAKABNR/9/1C5C1/P1P1P1P1P/9/9/"
        "p1p1p1p1p/1c5c1/9/rnbakabnr w - - 0 1"
    )

    ENDGAME_FEN_DB: Dict[str, str] = {}

    VALID_PIECE_CHARS = set("KRNCBAPkrncbap")
    FEN_ROW_COUNT = 10
    FEN_COL_COUNT = 9

    @staticmethod
    def fen_to_uci_pieces(fen: str, human_is_red: bool = False
                          ) -> List[Tuple[str, str]]:
        rows = fen.split()[0].split("/")
        pieces = []
        for rank_idx, row in enumerate(rows):
            col = 0
            for ch in row:
                if ch.isdigit():
                    col += int(ch)
                else:
                    rank = rank_idx if human_is_red else 9 - rank_idx
                    uci = f"{chr(ord('a') + col)}{rank}"
                    label = Config.INV_FEN_MAP.get(ch)
                    if label:
                        pieces.append((uci, label))
                    col += 1
        return pieces

    @staticmethod
    def validate_fen(fen: str) -> Tuple[bool, str]:
        if not fen or not isinstance(fen, str):
            return False, "FEN 为空或类型错误"

        parts = fen.strip().split()
        if len(parts) < 2:
            return False, f"FEN 字段不足，需要 ≥2 个字段，实际 {len(parts)} 个"

        placement = parts[0]
        rows = placement.split('/')
        if len(rows) != Config.FEN_ROW_COUNT:
            return False, f"期望 {Config.FEN_ROW_COUNT} 行，实际 {len(rows)} 行"

        for i, row in enumerate(rows):
            col_sum = 0
            for ch in row:
                if ch.isdigit():
                    col_sum += int(ch)
                elif ch in Config.VALID_PIECE_CHARS:
                    col_sum += 1
                else:
                    return False, f"第 {i+1} 行含非法字符 '{ch}'"
            if col_sum != Config.FEN_COL_COUNT:
                return False, (
                    f"第 {i+1} 行列数和为 {col_sum}，期望 {Config.FEN_COL_COUNT}")

        side = parts[1]
        if side not in ('w', 'b'):
            return False, f"走子方字段应为 'w' 或 'b'，实际 '{side}'"

        if len(parts) >= 5:
            if not parts[-2].isdigit():
                return False, f"半回合数字段应为数字，实际 '{parts[-2]}'"
        if len(parts) >= 6:
            if not parts[-1].isdigit():
                return False, f"全回合数字段应为数字，实际 '{parts[-1]}'"

        return True, ""


# 需要在文件末尾导入 cv2 以支持 STATIC_MATRIX 的初始化
import cv2
# 重新计算 STATIC_MATRIX（依赖于 cv2）
Config.STATIC_MATRIX = cv2.getPerspectiveTransform(Config.FIXED_PTS, Config.DST_PTS)
```

- [ ] **Step 2: 创建 `chess_engine/protocol_client.py`**

从 `Pikafish_auto4.py` 提取 ProtocolClient 类（第 190-338 行）。

```python
# -*- coding: utf-8 -*-
"""TCP 通信封装 —— 与 Qt 服务端交互"""

import socket
import threading
import queue
import time
from typing import Optional


class ProtocolClient:
    """TCP 通信封装 —— 与 Qt 服务端交互"""

    def __init__(self, host: str = "127.0.0.1", port: int = 8888):
        self._host = host
        self._port = port
        self._sock: Optional[socket.socket] = None
        self._queue: queue.Queue = queue.Queue()
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None
        self._shutdown_requested: bool = False

    def connect(self):
        """创建 socket 连接并启动后台接收线程"""
        self._sock = socket.socket()
        self._sock.connect((self._host, self._port))
        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()
        print(f"[协议] 已连接 {self._host}:{self._port}")

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
                if msg == "退出":
                    self._shutdown_requested = True
                    self._running = False
            except Exception as e:
                print(f"[协议] 接收异常: {e}")
                self._running = False
                break

    def send(self, msg: str):
        """发送消息到服务端"""
        if self._shutdown_requested:
            return
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
        while time.time() < deadline and self._running:
            remaining = max(0.1, deadline - time.time())
            try:
                msg = self._queue.get(timeout=min(1.0, remaining))
                if msg == "退出":
                    print("[协议] 收到退出，中断等待")
                    return False
                if msg == expected:
                    print(f"[匹配] {expected}")
                    return True
            except queue.Empty:
                pass
        print(f"[超时] 等待 {expected} 超时")
        return False

    def wait_for_any(self, *expected: str) -> Optional[str]:
        """阻塞等待多个消息中的任意一个，返回匹配的消息"""
        print(f"[等待] {list(expected)} ...")
        while self._running:
            try:
                msg = self._queue.get(timeout=1.0)
                if msg == "退出":
                    print("[协议] 收到退出，中断等待")
                    return None
                if msg in expected:
                    print(f"[匹配] {msg}")
                    return msg
            except queue.Empty:
                pass
        return None

    def wait_for_any_prefix(self, *prefixes: str) -> Optional[str]:
        """阻塞等待任意一个前缀匹配的消息，返回完整消息"""
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

    def wait_for_prefix(self, prefix: str, timeout: float = None) -> Optional[str]:
        """阻塞等待前缀匹配的消息"""
        print(f"[等待前缀] {prefix} ...")
        deadline = time.time() + (timeout if timeout else 999999)
        while time.time() < deadline and self._running:
            remaining = max(0.1, deadline - time.time())
            try:
                msg = self._queue.get(timeout=min(1.0, remaining))
                if msg == "退出":
                    print("[协议] 收到退出，中断等待")
                    return None
                if msg.startswith(prefix):
                    print(f"[匹配] {msg}")
                    return msg
            except queue.Empty:
                pass
        print(f"[超时] 等待前缀 '{prefix}' 超时")
        return None

    def shutdown_requested(self) -> bool:
        return self._shutdown_requested

    def peek(self) -> Optional[str]:
        """非阻塞读取一条消息"""
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

- [ ] **Step 3: 创建 `chess_engine/engine_controller.py`**

```python
# -*- coding: utf-8 -*-
"""UCI 象棋引擎进程管理 —— 含崩溃自动恢复"""

import subprocess
import threading
import queue
import time
from typing import Optional
from chess_engine.config import Config


class EngineController:
    """UCI 象棋引擎进程封装 —— 含崩溃自动恢复"""

    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.alive: bool = False
        self._line_queue: queue.Queue = queue.Queue()
        self._reader_thread: Optional[threading.Thread] = None
        self._level: int = 1

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
            self._reader_thread = threading.Thread(
                target=self._reader_loop, daemon=True)
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
        """从队列读取一行（带超时）"""
        try:
            return self._line_queue.get(timeout=timeout_ms / 1000.0)
        except queue.Empty:
            return ""

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
        movetime = Config.LEVEL_MOVETIME.get(level, 3000)
        print(f"[引擎] 握手完成，难度={level} (movetime={movetime}ms)")
        return True

    def set_level(self, level: int):
        """动态修改引擎难度"""
        self._level = max(1, min(4, level))
        print(f"[引擎] 难度已设置为 {self._level}")

    def get_best_move(self, fen_moves: str, initial_fen: str = None,
                      movetime_ms: int = None) -> str:
        """发送当前局面并获取最佳走法"""
        if not self.alive:
            raise RuntimeError("[引擎] 引擎不可用")

        if initial_fen:
            moves_part = f" moves {fen_moves}" if fen_moves.strip() else ""
            self.write(f"position fen {initial_fen}{moves_part}")
        elif fen_moves.strip():
            self.write(f"position startpos moves {fen_moves}")
        else:
            self.write("position startpos")

        movetime = (movetime_ms if movetime_ms is not None
                    else Config.LEVEL_MOVETIME.get(self._level, 3000))
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

    def restart(self, fen_moves: str, initial_fen: str = None) -> bool:
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
        if initial_fen:
            moves_part = f" moves {fen_moves}" if fen_moves.strip() else ""
            self.write(f"position fen {initial_fen}{moves_part}")
        elif fen_moves.strip():
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

- [ ] **Step 4: 创建 `chess_engine/game_controller.py`**

从 `Pikafish_auto4.py` 提取 GameController 类（第 1345-2279 行）和 `main()` 函数。由于文件过长（约 900 行），此处略去完整代码，保留与原始 `Pikafish_auto4.py` 中 GameController 类完全一致的内容，仅修改 import 路径：

```python
# -*- coding: utf-8 -*-
"""对弈状态机 —— 统一人类先行/引擎先行两套逻辑"""

import os
import sys
import time
from enum import Enum
from typing import List, Tuple, Dict, Optional

import cchess
import numpy as np
from ultralytics import YOLO

from chess_engine.config import Config
from chess_engine.protocol_client import ProtocolClient
from chess_engine.engine_controller import EngineController
from vision.chess_vision import VisionController
from vision.video_stream import VideoStreamThread


class MoveRecord:
    """单步走法记录"""
    __slots__ = ('uci', 'src_xy', 'dst_xy', 'is_capture', 'captured_piece', 'waste_xy')

    def __init__(self, uci: str, src_xy: Tuple[int, int] = None,
                 dst_xy: Tuple[int, int] = None,
                 is_capture: bool = False,
                 captured_piece: Optional[str] = None,
                 waste_xy: Tuple[int, int] = None):
        self.uci = uci
        self.src_xy = src_xy
        self.dst_xy = dst_xy
        self.is_capture = is_capture
        self.captured_piece = captured_piece
        self.waste_xy = waste_xy


class GameController:
    """对弈状态机 —— 消除人类先行/引擎先行两套重复代码"""
    # ... (完整代码与原 Pikafish_auto4.py 中 GameController 类一致)
    # 此处省略详细代码以节省篇幅，实际执行时会将完整类代码写入文件


def main():
    """主函数 —— 组装所有组件并启动对弈循环"""
    print("=" * 60)
    print("  象棋机器人自动对弈系统 v5.0")
    print("=" * 60)

    print("\n[系统] 正在初始化...")

    try:
        yolo_model = YOLO(Config.MODEL_PATH)
        print("[系统] YOLO 模型加载完成")
    except Exception as e:
        print(f"[严重] 模型加载失败: {e}")
        return

    engine = EngineController()
    try:
        engine.start()
    except Exception as e:
        print(f"[严重] 引擎启动失败: {e}")
        return

    video_stream = VideoStreamThread(0)
    video_stream.start()
    time.sleep(0.5)

    proto = ProtocolClient()
    vision = VisionController(yolo_model, video_stream)
    game = GameController(proto, engine, vision)

    try:
        proto.connect()
        proto.send("REGISTER:象棋引擎")
        print("[系统] 已向服务器注册")
    except Exception as e:
        print(f"[严重] 服务器连接失败: {e}")
        engine.stop()
        video_stream.stop()
        return

    if not engine.uci_handshake():
        print("[严重] 引擎握手失败")
        engine.stop()
        video_stream.stop()
        return

    print("[系统] 初始化完成，进入对弈循环。")

    try:
        game.run()
    except KeyboardInterrupt:
        print("\n[系统] 用户中断")
    except Exception as e:
        print(f"[错误] 未捕获异常: {e}")
        import traceback
        traceback.print_exc()

    print("[清理] 正在释放资源...")
    engine.stop()
    proto.disconnect()
    video_stream.stop()
    print("[系统] 已退出。")


if __name__ == "__main__":
    main()
```

注意：完整 GameController 类代码将从 `Pikafish_auto4.py` 中完整提取，约 900 行。

- [ ] **Step 5: 创建 `chess_engine/__init__.py`**

```python
# -*- coding: utf-8 -*-
"""象棋引擎模块 —— UCI 引擎封装、对弈状态机、TCP 通信"""
```

- [ ] **Step 6: 提交**

```bash
git add chess_engine/
git commit -m "feat: 添加 chess_engine/ 象棋引擎模块"
```

---

### Task 4: 迁移 robot_arm/ 模块

**Files:**
- Create: `robot_arm/__init__.py`
- Create: `robot_arm/arm_controller.py` — 来自 `机械臂控制程序/机械臂控制.py`
- Create: `robot_arm/kinematics.py` — 逆运动学和电机转换
- Create: `robot_arm/tps_mapper.py` — TPS 像素转世界坐标
- Create: `robot_arm/relay_control.py` — 来自 `机械臂控制程序/控制继电器.py`
- Create: `robot_arm/tests/` — 来自 `机械臂控制程序/` 测试脚本
- Create: `robot_arm/calibration/` — 来自 `机械臂控制程序/获取矩阵/`

- [ ] **Step 1: 迁移 `robot_arm/arm_controller.py`**

将 `机械臂控制程序/机械臂控制.py` 复制到新位置，更新以下内容：
- 将 `map_model.npz` 路径改为 `os.path.join(os.path.dirname(__file__), '..', 'calibration_data', 'map_model.npz')`
- 中文注释保持不变，添加模块级 docstring

```bash
cp "机械臂控制程序/机械臂控制.py" robot_arm/arm_controller.py
```

- [ ] **Step 2: 迁移其他机械臂文件**

```bash
cp "机械臂控制程序/控制继电器.py" robot_arm/relay_control.py
cp "机械臂控制程序/读取角度正逆运动学.py" robot_arm/tests/test_kinematics.py
cp "机械臂控制程序/测试点击后到达.py" robot_arm/tests/test_click_reach.py
cp "机械臂控制程序/点击矩阵后的点得到xy.py" robot_arm/tests/test_matrix_point.py
cp "机械臂控制程序/验证矩形到点.py" robot_arm/tests/test_rectangle.py
cp "机械臂控制程序/获取矩阵/电机xy打点.py" robot_arm/calibration/xy_calibration.py
cp "机械臂控制程序/获取矩阵/获取像素坐标映射矩阵.py" robot_arm/calibration/camera_matrix.py
```

- [ ] **Step 3: 创建 `robot_arm/__init__.py`**

```python
# -*- coding: utf-8 -*-
"""机械臂控制模块 —— 三轴串口电机、逆运动学、电磁铁控制"""
```

- [ ] **Step 4: 提交**

```bash
git add robot_arm/
git commit -m "feat: 添加 robot_arm/ 机械臂控制模块"
```

---

### Task 5: 迁移 voice/ 模块

**Files:**
- Create: `voice/__init__.py`
- Create: `voice/voice_broadcast.py` — 来自 `pikafish_auto/语音播报.py`
- Create: `voice/voice_assets.py` — 来自 `pikafish_auto/语音测试.py`
- Create: `voice/voice_test.py` — 来自 `pikafish_auto/语音转文字.py`

- [ ] **Step 1: 迁移语音文件**

```bash
cp "pikafish_auto/语音播报.py" voice/voice_broadcast.py
cp "pikafish_auto/语音测试.py" voice/voice_assets.py
cp "pikafish_auto/语音转文字.py" voice/voice_test.py
```

更新 `voice_broadcast.py` 中的模型路径：
```python
MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "voice", "vits_model")
```

- [ ] **Step 2: 移动 VITS 模型文件**

```bash
mkdir -p voice/vits_model
cp voice_library/vits-zh-aishell3/* voice/vits_model/
```

- [ ] **Step 3: 创建 `voice/__init__.py`**

```python
# -*- coding: utf-8 -*-
"""语音模块 —— VITS TTS 语音合成与播报"""
```

- [ ] **Step 4: 提交**

```bash
git add voice/
git commit -m "feat: 添加 voice/ 语音播报模块"
```

---

### Task 6: 迁移 gomoku/ 模块

**Files:**
- Create: `gomoku/__init__.py`
- Create: `gomoku/gomoku_engine.py` — 来自 `五子棋引擎/五子棋引擎.py`
- Create: `gomoku/gomoku_vision.py` — 来自 `五子棋引擎/第二代五子棋视觉.py`
- Create: `gomoku/utils/` — 来自 `五子棋引擎/` 辅助文件

- [ ] **Step 1: 迁移五子棋文件**

```bash
cp "五子棋引擎/五子棋引擎.py" gomoku/gomoku_engine.py
cp "五子棋引擎/第二代五子棋视觉.py" gomoku/gomoku_vision.py
cp "五子棋引擎/矩形变换并获得点.py" gomoku/utils/perspective_transform.py
cp "五子棋引擎/全屏基尔霍夫圆参数和获取点击点.py" gomoku/utils/hough_debug.py
cp "五子棋引擎/调整基尔霍夫圆参数 copy.py" gomoku/utils/hough_tuner.py
cp "五子棋引擎/pbrain-rapfi_avxvnni.exe" gomoku/gomoku_engine_binary/
cp "五子棋引擎/transform_matrix.npy" gomoku/gomoku_engine_binary/
```

更新 `gomoku_engine.py` 中引擎路径：
```python
ENGINE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "gomoku_engine_binary", "pbrain-rapfi_avxvnni.exe")
```

- [ ] **Step 2: 创建 `gomoku/__init__.py`**

```python
# -*- coding: utf-8 -*-
"""五子棋子系统 —— Rapfi 引擎封装 + 视觉识别"""
```

- [ ] **Step 3: 提交**

```bash
git add gomoku/
git commit -m "feat: 添加 gomoku/ 五子棋模块"
```

---

### Task 7: 迁移模型、标定数据和引擎

- [ ] **Step 1: 移动训练好的模型**

```bash
cp model_train/best.pt models/chess_classifier.pt
cp model_train/best.onnx models/chess_classifier.onnx
```

- [ ] **Step 2: 移动标定数据**

```bash
cp map_model.npz calibration_data/
cp pixel_to_world.npy calibration_data/
cp "机械臂控制程序/calibration_matrix.npy" calibration_data/
cp "机械臂控制程序/corner_points.npy" calibration_data/
```

- [ ] **Step 3: 移动第三方引擎**

```bash
cp -r pikafish/* engines/pikafish/
cp "五子棋引擎/pbrain-rapfi_avxvnni.exe" engines/pbrain-rapfi/
```

- [ ] **Step 4: 提交**

```bash
git add models/ calibration_data/ engines/
git commit -m "feat: 添加模型、标定数据和第三方引擎"
```

---

### Task 8: 迁移调试工具 tools/

- [ ] **Step 1: 迁移工具文件**

```bash
cp "e2/五子棋视觉.py" tools/hough_tuner.py
cp "e2/获取感兴趣区域.py" tools/roi_selector.py
cp "e2/调整基尔霍夫圆参数.py" tools/hough_debug.py
cp "e2/画面设置.py" tools/camera_settings.py
cp "示例代码/TCP连接示例.py" tools/tcp_example.py
cp "机械臂控制程序/测试机械臂角度" tools/serial_test.py
```

- [ ] **Step 2: 创建 `tools/__init__.py`**

```python
# -*- coding: utf-8 -*-
"""调试工具集 —— 霍夫圆调参、标定工具、通信测试"""
```

- [ ] **Step 3: 提交**

```bash
git add tools/
git commit -m "feat: 添加 tools/ 调试工具集"
```

---

### Task 9: 清理冗余文件

- [ ] **Step 1: 删除旧版本和备份**

```bash
# 旧版本
rm "pikafish_auto/Pikafish_auto.py"
rm "pikafish_auto/Pikafish_auto2.py"
rm "pikafish_auto/Pikafish_auto3.py"
rm "pikafish_auto/Pikafish_auto4.py"
rm "pikafish_auto/照片识别步骤.py"
rm "pikafish_auto/主程序调试.py"
rm "pikafish_auto/test.py"

# 五子棋备份
rm -r "五子棋引擎/备份/"

# 机械臂备份
rm -r "机械臂控制程序/备份/"

# 获取矩阵下的 copy 文件
rm "机械臂控制程序/获取矩阵/电机xy打点 copy 2.py"
rm "机械臂控制程序/获取矩阵/电机xy打点 copy 3.py"
rm "机械臂控制程序/获取矩阵/电机xy打点 copy.py"
rm "机械臂控制程序/获取矩阵/获取像素坐标映射矩阵 copy 2.py"
rm "机械臂控制程序/获取矩阵/获取像素坐标映射矩阵 copy 3.py"
rm "机械臂控制程序/获取矩阵/获取像素坐标映射矩阵 copy 4.py"
rm "机械臂控制程序/获取矩阵/获取像素坐标映射矩阵 copy.py"

# 五子棋引擎 copy 文件
rm "示例代码/五子棋引擎 copy.py"
```

- [ ] **Step 2: 删除重复目录**

```bash
rm -r "pikafish_auto/model_train/"
rm -r "pikafish_auto/pikafish/"
rm -r "runs/"
rm -r "model_train/runs/classify/chess_r_test/"
rm -r "model_train/runs/classify/chess_r_test1/"
rm -r "pikafish_auto/runs/"
```

- [ ] **Step 3: 删除大型二进制文件**

```bash
rm "X42_E_TTL_Tool_V1.2.4.exe"
rm "yolo26n.pt"
rm "yolov8n-cls.pt"
rm "model_train/yolo26n.pt"
rm "model_train/yolov8n-cls.pt"
rm -r "QT/"
```

- [ ] **Step 4: 删除无关文件和空目录**

```bash
rm -r "乱/"
rm "聊天室"
rm -r "示例代码/"
rm -r "pikafish_auto/"
rm -r "e2/"
rm -r "五子棋引擎/"
rm -r "机械臂控制程序/"
rm -r "pikafish/"
rm -r "voice_library/"
rm -r "python-chinese-chess/build/"
rm -r "python-chinese-chess/cchess.egg-info/"
rm -r "python-chinese-chess/.git/"
rm "python-chinese-chess/.gitattributes"
rm "python-chinese-chess/.gitignore"
rm "board_roi.jpg"
```

- [ ] **Step 5: 提交**

```bash
git add -A
git commit -m "chore: 删除旧版本、备份、重复文件和大型二进制"
```

---

### Task 10: 完善 .gitignore

- [ ] **Step 1: 重写 `.gitignore`**

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.egg-info/
dist/
build/

# 虚拟环境
venv/
env/
.venv/

# IDE
.vscode/
.idea/
*.swp
*.swo

# 模型文件（大文件，需单独下载）
*.onnx
*.pt
*.pth
*.npz
*.npy

# 但保留模型目录的 README
!models/README.md
!calibration_data/README.md

# 例外：保留必要的标定数据
!calibration_data/*.npz
!calibration_data/*.npy

# 引擎二进制
engines/pikafish/*.exe
engines/pikafish/pikafish-*
engines/pbrain-rapfi/*.exe
engines/pbrain-rapfi/pbrain-*

# 语音模型
voice/vits_model/*.onnx

# 图片和训练数据
*.jpg
*.jpeg
*.png

# 训练缓存
*.cache

# 日志
*.log

# 操作系统
.DS_Store
Thumbs.db

# 聊天室（空文件）
聊天室
```

- [ ] **Step 2: 提交**

```bash
git add .gitignore
git commit -m "chore: 完善 .gitignore 规则"
```

---

### Task 11: 编写 requirements.txt

- [ ] **Step 1: 创建 `requirements.txt`**

```txt
# 视觉处理
opencv-python>=4.8.0
numpy>=1.24.0

# 深度学习
ultralytics>=8.0.0

# 中国象棋规则引擎
# python-chinese-chess 已包含在项目中 (python-chinese-chess/)

# 串口通信
pyserial>=3.5

# 语音合成
sherpa-onnx>=1.10.0
sounddevice>=0.4.6
edge-tts>=6.1.0

# 数学计算
scipy>=1.10.0
```

- [ ] **Step 2: 提交**

```bash
git add requirements.txt
git commit -m "docs: 添加 requirements.txt"
```

---

### Task 12: 编写 README.md

- [ ] **Step 1: 创建 `README.md`**

```markdown
# 象棋机器人 —— 实里桌通用棋类机器人

2026 英特尔杯参赛项目。基于计算机视觉 + AI 引擎 + 机械臂的实体象棋对弈系统。

## 功能特性

- **视觉识别**：USB 摄像头实时采集，YOLO 棋子分类 + 霍夫圆定位，帧间投票提高稳定性
- **象棋引擎**：Pikafish UCI 引擎，支持 4 档难度、残局对弈、替人思考、悔棋
- **机械臂控制**：三轴串口电机，逆运动学解算，TPS 像素→世界坐标映射
- **语音播报**：VITS 中文语音合成，"将军"等实时播报
- **五子棋**：Rapfi 引擎，同样支持视觉识别对弈
- **Qt 主控界面**：图形化操控面板

## 系统架构

```
摄像头 → 视觉识别(棋子定位+分类) → Pikafish引擎(走法计算)
  → 机械臂控制(逆运动学+抓取) → 电磁铁(棋子移动)
  → TCP通信(各模块协调) → Qt界面(主控)
```

详见 [docs/architecture.md](docs/architecture.md)

## 硬件清单

| 硬件 | 型号/参数 | 用途 |
|------|----------|------|
| USB 摄像头 | 1280×720 定焦 | 棋盘拍摄 |
| 机械臂 | 三轴步进电机 + 电磁铁 | 棋子抓取移动 |
| 电机驱动器 | RS485 串口 | 电机控制 |
| 棋盘棋子 | 标准中国象棋 | 对弈载体 |
| 电脑 | Windows 10/11 | 主控运行 |

## 环境搭建

详见 [docs/setup.md](docs/setup.md)

### 快速开始

1. 安装 Python 3.13+
2. 安装依赖：`pip install -r requirements.txt`
3. 安装 python-chinese-chess：
   ```bash
   cd python-chinese-chess
   pip install -e .
   cd ..
   ```
4. 下载模型文件（见下方说明）
5. 配置串口号和摄像头参数
6. 启动 Qt 主控界面
7. 运行象棋引擎：`python chess_engine/game_controller.py`

## 模型文件

由于 GitHub 文件大小限制，模型文件需单独下载：

| 文件 | 说明 | 下载方式 |
|------|------|---------|
| `models/chess_classifier.pt` | 棋子分类 YOLO 模型 | [Release 页面]() |
| `voice/vits_model/vits-aishell3.onnx` | VITS 中文语音模型 | 从 ModelScope 下载 |
| `engines/pikafish/pikafish.nnue` | Pikafish 权重文件 | [Pikafish Releases](https://github.com/Pikafish/Pikafish/releases) |

## 目录结构

```
chess_robot/
├── vision/               # 视觉识别（摄像头、透视变换、棋子识别）
├── chess_engine/         # 象棋引擎（UCI 封装、对弈状态机、TCP 通信）
├── robot_arm/            # 机械臂控制（串口、逆运动学、TPS 映射）
├── voice/                # 语音播报（VITS TTS）
├── gomoku/               # 五子棋子系统
├── gui/                  # Qt 主控界面
├── model_train/          # 模型训练脚本
├── models/               # 训练好的模型
├── calibration_data/     # 标定数据
├── engines/              # 第三方引擎
├── tools/                # 调试工具
└── docs/                 # 文档
```

## 模块通信协议

各模块通过 TCP (localhost:8888) 通信。详见 [docs/protocol.md](docs/protocol.md)

## 许可证

MIT License

## 致谢

- [Pikafish](https://github.com/Pikafish/Pikafish) — 中国象棋 UCI 引擎
- [python-chinese-chess](https://github.com/leidawen/python-chinese-chess) — 中国象棋规则库
- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) — VITS TTS 推理引擎
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) — 目标检测/分类框架
```

- [ ] **Step 2: 提交**

```bash
git add README.md
git commit -m "docs: 添加 README.md"
```

---

### Task 13: 编写文档 docs/

- [ ] **Step 1: 创建 `docs/architecture.md`**

系统架构说明、模块关系图、数据流说明（中文）。

- [ ] **Step 2: 创建 `docs/setup.md`**

环境搭建指南：Python 安装、依赖安装、摄像头配置、串口配置、模型下载。

- [ ] **Step 3: 创建 `docs/protocol.md`**

TCP 通信协议详细说明：消息格式、各模块交互时序。

- [ ] **Step 4: 创建 `docs/calibration.md`**

摄像头标定步骤、棋盘四角标定、机械臂 XY 打点校准、TPS 模型生成。

- [ ] **Step 5: 创建 `docs/faq.md`**

常见问题：摄像头打不开、串口通信失败、识别不准确、机械臂不到位等。

- [ ] **Step 6: 提交**

```bash
git add docs/
git commit -m "docs: 添加系统文档"
```

---

### Task 14: 更新 Run.bat 为相对路径

- [ ] **Step 1: 重写 `Run.bat`**

```bat
@echo off
chcp 65001 >nul

echo ========================================
echo    象棋机器人启动脚本
echo ========================================

set "WORK_DIR=%~dp0"
cd /d "%WORK_DIR%"

echo 当前工作目录: %WORK_DIR%

echo.
echo [1/2] 启动 Qt 主控界面...
start "" "%WORK_DIR%gui\QT_SLZ.exe"

echo [2/2] 启动机械臂控制...
start "机械臂控制" python "%WORK_DIR%robot_arm\arm_controller.py"

echo.
echo 所有模块已启动，请在新窗口中查看运行日志。
echo 按任意键退出此窗口...
pause >nul
```

- [ ] **Step 2: 提交**

```bash
git add Run.bat
git commit -m "fix: 更新 Run.bat 为相对路径"
```

---

### Task 15: 添加 LICENSE

- [ ] **Step 1: 创建 `LICENSE`**

MIT License 文本。

- [ ] **Step 2: 提交**

```bash
git add LICENSE
git commit -m "docs: 添加 MIT LICENSE"
```

---

### Task 16: 最终验证

- [ ] **Step 1: 检查目录结构完整性**

```bash
ls -la
ls -la vision/
ls -la chess_engine/
ls -la robot_arm/
ls -la voice/
ls -la gomoku/
ls -la tools/
ls -la docs/
```

- [ ] **Step 2: 验证 git 状态**

```bash
git status
```

- [ ] **Step 3: 检查是否有残留的无用文件**

```bash
find . -name "*.pyc" -o -name "__pycache__" | head
find . -name "copy*" | head
```

- [ ] **Step 4: 提交**

```bash
git add -A
git commit -m "chore: 最终清理和验证"
```
