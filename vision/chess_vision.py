# -*- coding: utf-8 -*-
"""象棋视觉识别 —— YOLO + HoughCircles 棋盘识别、棋子定位、走法差分"""

import time
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np

from chess_engine.config import Config
from vision.board_utils import draw_board_grid


class VisionController:
    """视觉识别 —— YOLO + HoughCircles 棋盘识别"""

    # 人类废棋区 ROI（warped 图坐标，棋盘左侧）
    HUMAN_WASTE_ROI = (10, 0, 103, 625)  # (x1, y1, x2, y2)
    # 机器人废棋区 ROI（warped 图坐标，棋盘右侧）
    ROBOT_WASTE_ROI = (609, 2, 705, 623)  # (x1, y1, x2, y2)

    def __init__(self, yolo_model, video_stream):
        self._model = yolo_model
        self._stream = video_stream
        self._human_is_red: bool = False  # 人类执红时UCI映射反向

    def set_human_is_red(self, is_red: bool):
        self._human_is_red = is_red

    # ------------------------------------------------------------------
    # 棋盘识别
    # ------------------------------------------------------------------
    def recognize_board(self, warped_img: np.ndarray) -> List[List[str]]:
        """单帧识别：HoughCircles 寻圆 + YOLO 分类 → 10x9 矩阵"""
        board = [[None for _ in range(9)] for _ in range(10)]

        # 从视频线程读取霍夫圆参数（trackbar 与 waitKey 同线程，实时更新）
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

            # 边缘棋子处理：BORDER_REPLICATE 扩展而非 clamp
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

                # 可视化：蓝色圆圈 + 标签 + 置信度
                display = label.split('_')[-1]
                cv2.circle(warped_img, (cx, cy), r, (255, 0, 0), 2)
                # 上方棋子标签放在圆下方，下方棋子标签放在圆上方
                if cy < Config.WARP_H // 2:
                    text_y = cy + r + 14
                else:
                    text_y = cy - r - 5
                cv2.putText(warped_img, f"{display} {conf:.2f}",
                            (cx - 18, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)

                thresh = Config.CLASS_CONF_OVERRIDE.get(label, Config.CONF_THRESH)
                if conf > thresh:
                    # 高于阈值：实心绿点标记
                    cv2.circle(warped_img, (cx, cy), 3, (0, 255, 0), -1)

                    # 过滤废棋区：圆心在棋盘实际角点范围之外则丢弃
                    if (cx < Config.BOARD_LT[0] or cx > Config.BOARD_RT[0] or
                        cy < Config.BOARD_LT[1] or cy > Config.BOARD_LB[1]):
                        cv2.circle(warped_img, (cx, cy), r, (0, 0, 255), 1)
                        continue

                    # 映射到 10x9 网格
                    rel_x = (cx - Config.GRID_LT[0]) / max(1, (Config.GRID_RT[0] - Config.GRID_LT[0]))
                    rel_y = (cy - Config.GRID_LT[1]) / max(1, (Config.GRID_LB[1] - Config.GRID_LT[1]))

                    grid_c = min(8, max(0, int(round(rel_x * 8))))
                    grid_r = min(9, max(0, int(round(rel_y * 9))))

                    board[grid_r][grid_c] = label
                else:
                    # 低于阈值：红色小点标记（仅可视化，不进入棋盘矩阵）
                    cv2.circle(warped_img, (cx, cy), 3, (0, 0, 255), -1)

        # 同步到视频流供 Warped View 窗口展示（叠加网格线 + 识别标注）
        if hasattr(self._stream, 'annotated_warped'):
            self._stream.annotated_warped = draw_board_grid(warped_img)

        return board

    # ------------------------------------------------------------------
    # 帧间投票
    # ------------------------------------------------------------------
    def _vote_board(self, frames: int, threshold: int,
                    existing_votes: List[List[Dict]] = None
                    ) -> Tuple[List[List[str]], List[List[Dict]]]:
        """采集 frames 帧，累加到 votes 字典后产出候选棋盘。

        返回 (候选棋盘, 累加后的 votes)。
        """
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

        # 裁决：每个格子独立决定
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

    # 每种棋子类型的最大数量（开局初始值）
    _PIECE_TYPE_MAX = {
        'Kin': 1, 'Shi': 2, 'Ele': 2, 'Hor': 2, 'Car': 2, 'Can': 2, 'Paw': 5,
    }

    @staticmethod
    def _validate_piece_count(board: List[List[str]],
                               expected_red: int = None,
                               expected_black: int = None,
                               expected_rc: Dict[str, int] = None,
                               expected_bc: Dict[str, int] = None) -> Tuple[bool, str]:
        """按颜色+类型校验棋子数量合法性。

        将/帅必须恰好 1 个，其他棋种不超过开局最大数量。
        当提供 expected 时，总数和每种类型数量都必须精确匹配。
        返回 (是否通过, 失败原因)。
        """
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

        # 将/帅必须存在
        if rc.get('Kin', 0) != 1:
            return False, f"红帅数量={rc.get('Kin', 0)}，期望 1"
        if bc.get('Kin', 0) != 1:
            return False, f"黑将数量={bc.get('Kin', 0)}，期望 1"

        # 每种棋类不超过初始上限
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

        # 总数精确匹配
        if expected_red is not None and red != expected_red:
            return False, f"红方总数预期 {expected_red}，实际 {red}"
        if expected_black is not None and black != expected_black:
            return False, f"黑方总数预期 {expected_black}，实际 {black}"

        # 每种类型数量精确匹配（防止类型混淆，如少了一个车但多了一个炮）
        for ptype in VisionController._PIECE_TYPE_MAX:
            if expected_rc is not None and rc.get(ptype, 0) != expected_rc.get(ptype, 0):
                return False, (f"红{ptype}预期 {expected_rc.get(ptype, 0)}，"
                               f"实际 {rc.get(ptype, 0)}")
            if expected_bc is not None and bc.get(ptype, 0) != expected_bc.get(ptype, 0):
                return False, (f"黑{ptype}预期 {expected_bc.get(ptype, 0)}，"
                               f"实际 {bc.get(ptype, 0)}")

        return True, ""

    # ------------------------------------------------------------------
    def capture_stable_board(self, required_stable: int = None,
                             expected_red: int = None,
                             expected_black: int = None,
                             expected_rc: Dict[str, int] = None,
                             expected_bc: Dict[str, int] = None) -> Optional[List[List[str]]]:
        """帧间投票 + 棋子数校验，返回确认的棋盘矩阵。

        每格独立统计多帧棋子出现次数，超过阈值才确认。
        校验失败时追加帧数重新投票。

        Args:
            required_stable: 已弃用，保留兼容。
            expected_red: 预期的红方棋子总数。
            expected_black: 预期的黑方棋子总数。
            expected_rc: 预期的红方每种棋子类型数量。
            expected_bc: 预期的黑方每种棋子类型数量。
        """
        votes = [[{} for _ in range(9)] for _ in range(10)]
        threshold = Config.VOTE_THRESHOLD
        total_frames = 0

        print(f"[视觉] 帧间投票中 (初始 {Config.VOTE_FRAMES} 帧, 阈值 {threshold})...")

        # 初始投票
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

            print(f"[视觉] 校验失败: {reason} (已采 {total_frames} 帧, 红={red} 黑={black})，追加 {Config.VOTE_RETRY_EXTRA} 帧...")
            board, votes = self._vote_board(Config.VOTE_RETRY_EXTRA, threshold, votes)
            total_frames += Config.VOTE_RETRY_EXTRA

    @staticmethod
    def _board_mask(board: List[List[str]]) -> List[List[bool]]:
        """将棋盘矩阵转为布尔掩码，仅保留棋子有无，忽略类型"""
        return [[bool(p) for p in row] for row in board]

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

        # 详细调试输出
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
            # changed 优先于 added：_last_board 在目标格存了错误棋子，
            # 新棋子覆盖后表现为 "changed" 而非 "added"
            src, dst = removed[0], changed[0]
            print(f"[getMove] 变更移动: {src} -> {dst}")
        elif len(removed) == 1 and len(added) == 0 and len(changed) == 0:
            # 仅移除（被吃子），无其他变化 → 忽略
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
        """棋盘坐标 → UCI 坐标。
        人类执黑：cchess rank 0 = 红方底线 = 摄像头 row 9 → rank = 9 - row
        人类执红：红方在摄像头顶部 → rank = row
        """
        file_char = chr(ord('a') + col)
        rank = row if self._human_is_red else 9 - row
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
        """在指定 ROI 内多帧投票识别棋子，返回 [(label, x, y), ...].

        连续取 N=5 帧，每帧 HoughCircles 寻圆 + YOLO 分类。
        同一 (label, 位置) 出现在 >=3 帧才确认，返回 warped 图坐标。
        """
        N_FRAMES = 5
        VOTE_MIN = 3
        PROXIMITY = 15  # 相近位置阈值（像素）

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

    def _uci_to_pixel(self, uci_pos: str) -> Tuple[int, int]:
        """UCI 坐标 → 透视图像素坐标。
        人类执黑：摄像头 row = 9 - rank。
        人类执红：摄像头 row = rank。
        """
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
        """公开的 UCI→像素 转换（供 GameController 使用）"""
        return self._uci_to_pixel(uci_pos)

    def apply_uci_move(self, board: List[List[str]], uci_move: str) -> List[List[str]]:
        """在棋盘矩阵上直接应用 UCI 走法，返回新矩阵。

        人类执黑：row = 9 - rank。人类执红：row = rank。
        """
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
    def count_pieces_by_type(board: List[List[str]]) -> Tuple[Dict[str, int], Dict[str, int]]:
        """按棋子类型统计红黑数量，返回 (rc_dict, bc_dict)。"""
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
        """标准 FEN → 摄像头行序 10x9 矩阵。

        标准 FEN row 0 = 黑方底线 = 摄像头 row 0，直接映射。
        """
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
        """从摄像头行序矩阵构造标准 FEN 字符串。

        board[0] = 摄像头顶部 = 黑方底线 = 标准 FEN row 0。
        """
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
