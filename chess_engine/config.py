# -*- coding: utf-8 -*-
"""集中配置 —— 所有可调参数"""

import os
from typing import List, Tuple, Dict

import cv2
import numpy as np


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

    # 针对特定易漏棋子降低置信度门槛
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

    # 棋盘实际四个角点坐标（用于废棋区过滤，需大于 GRID 范围）
    BOARD_LT = (106, 2)
    BOARD_RT = (604, 2)
    BOARD_LB = (105, 624)
    BOARD_RB = (607, 624)

    # --- 视觉 ---
    STABLE_FRAMES = 3          # 保留兼容，旧的多帧一致性参数
    EDGE_PADDING = 3
    MAX_CAPTURE_ATTEMPTS = 40

    # 帧间投票参数
    VOTE_FRAMES = 15           # 投票总帧数
    VOTE_THRESHOLD = 8         # 最少确认票数 (> VOTE_FRAMES/2)
    VOTE_INTERVAL = 0.06       # 帧间间隔（秒）
    MAX_VOTE_RETRIES = 3       # 校验失败最大重试次数
    VOTE_RETRY_EXTRA = 5       # 每次重试追加的帧数

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
    # 人类执红时的标准开局：红方在上（摄像头顶部=人类侧）
    RED_STANDARD_OPENING_FEN = (
        "RNBAKABNR/9/1C5C1/P1P1P1P1P/9/9/"
        "p1p1p1p1p/1c5c1/9/rnbakabnr w - - 0 1"
    )

    @staticmethod
    def fen_to_uci_pieces(fen: str, human_is_red: bool = False
                          ) -> List[Tuple[str, str]]:
        """解析 FEN 行棋部分，返回 [(uci, label), ...]。

        human_is_red=False (默认): rank = 9 - fen_row (红方在底部)
        human_is_red=True: rank = fen_row (红方在顶部)
        """
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

    # 残局 FEN 库（键为残局名称，值为 FEN 字符串，待后续补充）
    ENDGAME_FEN_DB: Dict[str, str] = {}

    VALID_PIECE_CHARS = set("KRNCBAPkrncbap")
    FEN_ROW_COUNT = 10
    FEN_COL_COUNT = 9

    @staticmethod
    def validate_fen(fen: str) -> Tuple[bool, str]:
        """校验中国象棋 FEN 字符串是否合法。

        Args:
            fen: FEN 字符串，如
                 "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"

        Returns:
            (是否合法, 错误描述)。合法时错误描述为空字符串。
        """
        if not fen or not isinstance(fen, str):
            return False, "FEN 为空或类型错误"

        parts = fen.strip().split()
        if len(parts) < 2:
            return False, f"FEN 字段不足，需要 ≥2 个字段，实际 {len(parts)} 个"

        # --- 1. 棋子布局 ---
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

        # --- 2. 走子方 ---
        side = parts[1]
        if side not in ('w', 'b'):
            return False, f"走子方字段应为 'w' 或 'b'，实际 '{side}'"

        # --- 3. 半回合数 / 全回合数（可选但若存在须为数字） ---
        if len(parts) >= 5:
            if not parts[-2].isdigit():
                return False, f"半回合数字段应为数字，实际 '{parts[-2]}'"
        if len(parts) >= 6:
            if not parts[-1].isdigit():
                return False, f"全回合数字段应为数字，实际 '{parts[-1]}'"

        return True, ""
