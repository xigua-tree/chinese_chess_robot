# -*- coding: utf-8 -*-
"""棋盘绘制工具 —— 透视网格线与交叉点"""

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
