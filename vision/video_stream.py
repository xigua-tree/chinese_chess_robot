# -*- coding: utf-8 -*-
"""视频流采集 —— 独立线程摄像头读取与画面显示"""

import os
import threading
import time
from typing import Optional

import cv2
import numpy as np

from chess_engine.config import Config
from vision.board_utils import draw_board_grid


def _on_mouse_click(event, x, y, flags, param):
    """鼠标点击回调 —— 打印像素坐标"""
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"\n[点击坐标] X: {x}, Y: {y}")


def _hough_trackbar_noop(_val):
    """Hough Controls trackbar 回调（空函数，参数通过 getTrackbarPos 实时读取）"""
    pass


class VideoStreamThread(threading.Thread):
    """独立线程：实时采集摄像头、显示画面，为识别逻辑提供最新帧"""

    def __init__(self, camera_idx: int = 0):
        super().__init__()
        self.cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        #self.cap.set(cv2.CAP_PROP_EXPOSURE, -5)
        self.cap.set(cv2.CAP_PROP_SETTINGS, 0)
        self.ret = False
        self.frame = None
        self.warped = None
        self.annotated_warped = None  # VisionController 识别后写入，供显示
        self.stopped = False
        self.daemon = True
        # 霍夫圆参数（由本线程 trackbar 实时更新，VisionController 读取）
        self.hough_param1 = 49
        self.hough_param2 = 30
        self.hough_minDist = 32
        self.hough_minRadius = 15
        self.hough_maxRadius = 25

    def run(self):
        print("[系统] 视频监控已启动。按下 'q' 键退出。")
        cv2.namedWindow("Warped View")
        cv2.setMouseCallback("Warped View", _on_mouse_click)

        # 创建霍夫圆调节面板（必须与 waitKey 同一线程，否则 trackbar 无响应）
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
            # 有识别标注就用标注图，否则用纯网格图
            if self.annotated_warped is not None:
                cv2.imshow("Warped View", self.annotated_warped)
            else:
                cv2.imshow("Warped View", grid_img)
            self.warped = warped

            key = cv2.waitKey(1) & 0xFF

            # 实时读取 trackbar 参数（与 waitKey 同线程，值能正常更新）
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
