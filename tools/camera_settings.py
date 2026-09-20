# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os

# ---------- 全局变量 ----------
pts_src = []          # 存储鼠标点击的四个点
roi_ready = False     # 是否完成四点选择
matrix = None         # 透视变换矩阵
WARP_W, WARP_H = 707, 630  # 你指定的透视变换尺寸

def nothing(x):
    pass

def mouse_callback(event, x, y, flags, param):
    """鼠标回调：依次点击 左上 -> 右上 -> 左下 -> 右下"""
    global pts_src, roi_ready, matrix
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(pts_src) < 4:
            pts_src.append([x, y])
            print(f"点 {len(pts_src)} 已记录: ({x}, {y})")
            
        if len(pts_src) == 4:
            pts_dst = np.float32([[0, 0], [WARP_W, 0], [0, WARP_H], [WARP_W, WARP_H]])
            src = np.float32(pts_src)
            matrix = cv2.getPerspectiveTransform(src, pts_dst)
            roi_ready = True
            print(">>> 透视变换矩阵已生成")

def adjust_image(frame, brightness, contrast, saturation):
    """软件调节亮度和对比度和饱和度"""
    # 1. 对比度和亮度调节: f(x) = alpha*x + beta
    # alpha [1.0-3.0] 对比度, beta [0-100] 亮度
    alpha = contrast / 50.0  
    beta = brightness - 50
    adjusted = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    # 2. 饱和度调节
    hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV).astype("float32")
    (h, s, v) = cv2.split(hsv)
    s = s * (saturation / 50.0)
    s = np.clip(s, 0, 255)
    hsv = cv2.merge([h, s, v])
    adjusted = cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2BGR)
    
    return adjusted

def main():
    global pts_src, roi_ready, matrix

    # ---------- 初始化摄像头 ----------
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # ---------- 创建窗口与滑动条 ----------
    window_name = "Camera Debugging"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    # 创建滑动条 (默认值设在中间)
    cv2.createTrackbar("Brightness", window_name, 50, 100, nothing)
    cv2.createTrackbar("Contrast", window_name, 50, 100, nothing)
    cv2.createTrackbar("Saturation", window_name, 50, 100, nothing)
    cv2.createTrackbar("Exposure", window_name, 5, 20, nothing) # 硬件曝光调节

    print("\n[操作指引]")
    print("1. 在主窗口点击四点：左上 -> 右上 -> 左下 -> 右下")
    print("2. 使用滑动条实时调节画面质量")
    print("3. 按 'r' 重置标定，按 'q' 退出")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # --- 0. 获取滑动条参数 ---
        b = cv2.getTrackbarPos("Brightness", window_name)
        c = cv2.getTrackbarPos("Contrast", window_name)
        s = cv2.getTrackbarPos("Saturation", window_name)
        e = cv2.getTrackbarPos("Exposure", window_name)

        # --- 1. 硬件调节 (曝光) ---
        # 注意：部分摄像头可能不支持手动曝光或标志位不同
        cap.set(cv2.CAP_PROP_EXPOSURE, -e) 

        # --- 2. 软件图像调节 ---
        processed_frame = adjust_image(frame, b, c, s)
        disp_img = processed_frame.copy()

        # --- 3. 绘制点击引导 ---
        for i, pt in enumerate(pts_src):
            cv2.circle(disp_img, tuple(pt), 5, (0, 0, 255), -1)
            cv2.putText(disp_img, str(i+1), (pt[0]+10, pt[1]+10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # --- 4. 透视变换与识别 ---
        if roi_ready and matrix is not None:
            warped = cv2.warpPerspective(processed_frame, matrix, (WARP_W, WARP_H))
            
            # 霍夫圆检测预处理
            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 7)

            # 使用你指定的参数
            circles = cv2.HoughCircles(
                gray, cv2.HOUGH_GRADIENT, dp=1, 
                minDist=25, param1=200, param2=39, 
                minRadius=16, maxRadius=35
            )

            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    cv2.circle(warped, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    cv2.circle(warped, (i[0], i[1]), 2, (0, 0, 255), -1)
            
            cv2.imshow("Warped View", warped)

        # --- 5. 显示主窗口 ---
        cv2.imshow(window_name, disp_img)

        # 按键逻辑
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            pts_src = []
            roi_ready = False
            matrix = None
            print(">>> 已重置标定点位")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()