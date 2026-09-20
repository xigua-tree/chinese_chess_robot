import cv2
import numpy as np
import os

# 全局变量记录点位
pts_current = []

def nothing(x):
    pass

def click_handler(event, x, y, flags, param):
    global pts_current
    if event == cv2.EVENT_LBUTTONDOWN:
        pts_current.append([x, y])
        cv2.circle(param, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Step 1: Calibration", param)

def debug_hough_center_ref(target_filename="5.jpg"):
    global pts_current
    pts_current = []
    
    # 1. 路径处理
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_path = os.path.join(current_dir, target_filename)
    if not os.path.exists(target_path):
        target_path = os.path.join(current_dir, "e2", target_filename)

    raw_target = cv2.imdecode(np.fromfile(target_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if raw_target is None:
        print("图片读取失败！")
        return

    # 2. ROI 标定
    ROI_X, ROI_Y, ROI_W, ROI_H = 0, 4, 864, 954
    roi = raw_target[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W]
    display_img = roi.copy()

    cv2.imshow("Step 1: Calibration", display_img)
    cv2.setMouseCallback("Step 1: Calibration", click_handler, display_img)
    
    print("\n[操作指引] 请点击：左上 -> 右上 -> 左下 -> 右下")
    while len(pts_current) < 4:
        if cv2.waitKey(1) & 0xFF == ord('q'): return

    # 3. 透视变换
    side = 600
    height = int(side * 38 / 31)  # 735
    src = np.float32(pts_current)
    dst = np.float32([[0, 0], [side, 0], [0, height], [side, height]])
    M = cv2.getPerspectiveTransform(src, dst)
    img_warped = cv2.warpPerspective(roi, M, (side, height))
    
    # 预处理
    gray = cv2.cvtColor(img_warped, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)

    # 4. 创建调试窗口
    cv2.namedWindow("Step 2: Center Ref Debug")
    # 默认值
    cv2.createTrackbar("Param2", "Step 2: Center Ref Debug", 20, 100, nothing)
    cv2.createTrackbar("MinR", "Step 2: Center Ref Debug", 10, 50, nothing)
    cv2.createTrackbar("MaxR", "Step 2: Center Ref Debug", 26, 100, nothing)
    cv2.createTrackbar("MinDist", "Step 2: Center Ref Debug", 20, 100, nothing)

    print("\n>>> 调试模式启动！")
    print(">>> 画面正中心的【黄色圆圈】代表当前的 MinR（实心）和 MaxR（空心）。")
    print(">>> 请调节参数直到所有红圈都在，且没有空气棋子。")

    while True:
        p2 = cv2.getTrackbarPos("Param2", "Step 2: Center Ref Debug")
        min_r = cv2.getTrackbarPos("MinR", "Step 2: Center Ref Debug")
        max_r = cv2.getTrackbarPos("MaxR", "Step 2: Center Ref Debug")
        min_d = cv2.getTrackbarPos("MinDist", "Step 2: Center Ref Debug")
        
        p2 = max(1, p2)
        min_d = max(1, min_d)
        if max_r <= min_r: max_r = min_r + 1

        canvas = img_warped.copy()
        
        # --- 霍夫圆检测 ---
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, 
            minDist=min_d, param1=50, param2=p2, 
            minRadius=min_r, maxRadius=max_r
        )

        detected_count = 0
        if circles is not None:
            circles = np.uint16(np.around(circles))
            detected_count = len(circles[0])
            for i in circles[0, :]:
                # 绿色圈：识别出的棋子
                cv2.circle(canvas, (i[0], i[1]), i[2], (0, 255, 0), 2)
                cv2.circle(canvas, (i[0], i[1]), 2, (0, 0, 255), -1)

        # --- 优化：将参考基准圆放在画面正中间 ---
        ref_x, ref_y = side // 2, side // 2 # 中心坐标 (300, 300)
        # 绘制最大允许圆 (黄色空心)
        cv2.circle(canvas, (ref_x, ref_y), max_r, (0, 255, 255), 1) 
        # 绘制最小允许圆 (黄色半透明/填充)
        cv2.circle(canvas, (ref_x, ref_y), min_r, (0, 255, 255), 2) 
        # 添加文本说明
        cv2.putText(canvas, f"Ref: {min_r}-{max_r}", (ref_x-45, ref_y+max_r+15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # 数据显示
        info_txt = f"P2:{p2} | Count:{detected_count}"
        cv2.putText(canvas, info_txt, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Step 2: Center Ref Debug", canvas)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"\n--- 最终确认参数 ---")
            print(f"Param2: {p2}, MinRadius: {min_r}, MaxRadius: {max_r}, MinDist: {min_d}")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    debug_hough_center_ref("5.png")