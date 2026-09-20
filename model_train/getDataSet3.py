# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os

# ---------- 全局变量 ----------
WARP_W, WARP_H = 707, 630

FIXED_PTS = np.float32([
    [310, 102], [931, 95], [279, 679], [983, 664]
])
pts_dst = np.float32([[0, 0], [WARP_W, 0], [0, WARP_H], [WARP_W, WARP_H]])
matrix = cv2.getPerspectiveTransform(FIXED_PTS, pts_dst)

def nothing(x):
    pass

def main():
    global matrix

    # ---------- 参数配置 ----------
    train_dir = r"E:\chess_robot\chess_classify_data\train\B_Ele"
    val_dir = r"E:\chess_robot\chess_classify_data\val\B_Ele"
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # 统一计数：获取两个文件夹内图片的总数，确保编号连续
    existing_train = len([f for f in os.listdir(train_dir) if f.endswith('.jpg')])
    existing_val = len([f for f in os.listdir(val_dir) if f.endswith('.jpg')])
    sample_count = existing_train + existing_val
    
    roi_half_size = 24 

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_SETTINGS, 1) # 如果需要调驱动设置请取消注释

    # ---------- 控制面板 ----------
    cv2.namedWindow("Control Panel", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Control Panel", 400, 300)
    cv2.createTrackbar("Param1", "Control Panel", 49, 300, nothing)
    cv2.createTrackbar("Param2", "Control Panel", 25, 100, nothing)
    cv2.createTrackbar("MinDist", "Control Panel", 32, 100, nothing)
    cv2.createTrackbar("MinRad", "Control Panel", 15, 100, nothing)
    cv2.createTrackbar("MaxRad", "Control Panel", 25, 150, nothing)

    cv2.namedWindow("HoughCircles Collect")

    print(f"训练集路径: {train_dir}")
    print(f"验证集路径: {val_dir}")
    print(f"当前总样本数: {sample_count}")

    while True:
        ret, frame = cap.read()
        if not ret: break

        p1 = max(1, cv2.getTrackbarPos("Param1", "Control Panel"))
        p2 = max(1, cv2.getTrackbarPos("Param2", "Control Panel"))
        md = max(1, cv2.getTrackbarPos("MinDist", "Control Panel"))
        minR = cv2.getTrackbarPos("MinRad", "Control Panel")
        maxR = max(minR + 1, cv2.getTrackbarPos("MaxRad", "Control Panel"))

        current_circles = []

        warped = cv2.warpPerspective(frame, matrix, (WARP_W, WARP_H))
        tmp_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        tmp_fil = cv2.medianBlur(tmp_gray, 5)

        circles = cv2.HoughCircles(
            tmp_fil, cv2.HOUGH_GRADIENT, dp=1,
            minDist=md, param1=p1, param2=p2,
            minRadius=minR, maxRadius=maxR
        )

        warped_disp = warped.copy()
        if circles is not None:
            circles = np.round(circles[0, :]).astype(int)
            for (cx, cy, r) in circles:
                current_circles.append((cx, cy))
                cv2.circle(warped_disp, (cx, cy), r, (0, 255, 0), 2)

        cv2.imshow("Warped View", warped_disp)

        cv2.imshow("HoughCircles Collect", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('w'):
            if current_circles:
                t_saved, v_saved = 0, 0
                for (cx, cy) in current_circles:
                    x1, y1 = cx - roi_half_size, cy - roi_half_size
                    x2, y2 = cx + roi_half_size, cy + roi_half_size
                    
                    if x1 >= 0 and y1 >= 0 and x2 <= WARP_W and y2 <= WARP_H:
                        sample_roi = warped[y1:y2, x1:x2]
                        sample_count += 1
                        
                        # 【核心逻辑】每8张放1张到验证集 (val)
                        if sample_count % 8 == 0:
                            save_path = os.path.join(val_dir, f"B_Kin_v_{sample_count}.jpg")
                            v_saved += 1
                        else:
                            save_path = os.path.join(train_dir, f"B_Kin_t_{sample_count}.jpg")
                            t_saved += 1
                        
                        cv2.imwrite(save_path, sample_roi)
                
                print(f"本次保存完毕: 训练集+{t_saved}, 验证集+{v_saved}。当前总数: {sample_count}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()