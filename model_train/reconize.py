# -*- coding: utf-8 -*-
import cv2
import numpy as np
from ultralytics import YOLO

# ---------- 全局变量 ----------
pts_src = []
roi_ready = False
matrix = None
WARP_W, WARP_H = 707, 630

# 加载你训练好的模型路径 (请确保路径正确)
MODEL_PATH = "model_train/runs/classify/chess_r_test2/weights/best.pt"

try:
    model = YOLO(MODEL_PATH)
    print("模型加载成功！")
except:
    print("模型加载失败，请检查路径。")

def mouse_callback(event, x, y, flags, param):
    global pts_src, roi_ready, matrix
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(pts_src) < 4:
            pts_src.append([x, y])
            if len(pts_src) == 4:
                pts_dst = np.float32([[0, 0], [WARP_W, 0], [0, WARP_H], [WARP_W, WARP_H]])
                src = np.float32(pts_src)
                matrix = cv2.getPerspectiveTransform(src, pts_dst)
                roi_ready = True
                print("透视变换已就绪，开始实时识别...")

def main():
    global pts_src, roi_ready, matrix
    
    roi_half_size = 24  # 必须与训练时的切片尺寸逻辑保持一致
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_SETTINGS, 1) # 如果需要调驱动设置请取消注释
    cv2.namedWindow("Chess Recognition")
    cv2.setMouseCallback("Chess Recognition", mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret: break

        disp_img = frame.copy()
        # 绘制定位点
        for pt in pts_src:
            cv2.circle(disp_img, tuple(pt), 5, (0, 0, 255), -1)

        if roi_ready and matrix is not None:
            # 1. 透视变换得到矫正后的棋盘
            warped = cv2.warpPerspective(frame, matrix, (WARP_W, WARP_H))
            
            # 2. 预处理用于圆检测
            tmp_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            tmp_fil = cv2.medianBlur(tmp_gray, 5)

            # 3. 霍夫圆检测 (使用你之前调好的稳定参数)
            circles = cv2.HoughCircles(
                tmp_fil, cv2.HOUGH_GRADIENT, dp=1,
                minDist=32, param1=49, param2=35,
                minRadius=15, maxRadius=25
            )

            if circles is not None:
                circles = np.round(circles[0, :]).astype(int)
                
                # 遍历每一个检测到的圆进行 YOLO 识别
                for (cx, cy, r) in circles:
                    x1, y1 = cx - roi_half_size, cy - roi_half_size
                    x2, y2 = cx + roi_half_size, cy + roi_half_size
                    
                    # 边界检查防止越界
                    if x1 >= 0 and y1 >= 0 and x2 <= WARP_W and y2 <= WARP_H:
                        crop = warped[y1:y2, x1:x2]
                        
                        # --- YOLO 预测 ---
                        # verbose=False 关掉控制台的一堆输出
                        results = model.predict(crop, verbose=False) 
                        
                        if results:
                            res = results[0]
                            cls_id = res.probs.top1      # 类别索引
                            conf = res.probs.top1conf    # 置信度
                            label = res.names[cls_id]    # 类别名称 (如 R_Kin)

                            # 只有置信度大于 0.6 才显示，过滤干扰
                            if conf > 0.6:
                                # 在 warped 图上画圈和标签
                                color = (0, 255, 0) # 默认绿色
                                cv2.circle(warped, (cx, cy), r, color, 2)
                                cv2.putText(warped, f"{label} {conf:.2f}", (cx-r, cy-r-10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            cv2.imshow("Real-time Recognition", warped)

        cv2.imshow("Chess Recognition", disp_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('r'):
            pts_src = []
            roi_ready = False

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()