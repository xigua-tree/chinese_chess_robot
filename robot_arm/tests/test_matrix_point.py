import cv2
import numpy as np
import os

# ==========================================
# 1. 预设参数 (需与标定脚本一致)
# ==========================================
WARP_W, WARP_H = 707, 630

# 使用你之前提供的固定四角坐标
FIXED_PTS_SRC = np.float32([
    [306, 91], [952, 89], [262, 669], [974, 694]
])
PTS_DST = np.float32([[0, 0], [WARP_W, 0], [0, WARP_H], [WARP_W, WARP_H]])

# 自动生成透视矩阵 M (用于切图)
GLOBAL_M = cv2.getPerspectiveTransform(FIXED_PTS_SRC, PTS_DST)

tps_model = None  # 用于像素->物理坐标的 TPS 模型

# ==========================================
# 2. 加载 TPS 映射逻辑
# ==========================================
def load_tps_model(filename="map_model.npz"):
    if not os.path.exists(filename):
        print(f"\n[错误] 未找到模型文件 {filename}！请确保已生成该文件。")
        return None
    
    data = np.load(filename)
    p_pts = data['pixel_pts']
    w_pts = data['world_pts']
    
    tps = cv2.createThinPlateSplineShapeTransformer()
    ss = p_pts.reshape(1, -1, 2)
    td = w_pts.reshape(1, -1, 2)
    matches = [cv2.DMatch(i, i, 0) for i in range(len(p_pts))]
    tps.estimateTransformation(ss, td, matches)
    print(f">>> 成功加载 TPS 模型 (点数: {len(p_pts)})")
    return tps

def pixel_to_world(u, v):
    global tps_model
    if tps_model is None:
        tps_model = load_tps_model()
    if tps_model is None: return None

    pt = np.array([[[u, v]]], dtype=np.float32)
    _, transformed_pt = tps_model.applyTransformation(pt)
    return transformed_pt[0][0]

# ==========================================
# 3. 交互逻辑
# ==========================================
def click_warp(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        world = pixel_to_world(x, y)
        if world is not None:
            print(f"\n点击像素: ({x}, {y})")
            print(f"--- 预测机械坐标: X={world[0]:.2f}, Y={world[1]:.2f} ---")

def main():
    global tps_model
    tps_model = load_tps_model()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow("Warped")
    cv2.setMouseCallback("Warped", click_warp)

    print("\n[系统就绪]")
    print("直接在 'Warped' 窗口点击任意位置进行坐标测试。按 'q' 退出。")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 直接应用固定好的变换
        warped = cv2.warpPerspective(frame, GLOBAL_M, (WARP_W, WARP_H))
        
        # 加上一点辅助网格或文字提示，显得专业一点
        cv2.putText(warped, "Testing Mode: Click to get World XY", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Warped", warped)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()