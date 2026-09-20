import cv2
import numpy as np

# ===== 载入标定矩阵 =====
H = np.load("calibration_matrix.npy")
print("H loaded:\n", H)

# ===== 坐标转换 =====
def pixel_to_xy(u, v):
    pt = np.array([[[u, v]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, H)
    return float(out[0][0][0]), float(out[0][0][1])

# ===== 鼠标点击 =====
def mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:

        xw, yw = pixel_to_xy(x, y)

        print("\n================")
        print(f"像素: ({x},{y})")
        print(f"机械: ({xw:.2f},{yw:.2f})")
        print("================")

# ===== 摄像头 =====
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cv2.namedWindow("test")
cv2.setMouseCallback("test", mouse)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    cv2.imshow("test", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()