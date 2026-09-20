import cv2
import numpy as np

pts_current = []

def nothing(x):
    pass

# =========================
# 鼠标点击显示坐标
# =========================
def mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"[点击] ({x},{y})")

# =========================
# 摄像头初始化
# =========================
def init_camera(idx=0):
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
    cap.set(cv2.CAP_PROP_EXPOSURE, -6)
    return cap if cap.isOpened() else None

# =========================
# 主调参函数
# =========================
def debug_hough_all_circles(camera_index=0):

    cap = init_camera(camera_index)
    if cap is None:
        print("摄像头打开失败")
        return

    cv2.namedWindow("Full Frame Debug")
    cv2.namedWindow("Edge View")
    cv2.setMouseCallback("Full Frame Debug", mouse_cb)

    # ===== 滑块 =====
    cv2.createTrackbar("Param1", "Full Frame Debug", 50, 200, nothing)
    cv2.createTrackbar("Param2", "Full Frame Debug", 30, 100, nothing)
    cv2.createTrackbar("MinR", "Full Frame Debug", 20, 100, nothing)
    cv2.createTrackbar("MaxR", "Full Frame Debug", 40, 150, nothing)
    cv2.createTrackbar("MinDist", "Full Frame Debug", 40, 150, nothing)

    print(">>> 调参模式：显示所有圆")

    while True:

        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 1.5)

        # ===== 参数 =====
        p1 = max(1, cv2.getTrackbarPos("Param1", "Full Frame Debug"))
        p2 = max(1, cv2.getTrackbarPos("Param2", "Full Frame Debug"))
        min_r = cv2.getTrackbarPos("MinR", "Full Frame Debug")
        max_r = max(min_r + 1, cv2.getTrackbarPos("MaxR", "Full Frame Debug"))
        min_d = max(1, cv2.getTrackbarPos("MinDist", "Full Frame Debug"))

        # ===== 边缘图 =====
        edges = cv2.Canny(blurred, p1 // 2, p1)
        cv2.imshow("Edge View", edges)

        canvas = frame.copy()

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=min_d,
            param1=p1,
            param2=p2,
            minRadius=min_r,
            maxRadius=max_r
        )

        # =========================
        # 关键修改：全部画出来
        # =========================
        if circles is not None:

            circles = np.uint16(np.around(circles))

            for c in circles[0]:

                x, y, r = c

                # 圆边
                cv2.circle(canvas, (x, y), r, (0, 255, 0), 2)

                # 圆心
                cv2.circle(canvas, (x, y), 2, (0, 0, 255), 3)

                # 坐标文字
                cv2.putText(canvas,
                            f"{x},{y}",
                            (x + 5, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 0),
                            1)

        # ===== 信息 =====
        info = f"P1:{p1} P2:{p2} | circles:{0 if circles is None else len(circles[0])}"
        cv2.putText(canvas, info, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255), 2)

        cv2.imshow("Full Frame Debug", canvas)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# =========================
if __name__ == "__main__":
    debug_hough_all_circles(0)