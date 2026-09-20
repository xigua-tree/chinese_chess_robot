import cv2
import numpy as np

pts = []

def nothing(x):
    pass

# =========================
# 鼠标点击标定 + 输出
# =========================
def mouse_cb(event, x, y, flags, param):

    global pts

    if event == cv2.EVENT_LBUTTONDOWN:

        print(f"[点击] ({x},{y})")

        if len(pts) < 4:

            pts.append([x, y])

            print(f"标定点{len(pts)}: {x},{y}")

# =========================
# 摄像头
# =========================
def init_camera(idx=0):

    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
    cap.set(cv2.CAP_PROP_EXPOSURE, -6)

    return cap if cap.isOpened() else None

# =========================
# 主程序
# =========================
def run(camera_index=0):

    global pts

    cap = init_camera(camera_index)
    if cap is None:
        print("摄像头失败")
        return

    # ===== 窗口 =====
    cv2.namedWindow("View")
    cv2.namedWindow("Edge Debug")   # ⭐你要的黑白调试窗口
    cv2.setMouseCallback("View", mouse_cb)

    # ===== 滑块 =====
    cv2.createTrackbar("P1", "View", 50, 200, nothing)
    cv2.createTrackbar("P2", "View", 30, 100, nothing)
    cv2.createTrackbar("MinR", "View", 20, 100, nothing)
    cv2.createTrackbar("MaxR", "View", 40, 150, nothing)
    cv2.createTrackbar("MinD", "View", 40, 150, nothing)

    M = None
    calibrated = False

    REAL_W = 43.4
    REAL_H = 38.0

    OUT_H = 580
    OUT_W = int(OUT_H * REAL_W / REAL_H)

    print(f">>> 输出尺寸 {OUT_W} x {OUT_H}")

    print(">>> 点击4个点标定")

    while True:

        ret, frame = cap.read()
        if not ret:
            continue

        show = frame.copy()

        # =========================
        # 显示标定点
        # =========================
        for i, p in enumerate(pts):

            cv2.circle(show, tuple(p), 6, (0,0,255), -1)

            cv2.putText(show, str(i+1),
                        (p[0]+5, p[1]+5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0,0,255), 2)

        # =========================
        # 标定完成
        # =========================
        if len(pts) == 4 and not calibrated:

            src = np.float32(pts)

            dst = np.float32([
                [0, 0],
                [OUT_W, 0],
                [0, OUT_H],
                [OUT_W, OUT_H]
            ])

            M = cv2.getPerspectiveTransform(src, dst)

            calibrated = True

            print(">>> 标定完成")

        # =========================
        # warp
        # =========================
        if calibrated:
            frame = cv2.warpPerspective(frame, M, (OUT_W, OUT_H))

        # =========================
        # 灰度 + 模糊
        # =========================
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9,9), 1.5)

        # =========================
        # ⭐ Canny调试窗口（你要的）
        # =========================
        p1 = cv2.getTrackbarPos("P1", "View")

        edges = cv2.Canny(blur, p1//2, p1)
        cv2.imshow("Edge Debug", edges)

        # =========================
        # 霍夫参数
        # =========================
        p2 = cv2.getTrackbarPos("P2", "View")
        min_r = cv2.getTrackbarPos("MinR", "View")
        max_r = max(min_r+1, cv2.getTrackbarPos("MaxR", "View"))
        min_d = cv2.getTrackbarPos("MinD", "View")

        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=min_d,
            param1=p1,
            param2=p2,
            minRadius=min_r,
            maxRadius=max_r
        )

        # =========================
        # 画所有圆
        # =========================
        if circles is not None:

            circles = np.uint16(np.around(circles))

            for c in circles[0]:

                x, y, r = c

                cv2.circle(frame, (x,y), r, (0,255,0), 2)
                cv2.circle(frame, (x,y), 2, (0,0,255), 3)

                cv2.putText(frame,
                            f"{x},{y}",
                            (x+5,y-5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255,255,0),1)

        # =========================
        # UI信息
        # =========================
        cv2.putText(frame,
                    f"pts:{len(pts)} cal:{calibrated}",
                    (20,30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,(0,255,255),2)

        cv2.imshow("View", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# =========================
if __name__ == "__main__":
    run(0)