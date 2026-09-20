import cv2
import numpy as np

# ==========================================
# 透视变换参数
# ==========================================
WARP_W, WARP_H = 707, 630

# ==========================================
# 全局变量
# ==========================================
pts_src = []        # 原图四点
M = None            # 透视矩阵

# ==========================================
# 原图点击（选四个点）
# ==========================================
def click_src(event, x, y, flags, param):

    global pts_src, M

    if event == cv2.EVENT_LBUTTONDOWN and len(pts_src) < 4:

        pts_src.append([x, y])

        print(f"原图点 {len(pts_src)}: ({x}, {y})")

        # 满4点就计算矩阵
        if len(pts_src) == 4:

            pts_dst = np.float32([
                [0, 0],
                [WARP_W, 0],
                [0, WARP_H],
                [WARP_W, WARP_H]
            ])

            M = cv2.getPerspectiveTransform(
                np.float32(pts_src),
                pts_dst
            )

            print("\n>>> 透视矩阵计算完成")


# ==========================================
# 透视图点击（获取坐标）
# ==========================================
def click_warp(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:

        print(f"映射坐标: ({x}, {y})")


# ==========================================
# 主程序
# ==========================================
def main():

    global M

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    # 创建窗口
    cv2.namedWindow("Source")
    cv2.setMouseCallback("Source", click_src)

    cv2.namedWindow("Warped")
    cv2.setMouseCallback("Warped", click_warp)

    print("\n>>> 请点击四个点：左上 → 右上 → 左下 → 右下")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        display = frame.copy()

        # 画已选点
        for i, pt in enumerate(pts_src):

            cv2.circle(display, tuple(pt), 6, (0, 0, 255), -1)

            cv2.putText(
                display,
                str(i+1),
                (pt[0]+10, pt[1]+10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        cv2.imshow("Source", display)

        # 如果矩阵存在 → 生成透视图
        if M is not None:

            warped = cv2.warpPerspective(
                frame,
                M,
                (WARP_W, WARP_H)
            )

            cv2.imshow("Warped", warped)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()