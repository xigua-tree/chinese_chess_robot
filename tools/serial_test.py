import cv2
import math

# =========================
# 机械参数
# =========================
L1 = 247
L2 = 218

R1 = 4.40625
R2 = 5.6534722224

# =========================
# 运动学（修正后的）
# =========================
def fk_debug(m1, m2):

    # 1臂
    theta1 = m1 / R1

    # 2臂（差动）
    theta2 = (m2 / R2) - theta1

    # 2臂方向修正（逆时针为负）
    theta2 = -theta2

    # 末端
    t1 = math.radians(theta1)
    t2 = math.radians(theta1 + theta2)

    x = L1 * math.cos(t1) + L2 * math.cos(t2)
    y = L1 * math.sin(t1) + L2 * math.sin(t2)

    return theta1, theta2, x, y


# =========================
# 模拟输入（⚠️这里你后面要换成真实电机数据）
# =========================
m1, m2 = 465.18, 240.04


# =========================
# 摄像头
# =========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("摄像头打开失败")
    exit()


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # =========================
    # 计算当前角度
    # =========================
    theta1, theta2, x, y = fk_debug(m1, m2)

    # =========================
    # 画面叠加信息
    # =========================
    cv2.putText(frame, f"M1: {m1:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(frame, f"M2: {m2:.2f}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(frame, f"Theta1: {theta1:.2f} deg", (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)

    cv2.putText(frame, f"Theta2: {theta2:.2f} deg", (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)

    cv2.putText(frame, f"End: ({x:.1f}, {y:.1f})", (20, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    # =========================
    # 显示画面
    # =========================
    cv2.imshow("SCARA Debug View", frame)

    key = cv2.waitKey(1) & 0xFF

    # ESC退出
    if key == 27:
        break

    # =========================
    # 测试：用键盘模拟角度变化
    # =========================
    if key == ord('w'):
        m1 += 5
    if key == ord('s'):
        m1 -= 5
    if key == ord('i'):
        m2 += 5
    if key == ord('k'):
        m2 -= 5


cap.release()
cv2.destroyAllWindows()