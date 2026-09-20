import cv2
import numpy as np
import serial
import time
import math

# ==========================================
# 硬件与运动学参数
# ==========================================
CONFIG = {
    "L1": 247, "L2": 217.94,
    "RATIO1": 4.423828, "RATIO2": 3.670288,
    "DIR1": 1, "DIR2": 1,
    "SERIAL_PORT": "COM9",
    "BAUD_RATE": 115200,
    "INIT_A2_OFFSET": 90.0
}

# ==========================================
# 预设透视变换参数
# ==========================================
WARP_W, WARP_H = 707, 630

FIXED_PTS_SRC = np.float32([
    [311,102],
    [932,94],
    [279,680],
    [985,666]
])
PTS_DST = np.float32([[0, 0], [WARP_W, 0], [0, WARP_H], [WARP_W, WARP_H]])
GLOBAL_M = cv2.getPerspectiveTransform(FIXED_PTS_SRC, PTS_DST)

# ==========================================
# 霍夫圆检测函数
# ==========================================
def detect_circle_center(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=17,
        param1=148,
        param2=28,
        minRadius=7,
        maxRadius=29
    )
    if circles is None:
        return None

    circles = np.round(circles[0]).astype(int)
    best = max(circles, key=lambda c: c[2])
    x, y, r = best
    return (x, y, r)

# ==========================================
# 运动学计算逻辑 (把角度变成 XY)
# ==========================================
def get_current_xy(m1_raw, m2_raw):
    m1 = m1_raw * CONFIG["DIR1"]
    m2 = m2_raw * CONFIG["DIR2"]
    a1 = m1 / CONFIG["RATIO1"]
    a2_motor = m2 / CONFIG["RATIO2"]
    a2 = a2_motor - a1 + CONFIG["INIT_A2_OFFSET"]
    a2 = -1 * (a2 - 180)
    
    rad1, rad2 = math.radians(a1), math.radians(a2)
    x = CONFIG["L1"] * math.cos(rad1) + CONFIG["L2"] * math.cos(rad2)
    y = CONFIG["L1"] * math.sin(rad1) + CONFIG["L2"] * math.sin(rad2)
    return x, y

def read_motor_raw_angle(ser, mid):
    try:
        ser.reset_input_buffer()
        ser.write(bytes([mid, 0x36, 0x6B]))
        time.sleep(0.02)
        rx = ser.read(8)
        if len(rx) != 8: return None
        sign, pos = rx[2], int.from_bytes(rx[3:7], 'big')
        angle = pos * 360.0 / 65536.0
        return -angle if sign == 0x01 else angle
    except: return None

# ==========================================
# 全局变量
# ==========================================
recorded_data = []      # 存储格式: {'uv': [u,v], 'xy': [x,y]}
locked_pixel_uv = None  # 用于暂存第一次空格锁定的圆心像素

def main():
    global recorded_data, locked_pixel_uv
    try:
        ser = serial.Serial(CONFIG["SERIAL_PORT"], CONFIG["BAUD_RATE"], timeout=0.05)
    except:
        print("串口错误"); return

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cv2.namedWindow("Warped")

    print("\n>>> 棋子分步采集系统")
    print(">>> 步骤 1：看准圆心，按【空格】锁定像素")
    print(">>> 步骤 2：对准位置，再按【空格】记录坐标并绑定")

    while True:
        ret, frame = cap.read()
        if not ret: break
        warped = cv2.warpPerspective(frame, GLOBAL_M, (WARP_W, WARP_H))
        
        m1 = read_motor_raw_angle(ser, 1)
        m2 = read_motor_raw_angle(ser, 2)
        
        # 实时检测
        circle_info = detect_circle_center(warped)
        
        # 1. 绘制实时检测到的圆（蓝色）
        if circle_info is not None:
            cx, cy, cr = circle_info
            cv2.circle(warped, (cx, cy), cr, (255, 0, 0), 2)
            cv2.drawMarker(warped, (cx, cy), (255, 0, 0), cv2.MARKER_CROSS, 15, 2)
            
        # 2. 如果已经锁定了像素，把锁定的那个位置画成红十字，提醒你已经存下来了
        if locked_pixel_uv is not None:
            cv2.drawMarker(warped, locked_pixel_uv, (0, 0, 255), cv2.MARKER_CROSS, 25, 2)
            cv2.putText(warped, f"PIXEL LOCKED: {locked_pixel_uv}", (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 3. 实时显示机械臂当前的 XY
        if m1 is not None and m2 is not None:
            cur_x, cur_y = get_current_xy(m1, m2)
            cv2.putText(warped, f"REAL XY: {cur_x:.1f}, {cur_y:.1f}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 4. 绘制历史已完全记录的数据点（实心绿点）
        for d in recorded_data:
            cv2.circle(warped, tuple(d['uv']), 5, (0, 255, 0), -1)

        cv2.imshow("Warped", warped)
        key = cv2.waitKey(1) & 0xFF

        # 按空格键的分步处理逻辑
        if key == ord(' '):
            # 情况 A：还没有记录像素，这第一下空格是用来找圆并锁定像素的
            if locked_pixel_uv is None:
                if circle_info is not None:
                    locked_pixel_uv = (circle_info[0], circle_info[1])
                    print(f"[状态] 第1次空格：成功锁定圆心像素 {locked_pixel_uv}。请移动机械臂，准备按第2次空格记录坐标。")
                else:
                    print("[警告] 未检测到圆，无法锁定像素，请让棋子保持在视野内。")
            
            # 情况 B：像素已经锁定了，这第二下空格是用来读 XY 并合体保存的
            else:
                if m1 is not None and m2 is not None:
                    cur_x, cur_y = get_current_xy(m1, m2)
                    recorded_data.append({'uv': locked_pixel_uv, 'xy': [cur_x, cur_y]})
                    print(f"[成功] 第2次空格：记录坐标({cur_x:.2f}, {cur_y:.2f}) -> 成功绑定第 {len(recorded_data)} 组数据！")
                    # 关键：清除状态，复位回去等待下一次循环的第一个空格
                    locked_pixel_uv = None  
                else:
                    print("[错误] 电机数据读取失败，无法获取坐标，请保持连接重试。")

        # 按 q 或 s 退出并打印
        if key == ord('s') or key == ord('q'):
            if recorded_data:
                print("\n" + "="*40 + "\n采集到的 像素-XY 数据:\n")
                print("pixel_pts = np.array([")
                for d in recorded_data: print(f"    [{d['uv'][0]}, {d['uv'][1]}],")
                print("], dtype=np.float32)\n")
                
                print("world_pts = np.array([")
                for d in recorded_data: print(f"    [{d['xy'][0]:.2f}, {d['xy'][1]:.2f}],")
                print("], dtype=np.float32)\n" + "="*40)
            break

    cap.release()
    ser.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()