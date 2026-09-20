import cv2
import numpy as np
import serial
import time
import math

# ==========================================
# 硬件与运动学参数 (你的原始参数)
# ==========================================
CONFIG = {
    "L1": 247, "L2": 218,
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
    [306, 91], [952, 89], [262, 669], [974, 694]
])
PTS_DST = np.float32([[0, 0], [WARP_W, 0], [0, WARP_H], [WARP_W, WARP_H]])
GLOBAL_M = cv2.getPerspectiveTransform(FIXED_PTS_SRC, PTS_DST)

# ==========================================
# 运动学计算逻辑 (把角度变成 XY)
# ==========================================
def get_current_xy(m1_raw, m2_raw):
    """核心：将电机原始角度转换为物理 XY 坐标"""
    # 1. 电机角 -> 机械角
    m1 = m1_raw * CONFIG["DIR1"]
    m2 = m2_raw * CONFIG["DIR2"]
    a1 = m1 / CONFIG["RATIO1"]
    a2_motor = m2 / CONFIG["RATIO2"]
    a2 = a2_motor - a1 + CONFIG["INIT_A2_OFFSET"]
    a2 = -1 * (a2 - 180)
    
    # 2. 正运动学: 机械角 -> XY
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
current_pixel_uv = None
recorded_data = [] # 存储格式: {'uv': [u,v], 'xy': [x,y]}

def click_warp(event, x, y, flags, param):
    global current_pixel_uv
    if event == cv2.EVENT_LBUTTONDOWN:
        current_pixel_uv = (x, y)

def main():
    global current_pixel_uv, recorded_data
    try:
        ser = serial.Serial(CONFIG["SERIAL_PORT"], CONFIG["BAUD_RATE"], timeout=0.05)
    except:
        print("串口错误"); return

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cv2.namedWindow("Warped")
    cv2.setMouseCallback("Warped", click_warp)

    print("\n>>> 像素-XY 坐标采集系统 (按空格记录)")

    while True:
        ret, frame = cap.read()
        if not ret: break
        warped = cv2.warpPerspective(frame, GLOBAL_M, (WARP_W, WARP_H))
        
        m1 = read_motor_raw_angle(ser, 1)
        m2 = read_motor_raw_angle(ser, 2)
        
        # 实时计算并显示当前的 XY
        if m1 is not None and m2 is not None:
            cur_x, cur_y = get_current_xy(m1, m2)
            cv2.putText(warped, f"REAL XY: {cur_x:.1f}, {cur_y:.1f}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if current_pixel_uv:
                cv2.drawMarker(warped, current_pixel_uv, (0,0,255), cv2.MARKER_CROSS, 20, 2)

        for d in recorded_data:
            cv2.circle(warped, tuple(d['uv']), 5, (255, 0, 0), -1)

        cv2.imshow("Warped", warped)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            if m1 is not None and m2 is not None and current_pixel_uv is not None:
                cur_x, cur_y = get_current_xy(m1, m2)
                recorded_data.append({'uv': current_pixel_uv, 'xy': [cur_x, cur_y]})
                print(f"已记录: 像素{current_pixel_uv} -> 物理XY({cur_x:.2f}, {cur_y:.2f})")
                current_pixel_uv = None

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