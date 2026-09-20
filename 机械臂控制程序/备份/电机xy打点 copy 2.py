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
    "RATIO1": 4.3656, "RATIO2": 3.6478,
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
# 鼠标回调函数 (新增)
# ==========================================
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"[鼠标点击] 像素坐标: ({x}, {y})")

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
        param1=130,
        param2=25 ,
        minRadius=6,
        maxRadius=29
    )
    if circles is None:
        return None

    circles = np.round(circles[0]).astype(int)
    best = max(circles, key=lambda c: c[2])
    x, y, r = best
    return (x, y, r)

# ==========================================
# 运动学计算逻辑
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
    
    return x, y, a1, a2

def read_motor_raw_angle(ser, mid):
    try:
        time.sleep(0.05)
        ser.reset_input_buffer()
        ser.write(bytes([mid, 0x36, 0x6B]))
        time.sleep(0.05)
        rx = ser.read(8)
        if len(rx) != 8: return None
        sign, pos = rx[2], int.from_bytes(rx[3:7], 'big')
        angle = pos * 360.0 / 65536.0
        
        return -angle if sign == 0x01 else angle
    except: return None

# ==========================================
# 历史坐标数据预加载
# ==========================================
pixel_pts_init = np.array([
    [356, 306], [220, 454], [222, 160], [488, 162], [486, 452],
    [288, 378], [288, 234], [418, 234], [422, 380], [356, 380],
    [290, 306], [354, 232], [422, 308], [354, 454], [220, 306],
    [488, 306], [354, 160], [290, 452], [424, 452], [488, 378],
    [488, 234], [420, 162], [286, 158], [218, 232], [220, 380],
    [354, 594], [520, 532], [184, 534], [180, 94],  [520, 90],
    [576, 156], [576, 32],  [128, 32],  [132, 594], [574, 592],
    [296, 532], [410, 534], [126, 406], [576, 408], [126, 216],
    [578, 218], [576, 468], [128, 472], [240, 596], [466, 594],
    [240, 30],  [464, 30],  [350, 32],  [296, 94],  [408, 92],
       [32, 586],  [76, 584],  [76, 336],
    [32, 334],  [632, 30],  [672, 32],  [626, 324], [672, 320],
    [632, 586], [672, 584], [632, 76],  [672, 72],
    [632, 130], [676, 124], [666, 268], 
    [72, 78],   [30, 72],   [28, 124],  [74, 124],  [50, 172],
    [36, 230],  [444, 98],  [72, 274],  [30, 392],  [74, 416],
    [32, 454],  [70, 476],  [32, 516],  [660, 380], [630, 424],
    [668, 464], [630, 506], [668, 530],
], dtype=np.float32)

world_pts_init = np.array([
    [18.57, 225.27],  [-65.33, 149.55], [-65.58, 313.54], [94.99, 305.74],  [97.82, 131.63],
    [-21.78, 187.22], [-23.33, 270.59], [56.20, 265.35],  [59.65, 178.95],  [18.66, 183.01],
    [-22.44, 229.60], [18.26, 268.19],  [56.46, 221.80],  [17.57, 141.78],  [-64.60, 230.49],
    [97.40, 220.24],  [14.80, 309.71],  [-21.50, 144.97], [59.52, 135.55],  [96.61, 176.40],
    [95.95, 262.63],  [55.15, 306.96],  [-25.36, 311.43], [-63.92, 273.15], [-64.05, 190.43],
    [16.45, 61.20],   [114.24, 79.05],  [-92.48, 106.74], [-90.70, 350.76], [110.63, 344.67],
    [143.12, 305.51], [141.39, 380.59], [-126.29, 385.64],[-128.92, 73.95], [139.47, 39.60],
    [-20.25, 101.79], [51.15, 87.53],   [-122.73, 177.46],[146.98, 154.37], [-123.07, 282.68],
    [146.44, 267.82], [146.36, 116.09], [-124.25, 142.54],[-59.64, 72.41],  [81.34, 43.87],
    [-59.90, 385.49], [75.09, 380.92],  [8.00, 383.65],   [-22.98, 348.43], [43.62, 346.23],
    [-187.51, 78.17], [-161.61, 78.62], [-152.21, 216.29],
    [-180.56, 217.96],[173.33, 378.88], [195.98, 381.22], [175.88, 202.68], [200.84, 205.51],
    [172.74, 41.71],  [197.05, 43.11],   [176.00, 351.85], [196.75, 353.56],
    [176.35, 320.48], [201.08, 322.46],[197.95, 236.74], 
    [-159.07, 357.64],[-184.94, 362.41],[-183.27, 333.93],[-155.65, 332.22],[-169.46, 306.51],
    [-177.17, 274.26],[65.70, 342.05],  [-155.20, 249.72],[-182.74, 187.11],[-155.34, 172.10],
    [-182.07, 151.17],[-159.62, 138.92],[-184.55, 117.19],[195.09, 168.59], [176.38, 143.44],
    [199.36, 117.63], [173.97, 91.40],  [196.92, 76.61],
], dtype=np.float32)

recorded_data = [
    {'uv': [int(u), int(v)], 'xy': [x, y]} 
    for (u, v), (x, y) in zip(pixel_pts_init, world_pts_init)
]
locked_pixel_uv = None

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
    # 绑定鼠标回调
    cv2.setMouseCallback("Warped", mouse_callback)

    print("\n>>> 棋子分步采集系统")
    print(f">>> 提示：已成功加载 {len(recorded_data)} 组历史标定数据。")
    print(">>> 步骤 1：看准圆心，按【空格】锁定像素")
    print(">>> 步骤 2：对准位置，再按【空格】记录坐标并绑定")
    print(">>> 提示：点击画面可打印鼠标像素坐标")

    while True:
        ret, frame = cap.read()
        if not ret: break
        warped = cv2.warpPerspective(frame, GLOBAL_M, (WARP_W, WARP_H))
        
        m1 = read_motor_raw_angle(ser, 1)
        m2 = read_motor_raw_angle(ser, 2)
        
        circle_info = detect_circle_center(warped)
        
        if circle_info is not None:
            cx, cy, cr = circle_info
            cv2.circle(warped, (cx, cy), cr, (255, 0, 0), 2)
            cv2.drawMarker(warped, (cx, cy), (255, 0, 0), cv2.MARKER_CROSS, 15, 2)
            
        if locked_pixel_uv is not None:
            cv2.drawMarker(warped, locked_pixel_uv, (0, 0, 255), cv2.MARKER_CROSS, 25, 2)
            cv2.putText(warped, f"PIXEL LOCKED: {locked_pixel_uv}", (20, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if m1 is not None and m2 is not None:
            cur_x, cur_y, cur_a1, cur_a2 = get_current_xy(m1, m2)
            cv2.putText(warped, f"REAL XY: {cur_x:.1f}, {cur_y:.1f}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(warped, f"ANGLES : A1={cur_a1:.1f}, A2={cur_a2:.1f}", (20, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        for d in recorded_data:
            cv2.circle(warped, tuple(d['uv']), 5, (0, 255, 0), -1)

        cv2.imshow("Warped", warped)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            if locked_pixel_uv is None:
                if circle_info is not None:
                    locked_pixel_uv = (circle_info[0], circle_info[1])
                    print(f"[状态] 第1次空格：成功锁定圆心像素 {locked_pixel_uv}。请移动机械臂，准备按第2次空格记录坐标。")
                else:
                    print("[警告] 未检测到圆，无法锁定像素，请让棋子保持在视野内。")
            else:
                if m1 is not None and m2 is not None:
                    cur_x, cur_y, _, _ = get_current_xy(m1, m2) 
                    recorded_data.append({'uv': locked_pixel_uv, 'xy': [cur_x, cur_y]})
                    print(f"[成功] 第2次空格：记录坐标({cur_x:.2f}, {cur_y:.2f}) -> 成功绑定第 {len(recorded_data)} 组数据！")
                    locked_pixel_uv = None  
                else:
                    print("[错误] 电机数据读取失败，无法获取坐标，请保持连接重试。")

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