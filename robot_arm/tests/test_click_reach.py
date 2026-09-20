import cv2
import numpy as np
import os
import serial
import time
import math

# =========================
# 机械参数
# =========================
CFG={"L1":247,"L2":217.94,"R1":4.3656,"R2":3.6478,"D1":1,"D2":1,"A2OFF":90.0,"PORT":"COM9","BAUD":115200,}
PULSE_PER_REV=12800
SPEED=200
ACC=20

# =========================
# 透视参数
# =========================
W,H=707,630

SRC=np.float32([
   [311,102],
    [932,94],
    [279,680],
    [985,666]
])

DST=np.float32([
    [0,0],
    [W,0],
    [0,H],
    [W,H]
])

M=cv2.getPerspectiveTransform(SRC,DST)

# =========================
# 全局变量
# =========================
tps=None
ser=None
last_target=None


# =========================================================
# TPS加载
# =========================================================
def load_tps():

    if not os.path.exists("map_model.npz"):
        print("缺少TPS模型")
        return None

    data=np.load("map_model.npz")

    pixel_pts=data['pixel_pts']
    world_pts=data['world_pts']

    t=cv2.createThinPlateSplineShapeTransformer()

    src=pixel_pts.reshape(1,-1,2)
    dst=world_pts.reshape(1,-1,2)

    matches=[cv2.DMatch(i,i,0)
             for i in range(len(pixel_pts))]

    t.estimateTransformation(src,dst,matches)

    print("TPS OK")

    return t


# =========================================================
# 像素 → 世界坐标
# =========================================================
def px2world(u,v):

    global tps

    pt=np.array([[[u,v]]],dtype=np.float32)

    _,out=tps.applyTransformation(pt)

    wx,wy=out[0][0]

    return wx,wy


# =========================================================
# 逆运动学
# =========================================================
def IK(x,y):

    L1=CFG["L1"]
    L2=CFG["L2"]

    r2=x*x+y*y

    # 超出工作空间
    if r2>(L1+L2)**2:
        return None

    c2=(r2-L1*L1-L2*L2)/(2*L1*L2)

    c2=max(-1,min(1,c2))

    t2=math.acos(c2)

    k1=L1+L2*math.cos(t2)
    k2=L2*math.sin(t2)

    t1=math.atan2(y,x)-math.atan2(k2,k1)

    a1=math.degrees(t1)
    a2=math.degrees(t1+t2)

    return a1,a2


# =========================================================
# 机械角 → 电机角
# =========================================================
def arm2motor(a1,a2):

    m1=a1*CFG["R1"]

    a2m=(-a2+180)+a1-CFG["A2OFF"]

    m2=a2m*CFG["R2"]

    m1*=CFG["D1"]
    m2*=CFG["D2"]

    return m1,m2


# =========================================================
# 发送绝对位置
# =========================================================
def send_abs(mid,ang):

    direction=0x00 if ang>=0 else 0x01

    pulse=int(abs(ang)*PULSE_PER_REV/360)

    cmd=bytes([
        mid,
        0xFD,
        direction,

        (SPEED>>8)&0xFF,
        SPEED&0xFF,

        ACC,

        (pulse>>24)&0xFF,
        (pulse>>16)&0xFF,
        (pulse>>8)&0xFF,
        pulse&0xFF,

        0x01,
        0x00,
        0x6B
    ])

    ser.write(cmd)


# =========================================================
# 双电机移动
# =========================================================
def move_motor(m1,m2):

    send_abs(1,m1)

    time.sleep(0.02)

    send_abs(2,m2)

    time.sleep(0.02)


# =========================================================
# ⭐⭐⭐核心函数⭐⭐⭐
# 输入像素 → 电机到位
# =========================================================
def pixel_to_motor_move(u,v):

    global last_target

    # 像素 → 世界
    wx,wy=px2world(u,v)

    # 世界 → 关节角
    ik=IK(wx,wy)

    if ik is None:
        print("超出工作空间")
        return False

    a1,a2=ik

    # 关节 → 电机角
    m1,m2=arm2motor(a1,a2)

    # 执行运动
    move_motor(m1,m2)

    last_target=(u,v)

    print(
        f"PX({u},{v}) "
        f"XY({wx:.1f},{wy:.1f}) "
        f"A({a1:.1f},{a2:.1f})"
    )

    return True


# =========================================================
# 鼠标回调
# =========================================================
def mouse_callback(event,x,y,flags,param):

    if event!=cv2.EVENT_LBUTTONDOWN:
        return

    pixel_to_motor_move(x,y)


# =========================================================
# 系统初始化
# =========================================================
def init_system():

    global ser
    global tps

    # 串口初始化
    ser=serial.Serial(
        CFG["PORT"],
        CFG["BAUD"],
        timeout=0.05
    )

    print("串口OK")

    # TPS加载
    tps=load_tps()

    if tps is None:
        raise RuntimeError("TPS加载失败")


# =========================================================
# 摄像头显示循环
# =========================================================
def camera_loop():

    cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

    cv2.namedWindow("Warped")

    cv2.setMouseCallback(
        "Warped",
        mouse_callback
    )

    while True:

        ret,frame=cap.read()

        if not ret:
            break

        warped=cv2.warpPerspective(
            frame,
            M,
            (W,H)
        )

        # 显示最后目标点
        if last_target:

            cv2.circle(
                warped,
                last_target,
                5,
                (0,0,255),
                -1
            )

            cv2.putText(
                warped,
                str(last_target),
                (
                    last_target[0]+5,
                    last_target[1]-5
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0,0,255),
                1
            )

        cv2.putText(
            warped,
            "Click To Move",
            (10,30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2
        )

        cv2.imshow(
            "Warped",
            warped
        )

        key=cv2.waitKey(1)&0xFF

        if key==ord('q'):
            break

    cap.release()

    ser.close()

    cv2.destroyAllWindows()


# =========================================================
# 主函数（非常干净）
# =========================================================
def main():

    init_system()

    camera_loop()


# =========================================================
# 入口
# =========================================================
if __name__=="__main__":

    main()