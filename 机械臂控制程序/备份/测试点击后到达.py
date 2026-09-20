import cv2
import numpy as np
import os
import serial
import time
import math

# =========================
# 机械参数
# =========================
CFG={"L1":247,"L2":218,"R1":4.423828,"R2":3.670288,"D1":1,"D2":1,"PORT":"COM9","BAUD":115200,"A2OFF":90.0}
PULSE_PER_REV=12800
SPEED=200
ACC=20

# =========================
# 透视参数
# =========================
W,H=707,630
SRC=np.float32([[311,102],
    [932,94],
    [279,680],
    [985,666]])
DST=np.float32([[0,0],[W,0],[0,H],[W,H]])
M=cv2.getPerspectiveTransform(SRC,DST)

tps=None
ser=None
last_target=None

# =========================
# TPS加载
# =========================
def load_tps():
    if not os.path.exists("map_model.npz"):
        print("缺少TPS模型"); return None
    d=np.load("map_model.npz")
    p=d['pixel_pts']; w=d['world_pts']
    t=cv2.createThinPlateSplineShapeTransformer()
    ss=p.reshape(1,-1,2); dd=w.reshape(1,-1,2)
    matches=[cv2.DMatch(i,i,0) for i in range(len(p))]
    t.estimateTransformation(ss,dd,matches)
    print("TPS OK")
    return t

# =========================
# 像素→世界
# =========================
def px2world(u,v):
    pt=np.array([[[u,v]]],dtype=np.float32)
    _,out=tps.applyTransformation(pt)
    return out[0][0]

# =========================
# 逆运动学
# =========================
def IK(x,y):
    L1,L2=CFG["L1"],CFG["L2"]
    r2=x*x+y*y
    if r2>(L1+L2)**2: return None  # 超范围
    c2=(r2-L1*L1-L2*L2)/(2*L1*L2)
    c2=max(-1,min(1,c2))
    t2=math.acos(c2)
    k1=L1+L2*math.cos(t2)
    k2=L2*math.sin(t2)
    t1=math.atan2(y,x)-math.atan2(k2,k1)
    a1=math.degrees(t1)
    a2=math.degrees(t1+t2)
    return a1,a2

# =========================
# 机械角→电机角
# =========================
def arm2motor(a1,a2):
    m1=a1*CFG["R1"]
    a2m=(-a2+180)+a1-CFG["A2OFF"]
    m2=a2m*CFG["R2"]
    m1*=CFG["D1"]
    m2*=CFG["D2"]
    return m1,m2

# =========================
# 绝对控制发送
# =========================
def send_abs(mid,ang):
    direction=0x00 if ang>=0 else 0x01
    pulse=int(abs(ang)*PULSE_PER_REV/360)
    cmd=bytes([
        mid,0xFD,direction,
        (SPEED>>8)&0xFF,SPEED&0xFF,
        ACC,
        (pulse>>24)&0xFF,
        (pulse>>16)&0xFF,
        (pulse>>8)&0xFF,
        pulse&0xFF,
        0x01,0x00,0x6B
    ])
    ser.write(cmd)

def move(m1,m2):
    send_abs(1,m1)
    time.sleep(0.02)
    send_abs(2,m2)
    time.sleep(0.02)

# =========================
# 鼠标点击
# =========================
def mouse(event,x,y,flags,param):
    global last_target
    if event!=cv2.EVENT_LBUTTONDOWN: return
    world=px2world(x,y)
    if world is None: return
    wx,wy=world
    ik=IK(wx,wy)
    if ik is None:
        print("超出工作空间"); return
    a1,a2=ik
    m1,m2=arm2motor(a1,a2)
    move(m1,m2)
    last_target=(x,y)
    print(f"PX({x},{y}) XY({wx:.1f},{wy:.1f}) A({a1:.1f},{a2:.1f})")

# =========================
# 主程序
# =========================
def main():
    global ser,tps
    ser=serial.Serial(CFG["PORT"],CFG["BAUD"],timeout=0.05)
    tps=load_tps()

    cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

    cv2.namedWindow("Warped")
    cv2.setMouseCallback("Warped",mouse)

    while True:
        ret,frame=cap.read()
        if not ret: break

        warped=cv2.warpPerspective(frame,M,(W,H))

        if last_target:
            cv2.circle(warped,last_target,5,(0,0,255),-1)
            cv2.putText(warped,str(last_target),
                        (last_target[0]+5,last_target[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)

        cv2.putText(warped,"Click To Move",
                    (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,(0,255,0),2)

        cv2.imshow("Warped",warped)

        if cv2.waitKey(1)&0xFF==ord('q'):
            break

    cap.release()
    ser.close()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()