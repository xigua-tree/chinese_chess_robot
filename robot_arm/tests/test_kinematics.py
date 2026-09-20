import serial, struct, time, math

# =========================
# 硬件参数
# =========================
CONFIG = {
    "L1":247, "L2":218,
    "RATIO1":4.423828, "RATIO2":3.670288,
    "DIR1":1, "DIR2":1,
    "SERIAL_PORT":"COM9",
    "BAUD_RATE":115200,
    "INIT_A2_OFFSET":90.0
}

# =========================
# 电机角 → 机械角
# =========================
def motor_to_arm_angles(m1_val,m2_val):

    m1=m1_val*CONFIG["DIR1"]
    m2=m2_val*CONFIG["DIR2"]

    a1=m1/CONFIG["RATIO1"]
    a2_motor=m2/CONFIG["RATIO2"]

    a2=a2_motor-a1+CONFIG["INIT_A2_OFFSET"]
    a2=-1*(a2-180)

    return a1,a2


# =========================
# 机械角 → 电机角 (新增)
# =========================
def arm_to_motor_angles(a1,a2):

    m1=a1*CONFIG["RATIO1"]

    a2_motor=(-a2+180)+a1-CONFIG["INIT_A2_OFFSET"]

    m2=a2_motor*CONFIG["RATIO2"]

    m1*=CONFIG["DIR1"]
    m2*=CONFIG["DIR2"]

    return m1,m2


# =========================
# 正运动学
# =========================
def forward_kinematics(a1_deg,a2_deg):

    a1=math.radians(a1_deg)
    a2=math.radians(a2_deg)

    x=CONFIG["L1"]*math.cos(a1)+CONFIG["L2"]*math.cos(a2)
    y=CONFIG["L1"]*math.sin(a1)+CONFIG["L2"]*math.sin(a2)

    return x,y


# =========================
# 逆运动学（新增核心）
# =========================
def inverse_kinematics(x,y):

    L1=CONFIG["L1"]
    L2=CONFIG["L2"]

    r2=x*x+y*y

    cos_t2=(r2-L1*L1-L2*L2)/(2*L1*L2)

    if cos_t2>1: cos_t2=1
    if cos_t2<-1: cos_t2=-1

    t2=math.acos(cos_t2)  # 肘下解

    k1=L1+L2*math.cos(t2)
    k2=L2*math.sin(t2)

    t1=math.atan2(y,x)-math.atan2(k2,k1)

    a1=math.degrees(t1)
    a2=math.degrees(t1+t2)  # 转绝对角

    return a1,a2


# =========================
# 读取电机角度
# =========================
def read_motor_raw_angle(ser,mid):

    try:
        ser.reset_input_buffer()
        ser.write(bytes([mid,0x36,0x6B]))
        time.sleep(0.05)

        rx=ser.read(8)
        if len(rx)!=8: return None

        sign=rx[2]
        pos=int.from_bytes(rx[3:7],'big',signed=False)

        angle=pos*360.0/65536.0
        if sign==0x01: angle=-angle

        return angle

    except Exception as e:
        print("err:",e)
        return None


def set_motor_enable(ser,mid,enable):

    state=0x01 if enable else 0x00
    cmd=bytes([mid,0xF3,0xAB,state,0x00,0x6B])

    ser.write(cmd)
    time.sleep(0.05)


# =========================
# 主循环
# =========================
def run_angle_monitor():

    try:
        ser=serial.Serial(CONFIG["SERIAL_PORT"],CONFIG["BAUD_RATE"],timeout=0.05)
    except:
        return

    set_motor_enable(ser,1,False)
    set_motor_enable(ser,2,False)

    print("\n--- 调试模式 ---")

    try:
        while True:

            m1=read_motor_raw_angle(ser,1)
            m2=read_motor_raw_angle(ser,2)

            if m1 is not None and m2 is not None:

                # 正解
                a1,a2=motor_to_arm_angles(m1,m2)

                x,y=forward_kinematics(a1,a2)

                # 逆解
                ia1,ia2=inverse_kinematics(x,y)

                # 再转回电机角
                rm1,rm2=arm_to_motor_angles(ia1,ia2)

                # 误差
                e1=rm1-m1
                e2=rm2-m2

                print(
                    f"\r[RAW] M1:{m1:8.4f} M2:{m2:8.4f} | "
                    f"[XY] x:{x:7.1f} y:{y:7.1f} | "
                    f"[IK] a1:{ia1:7.2f} a2:{ia2:7.2f} | "
                    f"[ERR] e1:{e1:7.3f} e2:{e2:7.3f}",
                    end=""
                )

            time.sleep(0.05)

    except KeyboardInterrupt:
        pass

    finally:
        ser.close()


if __name__=="__main__":
    run_angle_monitor()