import socket,threading,queue,time,serial,math,numpy as np,cv2,os

# ========================= 机械参数 =========================
CFG={"L1":247,"L2":217.94,"R1":4.3656,"R2":3.6478,"D1":1,"D2":1,"A2OFF":90.0}
PULSE_PER_REV=12800

# ========================= 运动参数 =========================  

SPEED = 2000     # 速度 (0~65535)
ACC   = 0      # 加速度 (0~255)
ACC_Z = 249       # Z轴加速度 (0~255)
# Z轴 pulse绝对位置（不是角度）
Z_UP=430000
Z_DOWN=676000

# ========================= 串口初始化 =========================
ser=serial.Serial("COM9",115200,timeout=0.05)

# ========================= 任务队列 =========================
task_queue=queue.Queue()

# ========================= TPS加载 =========================
def load_tps():  # 加载像素→世界TPS模型
    if not os.path.exists("map_model.npz"):
        print("缺少TPS模型");return None
    d=np.load("map_model.npz")
    p=d['pixel_pts'];w=d['world_pts']
    t=cv2.createThinPlateSplineShapeTransformer()
    ss=p.reshape(1,-1,2);dd=w.reshape(1,-1,2)
    matches=[cv2.DMatch(i,i,0) for i in range(len(p))]
    t.estimateTransformation(ss,dd,matches)
    print("TPS OK");return t

tps=load_tps()

# ========================= TCP解析 =========================
def parse_command(msg:str):  # 解析TCP命令
    msg=msg.strip()
    try:
        if msg.startswith("棋子移动:"):
            data=msg.split(":")[1]
            p1,p2=data.split("),(")
            p1=p1.replace("(","");p2=p2.replace(")","")
            x1,y1=p1.split(",");x2,y2=p2.split(",")
            return [0,int(x1),int(y1),int(x2),int(y2)]
        else:
            print("未知指令:",msg);return None
    except Exception as e:
        print("解析失败:",e);return None

# ========================= TCP接收 =========================
buffer=""
def recv_thread(client):  # TCP接收线程
    global buffer
    while True:
        try:
            data=client.recv(1024)
            if not data:print("服务器断开");break
            buffer+=data.decode()
            while "\n" in buffer:
                line,buffer=buffer.split("\n",1)
                line=line.strip()
                if not line:continue
                print("收到:",line)
                task=parse_command(line)
                if task:
                    task_queue.put(task)
                    print("加入队列:",task)
        except Exception as e:
            print("接收异常:",e);break

# ========================= TCP发送 =========================
def send_msg(client,msg):  # 发送TCP反馈
    try:client.send((msg+"\n").encode())
    except Exception as e:print("发送失败:",e)

# ========================= TCP连接 =========================
client=socket.socket()
client.connect(("127.0.0.1",8888))
client.send("REGISTER:控制系统\n".encode())
print("连接成功")
recv_t=threading.Thread(target=recv_thread,args=(client,))
recv_t.daemon=True
recv_t.start()

# ========================= 磁铁控制 =========================
def magnet_off():  # 打开电磁铁
    ser.setRTS(True);ser.setDTR(False)

def magnet_on():  # 关闭电磁铁
    ser.setRTS(False);ser.setDTR(False)

# ========================= XY电机角度控制 =========================
def move_motor_angle(mid,angle):  # XY轴角度→pulse控制
    direction=0x00 if angle>=0 else 0x01
    pulse=int(abs(angle)*PULSE_PER_REV/360)
    cmd=bytes([mid,0xFD,direction,0x01,0x2C,10,
               (pulse>>24)&0xFF,(pulse>>16)&0xFF,
               (pulse>>8)&0xFF,pulse&0xFF,
               0x01,0x00,0x6B])
    ser.write(cmd);time.sleep(0.03)

# ========================= Z电机pulse控制 =========================
def move_motor_pulse(mid,pulse):  # Z轴pulse控制（带速度）
    direction=0x01
    pulse=abs(int(pulse))
    v1=(SPEED>>8)&0xFF
    v2=SPEED&0xFF
    acc = ACC
    if(mid==3):
        acc = ACC_Z
    cmd=bytes([
        mid,
        0xFD,
        direction,
        v1,v2,        # ⭐速度
        acc,          # ⭐加速度
        (pulse>>24)&0xFF,
        (pulse>>16)&0xFF,
        (pulse>>8)&0xFF,
        pulse&0xFF,
        0x01,
        0x00,
        0x6B
    ])
    ser.write(cmd)
    time.sleep(0.03)

# ========================= 等待电机完成 =========================
def wait_all_motor_done(ser,motor_ids,timeout=2):  # 等待全部电机到位
    arrived={mid:False for mid in motor_ids}
    start=time.time()
    while True:
        if time.time()-start>timeout:
            print("等待超时");return False
        rx=ser.read(4)
        if len(rx)!=4:continue
        motor_id,cmd,status=rx[0],rx[1],rx[2]
        if cmd==0xFD and status==0x9F:
            if motor_id in arrived:
                arrived[motor_id]=True
                print(f"Motor {motor_id} 到位")
        if all(arrived.values()):
            print("全部电机到位");return True

# ========================= 三轴同步运动 =========================
def move_3motor_and_wait(m1,m2,z):  # XY角度+Z pulse
    ser.reset_input_buffer()
    move_motor_angle(1,m1)
    move_motor_angle(2,m2)
    move_motor_pulse(3,z)
    return wait_all_motor_done(ser,[1,2,3])

# ========================= 逆运动学 =========================
def inverse_kinematics(x,y):  # XY→关节角
    L1,L2=CFG["L1"],CFG["L2"]
    r2=x*x+y*y
    if r2>(L1+L2)**2:
        print("超范围");return None
    c2=(r2-L1*L1-L2*L2)/(2*L1*L2)
    c2=max(-1,min(1,c2))
    t2=math.acos(c2)
    k1=L1+L2*math.cos(t2)
    k2=L2*math.sin(t2)
    t1=math.atan2(y,x)-math.atan2(k2,k1)
    a1=math.degrees(t1)
    a2=math.degrees(t1+t2)
    return a1,a2

# ========================= 机械角转电机角 =========================
def arm_to_motor(a1,a2):  # 齿轮比转换
    m1=a1*CFG["R1"]
    a2m=(-a2+180)+a1-CFG["A2OFF"]
    m2=a2m*CFG["R2"]
    m1*=CFG["D1"];m2*=CFG["D2"]
    return m1,m2

# ========================= 像素转世界 =========================
def pixel_to_world(u,v):  # TPS转换
    pt=np.array([[[u,v]]],dtype=np.float32)
    _,out=tps.applyTransformation(pt)
    return out[0][0]

# ========================= 到指定像素 =========================
def move_pixel(u,v,z):  # pixel→运动
    wx,wy=pixel_to_world(u,v)
    
    # ---------------- 教师添加：动态Z轴下垂补偿算法 ----------------
    # 1. 计算当前目标点到原点(0,0)的距离
    distance = math.sqrt(wx**2 + wy**2)
    
    # 2. 根据你的测试数据计算补偿比例参数 (K)
    # 测试点: (7.7, 376.5), Z轴脉冲偏差: 676000 - 575000 = 101000
    ref_distance = math.sqrt(7.7**2 + 376.5**2) 
    pulse_diff = 70000
    compensation_ratio = pulse_diff / ref_distance
    
    # 3. 计算当前距离需要的Z轴补偿量（向外伸出越多，扣除的脉冲越多，机械臂越往上抬）
    z_offset = distance * compensation_ratio
    
    # 4. 得到真实需要发送的 Z 脉冲值
    actual_z = int(z - z_offset)
    # ---------------------------------------------------------------
    
    ik=inverse_kinematics(wx,wy)
    if ik is None:return False
    a1,a2=ik
    m1,m2=arm_to_motor(a1,a2)
    return move_3motor_and_wait(m1,m2,actual_z)

# ========================= 回原点 =========================
def move_home():  # XY回0 Z抬起
    print("回到原点")
    return move_3motor_and_wait(0,0,Z_UP)

# ========================= 棋子移动流程 =========================
def execute_pixel_move(x1,y1,x2,y2):  # 抓取→移动→释放
    print(f"棋子移动 ({x1},{y1})->({x2},{y2})")
    if not move_pixel(x1,y1,Z_UP):return False
    if not move_pixel(x1,y1,Z_DOWN):return False
    magnet_on();time.sleep(0.2)
    if not move_pixel(x1,y1,Z_UP):return False
    if not move_pixel(x2,y2,Z_UP):return False
    if not move_pixel(x2,y2,Z_DOWN):return False
    magnet_off();time.sleep(0.2)
    move_pixel(x2,y2,Z_UP)

    return True

move_home()
# ========================= 主循环 =========================
while True:
    task=task_queue.get()
    task_type=task[0]

    if task_type==0:
        x1,y1,x2,y2=task[1:]
        execute_pixel_move(x1,y1,x2,y2)

    task_queue.task_done()

    if task_queue.empty():
        print("所有任务执行完成")
        move_home()        
        time.sleep(1)
        send_msg(client,"运动完成")