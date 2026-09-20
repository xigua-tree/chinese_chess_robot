import socket
import threading

# ===============================
# 接收线程（一直监听Qt消息）
# ===============================
def recv_thread(client):
    while True:
        try:
            data = client.recv(1024)
            if not data:
                print("服务器断开连接")
                break
            msg = data.decode().strip()
            print("收到:", msg)
        except Exception as e:
            print("接收异常:", e)
            break

#这连接服务器并注册名字为象棋引擎
client = socket.socket()                        # 创建socket
client.connect(("127.0.0.1", 8888))             # 连接服务器
client.send("REGISTER:五子棋视觉\n".encode())      # 注册名字为象棋引擎！
print("连接服务端成功")

# 启动接收线程
recv_t = threading.Thread(
    target=recv_thread,
    args=(client,)
)
recv_t.daemon = True   # 主线程退出时自动关闭
recv_t.start()

# 循环发送消息
while True:

    msg = input("发送:")
    client.send((msg + "\n").encode()) # 记得加 \n（按行解析）

    if msg == "exit":
        break
client.close()