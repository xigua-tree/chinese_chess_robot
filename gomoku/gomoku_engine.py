import subprocess
import socket
import threading
import os
import time

ENGINE_PATH = r"C:/Kele/2026Inter/2026---/五子棋引擎/pbrain-rapfi_avxvnni.exe"


class GomokuRobot:

    def __init__(self, path, timeout_ms):

        self.proc = subprocess.Popen(
            path,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        self.board = [[0]*15 for _ in range(15)]

        self.send("START 15")
        self.send(f"INFO timeout_turn {timeout_ms}")
        self.send("INFO timeout_match 1000000")


    def send(self, cmd):

        try:
            self.proc.stdin.write(cmd + "\n")
            self.proc.stdin.flush()
        except:
            pass


    def print_board(self):

        print("\n--- 当前棋盘 (X:玩家, O:引擎) ---")

        header = "   " + " ".join([f"{i:x}" for i in range(15)])
        print(header)

        for r in range(15):

            line = f"{r:x}  "

            for c in range(15):

                cell = self.board[c][r]

                if cell == 1:
                    line += "X "
                elif cell == 2:
                    line += "O "
                else:
                    line += ". "

            print(line)

        print("--------------------------------\n")


    def check_winner(self, x, y):

        color = self.board[x][y]

        if color == 0:
            return False

        directions = [
            (1,0),
            (0,1),
            (1,1),
            (1,-1)
        ]

        for dx,dy in directions:

            count = 1

            nx,ny = x+dx,y+dy

            while 0<=nx<15 and 0<=ny<15 and self.board[nx][ny]==color:
                count += 1
                nx += dx
                ny += dy

            nx,ny = x-dx,y-dy

            while 0<=nx<15 and 0<=ny<15 and self.board[nx][ny]==color:
                count += 1
                nx -= dx
                ny -= dy

            if count >= 5:
                return True

        return False


    def handle_engine_reply(self):

        while True:

            line = self.proc.stdout.readline()

            if not line:
                continue

            line = line.strip()

            if "," in line and not line.startswith("DEBUG"):

                try:

                    x,y = map(int,line.split(","))

                    self.board[x][y] = 2

                    self.print_board()

                    return x,y

                except ValueError:
                    continue


    def player_move(self, x, y):

        self.board[x][y] = 1

        self.print_board()

        if self.check_winner(x,y):
            return "HUMAN_WIN",None,None

        self.send(f"TURN {x},{y}")

        rx,ry = self.handle_engine_reply()

        if self.check_winner(rx,ry):
            return "ENGINE_WIN",rx,ry

        return "CONTINUE",rx,ry


    def terminate(self):

        try:

            print(">>> 正在关闭引擎")

            self.send("END")

            time.sleep(0.1)

            self.proc.kill()

            self.proc.wait(timeout=1)

            print(">>> 引擎已关闭")

        except Exception as e:

            print("终止引擎异常:", e)



# ================= 全局状态 =================

bot = None
game_running = False
engine_first = False


def send_to_server(msg):

    try:
        client.send((msg+"\n").encode())
    except:
        pass


def handle_command(msg):

    global bot
    global game_running
    global engine_first

    # 调试打印（关键）
    print("RAW:",repr(msg))
    print("收到:", msg)

    msg = msg.strip()

    # ================= 退出程序 =================

    if msg == "退出":

        print(">>> 收到退出程序")

        try:
            if bot:
                bot.terminate()
        except:
            pass

        try:
            client.close()
        except:
            pass

        os._exit(0)


    # ================= 引擎执黑 =================

    if msg == "YQZH":

        engine_first = True

        print(">>> 设置引擎先手")

        return


    # ================= 开始对弈 =================

    if msg.startswith("开始对弈:"):

        if game_running:
            print(">>> 对局已存在")
            return

        try:

            timeout = int(msg.split(":")[1])

            print(f"启动新对局 timeout={timeout}")

            bot = GomokuRobot(
                ENGINE_PATH,
                timeout
            )

            game_running = True

            # 引擎先手

            if engine_first:

                time.sleep(0.3)

                bot.send("BEGIN")

                rx,ry = bot.handle_engine_reply()

                send_to_server(
                    f"引擎落子:{rx},{ry}"
                )

                engine_first = False

                print(">>> 引擎已先手")

            else:

                print(">>> 等待玩家落子")

        except Exception as e:

            print("初始化引擎失败:", e)


    # ================= 玩家落子 =================

    if msg.startswith("落子:"):

        if not game_running or not bot:
            print(">>> 当前没有对局")
            return

        try:

            pos = msg.split(":")[1]

            x,y = map(int,pos.split(","))

            status,rx,ry = bot.player_move(x,y)

            if status == "HUMAN_WIN":

                send_to_server("人类赢")

                bot.terminate()

                bot = None

                game_running = False

                print(">>> 人类胜利")

            elif status == "ENGINE_WIN":
                
                # 【修复核心】先将引擎绝杀的这一步发送给服务端，再发送人类输
                send_to_server(
                    f"引擎落子:{rx},{ry}"
                )

                send_to_server("人类输")

                bot.terminate()

                bot = None

                game_running = False

                print(">>> 引擎胜利")

            else:

                send_to_server(
                    f"引擎落子:{rx},{ry}"
                )

                print(">>> 等待玩家落子")

        except Exception as e:

            print("处理落子异常:", e)


    # ================= 退出对局（关键修复） =================

    if msg == "退出对局":

        print(">>> 收到退出对局")

        try:

            if bot:

                bot.terminate()

                bot = None

        except Exception as e:

            print("退出异常:",e)

        game_running = False

        print(">>> 对局已结束")

        return



# ================= TCP接收线程 =================

def recv_thread():

    while True:

        try:

            data = client.recv(1024)

            if not data:
                break

            msgs = data.decode().split("\n")

            for msg in msgs:

                msg = msg.strip()

                if msg:

                    handle_command(msg)

        except Exception as e:

            print("接收异常:", e)

            break



# ================= TCP连接 =================

client = socket.socket()

try:

    print("五子棋引擎")

    client.connect(("127.0.0.1",8888))

    client.send(
        "REGISTER:五子棋引擎\n".encode()
    )

    print("连接服务端成功")

    threading.Thread(
        target=recv_thread,
        daemon=True
    ).start()

    while True:

        time.sleep(1)

except Exception as e:

    print("连接失败:", e)