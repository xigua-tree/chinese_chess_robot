import cv2, numpy as np, socket, threading, os

# ================= 一级透视（固定） =================

W, H = 707, 630
SRC = np.float32([[310, 102], [931, 95], [279, 679], [983, 664]])
DST = np.float32([[0, 0], [W, 0], [0, H], [W, H]])
M_base = cv2.getPerspectiveTransform(SRC, DST)

# ================= 参数 =================

GRID_SIZE = 15

HOUGH_CONFIG = {
    "p2": 20,
    "minR": 9,
    "maxR": 23,
    "minD": 20
}

# 棋盘标定点
pts_board = [[590, 570], [121, 568], [591, 57], [122, 53]]
board_ready = True
board_points = None
poly_board = None  # 用于界定棋盘内外区域的凸包

current_status = "IDLE"

client = None
engine_move = None

# ===== 新增：颜色与状态追踪 =====
global_board_matrix = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
engine_color = None  # "WHITE" 或 "BLACK"
COLOR_THRESHOLD = 120

# ================= 摆棋区域 =================

LEFT_AREA = [
    (0, 2),
    (104, 4),
    (4, 627),
    (101, 626)
]

RIGHT_AREA = [
    (603, 3),
    (705, 1),
    (605, 625),
    (702, 628)
]

arrange_mode = None

# ================= 鼠标标定 =================

def click_board(event, x, y, flags, param):
    global pts_board, board_ready, poly_board
    global client

    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"[CLICK] ({x},{y})")

        # ===== 棋盘标定 =====
        if not board_ready:
            pts_board.append([x, y])
            print(f"棋盘点 {len(pts_board)}: ({x},{y})")

            if len(pts_board) == 4:
                board_ready = True
                generate_board_points()
                poly_board = cv2.convexHull(np.array(pts_board, dtype=np.int32))
                print(">>> 棋盘标定完成")

                try:
                    client.send("准备完毕\n".encode())
                    print(">>> 已发送:准备完毕")
                except:
                    print("TCP发送失败")

# ================= 生成棋盘225点 =================

def generate_board_points():
    global board_points, poly_board

    p0 = np.array(pts_board[0])
    p1 = np.array(pts_board[1])
    p2 = np.array(pts_board[2])
    p3 = np.array(pts_board[3])

    board_points = []
    for r in range(GRID_SIZE):
        t = r / (GRID_SIZE - 1)
        left = p0 * (1 - t) + p2 * t
        right = p1 * (1 - t) + p3 * t
        row_pts = []
        for c in range(GRID_SIZE):
            s = c / (GRID_SIZE - 1)
            pt = left * (1 - s) + right * s
            row_pts.append(pt)
        board_points.append(row_pts)

    poly_board = cv2.convexHull(np.array(pts_board, dtype=np.int32))

# ================= 画虚线 =================

def draw_dashed(img, p1, p2):
    dash = 10
    dist = int(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
    for i in range(0, dist, dash * 2):
        r1 = i / dist
        r2 = (i + dash) / dist
        if r2 > 1: r2 = 1
        x1 = int(p1[0] + (p2[0] - p1[0]) * r1)
        y1 = int(p1[1] + (p2[1] - p1[1]) * r1)
        x2 = int(p1[0] + (p2[0] - p1[0]) * r2)
        y2 = int(p1[1] + (p2[1] - p1[1]) * r2)
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

# ================= 画棋盘 =================

def draw_board(img):
    if len(pts_board) != 4: return img
    p0 = np.array(pts_board[0])
    p1 = np.array(pts_board[1])
    p2 = np.array(pts_board[2])
    p3 = np.array(pts_board[3])
    for i in range(GRID_SIZE):
        t = i / (GRID_SIZE - 1)
        left = (p0 * (1 - t) + p2 * t).astype(int)
        right = (p1 * (1 - t) + p3 * t).astype(int)
        top = (p0 * (1 - t) + p1 * t).astype(int)
        bottom = (p2 * (1 - t) + p3 * t).astype(int)
        draw_dashed(img, left, right)
        draw_dashed(img, top, bottom)
    return img

# ================= 圆检测与灰度提取 =================

def detect_circles(frame):
    warped = cv2.warpPerspective(frame, M_base, (W, H))
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 7)

    centers = []
    circles = cv2.HoughCircles(
        gray_blur, cv2.HOUGH_GRADIENT, dp=1, minDist=HOUGH_CONFIG["minD"],
        param1=50, param2=HOUGH_CONFIG["p2"],
        minRadius=HOUGH_CONFIG["minR"], maxRadius=HOUGH_CONFIG["maxR"]
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for c in circles[0, :]:
            centers.append((c[0], c[1]))

    return centers, warped, gray

# ================= 圆→棋盘矩阵 =================

def circles_to_matrix(centers):
    matrix = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    if board_points is None: return matrix

    for cx, cy in centers:
        min_d = 9999
        best = None
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                px, py = board_points[r][c]
                d = np.hypot(cx - px, cy - py)
                if d < min_d:
                    min_d = d
                    best = (r, c)
        if best and min_d < 20:
            r, c = best
            matrix[r][c] = 1
    return matrix

# ================= 点是否在矩形区域 =================

def point_in_rect(x, y, rect):
    xs = [p[0] for p in rect]
    ys = [p[1] for p in rect]

    return (
        min(xs) <= x <= max(xs) and
        min(ys) <= y <= max(ys)
    )

# ================= 生成散棋摆放位置 =================

def generate_spare_positions(area_pts, count):

    x1 = min(p[0] for p in area_pts)
    x2 = max(p[0] for p in area_pts)

    y1 = min(p[1] for p in area_pts)
    y2 = max(p[1] for p in area_pts)

    positions = []

    # ================= 参数 =================

    PIECE_RADIUS = 14

    # 距离边缘至少4像素
    EDGE_MARGIN = 10

    # 棋子之间间隔5像素
    PIECE_GAP = 16

    # 实际步进
    step = PIECE_RADIUS * 2 + PIECE_GAP

    # 起始位置
    start_x = x1 + PIECE_RADIUS + EDGE_MARGIN
    start_y = y1 + PIECE_RADIUS + EDGE_MARGIN

    # 最大边界
    max_x = x2 - PIECE_RADIUS - EDGE_MARGIN
    max_y = y2 - PIECE_RADIUS - EDGE_MARGIN

    # 一行能放几个
    # cols = max(1, int((max_x - start_x) // step) + 1)
    cols = 2
    for i in range(count):

        row = i // cols
        col = i % cols

        px = int(start_x + col * step)
        py = int(start_y + row * step)

        # 超边界停止
        if py > max_y:
            break

        positions.append((px, py))

    return positions

# ================= 收集区域已有散棋 =================

def collect_spare_occupied(centers):
    occupied = []

    for cx, cy in centers:

        if point_in_rect(cx, cy, LEFT_AREA):
            occupied.append((cx, cy))

        elif point_in_rect(cx, cy, RIGHT_AREA):
            occupied.append((cx, cy))

    return occupied

# ================= 摆棋 =================

def arrange_pieces(centers, gray, white_left=True):

    global client

    board_white = []
    board_black = []

    for cx, cy in centers:

        if poly_board is None:
            continue

        dist = cv2.pointPolygonTest(poly_board, (cx, cy), True)

        if dist < -15:
            continue

        y1, y2 = max(0, cy-3), min(H, cy+4)
        x1, x2 = max(0, cx-3), min(W, cx+4)

        intensity = np.mean(gray[y1:y2, x1:x2])

        piece_color = "WHITE" if intensity > COLOR_THRESHOLD else "BLACK"

        if piece_color == "WHITE":
            board_white.append((cx, cy))
        else:
            board_black.append((cx, cy))

    if white_left:
        white_area = LEFT_AREA
        black_area = RIGHT_AREA
    else:
        white_area = RIGHT_AREA
        black_area = LEFT_AREA

    occupied = collect_spare_occupied(centers)

    white_targets_all = generate_spare_positions(
        white_area,
        len(board_white) + len(occupied) + 20
    )

    black_targets_all = generate_spare_positions(
        black_area,
        len(board_black) + len(occupied) + 20
    )

    def filter_targets(targets):

        valid = []

        for tx, ty in targets:

            ok = True

            for ox, oy in occupied:

                if np.hypot(tx - ox, ty - oy) < 24:
                    ok = False
                    break

            if ok:
                valid.append((tx, ty))

        return valid

    white_targets = filter_targets(white_targets_all)
    black_targets = filter_targets(black_targets_all)

    send_list = []

    for i, (sx, sy) in enumerate(board_white):

        if i >= len(white_targets):
            break

        tx, ty = white_targets[i]

        send_list.append(
            f"棋子移动:({sx},{sy}),({tx},{ty})\n"
        )

    for i, (sx, sy) in enumerate(board_black):

        if i >= len(black_targets):
            break

        tx, ty = black_targets[i]

        send_list.append(
            f"棋子移动:({sx},{sy}),({tx},{ty})\n"
        )

    for msg in send_list:
        client.send(msg.encode())
        print(msg.strip())

# ================= TCP线程 =================

def recv_thread(client):
    global current_status, engine_move, engine_color
    global global_board_matrix
    global arrange_mode

    while True:
        try:
            data = client.recv(1024)
            if not data:
                print("服务器断开")
                os._exit(0)

            msgs = data.decode().strip().split('\n')
            for msg in msgs:
                msg = msg.strip()
                if not msg: continue

                print("\n[SERVER] ->", msg)

                if "退出" in msg:
                    os._exit(0)

                elif "重置" in msg:
                    current_status = "IDLE"
                    global_board_matrix.fill(0)
                    print(">>> 全局棋局已重置")

                elif "引擎执白" in msg:
                    engine_color = "WHITE"
                    print(">>> 游戏状态更新: 引擎执白棋")

                elif "引擎执黑" in msg:
                    engine_color = "BLACK"
                    print(">>> 游戏状态更新: 引擎执黑棋")

                elif "等待人类落子" in msg:
                    current_status = "CAPTURE_BEFORE"

                elif "人类落子完成" in msg:
                    current_status = "CAPTURE_AFTER"

                elif "引擎落子" in msg:
                    engine_move = msg

                elif "摆棋白左" in msg:
                    arrange_mode = "WHITE_LEFT"

                elif "摆棋白右" in msg:
                    arrange_mode = "WHITE_RIGHT"

        except Exception as e:
            print("接收异常:", e)
            os._exit(0)

# ================= 主程序 =================


def main():
    global current_status, client, engine_move, engine_color
    global global_board_matrix
    global arrange_mode

    generate_board_points()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_SETTINGS, 0)
    client = socket.socket()
    client.connect(("127.0.0.1", 8888))
    client.send("REGISTER:五子棋视觉\n".encode())

    try:

        client.send("准备完毕\n".encode())
        print(">>> 已发送:准备完毕")
    except:
        print("TCP发送失败")

    t = threading.Thread(target=recv_thread, args=(client,), daemon=True)
    t.start()

    cv2.namedWindow("Gomoku_Main")
    cv2.setMouseCallback("Gomoku_Main", click_board)

    while True:
        ret, frame = cap.read()
        if not ret: break

        centers, warped, gray = detect_circles(frame)
        display = warped.copy()

        if board_ready:
            display = draw_board(display)

        for cx, cy in centers:

            y1, y2 = max(0, cy-3), min(H, cy+4)
            x1, x2 = max(0, cx-3), min(W, cx+4)

            intensity = np.mean(gray[y1:y2, x1:x2])

            piece_color = "WHITE" if intensity > COLOR_THRESHOLD else "BLACK"

            is_spare = False

            if poly_board is not None:
                dist = cv2.pointPolygonTest(poly_board, (cx, cy), True)
                if dist < -15:
                    is_spare = True

            ring_color = (255, 255, 255) if piece_color == "WHITE" else (0, 0, 0)

            if is_spare:
                cv2.circle(display, (cx, cy), 14, ring_color, -1)
                cv2.circle(display, (cx, cy), 18, (0, 255, 255), 2)
            else:
                cv2.circle(display, (cx, cy), 12, ring_color, 2)

        # ===== 摆棋 =====

        if arrange_mode == "WHITE_LEFT":
            arrange_pieces(centers, gray, True)
            arrange_mode = None

        elif arrange_mode == "WHITE_RIGHT":
            arrange_pieces(centers, gray, False)
            arrange_mode = None

        # ===== BEFORE =====

        if current_status == "CAPTURE_BEFORE":
            if not board_ready:
                current_status = "IDLE"
                continue

            print(">>> 收到等待落子信号，已启用全局棋局数组追踪...")
            current_status = "WAITING_MOVE"

        # ===== AFTER =====

        elif current_status == "CAPTURE_AFTER":

            mat_after = circles_to_matrix(centers)
            diff = mat_after - global_board_matrix

            found = None

            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    if diff[r][c] == 1:
                        found = (r, c)

            if found:
                r, c = found

                msg = f"落子:{c},{r}\n"

                client.send(msg.encode())

                print(f">>> 发送人类落子坐标:{c},{r}")

                global_board_matrix[r][c] = 1

            else:
                client.send("无变化\n".encode())

            current_status = "IDLE"

        # ===== 引擎落子 =====

        if engine_move is not None:

            rc = engine_move.split(":")[1]
            parts = rc.split(",")

            c, r = int(parts[0]), int(parts[1])

            if global_board_matrix[r][c] == 0:
                global_board_matrix[r][c] = 1
                print(f">>> 全局棋盘已记录引擎落子: {c},{r}")

            px, py = board_points[r][c]

            spare_list = []

            if board_ready and poly_board is not None:

                for cx, cy in centers:

                    dist = cv2.pointPolygonTest(poly_board, (cx, cy), True)

                    if dist < -15:

                        y1, y2 = max(0, cy-3), min(H, cy+4)
                        x1, x2 = max(0, cx-3), min(W, cx+4)

                        intensity = np.mean(gray[y1:y2, x1:x2])

                        c_color = "WHITE" if intensity > COLOR_THRESHOLD else "BLACK"

                        if engine_color is None or c_color == engine_color:
                            spare_list.append((cx, cy))

            if len(spare_list) > 0:

                sx, sy = spare_list[0]

                msg = f"棋子移动:({sx},{sy}),({int(px)},{int(py)})\n"

                client.send(msg.encode())

                print(msg)

                engine_move = None

        cv2.imshow("Gomoku_Main", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()