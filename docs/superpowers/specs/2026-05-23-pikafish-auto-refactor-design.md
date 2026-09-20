# Pikafish_auto4.py 重构设计

## 目标

将 Pikafish_auto4.py（1058 行单文件）重构为类化结构，保持单文件部署便利性，同时提升完整性和可靠性。

## 核心需求

1. **棋盘边缘识别改进**：边缘棋子偶尔漏识别导致走法计算错误
2. **引擎难度设置**：支持 4 档难度，通过 `go movetime` 控制
3. **通信协议重构**：新消息格式、新坐标格式、增加"运动完成"同步
4. **引擎崩溃自动恢复**：引擎进程异常退出时重启并恢复局面
5. **悔棋接口预留**：架构层面预留悔棋功能接入点

## 架构：5 类 + 1 入口

### `Config` — 集中配置

所有可调参数集中为类属性，消除散落的全局常量：

- 模型路径、置信度阈值
- 引擎路径、难度→movetime 映射 `{1: 500, 2: 1500, 3: 3000, 4: 6000}`
- 棋盘定位坐标、透视变换矩阵、网格参数
- 视觉参数：稳定帧数、边缘扩展 padding、最大重试次数
- 棋子映射表（FEN_MAP / CHINESE_MAP / INV_FEN_MAP）
- 网络参数

### `ProtocolClient` — TCP 通信

```
connect(host, port, identifier)    → 创建 socket + 启动后台 recv 线程
send(msg)                          → 按新格式发送
wait_for(expected, timeout)        → 阻塞等待精确匹配
wait_for_prefix(prefix, timeout)   → 前缀匹配，返回完整消息（用于解析难度参数）
peek()                             → 非阻塞读
is_connected()                     → 连接状态
reconnect()                        → 断线重连
```

**协议变更**：

| 场景 | 旧格式 | 新格式 |
|------|--------|--------|
| 注册 | `REGISTER:象棋引擎` | `象棋引擎` (标识符) |
| 初始化完成 | 无 | `准备完毕` |
| 开始对弈 | `开始对弈` | `开始对弈N` (N=难度1-4) |
| 请求落子 | `请玩家落子` | `请玩家落子` (不变) |
| 落子完成 | `玩家落子完成` | `玩家落子完成` (不变) |
| 坐标移动 | `象棋落子坐标:src->dst` | `棋子移动:(x1,y1),(x2,y2)` |
| 机械臂完成 | 无 | `运动完成` |
| 悔棋(预留) | 无 | `悔棋` |

修复：`recv_t.daemon = True` → `daemon`

### `EngineController` — UCI 引擎管理

```
start()                              → 启动 pikafish 子进程
uci_handshake(level)                 → uci → uciok → isready → readyok
set_level(level)                     → 动态修改 movetime
get_best_move(fen_moves) -> str      → position + go movetime → 解析 bestmove
restart(fen_moves)                   → 崩溃后重启并恢复棋盘状态
is_alive() -> bool                   → 进程存活检测
stop()                               → 优雅退出
```

**改进**：
- 用 `select.select()` 替代每次新建线程读 stdout，减少线程开销
- 进程崩溃时自动检测并通过 `restart(fen)` 恢复当前局面
- `restart` 流程：kill 旧进程 → 重新 Popen → uci_handshake → position fen

### `VisionController` — 视觉识别

```
capture_stable_board(frames=3)        → 多帧稳定校验，返回 10x9 矩阵
capture_any_board()                   → 单帧识别（用于快速重试）
get_move(prev, curr) -> str           → 差分计算人类 UCI 走法
detect_side(board) -> str             → 检测人类执子方 ("red"/"black")
detect_piece_at(uci, frame)           → 指定坐标棋子定位 + 类型识别
draw_overlay(warped)                  → 可视化网格
count_pieces(board)                   → 统计红黑棋子数
board_to_fen(board) -> str            → 10x9 矩阵 → FEN
board_from_fen(fen) -> List[List[str]] → FEN → 10x9 矩阵
```

**边缘棋子改进**：
- ROI 裁剪边界使用 `cv2.BORDER_REPLICATE` 扩展而非 clamp
- 网格映射使用浮点比较替代 `int(round())` 硬截断
- 棋子总数校验：识别结果总数 ≠ 32 时触发重新采样

### `GameController` — 对弈状态机

状态枚举：
```
INIT → READY → WAIT_START → HUMAN_MOVE ⇄ ENGINE_MOVE → GAME_OVER → READY → ...
                                ↑            ↓
                                └── UNDO ────┘ (预留)
```

**方法**：
```
run()                                → 状态机主循环
_handle_human_move()                 → 识别 + 校验 + 错误重试
_handle_engine_move()                → 计算 + 发送坐标 + 吃子处理
_send_move_to_robot(uci, capture)    → 构造并发送坐标指令
_check_game_over(last_color)         → 将死检测
_reset_for_new_game()                → 重置棋盘状态和计数器
request_undo() -> str|None           → 悔棋（预留接口）
push_move_record(uci, src, dst)      → 记录走法历史
```

**消除的重复代码**：原 `run_chess_loop` 中人类先走/引擎先走两套几乎相同的 while 循环合并为一个状态机。

## 关键 Bug 修复

| Bug | 修复 |
|-----|------|
| `recv_t.daemon = True` 拼写错误 | → `daemon` |
| `engine_move` 在人类先走分支中引用前未定义 | 状态机保证顺序 |
| `os._exit(0)` 跳过资源释放 | 统一走 GAME_OVER → 资源释放路径 |
| `check_game_over` 的 flag 参数逻辑错误 | 用 board.turn 直接判断 |
| `use_chess_count` 对局间不重置 | 在 `_reset_for_new_game()` 中重置 |
| 引擎崩溃后静默失效 | `EngineController.restart(fen)` |
| `readline_timeout` 每次新建线程 | 改用 `select.select()` |

## 悔棋接口预留

```python
class GameController:
    move_history: List[MoveRecord]  # [(uci, src_xy, dst_xy), ...]

    def request_undo(self) -> str | None:
        """撤销步，返回机械臂复位坐标指令"""
        if len(self.move_history) < 2:
            return None
        m2 = self.move_history.pop()  # 引擎走法
        m1 = self.move_history.pop()  # 人类走法
        # 重建引擎棋盘状态到悔棋前
        self._engine.restart(self._fen_before_last_human_move)
        return f"棋子移动:{m2.dst},{m2.src};棋子移动:{m1.dst},{m1.src}"
```

## 测试策略

- **视觉模块**：离线棋盘图片测试 `capture_stable_board`，验证边缘棋子不遗漏
- **引擎模块**：单独测试握手 + 走法生成 + 崩溃重启恢复后 fen 一致
- **协议模块**：mock 服务器验证消息格式正确性
- **状态机**：模拟正常对局流程 + 悔棋流程 + 引擎崩溃恢复流程
