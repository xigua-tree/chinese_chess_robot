# -*- coding: utf-8 -*-
"""UCI 象棋引擎进程管理 —— 含崩溃自动恢复"""

import queue
import subprocess
import threading
import time
from typing import Optional

from chess_engine.config import Config


class EngineController:
    """UCI 象棋引擎进程封装 —— 含崩溃自动恢复"""

    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.alive: bool = False
        self._line_queue: queue.Queue = queue.Queue()
        self._reader_thread: Optional[threading.Thread] = None
        self._level: int = 1

    # ------------------------------------------------------------------
    def start(self):
        """启动引擎子进程并创建持久化读取线程"""
        try:
            self.process = subprocess.Popen(
                [Config.PIKAFISH_PATH],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            self.alive = True
            self._reader_thread = threading.Thread(
                target=self._reader_loop, daemon=True)
            self._reader_thread.start()
            print("[引擎] 进程已启动")
        except Exception as e:
            raise RuntimeError(f"[引擎] CreateProcess 失败: {e}")

    def _reader_loop(self):
        """持久化线程：持续读取引擎 stdout 行并放入队列"""
        while self.alive:
            try:
                line = self.process.stdout.readline()
                if line:
                    self._line_queue.put(line.strip())
                else:
                    self.alive = False
                    break
            except Exception:
                self.alive = False
                break

    # ------------------------------------------------------------------
    def write(self, line: str):
        """向引擎发送命令"""
        if not self.alive:
            return
        try:
            self.process.stdin.write(line + "\n")
            self.process.stdin.flush()
            print(f"[→Engine] {line}")
        except (BrokenPipeError, OSError):
            self.alive = False
            print("[引擎] 写入失败，进程可能已退出")

    def readline(self, timeout_ms: int = 10000) -> str:
        """从队列读取一行（带超时）"""
        try:
            return self._line_queue.get(timeout=timeout_ms / 1000.0)
        except queue.Empty:
            return ""

    # ------------------------------------------------------------------
    def uci_handshake(self, level: int = 1) -> bool:
        """UCI 协议握手并设置难度"""
        self._level = level
        self.write("uci")
        for _ in range(50):
            line = self.readline(3000)
            if line == "uciok":
                break
            if not self.alive:
                return False
        self.write("isready")
        for _ in range(50):
            line = self.readline(3000)
            if line == "readyok":
                break
            if not self.alive:
                return False
        movetime = Config.LEVEL_MOVETIME.get(level, 3000)
        print(f"[引擎] 握手完成，难度={level} (movetime={movetime}ms)")
        return True

    def set_level(self, level: int):
        """动态修改引擎难度"""
        self._level = max(1, min(4, level))
        print(f"[引擎] 难度已设置为 {self._level}")

    # ------------------------------------------------------------------
    def get_best_move(self, fen_moves: str, initial_fen: str = None,
                      movetime_ms: int = None) -> str:
        """发送当前局面并获取最佳走法。movetime_ms 可覆盖等级配置。"""
        if not self.alive:
            raise RuntimeError("[引擎] 引擎不可用")

        if initial_fen:
            moves_part = f" moves {fen_moves}" if fen_moves.strip() else ""
            self.write(f"position fen {initial_fen}{moves_part}")
        elif fen_moves.strip():
            self.write(f"position startpos moves {fen_moves}")
        else:
            self.write("position startpos")

        movetime = (movetime_ms if movetime_ms is not None
                    else Config.LEVEL_MOVETIME.get(self._level, 3000))
        self.write(f"go movetime {movetime}")

        deadline = time.time() + (movetime + 5000) / 1000.0
        while time.time() < deadline:
            remaining_ms = max(100, int((deadline - time.time()) * 1000))
            line = self.readline(remaining_ms)
            if line.startswith("bestmove"):
                parts = line.split()
                if len(parts) >= 2:
                    return parts[1]
            if not self.alive:
                raise RuntimeError("[引擎] 搜索期间引擎崩溃")
        return ""

    # ------------------------------------------------------------------
    def restart(self, fen_moves: str, initial_fen: str = None) -> bool:
        """崩溃恢复：杀掉旧进程、启动新进程、恢复局面"""
        print("[引擎] 正在重启...")
        self.stop()
        try:
            self.start()
        except Exception as e:
            print(f"[引擎] 重启失败: {e}")
            return False
        if not self.uci_handshake(self._level):
            return False
        if initial_fen:
            moves_part = f" moves {fen_moves}" if fen_moves.strip() else ""
            self.write(f"position fen {initial_fen}{moves_part}")
        elif fen_moves.strip():
            self.write(f"position startpos moves {fen_moves}")
        print("[引擎] 重启完成，局面已同步")
        return True

    def is_alive(self) -> bool:
        """检测引擎进程是否存活"""
        if not self.alive:
            return False
        if self.process and self.process.poll() is not None:
            self.alive = False
        return self.alive

    # ------------------------------------------------------------------
    def stop(self):
        """优雅停止引擎进程"""
        if not self.alive:
            return
        try:
            self.write("quit")
        except Exception:
            pass
        time.sleep(0.2)
        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
                time.sleep(0.1)
                self.process.kill()
                self.process.wait()
            except Exception:
                pass
        self.alive = False
        print("[引擎] 已停止")
