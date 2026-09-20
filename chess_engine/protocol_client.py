# -*- coding: utf-8 -*-
"""TCP 通信封装 —— 与 Qt 服务端交互"""

import queue
import socket
import threading
import time
from typing import Optional

from chess_engine.config import Config


class ProtocolClient:
    """TCP 通信封装 —— 与 Qt 服务端交互"""

    def __init__(self):
        self._sock: Optional[socket.socket] = None
        self._queue: queue.Queue = queue.Queue()
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None
        self._shutdown_requested: bool = False

    def connect(self, host: str = None, port: int = None):
        """创建 socket 连接并启动后台接收线程"""
        host = host or Config.SERVER_HOST
        port = port or Config.SERVER_PORT
        self._sock = socket.socket()
        self._sock.connect((host, port))
        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()
        print(f"[协议] 已连接 {host}:{port}")

    def _recv_loop(self):
        """后台线程：持续监听服务端消息"""
        while self._running:
            try:
                data = self._sock.recv(1024)
                if not data:
                    print("[协议] 服务器断开连接")
                    self._running = False
                    break
                msg = data.decode().strip()
                print(f"[收到] {msg}")
                self._queue.put(msg)
                if msg == "退出":
                    self._shutdown_requested = True
                    self._running = False
            except Exception as e:
                print(f"[协议] 接收异常: {e}")
                self._running = False
                break

    def send(self, msg: str):
        """发送消息到服务端"""
        if self._shutdown_requested:
            return
        if not self._sock:
            print("[协议] 未连接，无法发送")
            return
        try:
            self._sock.send((msg + "\n").encode())
            print(f"[发送] {msg}")
        except Exception as e:
            print(f"[协议] 发送异常: {e}")
            self._running = False

    def wait_for(self, expected: str, timeout: float = None) -> bool:
        """阻塞等待精确匹配的消息"""
        print(f"[等待] {expected} ...")
        deadline = time.time() + (timeout if timeout else 999999)
        while time.time() < deadline and self._running:
            remaining = max(0.1, deadline - time.time())
            try:
                msg = self._queue.get(timeout=min(1.0, remaining))
                if msg == "退出":
                    print("[协议] 收到退出，中断等待")
                    return False
                if msg == expected:
                    print(f"[匹配] {expected}")
                    return True
            except queue.Empty:
                pass
        print(f"[超时] 等待 {expected} 超时")
        return False

    def wait_for_any(self, *expected: str) -> Optional[str]:
        """阻塞等待多个消息中的任意一个，返回匹配的消息。"""
        print(f"[等待] {list(expected)} ...")
        while self._running:
            try:
                msg = self._queue.get(timeout=1.0)
                if msg == "退出":
                    print("[协议] 收到退出，中断等待")
                    return None
                if msg in expected:
                    print(f"[匹配] {msg}")
                    return msg
            except queue.Empty:
                pass
        return None

    def wait_for_any_prefix(self, *prefixes: str) -> Optional[str]:
        """阻塞等待任意一个前缀匹配的消息，返回完整消息。"""
        print(f"[等待前缀] {list(prefixes)} ...")
        while self._running:
            try:
                msg = self._queue.get(timeout=1.0)
                if msg == "退出":
                    print("[协议] 收到退出，中断等待")
                    return None
                for p in prefixes:
                    if msg.startswith(p):
                        print(f"[匹配] {msg}")
                        return msg
            except queue.Empty:
                pass
        return None

    def wait_for_prefix(self, prefix: str, timeout: float = None) -> Optional[str]:
        """阻塞等待前缀匹配的消息，返回完整消息"""
        print(f"[等待前缀] {prefix} ...")
        deadline = time.time() + (timeout if timeout else 999999)
        while time.time() < deadline and self._running:
            remaining = max(0.1, deadline - time.time())
            try:
                msg = self._queue.get(timeout=min(1.0, remaining))
                if msg == "退出":
                    print("[协议] 收到退出，中断等待")
                    return None
                if msg.startswith(prefix):
                    print(f"[匹配] {msg}")
                    return msg
            except queue.Empty:
                pass
        print(f"[超时] 等待前缀 '{prefix}' 超时")
        return None

    def shutdown_requested(self) -> bool:
        return self._shutdown_requested

    def peek(self) -> Optional[str]:
        """非阻塞读取一条消息（用于检测悔棋等中断信号）"""
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def is_connected(self) -> bool:
        return self._running

    def disconnect(self):
        """断开连接并清理资源"""
        self._running = False
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        print("[协议] 已断开连接")
