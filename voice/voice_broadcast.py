# -*- coding: utf-8 -*-
"""语音播报客户端 —— 连接服务端，收到"文字：xx"时用 VITS 语音播放 xx"""

import socket
import threading
import queue
import time
import os
import numpy as np
import sherpa_onnx
import sounddevice as sd

# --- 配置 ---
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8888

# 语音模型路径
MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "voice_library", "vits-zh-aishell3")
MODEL_FILE = os.path.join(MODEL_DIR, "vits-aishell3.onnx")

# TTS 参数
TTS_SPEED = 1.2      # 语速
TTS_GAIN = 8.0       # 音量增益
TTS_SID = 129         # 音色 ID


class VoiceClient:
    """TCP 客户端 —— 监听服务端文字消息并语音播报"""

    def __init__(self):
        self._sock: socket.socket | None = None
        self._queue: queue.Queue = queue.Queue()
        self._running: bool = False
        self._thread: threading.Thread | None = None
        self._tts: sherpa_onnx.OfflineTts | None = None

    # ------------------------------------------------------------------
    # TTS 初始化
    # ------------------------------------------------------------------
    def init_tts(self) -> bool:
        """加载 VITS 语音合成引擎"""
        if not os.path.exists(MODEL_FILE):
            print(f"[错误] 模型文件不存在: {MODEL_FILE}")
            return False

        print("[语音] 正在加载 TTS 引擎...")
        try:
            tts_config = sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                    vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                        model=MODEL_FILE,
                        lexicon=os.path.join(MODEL_DIR, "lexicon.txt"),
                        tokens=os.path.join(MODEL_DIR, "tokens.txt"),
                    ),
                    num_threads=2,
                )
            )
            self._tts = sherpa_onnx.OfflineTts(tts_config)
            print("[语音] TTS 引擎加载完成")
            return True
        except Exception as e:
            print(f"[错误] TTS 初始化失败: {e}")
            return False

    def speak(self, text: str):
        """合成并播放语音"""
        if not self._tts or not text.strip():
            return
        try:
            audio = self._tts.generate(text, sid=TTS_SID, speed=TTS_SPEED)
            samples = np.array(audio.samples)
            if samples.size > 0:
                samples = samples * TTS_GAIN
                samples = np.clip(samples, -1.0, 1.0)
                sd.play(samples, audio.sample_rate)
                sd.wait()
                print(f"[语音] 播报完成: {text}")
        except Exception as e:
            print(f"[语音] 播报失败: {e}")

    # ------------------------------------------------------------------
    # 网络通信
    # ------------------------------------------------------------------
    def connect(self):
        """连接服务端并启动接收线程"""
        self._sock = socket.socket()
        try:
            self._sock.connect((SERVER_HOST, SERVER_PORT))
        except Exception as e:
            print(f"[网络] 连接失败: {e}")
            return False

        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()
        print(f"[网络] 已连接 {SERVER_HOST}:{SERVER_PORT}")
        return True
s
    def _recv_loop(self):
        """后台接收线程"""
        while self._running:
            try:
                data = self._sock.recv(1024)
                if not data:
                    print("[网络] 服务器断开连接")
                    self._running = False
                    break
                msg = data.decode().strip()
                print(f"[收到] {msg}")
                self._queue.put(msg)
            except Exception as e:
                if self._running:
                    print(f"[网络] 接收异常: {e}")
                self._running = False
                break

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    def run(self):
        """主循环：监听消息，匹配"文字：xx"并播报"""
        print("\n" + "=" * 40)
        print("  象棋语音播报客户端")
        print("=" * 40)
        print(f"  服务端: {SERVER_HOST}:{SERVER_PORT}")
        print(f"  音色 ID: {TTS_SID}")
        print("-" * 40)
        print("  等待服务端文字消息...")
        print("  按 Ctrl+C 退出")
        print("-" * 40 + "\n")

        while self._running:
            try:
                msg = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            # 匹配"文字：xx"
            if msg.startswith("语音:"):
                text = msg[3:].strip()  # 去掉"文字："前缀
                if text:
                    self.speak(text)

    def shutdown(self):
        """关闭连接"""
        self._running = False
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
        print("[语音] 已退出")


def main():
    client = VoiceClient()

    if not client.init_tts():
        return

    if not client.connect():
        return

    try:
        client.run()
    except KeyboardInterrupt:
        print("\n[语音] 用户中断")
    finally:
        client.shutdown()


if __name__ == "__main__":
    main()
