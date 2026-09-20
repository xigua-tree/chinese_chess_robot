import sherpa_onnx
import sounddevice as sd
import numpy as np
import os
import msvcrt

# --- 配置区 ---
MODEL_DIR = "E:/chess_robot/project/oo/2026---/voice_library/vits-zh-aishell3"
MODEL_FILE = os.path.join(MODEL_DIR, "vits-aishell3.onnx")
TARGET_DEVICE_ID = 5 

# 你的精选名单
CANDIDATES = [129]

class VoiceBattle:
    def __init__(self):
        print("正在初始化引擎...")
        tts_config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=MODEL_FILE,
                    lexicon=os.path.join(MODEL_DIR, "lexicon.txt"),
                    tokens=os.path.join(MODEL_DIR, "tokens.txt"),
                ),
                num_threads=4,
            )
        )
        self.tts = sherpa_onnx.OfflineTts(tts_config)
        self.index = 0

    def play_battle(self, mode="battle"):
        sid = CANDIDATES[self.index]
        
        # 语气增强逻辑
        if mode == "battle":
            text = "将军！"
            speed = 1.5  # 语速加快，显得肯定、急促
            gain = 18.0  # 威力加强
            power = 0.8  # 动态压缩系数，越小声音越“硬”
        else:
            text = "炮二平五。"
            speed = 1.0
            gain = 6.0
            power = 1.0

        print(f"\r测试音色 ID: {sid} | 模式: {mode}      ", end="", flush=True)
        
        try:
            # 生成音频
            audio = self.tts.generate(text, sid=sid, speed=speed)
            samples = np.array(audio.samples)
            
            if samples.size > 0:
                # 核心处理：让语气更硬、更肯定
                if mode == "battle":
                    # 提升弱信号强度，模拟喊出来的爆发感
                    samples = np.sign(samples) * (np.abs(samples) ** power)
                
                samples = samples * gain
                samples = np.clip(samples, -1.0, 1.0)
                
                sd.play(samples, audio.sample_rate, device=TARGET_DEVICE_ID)
            
        except Exception as e:
            print(f"\n错误: {e}")

    def run(self):
        print("\n" + "="*40)
        print("   象棋精选音色“将军”气势对比工具")
        print("="*40)
        print(f"当前候选名单: {CANDIDATES}")
        print("-" * 40)
        print("按 [E] : 下一个候选 ID")
        print("按 [Q] : 上一个候选 ID")
        print("按 [W] : 【战斗模式】播放：将——军！")
        print("按 [S] : 【平稳模式】播放：炮二平五")
        print("按 [ESC] : 退出")
        print("-" * 40)

        while True:
            if msvcrt.kbhit():
                key = msvcrt.getch()
                if ord(key) == 27: break
                
                char = key.decode(errors='ignore').lower()
                if char == 'e':
                    self.index = (self.index + 1) % len(CANDIDATES)
                    self.play_battle("battle")
                elif char == 'q':
                    self.index = (self.index - 1) % len(CANDIDATES)
                    self.play_battle("battle")
                elif char == 'w':
                    self.play_battle("battle")
                elif char == 's':
                    self.play_battle("normal")

if __name__ == "__main__":
    if not os.path.exists(MODEL_FILE):
        print(f"路径错误: {MODEL_FILE}")
    else:
        VoiceBattle().run()