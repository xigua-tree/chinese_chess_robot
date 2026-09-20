import edge_tts
import asyncio
import os

# 定义你需要的象棋术语
CHESS_PHRASES = [
    "将军！", "绝杀！", "炮二平五", "马八进七", "车一平二", 
    "该你走啦。", "这步棋真精彩。"
]

VOICE = "zh-CN-Xiaoxiao-Neural"

async def generate_assets():
    if not os.path.exists("voice_library"):
        os.makedirs("voice_library")

    for text in CHESS_PHRASES:
        output_path = f"voice_library/{text}.mp3"
        print(f"正在尝试生成: {text}...")
        
        try:
            # rate=+15% 语速轻快显得俏皮
            # pitch=+10% 音调升高，增加甜美度（像少女）
            communicate = edge_tts.Communicate(text, VOICE, rate="+15%", pitch="+10%")
            await communicate.save(output_path)
            print(f"✅ 成功生成: {output_path}")
        except Exception as e:
            print(f"❌ 生成 {text} 失败: {e}")
            print("提示：请检查网络是否能访问微软服务器，或尝试关闭魔法上网插件再试。")

if __name__ == "__main__":
    # 针对 Windows 的 asyncio 事件循环兼容性修复
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(generate_assets())