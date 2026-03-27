import re
import threading
import time
import queue
import sys
import speech_recognition as sr
from googletrans import Translator

# 重要：必须使用 googletrans==3.0.0 (同步版)
# 安装命令: pip install googletrans==3.0.0

# 初始化组件
recognizer = sr.Recognizer()
translator = Translator()
audio_queue = queue.Queue()
is_running = True


def is_valid_audio(audio):
    """智能检测有效语音（避免静音/噪声）"""
    try:
        # 检测音频能量阈值（静音<100，有效语音>300）
        energy = max(audio.frame_data)
        return energy > 300
    except:
        return False


def listen_for_audio():
    """持续监听麦克风，智能降噪 + 语音检测"""
    with sr.Microphone() as source:
        # 关键优化：自适应噪声环境
        recognizer.adjust_for_ambient_noise(source, duration=2.0)
        print("🎤 正在监听英文语音... (按Ctrl+C退出)")
        print("💡 请在安静环境说话，麦克风距离嘴部15cm")

        while is_running:
            try:
                # 1. 优化语音捕获参数
                audio = recognizer.listen(source, timeout=1.5, phrase_time_limit=6)

                # 2. 语音有效性检测（关键！）
                if not is_valid_audio(audio):
                    continue

                audio_queue.put(audio)

            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                print(f"🎤 麦克风错误: {str(e)}")


def process_audio():
    """处理语音队列，智能翻译 + 健壮性处理"""
    while is_running:
        if audio_queue.empty():
            time.sleep(0.1)
            continue

        audio = audio_queue.get()
        try:
            # 1. 优先英文识别
            text = recognizer.recognize_google(audio, language='en-US')
            print(f"🗣️ 识别: {text}")

            # 2. 严格清理识别结果（移除无效字符）
            text = re.sub(r'[^\w\s.,!?\'"]', '', text).strip()
            if not text or len(text) < 3:  # 跳过太短的句子
                print("❌ 跳过无效语音（内容太短）\n")
                continue

            # 3. 翻译增强：增加重试 + 响应验证
            translation = None
            for retry in range(3):
                try:
                    translation = translator.translate(text, dest='zh-cn')
                    # 关键：验证翻译结果是否有效
                    if translation and hasattr(translation, 'text') and translation.text:
                        break
                except Exception as e:
                    print(f"🔄 翻译重试 {retry + 1}/3: {str(e)}")
                    time.sleep(0.5)

            if not translation or not hasattr(translation, 'text') or not translation.text:
                print("❌ 翻译失败（重试3次）\n")
                continue

            print(f"✅ 翻译: {translation.text}\n")

        except sr.UnknownValueError:
            print("❌ 语音识别失败（请说清楚些）\n")
        except Exception as e:
            print(f"❌ 翻译错误: {str(e)}\n")
        finally:
            audio_queue.task_done()


def main():
    global is_running

    # 检查 googletrans 版本（必须3.0.0）
    try:
        import googletrans
        if googletrans.__version__ != '3.0.0':
            print("❌ 错误: 请安装 googletrans==3.0.0")
            print("运行: pip uninstall -y googletrans && pip install googletrans==3.0.0")
            sys.exit(1)
    except:
        print("❌ 请先安装 googletrans==3.0.0")
        print("运行: pip install googletrans==3.0.0")
        sys.exit(1)

    # 启动语音监听线程
    listen_thread = threading.Thread(target=listen_for_audio, daemon=True)
    listen_thread.start()

    # 启动音频处理线程
    process_thread = threading.Thread(target=process_audio, daemon=True)
    process_thread.start()

    print("\n🚀 实时英文语音 → 中文翻译系统已启动")
    print("👉 请说英文内容（靠近麦克风，保持安静）")
    print("👉 按 Ctrl+C 退出程序\n")

    try:
        while is_running:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 正在关闭系统...")
        is_running = False


if __name__ == "__main__":
    main()