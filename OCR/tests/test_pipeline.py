"""
端到端测试脚本
"""
import os
import sys
import json
import tempfile
import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# 单元测试（不需要启动服务）
# ============================================================
class TestModules:
    """模块级别测试"""

    def test_chinese_to_arabic(self):
        """测试中文数字转换"""
        from modules.asr_service import ASRService
        from config.settings import config
        asr = ASRService(config.whisper)

        cases = [
            ("二零二一零零三七", "20210037"),
            ("二〇二三一二三四", "20231234"),
            ("幺二三四五六七八", "12345678"),
            ("S二零二一零零三七", "S20210037"),
        ]
        for text, expected in cases:
            result = asr.extract_id_from_text(text)
            assert result == expected, f"Failed: '{text}' -> '{result}', expected '{expected}'"
            print(f"  ✓ '{text}' -> '{result}'")

    def test_matching_engine(self):
        """测试匹配引擎"""
        from modules.matching_engine import MatchingEngine
        from config.settings import config
        engine = MatchingEngine(config.matching)

        # 精确匹配
        result = engine.verify("20210037", "20210037")
        assert result['verdict'] == "MATCH"
        print(f"  ✓ 精确匹配: {result['verdict']}")

        # 混淆字符
        result = engine.verify("2021O037", "20210037")
        assert result['verdict'] == "MATCH"
        print(f"  ✓ 混淆字符 (O->0): {result['verdict']}")

        # 不匹配
        result = engine.verify("20210037", "20215599")
        assert result['verdict'] == "NO_MATCH"
        print(f"  ✓ 不匹配: {result['verdict']}")

        # 模糊匹配 (1位差异)
        result = engine.verify("20210037", "20210087")
        print(f"  ✓ 模糊匹配 (1位差异): {result['verdict']}, similarity={result['similarity']}")

    def test_audio_preprocessor(self):
        """测试音频预处理"""
        from modules.audio_preprocessor import AudioPreprocessor
        from config.settings import config

        preprocessor = AudioPreprocessor(config.audio)

        # 生成测试音频
        import numpy as np
        try:
            import soundfile as sf
        except ImportError:
            print("  ⚠ soundfile未安装，跳过音频测试")
            return

        sr = 44100
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t) * 0.5  # 440Hz正弦波

        test_path = os.path.join(tempfile.gettempdir(), "test_audio.wav")
        sf.write(test_path, audio, sr)

        processed_path, metadata = preprocessor.process(test_path)
        assert os.path.exists(processed_path)
        assert metadata['sample_rate'] == 16000
        print(f"  ✓ 音频预处理: {metadata}")

        os.remove(test_path)

    def test_image_preprocessor(self):
        """测试图像预处理"""
        from modules.image_preprocessor import ImagePreprocessor
        from config.settings import config
        import numpy as np
        import cv2

        preprocessor = ImagePreprocessor(config.image)

        # 生成测试图像
        img = np.ones((400, 600, 3), dtype=np.uint8) * 255
        cv2.putText(img, "Student ID: 20210037", (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        test_path = os.path.join(tempfile.gettempdir(), "test_card.png")
        cv2.imwrite(test_path, img)

        processed_path, metadata = preprocessor.process(test_path)
        assert os.path.exists(processed_path)
        print(f"  ✓ 图像预处理: {metadata}")

        os.remove(test_path)


# ============================================================
# 集成测试（需要启动服务）
# ============================================================
class TestAPI:
    """API集成测试"""

    BASE_URL = "http://localhost:8000"

    def test_health(self):
        """测试健康检查"""
        resp = requests.get(f"{self.BASE_URL}/health")
        assert resp.status_code == 200
        print(f"  ✓ 健康检查: {resp.json()}")

    def test_full_verification(self):
        """完整验证流程测试"""
        import numpy as np
        import cv2
        try:
            import soundfile as sf
        except ImportError:
            print("  ⚠ soundfile未安装，跳过集成测试")
            return

        # 生成测试音频
        sr = 16000
        t = np.linspace(0, 2, sr * 2)
        audio = np.sin(2 * np.pi * 440 * t) * 0.5
        audio_path = os.path.join(tempfile.gettempdir(), "test_verify.wav")
        sf.write(audio_path, audio, sr)

        # 生成测试图像
        img = np.ones((400, 600, 3), dtype=np.uint8) * 255
        cv2.putText(img, "Student ID: 20210037", (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        image_path = os.path.join(tempfile.gettempdir(), "test_card_verify.png")
        cv2.imwrite(image_path, img)

        # 发送请求
        with open(audio_path, 'rb') as af, open(image_path, 'rb') as imf:
            files = {
                'audio': ('test.wav', af, 'audio/wav'),
                'image': ('test.png', imf, 'image/png'),
            }
            data = {'use_llm': 'false', 'language': 'zh'}
            resp = requests.post(f"{self.BASE_URL}/api/verify", files=files, data=data)

        print(f"  Status: {resp.status_code}")
        if resp.status_code == 200:
            result = resp.json()
            print(f"  ✓ 验证结果: {result['final_verdict']}")
            print(f"    ASR ID: {result['asr_result'].get('extracted_id', 'N/A')}")
            print(f"    OCR ID: {result['ocr_result'].get('extracted_id', 'N/A')}")
            print(f"    耗时: {result['processing_time_ms']}ms")
        else:
            print(f"  ✗ 请求失败: {resp.text}")

        os.remove(audio_path)
        os.remove(image_path)


def run_unit_tests():
    """运行单元测试"""
    print("\n" + "=" * 50)
    print(" 单元测试")
    print("=" * 50)

    tests = TestModules()

    print("\n[1] 中文数字转换测试:")
    tests.test_chinese_to_arabic()

    print("\n[2] 匹配引擎测试:")
    tests.test_matching_engine()

    print("\n[3] 音频预处理测试:")
    tests.test_audio_preprocessor()

    print("\n[4] 图像预处理测试:")
    tests.test_image_preprocessor()

    print("\n✅ 所有单元测试通过!")


def run_integration_tests():
    """运行集成测试"""
    print("\n" + "=" * 50)
    print(" 集成测试 (请确保服务已启动)")
    print("=" * 50)

    tests = TestAPI()

    print("\n[1] 健康检查:")
    try:
        tests.test_health()
    except requests.exceptions.ConnectionError:
        print("  ✗ 无法连接到服务，请先启动: python main.py")
        return

    print("\n[2] 完整验证流程:")
    tests.test_full_verification()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['unit', 'integration', 'all'], default='unit')
    args = parser.parse_args()

    if args.mode in ('unit', 'all'):
        run_unit_tests()
    if args.mode in ('integration', 'all'):
        run_integration_tests()