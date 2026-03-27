"""
语音识别模块 (ASR - Automatic Speech Recognition)
基于 OpenAI Whisper / faster-whisper
"""
import re
import logging
from typing import Tuple, Optional, List, Dict


logger = logging.getLogger(__name__)


# 中文数字映射表
CHINESE_DIGIT_MAP = {
    '零': '0', '〇': '0', '洞': '0',
    '一': '1', '壹': '1', '幺': '1', '拐': '7',
    '二': '2', '贰': '2', '两': '2',
    '三': '3', '叁': '3',
    '四': '4', '肆': '4',
    '五': '5', '伍': '5',
    '六': '6', '陆': '6',
    '七': '7', '柒': '7',
    '八': '8', '捌': '8',
    '九': '9', '玖': '9',
}

# 英文数字词映射
ENGLISH_DIGIT_MAP = {
    'zero': '0', 'oh': '0',
    'one': '1', 'two': '2', 'three': '3',
    'four': '4', 'five': '5', 'six': '6',
    'seven': '7', 'eight': '8', 'nine': '9',
    'double': '',  # 需特殊处理
    'triple': '',  # 需特殊处理
}


class ASRService:
    """语音识别服务"""

    def __init__(self, config):
        self.config = config
        self.model = None

    def load_model(self):
        """延迟加载模型"""
        if self.model is not None:
            return

        logger.info(f"正在加载 Whisper 模型: {self.config.model_size}")

        try:
            # 优先使用 faster-whisper（更快、更省内存）
            from faster_whisper import WhisperModel
            self.model = WhisperModel(
                self.config.model_size,
                device=self.config.device,
                compute_type=self.config.compute_type,
            )
            self._engine = "faster-whisper"
            logger.info(f"faster-whisper 加载成功 (device={self.config.device})")

        except ImportError:
            logger.info("faster-whisper 不可用，使用原版 whisper")
            import whisper
            self.model = whisper.load_model(
                self.config.model_size,
                device=self.config.device
            )
            self._engine = "whisper"
            logger.info(f"whisper 加载成功 (device={self.config.device})")

    def transcribe(self, audio_path: str) -> Dict:
        """
        语音转文字

        Returns:
            dict: {
                'raw_text': 原始转写文本,
                'extracted_id': 提取的学号,
                'segments': 分段信息,
                'language': 检测语言,
                'confidence': 置信度
            }
        """
        self.load_model()
        logger.info(f"开始转写音频: {audio_path}")

        if self._engine == "faster-whisper":
            result = self._transcribe_faster(audio_path)
        else:
            result = self._transcribe_original(audio_path)

        # 从转写文本中提取学号
        extracted_id = self.extract_id_from_text(result['raw_text'])
        result['extracted_id'] = extracted_id

        logger.info(f"转写结果: text='{result['raw_text']}', id='{extracted_id}'")
        return result

    def _transcribe_faster(self, audio_path: str) -> Dict:
        """使用 faster-whisper 转写"""
        segments, info = self.model.transcribe(
            audio_path,
            language=self.config.language,
            beam_size=self.config.beam_size,
            best_of=self.config.best_of,
            temperature=self.config.temperature,
            initial_prompt=self.config.initial_prompt,
            vad_filter=self.config.vad_filter,
            vad_parameters=self.config.vad_parameters,
            word_timestamps=True,
        )

        segment_list = []
        full_text = ""
        total_logprob = 0
        total_tokens = 0

        for seg in segments:
            segment_list.append({
                'start': round(seg.start, 2),
                'end': round(seg.end, 2),
                'text': seg.text.strip(),
                'avg_logprob': round(seg.avg_logprob, 4),
                'no_speech_prob': round(seg.no_speech_prob, 4),
            })
            full_text += seg.text
            total_logprob += seg.avg_logprob
            total_tokens += 1

        avg_confidence = 0.0
        if total_tokens > 0:
            # 将 avg_logprob 转为 0-1 的置信度
            import math
            avg_logprob = total_logprob / total_tokens
            avg_confidence = math.exp(avg_logprob)

        return {
            'raw_text': full_text.strip(),
            'segments': segment_list,
            'language': info.language,
            'language_probability': round(info.language_probability, 4),
            'confidence': round(avg_confidence, 4),
            'duration': round(info.duration, 2),
        }

    def _transcribe_original(self, audio_path: str) -> Dict:
        """使用原版 whisper 转写"""
        result = self.model.transcribe(
            audio_path,
            language=self.config.language,
            beam_size=self.config.beam_size,
            best_of=self.config.best_of,
            temperature=self.config.temperature,
            initial_prompt=self.config.initial_prompt,
        )

        import math
        segments = []
        total_logprob = 0
        total_count = 0

        for seg in result.get('segments', []):
            segments.append({
                'start': round(seg['start'], 2),
                'end': round(seg['end'], 2),
                'text': seg['text'].strip(),
                'avg_logprob': round(seg.get('avg_logprob', 0), 4),
                'no_speech_prob': round(seg.get('no_speech_prob', 0), 4),
            })
            total_logprob += seg.get('avg_logprob', 0)
            total_count += 1

        avg_confidence = math.exp(total_logprob / max(total_count, 1))

        return {
            'raw_text': result['text'].strip(),
            'segments': segments,
            'language': result.get('language', self.config.language),
            'language_probability': 0.0,
            'confidence': round(avg_confidence, 4),
            'duration': 0.0,
        }

    def extract_id_from_text(self, text: str) -> str:
        """
        从转写文本中提取学号

        处理情况:
        1. 纯数字："20210037"
        2. 中文数字："二零二一零零三七"
        3. 混合："二零21零零37"
        4. 带空格/标点："2021 0037"
        5. 带无关文字："我的学号是20210037"
        """
        if not text:
            return ""

        # Step 1: 清理文本
        cleaned = self._clean_text(text)

        # Step 2: 中文数字转阿拉伯数字
        converted = self._chinese_to_arabic(cleaned)

        # Step 3: 英文数字词转阿拉伯数字
        converted = self._english_words_to_digits(converted)

        # Step 4: 提取数字序列
        extracted = self._extract_number_sequence(converted)

        return extracted

    def _clean_text(self, text: str) -> str:
        """清理文本中的无关字符"""
        # 移除常见的无关前缀
        removals = [
            '我的学号是', '学号是', '我的学号', '学号',
            '我是', '号码是', '号码',
            'my student id is', 'my id is', 'student id',
        ]
        cleaned = text.strip()
        for r in removals:
            cleaned = cleaned.replace(r, '')

        # 移除标点符号（保留字母和数字和中文）
        cleaned = re.sub(r'[，。！？、；：""''【】《》\-\.\,\!\?\;\:]', ' ', cleaned)
        return cleaned.strip()

    def _chinese_to_arabic(self, text: str) -> str:
        """中文数字转阿拉伯数字"""
        result = text
        for cn, ar in CHINESE_DIGIT_MAP.items():
            result = result.replace(cn, ar)
        return result

    def _english_words_to_digits(self, text: str) -> str:
        """英文数字词转阿拉伯数字"""
        result = text.lower()

        # 处理 "double X" -> "XX"
        result = re.sub(r'double\s+(\d)', r'\1\1', result)
        result = re.sub(r'triple\s+(\d)', r'\1\1\1', result)

        for word, digit in ENGLISH_DIGIT_MAP.items():
            if digit:  # 跳过 double/triple
                result = re.sub(r'\b' + word + r'\b', digit, result)

        return result

    def _extract_number_sequence(self, text: str) -> str:
        """提取数字序列"""
        # 保留字母前缀（如 S20210037）
        # 先尝试匹配带字母前缀的学号
        pattern_with_prefix = re.findall(r'[A-Za-z]\d{7,12}', text)
        if pattern_with_prefix:
            return pattern_with_prefix[0].upper()

        # 提取所有数字并拼接
        digits = re.findall(r'\d+', text)
        if digits:
            combined = ''.join(digits)
            return combined

        return ""