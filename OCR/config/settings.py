"""
全局配置文件
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent

@dataclass
class LLMConfig:
    api_key: str = "sk-d042a13c89ce4e55ab2a1b0267aeaed2"

@dataclass
class WhisperConfig:
    """Whisper语音识别配置"""
    model_size: str = "small"          # tiny/base/small/medium/large-v3
    language: str = "zh"                # 默认中文
    device: str = "cpu"                # cuda / cpu
    compute_type: str = "float"       # float16 / int8 / float32
    beam_size: int = 5
    best_of: int = 5
    temperature: float = 0.0
    initial_prompt: str = "以下是一段学号数字朗读，学号由数字组成，例如20210001、S20230042。"
    vad_filter: bool = True
    vad_parameters: dict = field(default_factory=lambda: {
        "threshold": 0.5,
        "min_speech_duration_ms": 250,
        "max_speech_duration_s": 30,
        "min_silence_duration_ms": 200,
    })


@dataclass
class OCRConfig:
    """PaddleOCR配置"""
    # use_angle_cls: bool = True
    lang: str = "ch"                    # ch / en
    use_gpu: bool = True
    det_model_dir: Optional[str] = None
    rec_model_dir: Optional[str] = None
    cls_model_dir: Optional[str] = None
    det_db_thresh: float = 0.3
    det_db_box_thresh: float = 0.5
    # drop_score: float = 0.5
    # 学号正则模式列表（根据学校实际格式调整）
    id_patterns: list = field(default_factory=lambda: [
        r'[A-Za-z]?\d{8,12}',          # 通用: 8-12位数字，可选字母前缀
        r'[Ss]\d{8,10}',               # S20210001 格式
        r'20[12]\d{5,8}',              # 20开头的年份格式
    ])


@dataclass
class LLMConfig:
    """LLM配置"""
    provider: str = "deepseek"            # openai / local / dashscope
    model: str = "deepseek-chat"              # gpt-4o / qwen-plus / glm-4
    api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    base_url: Optional[str] = "https://api.deepseek.com"     # 自定义API地址(本地部署)
    temperature: float = 0.1
    max_tokens: int = 1024
    timeout: int = 30
    # 分级调用阈值
    enable_llm: bool = True
    llm_trigger_threshold: float = 0.85  # 相似度低于此值时触发LLM


@dataclass
class MatchingConfig:
    """匹配引擎配置"""
    exact_match_threshold: float = 1.0
    fuzzy_match_threshold: float = 0.85
    high_confidence_threshold: float = 0.95
    max_edit_distance: int = 2
    # 常见OCR/ASR混淆字符对
    confusion_pairs: dict = field(default_factory=lambda: {
        'O': '0', 'o': '0',
        'l': '1', 'I': '1', 'i': '1',
        'Z': '2', 'z': '2',
        'S': '5', 's': '5',
        'B': '8', 'b': '8',
        'G': '6', 'g': '9',
        'q': '9', 'D': '0',
    })


@dataclass
class AudioConfig:
    """音频预处理配置"""
    target_sample_rate: int = 16000     # Whisper要求16kHz
    target_channels: int = 1            # 单声道
    max_duration_seconds: int = 30      # 最长录音
    min_duration_seconds: float = 0.5   # 最短录音
    noise_reduce: bool = True
    normalize: bool = True
    supported_formats: list = field(default_factory=lambda: [
        '.wav', '.mp3', '.flac', '.m4a', '.ogg', '.webm'
    ])


@dataclass
class ImageConfig:
    """图像预处理配置"""
    max_image_size: tuple = (2000, 2000)
    min_image_size: tuple = (200, 100)
    target_dpi: int = 300
    enable_deskew: bool = True
    enable_denoise: bool = True
    enable_contrast_enhance: bool = True
    supported_formats: list = field(default_factory=lambda: [
        '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'
    ])


@dataclass
class AppConfig:
    """应用总配置"""
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    image: ImageConfig = field(default_factory=ImageConfig)

    # 服务配置
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    upload_dir: str = str(BASE_DIR / "uploads")
    log_dir: str = str(BASE_DIR / "logs")
    log_level: str = "INFO"

    def __post_init__(self):
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)


# 全局配置实例
config = AppConfig()