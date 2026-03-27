import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent.parent

@dataclass
class WhisperConfig:
    model_size: str = "small"
    language: str = "zh"
    device: str = "cuda"               # 强制默认使用 GPU
    compute_type: str = "float16"
    beam_size: int = 5
    best_of: int = 5
    temperature: float = 0.0
    initial_prompt: str = "以下是一段学号数字朗读，学号由数字组成，例如20210001、S20230042。"
    vad_filter: bool = True
    vad_parameters: dict = field(default_factory=lambda: {
        "threshold": 0.5, "min_speech_duration_ms": 250,
        "max_speech_duration_s": 30, "min_silence_duration_ms": 200,
    })

@dataclass
class OCRConfig:
    lang: str = "ch"
    use_gpu: bool = True                # 强制开启 GPU
    det_model_dir: Optional[str] = None
    rec_model_dir: Optional[str] = None
    cls_model_dir: Optional[str] = None
    det_db_thresh: float = 0.3
    det_db_box_thresh: float = 0.5
    id_patterns: list = field(default_factory=lambda: [
        r'[A-Za-z]?\d{8,12}',
        r'[Ss]\d{8,10}',
        r'20[12]\d{5,8}',
    ])

@dataclass
class LLMConfig:
    provider: str = "deepseek"
    model: str = "deepseek-chat"
    api_key: str = os.getenv("DEEPSEEK_API_KEY", "xxxxx")
    base_url: Optional[str] = "https://api.deepseek.com"
    temperature: float = 0.1
    max_tokens: int = 1024
    timeout: int = 30
    enable_llm: bool = True
    llm_trigger_threshold: float = 0.85

@dataclass
class MatchingConfig:
    exact_match_threshold: float = 1.0
    fuzzy_match_threshold: float = 0.85
    high_confidence_threshold: float = 0.95
    max_edit_distance: int = 2
    confusion_pairs: dict = field(default_factory=lambda: {
        'O': '0', 'o': '0', 'l': '1', 'I': '1', 'i': '1',
        'Z': '2', 'z': '2', 'S': '5', 's': '5',
        'B': '8', 'b': '8', 'G': '6', 'g': '9', 'q': '9', 'D': '0',
    })

@dataclass
class AudioConfig:
    target_sample_rate: int = 16000
    target_channels: int = 1
    max_duration_seconds: int = 30
    min_duration_seconds: float = 0.5
    noise_reduce: bool = True
    normalize: bool = True
    supported_formats: list = field(default_factory=lambda: [
        '.wav', '.mp3', '.flac', '.m4a', '.ogg', '.webm'
    ])

@dataclass
class ImageConfig:
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
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    image: ImageConfig = field(default_factory=ImageConfig)

    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    upload_dir: str = str(BASE_DIR / "uploads")
    log_dir: str = str(BASE_DIR / "logs")
    log_level: str = "INFO"

    def __post_init__(self):
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

config = AppConfig()