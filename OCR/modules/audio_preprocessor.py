"""
音频预处理模块
- 格式转换
- 重采样至16kHz
- 降噪
- 音量归一化
- 静音检测与裁剪
"""
import os
import logging
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """音频预处理器"""

    def __init__(self, config):
        self.config = config
        self._check_dependencies()

    def _check_dependencies(self):
        """检查依赖库"""
        try:
            import librosa
            import soundfile
            self.librosa = librosa
            self.soundfile = soundfile
        except ImportError:
            raise ImportError("请安装: pip install librosa soundfile")

        try:
            import noisereduce
            self.noisereduce = noisereduce
        except ImportError:
            logger.warning("noisereduce未安装，降噪功能不可用: pip install noisereduce")
            self.noisereduce = None

    def process(self, audio_path: str) -> Tuple[str, dict]:
        """
        完整音频预处理流程

        Args:
            audio_path: 输入音频文件路径

        Returns:
            processed_path: 处理后的音频文件路径
            metadata: 音频元数据
        """
        logger.info(f"开始处理音频: {audio_path}")

        # 1. 验证文件
        self._validate_file(audio_path)

        # 2. 加载音频
        audio, sr = self._load_audio(audio_path)

        # 3. 转为单声道
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
            logger.info("已转换为单声道")

        # 4. 重采样
        if sr != self.config.target_sample_rate:
            audio = self.librosa.resample(
                audio,
                orig_sr=sr,
                target_sr=self.config.target_sample_rate
            )
            sr = self.config.target_sample_rate
            logger.info(f"已重采样至 {sr}Hz")

        # 5. 降噪
        if self.config.noise_reduce and self.noisereduce:
            audio = self._reduce_noise(audio, sr)

        # 6. 音量归一化
        if self.config.normalize:
            audio = self._normalize_volume(audio)

        # 7. 静音裁剪
        audio = self._trim_silence(audio, sr)

        # 8. 时长验证
        duration = len(audio) / sr
        if duration < self.config.min_duration_seconds:
            raise ValueError(f"音频太短: {duration:.2f}秒 < {self.config.min_duration_seconds}秒")
        if duration > self.config.max_duration_seconds:
            audio = audio[:int(self.config.max_duration_seconds * sr)]
            logger.warning(f"音频已截断至 {self.config.max_duration_seconds}秒")
            duration = self.config.max_duration_seconds

        # 9. 保存处理后的音频
        output_path = self._save_processed(audio, sr, audio_path)

        metadata = {
            "original_path": audio_path,
            "processed_path": output_path,
            "sample_rate": sr,
            "duration": round(duration, 2),
            "samples": len(audio),
            "rms_db": round(float(20 * np.log10(np.sqrt(np.mean(audio**2)) + 1e-10)), 2),
        }

        logger.info(f"音频处理完成: 时长={duration:.2f}s, 采样率={sr}Hz")
        return output_path, metadata

    def _validate_file(self, path: str):
        """验证音频文件"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"音频文件不存在: {path}")

        ext = Path(path).suffix.lower()
        if ext not in self.config.supported_formats:
            raise ValueError(f"不支持的格式 {ext}，支持: {self.config.supported_formats}")

        file_size = os.path.getsize(path)
        if file_size == 0:
            raise ValueError("音频文件为空")
        if file_size > 50 * 1024 * 1024:  # 50MB
            raise ValueError(f"文件过大: {file_size / 1024 / 1024:.1f}MB > 50MB")

    def _load_audio(self, path: str):
        """加载音频文件，支持 webm/m4a 等格式"""
        import subprocess
        import tempfile
        from pathlib import Path

        ext = Path(path).suffix.lower()

        # webm/m4a/ogg 等格式先用 ffmpeg 转成 wav
        if ext in ['.webm', '.m4a', '.ogg', '.opus', '.aac', '.wma']:
            wav_path = os.path.join(
                tempfile.gettempdir(),
                Path(path).stem + "_converted.wav"
            )
            try:
                result = subprocess.run(
                    [
                        'ffmpeg', '-y',
                        '-i', path,
                        '-ar', '16000',
                        '-ac', '1',
                        '-f', 'wav',
                        wav_path
                    ],
                    capture_output=True,
                    timeout=30
                )
                if result.returncode != 0:
                    raise RuntimeError(f"ffmpeg错误: {result.stderr.decode()}")

                logger.info(f"格式转换: {ext} -> .wav")
                path = wav_path

            except FileNotFoundError:
                raise RuntimeError(
                    "需要安装 ffmpeg！\n"
                    "运行: conda install ffmpeg -c conda-forge"
                )

        # 加载 wav/mp3/flac
        try:
            audio, sr = self.librosa.load(path, sr=None, mono=False)
            return audio, sr
        except Exception as e:
            raise RuntimeError(f"无法加载音频: {e}")

    def _reduce_noise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """降噪处理"""
        try:
            reduced = self.noisereduce.reduce_noise(
                y=audio,
                sr=sr,
                prop_decrease=0.7,
                stationary=True
            )
            logger.info("降噪处理完成")
            return reduced
        except Exception as e:
            logger.warning(f"降噪失败，使用原始音频: {e}")
            return audio

    def _normalize_volume(self, audio: np.ndarray) -> np.ndarray:
        """音量归一化到 [-1, 1]"""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.95
        return audio

    def _trim_silence(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """裁剪首尾静音"""
        try:
            trimmed, _ = self.librosa.effects.trim(
                audio,
                top_db=25,
                frame_length=2048,
                hop_length=512
            )
            logger.info(f"静音裁剪: {len(audio)/sr:.2f}s -> {len(trimmed)/sr:.2f}s")
            return trimmed
        except Exception:
            return audio

    def _save_processed(self, audio: np.ndarray, sr: int, original_path: str) -> str:
        """保存处理后的音频为WAV"""
        stem = Path(original_path).stem
        output_path = os.path.join(
            tempfile.gettempdir(),
            f"{stem}_processed.wav"
        )
        self.soundfile.write(output_path, audio, sr, subtype='PCM_16')
        return output_path