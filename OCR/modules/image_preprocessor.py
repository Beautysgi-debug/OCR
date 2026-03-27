import os
import logging
from pathlib import Path
from typing import Tuple, Optional
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class ImagePreprocessor:
    def __init__(self, config):
        self.config = config

    def process(self, image_path: str):
        logger.info(f"开始处理图像: {image_path}")
        
        # 兼容中文路径的读取
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            logger.warning(f"无法读取图像: {image_path}")
            return image_path, {"note": "无法读取图像，使用原图"}

        try:
            image = self._resize_if_needed(image)
        except Exception as e:
            logger.warning(f"resize失败: {e}")

        output_path = self._save_processed(image, image_path)
        return output_path, {"original_path": image_path, "processed_path": output_path}

    def _resize_if_needed(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        max_h, max_w = self.config.max_image_size
        min_h, min_w = self.config.min_image_size

        if h < min_h or w < min_w:
            scale = max(min_h / h, min_w / w)
            new_w, new_h = int(w * scale), int(h * scale)
            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        elif h > max_h or w > max_w:
            scale = min(max_h / h, max_w / w)
            new_w, new_h = int(w * scale), int(h * scale)
            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return image

    def _save_processed(self, image, original_path, suffix="_processed"):
        stem = Path(original_path).stem
        output_dir = Path(original_path).parent
        output_path = str(output_dir / f"{stem}{suffix}.png")
        
        # 兼容中文路径的保存
        try:
            cv2.imencode('.png', image)[1].tofile(output_path)
            return output_path
        except Exception as e:
            logger.warning(f"保存失败: {e}")
            return original_path