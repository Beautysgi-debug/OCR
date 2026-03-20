"""
图像预处理模块
- 尺寸调整
- 倾斜校正
- 去噪
- 对比度增强
- ROI区域检测
"""
import os
import logging
import tempfile
from pathlib import Path
from typing import Tuple, Optional, List

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """图像预处理器"""

    def __init__(self, config):
        self.config = config

    def process(self, image_path: str):
        logger.info(f"开始处理图像: {image_path}")

        # 1. 验证文件
        self._validate_file(image_path)

        # 2. 加载图像
        image = cv2.imread(image_path)
        if image is None:
            # 读取失败，直接返回原图路径
            logger.warning(f"cv2.imread 无法读取: {image_path}，返回原图")
            return image_path, {
                "original_path": image_path,
                "processed_path": image_path,
                "note": "无法读取图像，使用原图"
            }

        original_shape = image.shape[:2]

        # 3-6 各处理步骤加 try-except
        try:
            image = self._resize_if_needed(image)
        except Exception as e:
            logger.warning(f"resize失败: {e}")

        if self.config.enable_deskew:
            try:
                image, angle = self._deskew(image)
            except Exception as e:
                logger.warning(f"deskew失败: {e}")
                angle = 0.0
        else:
            angle = 0.0

        if self.config.enable_denoise:
            try:
                image = self._denoise(image)
            except Exception as e:
                logger.warning(f"denoise失败: {e}")

        if self.config.enable_contrast_enhance:
            try:
                image = self._enhance_contrast(image)
            except Exception as e:
                logger.warning(f"enhance失败: {e}")

        # 7. 保存
        try:
            output_path = self._save_processed(image, image_path)
        except Exception as e:
            logger.warning(f"保存处理图像失败: {e}，使用原图")
            output_path = image_path

        metadata = {
            "original_path": image_path,
            "processed_path": output_path,
            "original_size": original_shape,
            "processed_size": image.shape[:2],
            "deskew_angle": round(angle, 2),
        }

        logger.info(f"图像处理完成")
        return output_path, metadata

    def _validate_file(self, path: str):
        """验证图像文件"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"图像文件不存在: {path}")

        ext = Path(path).suffix.lower()
        if ext not in self.config.supported_formats:
            raise ValueError(f"不支持的格式: {ext}")

        file_size = os.path.getsize(path)
        if file_size == 0:
            raise ValueError("图像文件为空")
        if file_size > 20 * 1024 * 1024:
            raise ValueError(f"文件过大: {file_size / 1024 / 1024:.1f}MB > 20MB")

    def _resize_if_needed(self, image: np.ndarray) -> np.ndarray:
        """必要时调整尺寸"""
        h, w = image.shape[:2]
        max_h, max_w = self.config.max_image_size
        min_h, min_w = self.config.min_image_size

        if h < min_h or w < min_w:
            # 图像太小，放大
            scale = max(min_h / h, min_w / w)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            logger.info(f"放大图像: ({w},{h}) -> ({new_w},{new_h})")

        elif h > max_h or w > max_w:
            # 图像太大，缩小
            scale = min(max_h / h, max_w / w)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            logger.info(f"缩小图像: ({w},{h}) -> ({new_w},{new_h})")

        return image

    def _deskew(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """倾斜校正"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.bitwise_not(gray)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

            coords = np.column_stack(np.where(thresh > 0))
            if len(coords) < 10:
                return image, 0.0

            angle = cv2.minAreaRect(coords)[-1]

            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle

            # 只在倾斜角度较小时校正（避免误校正）
            if abs(angle) > 15:
                logger.warning(f"倾斜角度过大({angle:.1f}°)，跳过校正")
                return image, 0.0

            if abs(angle) > 0.5:
                h, w = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(
                    image, M, (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE
                )
                logger.info(f"倾斜校正: {angle:.2f}°")
                return rotated, angle

            return image, angle

        except Exception as e:
            logger.warning(f"倾斜校正失败: {e}")
            return image, 0.0

    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """去噪"""
        try:
            denoised = cv2.fastNlMeansDenoisingColored(
                image, None,
                h=6, hForColorComponents=6,
                templateWindowSize=7,
                searchWindowSize=21
            )
            return denoised
        except Exception:
            return image

    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """对比度增强 - CLAHE"""
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel, a, b = cv2.split(lab)

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_l = clahe.apply(l_channel)

            merged = cv2.merge([enhanced_l, a, b])
            result = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
            return result
        except Exception:
            return image

    def _prepare_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        为OCR专门优化的图像处理
        """
        # 转灰度
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 自适应二值化
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=15,
            C=8
        )

        # 轻度膨胀使文字更清晰
        kernel = np.ones((1, 1), np.uint8)
        processed = cv2.dilate(binary, kernel, iterations=1)

        # 转回3通道（PaddleOCR需要）
        result = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        return result

    def _save_processed(self, image, original_path, suffix="_processed"):
        """保存处理后的图像"""
        import numpy as np

        stem = Path(original_path).stem

        # 保存到 uploads 目录而不是 temp（避免权限问题）
        output_dir = Path(original_path).parent
        output_path = str(output_dir / f"{stem}{suffix}.png")

        try:
            success = cv2.imwrite(output_path, image)
            if success:
                logger.info(f"图像已保存: {output_path}")
                return output_path
            else:
                logger.warning(f"cv2.imwrite 返回 False")
                return original_path
        except Exception as e:
            logger.warning(f"保存失败: {e}")
            return original_path

    def detect_card_region(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        检测学生证/ID卡区域 (ROI)
        使用轮廓检测找到卡片区域
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 200)

        contours, _ = cv2.findContours(
            edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        # 按面积排序，找最大的矩形轮廓
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in contours[:5]:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            if len(approx) == 4:
                # 透视变换
                pts = approx.reshape(4, 2)
                card = self._four_point_transform(image, pts)
                return card

        return None

    def _four_point_transform(self, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """四点透视变换"""
        rect = self._order_points(pts.astype(np.float32))
        (tl, tr, br, bl) = rect

        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0], [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """按 左上、右上、右下、左下 排列四个角点"""
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        d = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(d)]
        rect[3] = pts[np.argmax(d)]
        return rect