"""
OCR识别模块
基于 PaddleOCR，带有Tesseract备选
"""
import re
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class OCRService:
    """OCR识别服务"""

    def __init__(self, config):
        self.config = config
        self.engine = None
        self.backup_engine = None

    def load_model(self):
        """延迟加载OCR模型"""
        if self.engine is not None:
            return

        # ====== 优先用 EasyOCR（更稳定）======
        try:
            import easyocr
            self.engine = easyocr.Reader(
                ['ch_sim', 'en'],
                gpu=True,  # 有GPU就用GPU
                verbose=False
            )
            self._engine_name = "EasyOCR"
            logger.info("EasyOCR 加载成功")
            return
        except ImportError:
            logger.info("EasyOCR 不可用")
        except Exception as e:
            logger.warning(f"EasyOCR 加载失败: {e}")
            # GPU不可用时用CPU重试
            try:
                import easyocr
                self.engine = easyocr.Reader(
                    ['ch_sim', 'en'],
                    gpu=False,
                    verbose=False
                )
                self._engine_name = "EasyOCR"
                logger.info("EasyOCR 加载成功 (CPU模式)")
                return
            except Exception:
                pass

        # ====== 备选：PaddleOCR ======
        try:
            from paddleocr import PaddleOCR
            self.engine = PaddleOCR(lang='ch')
            self._engine_name = "PaddleOCR"
            logger.info("PaddleOCR 加载成功")
            return
        except Exception as e:
            logger.warning(f"PaddleOCR 加载失败: {e}")

        raise ImportError(
            "没有可用的OCR引擎！请安装:\n"
            "  pip install easyocr\n"
            "  或 pip install paddlepaddle paddleocr"
        )

    def recognize(self, image_path: str) -> Dict:
        """
        识别图像中的文字

        Returns:
            dict: {
                'all_texts': 所有识别到的文本,
                'extracted_id': 提取的学号,
                'confidence': 置信度,
                'id_bbox': 学号区域坐标,
                'engine': 使用的引擎
            }
        """
        self.load_model()
        logger.info(f"开始OCR识别: {image_path}")

        # 主引擎识别
        if self._engine_name == "PaddleOCR":
            texts = self._recognize_paddle(image_path)
        else:
            texts = self._recognize_easyocr(image_path)

        # 提取学号
        student_id, id_confidence, id_bbox = self._extract_student_id(texts)

        # 如果主引擎未提取到，尝试备选引擎
        if not student_id and self.backup_engine:
            logger.info("主引擎未找到学号，尝试Tesseract")
            backup_texts = self._recognize_tesseract(image_path)
            texts.extend(backup_texts)
            student_id, id_confidence, id_bbox = self._extract_student_id(texts)

        result = {
            'all_texts': texts,
            'extracted_id': student_id or "",
            'confidence': round(id_confidence, 4),
            'id_bbox': id_bbox,
            'engine': self._engine_name,
            'total_text_blocks': len(texts),
        }

        logger.info(f"OCR结果: id='{student_id}', confidence={id_confidence:.4f}, "
                     f"共识别 {len(texts)} 个文本块")
        return result

    def _recognize_easyocr(self, image_path: str) -> list:
        """EasyOCR识别"""
        import cv2
        import numpy as np

        img = cv2.imread(image_path)
        if img is None:
            logger.warning(f"cv2无法读取图片: {image_path}")
            return []

        results = self.engine.readtext(img)
        texts = []
        for (bbox, text, confidence) in results:
            # ====== 把 numpy 类型全部转成 Python 原生类型 ======
            clean_bbox = []
            for point in bbox:
                clean_bbox.append([int(point[0]), int(point[1])])
            # =================================================

            texts.append({
                'text': str(text),
                'confidence': float(confidence),
                'bbox': clean_bbox,
                'source': 'EasyOCR'
            })
        return texts

    def _recognize_easyocr(self, image_path: str) -> List[Dict]:
        """EasyOCR识别"""
        results = self.engine.readtext(image_path)
        texts = []

        for (bbox, text, confidence) in results:
            texts.append({
                'text': text,
                'confidence': float(confidence),
                'bbox': bbox,
                'source': 'EasyOCR'
            })

        return texts

    def _recognize_tesseract(self, image_path: str) -> List[Dict]:
        """Tesseract识别"""
        try:
            from PIL import Image
            img = Image.open(image_path)

            # 中文+英文+数字
            data = self.backup_engine.image_to_data(
                img,
                lang='chi_sim+eng',
                output_type=self.backup_engine.Output.DICT,
                config='--psm 6'
            )

            texts = []
            n = len(data['text'])
            for i in range(n):
                text = data['text'][i].strip()
                conf = int(data['conf'][i])
                if text and conf > 30:
                    texts.append({
                        'text': text,
                        'confidence': conf / 100.0,
                        'bbox': [
                            [data['left'][i], data['top'][i]],
                            [data['left'][i] + data['width'][i], data['top'][i]],
                            [data['left'][i] + data['width'][i], data['top'][i] + data['height'][i]],
                            [data['left'][i], data['top'][i] + data['height'][i]],
                        ],
                        'source': 'Tesseract'
                    })
            return texts

        except Exception as e:
            logger.warning(f"Tesseract识别失败: {e}")
            return []

    def _extract_student_id(self, texts: List[Dict]) -> Tuple[Optional[str], float, Optional[list]]:
        """
        从OCR结果中提取学号

        策略:
        1. 关键词定位: 找到"学号"/"Student ID"旁边的数字
        2. 格式匹配: 匹配学号格式的数字串
        3. 空间定位: 在"学号"标签右侧或下方找数字
        """

        # === 策略1: 关键词定位 ===
        id_keywords = ['学号', '学 号', 'Student ID', 'ID', 'No.', '编号', '学生编号']

        for item in texts:
            text = item['text']
            for keyword in id_keywords:
                if keyword.lower() in text.lower():
                    # 在同一文本块中查找数字
                    after_keyword = text[text.lower().find(keyword.lower()) + len(keyword):]
                    numbers = re.findall(r'[A-Za-z]?\d{6,12}', after_keyword)
                    if numbers:
                        return self._clean_id(numbers[0]), item['confidence'], item['bbox']

        # === 策略2: 空间关联 - 找"学号"标签附近的数字 ===
        keyword_items = []
        number_items = []

        for item in texts:
            text = item['text']
            is_keyword = any(kw.lower() in text.lower() for kw in id_keywords)
            if is_keyword:
                keyword_items.append(item)

            # 检查是否为数字串
            for pattern in self.config.id_patterns:
                if re.search(pattern, text):
                    number_items.append(item)
                    break

        if keyword_items and number_items:
            # 找离关键词最近的数字
            for kw_item in keyword_items:
                kw_center = self._get_bbox_center(kw_item['bbox'])
                best_num = None
                best_dist = float('inf')

                for num_item in number_items:
                    num_center = self._get_bbox_center(num_item['bbox'])
                    dist = ((kw_center[0] - num_center[0])**2 +
                            (kw_center[1] - num_center[1])**2) ** 0.5
                    if dist < best_dist:
                        best_dist = dist
                        best_num = num_item

                if best_num:
                    id_str = re.findall(r'[A-Za-z]?\d{6,12}', best_num['text'])
                    if id_str:
                        return self._clean_id(id_str[0]), best_num['confidence'], best_num['bbox']

        # === 策略3: 纯格式匹配 - 在所有文本中找符合学号格式的 ===
        candidates = []
        for item in texts:
            text = item['text']
            for pattern in self.config.id_patterns:
                matches = re.findall(pattern, text)
                for m in matches:
                    candidates.append({
                        'id': self._clean_id(m),
                        'confidence': item['confidence'],
                        'bbox': item['bbox'],
                        'length': len(m),
                    })

        if candidates:
            # 优先选择更长的、置信度更高的
            candidates.sort(key=lambda x: (x['length'], x['confidence']), reverse=True)
            best = candidates[0]
            return best['id'], best['confidence'], best['bbox']

        return None, 0.0, None

    def _clean_id(self, id_str: str) -> str:
        """清理学号字符串"""
        # 去除空格
        cleaned = id_str.replace(' ', '').replace('-', '').replace('_', '')
        # 大写字母前缀
        cleaned = cleaned.upper()
        return cleaned

    def _get_bbox_center(self, bbox) -> Tuple[float, float]:
        """获取bbox中心点"""
        try:
            if isinstance(bbox[0], list):
                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
            else:
                xs = [bbox[0], bbox[2]]
                ys = [bbox[1], bbox[3]]
            return (sum(xs) / len(xs), sum(ys) / len(ys))
        except (IndexError, TypeError):
            return (0, 0)