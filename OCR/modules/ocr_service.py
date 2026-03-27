"""
OCR识别模块（增强版）
"""
import re
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class OCRService:
    """OCR识别服务（增强版）"""

    def __init__(self, config):
        self.config = config
        self.engine = None
        self.backup_engine = None
        self._engine_name = ""
        self.enhancer = None

    def load_model(self):
        if self.engine is not None:
            return

        # 加载OCR增强器
        try:
            from modules.ocr_enhancer import OCREnhancer
            self.enhancer = OCREnhancer()
            logger.info("OCR增强器已加载")
        except Exception as e:
            logger.warning(f"OCR增强器加载失败: {e}")

        # EasyOCR
        try:
            import easyocr
            self.engine = easyocr.Reader(
                ['ch_sim', 'en'],
                gpu=False,
                verbose=False
            )
            self._engine_name = "EasyOCR"
            logger.info("EasyOCR 加载成功")
            return
        except ImportError:
            logger.info("EasyOCR 不可用")
        except Exception as e:
            logger.warning(f"EasyOCR GPU模式失败，尝试CPU: {e}")
            try:
                import easyocr
                self.engine = easyocr.Reader(['ch_sim', 'en'], gpu=False, verbose=False)
                self._engine_name = "EasyOCR"
                logger.info("EasyOCR 加载成功 (CPU)")
                return
            except Exception:
                pass

        # PaddleOCR
        try:
            from paddleocr import PaddleOCR
            self.engine = PaddleOCR(lang='ch')
            self._engine_name = "PaddleOCR"
            logger.info("PaddleOCR 加载成功")
            return
        except Exception as e:
            logger.warning(f"PaddleOCR 加载失败: {e}")

        raise ImportError("没有可用的OCR引擎")

    def recognize(self, image_path: str, use_enhancement: bool = True) -> Dict:
        """
        识别图像中的文字（增强版）
        """
        self.load_model()
        logger.info(f"开始OCR识别: {image_path}")

        import cv2
        import numpy as np

        # 读取图像
        if isinstance(image_path, np.ndarray):
            img = image_path
        else:
            img = cv2.imread(image_path)
            if img is None:
                logger.warning(f"无法读取: {image_path}")
                return self._empty_result()

        # 是否使用增强识别
        if use_enhancement and self.enhancer:
            result = self._enhanced_recognize(img)
        else:
            result = self._basic_recognize(img)

        logger.info(f"OCR结果: id='{result.get('extracted_id')}', "
                     f"confidence={result.get('confidence', 0):.4f}")
        return result

    def _basic_recognize(self, img) -> dict:
        """基础识别（加旋转重试）"""
        import cv2

        # 第一次：原图识别
        if self._engine_name == "EasyOCR":
            texts = self._recognize_easyocr(img)
        elif self._engine_name == "PaddleOCR":
            texts = self._recognize_paddle(img)
        else:
            texts = []

        student_id, confidence, bbox = self._extract_student_id(texts)

        # 如果原图没识别到学号，尝试旋转
        if not student_id:
            logger.info("原图未识别到学号，尝试旋转识别...")

            rotations = [
                ("顺时针90°", cv2.ROTATE_90_CLOCKWISE),
                ("逆时针90°", cv2.ROTATE_90_COUNTERCLOCKWISE),
                ("180°", cv2.ROTATE_180),
            ]

            for name, rotation in rotations:
                rotated = cv2.rotate(img, rotation)

                if self._engine_name == "EasyOCR":
                    rot_texts = self._recognize_easyocr(rotated)
                elif self._engine_name == "PaddleOCR":
                    rot_texts = self._recognize_paddle(rotated)
                else:
                    rot_texts = []

                rot_id, rot_conf, rot_bbox = self._extract_student_id(rot_texts)

                if rot_id:
                    logger.info(f"旋转{name}后识别到学号: {rot_id}")
                    texts.extend(rot_texts)
                    student_id = rot_id
                    confidence = rot_conf
                    bbox = rot_bbox
                    break

        return {
            'all_texts': texts,
            'extracted_id': student_id or "",
            'confidence': round(float(confidence), 4),
            'id_bbox': bbox,
            'engine': self._engine_name,
            'total_text_blocks': len(texts),
            'method': 'basic',
        }

    def _enhanced_recognize(self, img) -> Dict:
        """增强识别：多版本预处理 + 投票"""
        # 1. 先尝试透视校正
        try:
            corrected = self.enhancer.correct_perspective(img)
        except Exception:
            corrected = img

        # 2. 多版本投票识别
        vote_result = self.enhancer.recognize_with_voting(
            corrected, self.engine, self._engine_name
        )

        best_id = vote_result.get('best_id', '')
        confidence = vote_result.get('confidence', 0)
        all_texts = vote_result.get('all_texts', [])

        # 3. 后处理校正
        if best_id:
            corrected_id, notes = self.enhancer.post_correct(best_id)
            if corrected_id != best_id:
                logger.info(f"后处理校正: '{best_id}' → '{corrected_id}'")
                best_id = corrected_id

        # 4. 如果增强识别没结果，回退到基础识别
        if not best_id:
            logger.info("增强识别未找到学号，回退到基础识别")
            basic_result = self._basic_recognize(img)
            basic_result['method'] = 'basic_fallback'
            return basic_result

        # 去重文本
        unique_texts = []
        seen = set()
        for t in all_texts:
            key = t['text'].strip()
            if key and key not in seen:
                seen.add(key)
                unique_texts.append(t)

        return {
            'all_texts': unique_texts[:15],
            'extracted_id': best_id,
            'confidence': round(float(confidence), 4),
            'id_bbox': None,
            'engine': self._engine_name,
            'total_text_blocks': len(unique_texts),
            'method': 'enhanced_voting',
            'vote_count': vote_result.get('vote_count', 0),
            'total_versions': vote_result.get('total_versions', 0),
            'candidates': vote_result.get('all_candidates', []),
        }

    def _recognize_easyocr(self, image_path_or_array) -> list:
        """EasyOCR识别"""
        import cv2
        import numpy as np

        if isinstance(image_path_or_array, np.ndarray):
            img = image_path_or_array
        elif isinstance(image_path_or_array, str):
            img = cv2.imread(image_path_or_array)
            if img is None:
                return []
        else:
            return []

        results = self.engine.readtext(img)
        texts = []
        for (bbox, text, confidence) in results:
            clean_bbox = [[int(p[0]), int(p[1])] for p in bbox]
            texts.append({
                'text': str(text),
                'confidence': float(confidence),
                'bbox': clean_bbox,
                'source': 'EasyOCR'
            })
        return texts

    def _recognize_paddle(self, image_path_or_array) -> list:
        """PaddleOCR识别"""
        results = self.engine.ocr(image_path_or_array, cls=True)
        texts = []
        if results and results[0]:
            for line in results[0]:
                texts.append({
                    'text': line[1][0],
                    'confidence': float(line[1][1]),
                    'bbox': line[0],
                    'source': 'PaddleOCR'
                })
        return texts

    def _extract_student_id(self, texts: list):
        """从OCR结果中提取学号"""
        import re

        # 关键词列表
        id_keywords = [
            '学号', '学 号', 'Student ID', 'ID', 'No.', '编号', '学生编号',
            'Stu ID', 'SID', '学籍号', '考号',
            '卡号', '卡 号', 'Card No', 'Card ID',
            '工号', '职工号', '员工号',
            '证号', '证件号', '账号',
            'Number', 'NO',
        ]

        # 排除词
        exclude_keywords = [
            '电话', '手机', 'Tel', 'Phone', 'Fax',
            '邮编', '邮政', 'Zip',
            '有效期', '日期', 'Date',
            '校园卡通中心',
        ]

        # 学号正则
        id_patterns = [
            r'[A-Za-z]{1,4}\d{6,12}',
            r'[A-Za-z]?\d{8,12}',
            r'20[12]\d{5,8}',
            r'\d{6,12}',
        ]

        # ====== 第0步：过滤干扰文本 ======
        filtered_texts = []
        for item in texts:
            text = item['text']
            is_excluded = False

            # 检查排除关键词
            for ex_kw in exclude_keywords:
                if ex_kw.lower() in text.lower():
                    is_excluded = True
                    break

            # 排除电话号码格式
            if re.match(r'^\d{3,4}[-\s]\d{7,8}$', text.strip()):
                is_excluded = True

            if not is_excluded:
                filtered_texts.append(item)

        # ====== 第1步：关键词定位 ======
        for item in filtered_texts:
            text = item['text']
            for kw in id_keywords:
                if kw.lower() in text.lower():
                    idx = text.lower().find(kw.lower())
                    after = text[idx + len(kw):]
                    after = re.sub(r'^[\s:：\-_]+', '', after)
                    for pattern in id_patterns:
                        m = re.search(pattern, after)
                        if m:
                            candidate = self._clean_id(m.group())
                            if not self._is_phone_number(candidate):
                                return candidate, item['confidence'], item.get('bbox')

        # ====== 第2步：空间关联 ======
        kw_items = []
        num_items = []
        for item in filtered_texts:
            is_kw = any(kw.lower() in item['text'].lower() for kw in id_keywords)
            if is_kw:
                kw_items.append(item)
            for p in id_patterns[:3]:
                if re.search(p, item['text']):
                    num_items.append(item)
                    break

        if kw_items and num_items:
            for ki in kw_items:
                kc = self._bbox_center(ki.get('bbox'))
                best = None
                best_d = float('inf')
                for ni in num_items:
                    nc = self._bbox_center(ni.get('bbox'))
                    d = ((kc[0] - nc[0]) ** 2 + (kc[1] - nc[1]) ** 2) ** 0.5
                    if d < best_d:
                        best_d = d
                        best = ni
                if best:
                    for p in id_patterns:
                        m = re.search(p, best['text'])
                        if m:
                            candidate = self._clean_id(m.group())
                            if not self._is_phone_number(candidate):
                                return candidate, best['confidence'], best.get('bbox')

        # ====== 第3步：格式匹配 ======
        candidates = []
        for item in filtered_texts:
            for p in id_patterns[:3]:
                for m in re.finditer(p, item['text']):
                    candidate = self._clean_id(m.group())
                    if not self._is_phone_number(candidate):
                        candidates.append({
                            'id': candidate,
                            'confidence': item['confidence'],
                            'bbox': item.get('bbox'),
                            'length': len(m.group()),
                        })

        if candidates:
            candidates.sort(key=lambda x: (x['length'], x['confidence']), reverse=True)
            best = candidates[0]
            return best['id'], best['confidence'], best['bbox']

        # ====== 第4步：单文本块 ======
        if len(filtered_texts) == 1:
            text = filtered_texts[0]['text'].strip()
            cleaned = re.sub(r'[\s\-_:：]', '', text)
            if 4 <= len(cleaned) <= 15 and any(c.isdigit() for c in cleaned):
                if not self._is_phone_number(cleaned):
                    return (self._clean_id(cleaned),
                            filtered_texts[0]['confidence'],
                            filtered_texts[0].get('bbox'))

        return None, 0.0, None

    def _is_phone_number(self, text: str) -> bool:
        """判断是否为电话号码"""
        import re
        cleaned = re.sub(r'[\s\-_]', '', text)

        # 固话：0xxx-xxxxxxx
        if re.match(r'^0\d{2,3}\d{7,8}$', cleaned):
            return True

        # 手机号：1xx xxxx xxxx
        if re.match(r'^1[3-9]\d{9}$', cleaned):
            return True

        # 400/800
        if re.match(r'^[48]00\d{7,8}$', cleaned):
            return True

        return False

    def _clean_id(self, id_str: str) -> str:
        """清理学号"""
        import re
        return re.sub(r'[\s\-_]', '', id_str).upper()

    def _bbox_center(self, bbox):
        """获取bbox中心点"""
        try:
            if isinstance(bbox[0], list):
                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
            else:
                xs = [bbox[0], bbox[2]]
                ys = [bbox[1], bbox[3]]
            return (sum(xs) / len(xs), sum(ys) / len(ys))
        except Exception:
            return (0, 0)

    def _empty_result(self):
        return {
            'all_texts': [], 'extracted_id': '', 'confidence': 0,
            'id_bbox': None, 'engine': self._engine_name,
            'total_text_blocks': 0, 'method': 'empty',
        }