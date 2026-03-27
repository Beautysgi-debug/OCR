"""
OCR增强模块
- 高级图像预处理
- 多引擎融合
- 结果后处理与校正
- LLM辅助纠错
"""
import re
import logging
from typing import Dict, List, Optional, Tuple
from collections import Counter

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class OCREnhancer:
    """OCR增强器"""

    def __init__(self):
        self.correction_map = {
            'O': '0', 'o': '0', 'D': '0',
            'l': '1', 'I': '1', 'i': '1', '|': '1',
            'Z': '2', 'z': '2',
            'S': '5', 's': '5',
            'B': '8', 'b': '8',
            'G': '6', 'g': '9', 'q': '9',
        }

    # ============================================================
    # 1. 高级图像预处理
    # ============================================================
    def enhance_image(self, image: np.ndarray) -> List[np.ndarray]:
        """
        生成多种预处理版本，提高识别成功率
        返回多张处理后的图片，每张用不同方法处理
        """
        results = []

        # 版本1：原图（基准）
        results.append(image.copy())

        # 版本2：灰度 + OTSU二值化
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results.append(cv2.cvtColor(binary_otsu, cv2.COLOR_GRAY2BGR))

        # 版本3：灰度 + 自适应二值化
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 8
        )
        results.append(cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR))

        # 版本4：CLAHE增强 + 锐化
        enhanced = self._clahe_enhance(image)
        sharpened = self._sharpen(enhanced)
        results.append(sharpened)

        # 版本5：放大2倍（小图片时特别有用）
        h, w = image.shape[:2]
        if h < 500 or w < 500:
            upscaled = cv2.resize(image, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
            # 放大后再锐化
            upscaled = self._sharpen(upscaled)
            results.append(upscaled)

        # 版本6：去噪 + 形态学处理
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 6, 6, 7, 21)
        results.append(denoised)

        # 版本7：反色（白底黑字 → 黑底白字）
        inverted = cv2.bitwise_not(image)
        results.append(inverted)

        # ====== 新增：旋转版本（处理竖排文字）======

        # 顺时针旋转90度
        rotated_cw = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        results.append(rotated_cw)

        # 逆时针旋转90度
        rotated_ccw = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        results.append(rotated_ccw)

        # 旋转90度 + CLAHE增强
        rotated_enhanced = self._clahe_enhance(rotated_cw)
        results.append(rotated_enhanced)

        logger.info(f"生成了 {len(results)} 种预处理版本（含旋转）")
        return results


    def _clahe_enhance(self, image: np.ndarray) -> np.ndarray:
        """CLAHE对比度增强"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    def _sharpen(self, image: np.ndarray) -> np.ndarray:
        """锐化"""
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=np.float32)
        return cv2.filter2D(image, -1, kernel)

    def correct_perspective(self, image: np.ndarray) -> np.ndarray:
        """
        透视校正：检测卡片四边，做透视变换
        适用于拍照角度倾斜的情况
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 30, 150)

        # 膨胀边缘使其连续
        kernel = np.ones((3, 3), np.uint8)
        edged = cv2.dilate(edged, kernel, iterations=2)

        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image

        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for c in contours[:5]:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            if len(approx) == 4:
                # 检查面积是否足够大（至少占图片20%）
                area = cv2.contourArea(approx)
                img_area = image.shape[0] * image.shape[1]
                if area > img_area * 0.2:
                    pts = approx.reshape(4, 2).astype(np.float32)
                    warped = self._four_point_transform(image, pts)
                    logger.info("透视校正成功")
                    return warped

        return image

    def _four_point_transform(self, image, pts):
        """四点透视变换"""
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # 左上
        rect[2] = pts[np.argmax(s)]  # 右下
        d = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(d)]  # 右上
        rect[3] = pts[np.argmax(d)]  # 左下

        (tl, tr, br, bl) = rect
        w = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
        h = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))

        dst = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, M, (w, h))

    # ============================================================
    # 2. 多引擎融合识别
    # ============================================================
    def multi_engine_recognize(self, image: np.ndarray, engines: dict) -> List[Dict]:
        """
        用多个引擎识别同一张图，融合结果

        Args:
            image: 图像
            engines: {'easyocr': engine, 'tesseract': engine, ...}

        Returns:
            融合后的文本列表
        """
        all_results = {}

        for engine_name, engine in engines.items():
            try:
                if engine_name == 'easyocr':
                    texts = self._run_easyocr(image, engine)
                elif engine_name == 'tesseract':
                    texts = self._run_tesseract(image, engine)
                else:
                    continue

                all_results[engine_name] = texts
                logger.info(f"{engine_name}: 识别到 {len(texts)} 个文本块")

            except Exception as e:
                logger.warning(f"{engine_name} 识别失败: {e}")

        # 融合结果
        if not all_results:
            return []

        return self._merge_results(all_results)

    def _run_easyocr(self, image, engine):
        results = engine.readtext(image)
        texts = []
        for (bbox, text, conf) in results:
            texts.append({
                'text': str(text),
                'confidence': float(conf),
                'bbox': [[int(p[0]), int(p[1])] for p in bbox],
                'source': 'EasyOCR'
            })
        return texts

    def _run_tesseract(self, image, engine):
        """运行Tesseract"""
        try:
            from PIL import Image
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            data = engine.image_to_data(
                pil_img, lang='chi_sim+eng',
                output_type=engine.Output.DICT, config='--psm 6'
            )
            texts = []
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                conf = int(data['conf'][i])
                if text and conf > 30:
                    texts.append({
                        'text': text,
                        'confidence': conf / 100.0,
                        'bbox': [[data['left'][i], data['top'][i]],
                                 [data['left'][i]+data['width'][i], data['top'][i]],
                                 [data['left'][i]+data['width'][i], data['top'][i]+data['height'][i]],
                                 [data['left'][i], data['top'][i]+data['height'][i]]],
                        'source': 'Tesseract'
                    })
            return texts
        except Exception as e:
            logger.warning(f"Tesseract失败: {e}")
            return []

    def _merge_results(self, all_results: dict) -> List[Dict]:
        """融合多引擎结果"""
        merged = []
        seen_texts = set()

        for engine_name, texts in all_results.items():
            for t in texts:
                # 简单去重：相同文本只保留置信度最高的
                normalized = t['text'].strip().lower()
                if normalized not in seen_texts:
                    seen_texts.add(normalized)
                    merged.append(t)
                else:
                    # 如果已存在，更新为置信度更高的版本
                    for existing in merged:
                        if existing['text'].strip().lower() == normalized:
                            if t['confidence'] > existing['confidence']:
                                existing.update(t)
                            break

        # 按置信度排序
        merged.sort(key=lambda x: x['confidence'], reverse=True)
        return merged

    # ============================================================
    # 3. 多版本图像识别 + 投票
    # ============================================================
    def recognize_with_voting(self, image: np.ndarray, ocr_engine,
                              engine_name: str = "EasyOCR") -> Dict:
        """
        对多种预处理版本的图像分别识别，投票选最优结果

        Returns:
            {
                'best_id': 最佳学号,
                'confidence': 置信度,
                'vote_count': 得票数,
                'all_candidates': 所有候选结果,
                'all_texts': 所有识别到的文字
            }
        """
        # 生成多个预处理版本
        versions = self.enhance_image(image)

        all_ids = []         # 所有识别到的学号
        all_texts = []       # 所有识别到的文字
        id_confidences = {}  # 学号 → 最高置信度

        for idx, img_version in enumerate(versions):
            try:
                if engine_name == "EasyOCR":
                    results = ocr_engine.readtext(img_version)
                    texts = []
                    for (bbox, text, conf) in results:
                        texts.append({
                            'text': str(text),
                            'confidence': float(conf),
                            'bbox': [[int(p[0]), int(p[1])] for p in bbox],
                            'source': f'EasyOCR_v{idx}'
                        })
                else:
                    continue

                all_texts.extend(texts)

                # 从这个版本提取学号
                student_id = self._extract_id_from_texts(texts)
                if student_id:
                    all_ids.append(student_id)
                    # 记录最高置信度
                    max_conf = max([t['confidence'] for t in texts], default=0)
                    if student_id not in id_confidences or max_conf > id_confidences[student_id]:
                        id_confidences[student_id] = max_conf

            except Exception as e:
                logger.warning(f"版本{idx}识别失败: {e}")

        if not all_ids:
            return {
                'best_id': None,
                'confidence': 0,
                'vote_count': 0,
                'all_candidates': [],
                'all_texts': all_texts
            }

        # 投票：选出票最多的学号
        counter = Counter(all_ids)
        best_id, vote_count = counter.most_common(1)[0]

        # 所有候选
        candidates = [
            {'id': cid, 'votes': count, 'confidence': id_confidences.get(cid, 0)}
            for cid, count in counter.most_common()
        ]

        logger.info(f"投票结果: best='{best_id}' (票数={vote_count}/{len(all_ids)})")
        return {
            'best_id': best_id,
            'confidence': id_confidences.get(best_id, 0),
            'vote_count': vote_count,
            'total_versions': len(versions),
            'all_candidates': candidates,
            'all_texts': all_texts
        }

    def _extract_id_from_texts(self, texts: list) -> Optional[str]:
        """从文本列表中提取学号"""
        id_patterns = [
            r'[A-Za-z]{1,4}\d{6,12}',
            r'[A-Za-z]?\d{8,12}',
            r'20[12]\d{5,8}',
            r'\d{6,12}',
        ]

        # 先找关键词旁边的
        keywords = ['学号', '学 号', 'ID', 'No', '编号']
        for t in texts:
            text = t['text']
            for kw in keywords:
                if kw.lower() in text.lower():
                    after = text[text.lower().find(kw.lower()) + len(kw):]
                    after = re.sub(r'^[\s:：\-_]+', '', after)
                    for pattern in id_patterns:
                        m = re.search(pattern, after)
                        if m:
                            return self._clean_id(m.group())

        # 再用格式匹配
        for t in texts:
            for pattern in id_patterns[:3]:  # 不用最宽松的那个
                m = re.search(pattern, t['text'])
                if m:
                    return self._clean_id(m.group())

        return None

    def _clean_id(self, id_str: str) -> str:
        cleaned = re.sub(r'[\s\-_]', '', id_str).upper()
        return cleaned

    # ============================================================
    # 4. 后处理校正
    # ============================================================
    def post_correct(self, ocr_id: str) -> Tuple[str, str]:
        """
        对OCR识别结果做后处理校正

        Returns:
            (corrected_id, correction_notes)
        """
        if not ocr_id:
            return "", "空输入"

        original = ocr_id
        corrected = ocr_id

        # 规则1：如果主体是数字，把混入的字母替换为对应数字
        digit_count = sum(1 for c in corrected if c.isdigit())
        if len(corrected) > 0 and digit_count / len(corrected) > 0.5:
            result = []
            for c in corrected:
                if c in self.correction_map:
                    result.append(self.correction_map[c])
                else:
                    result.append(c)
            corrected = ''.join(result)

        # 规则2：去掉前后的特殊字符
        corrected = re.sub(r'^[^A-Za-z0-9]+', '', corrected)
        corrected = re.sub(r'[^A-Za-z0-9]+$', '', corrected)

        # 规则3：如果以年份开头（如2021），确保是4位
        year_match = re.match(r'^(20[12]\d)', corrected)
        if year_match:
            # 检查年份合理性
            year = int(year_match.group(1))
            if year < 2015 or year > 2030:
                pass  # 可疑年份，不做处理

        notes = ""
        if corrected != original:
            notes = f"校正: '{original}' → '{corrected}'"
            logger.info(notes)

        return corrected, notes

    # ============================================================
    # 5. 多帧融合（摄像头模式）
    # ============================================================
    def multi_frame_fusion(self, frame_results: List[str],
                           min_agreement: int = 3) -> Optional[str]:
        """
        多帧结果融合
        当连续多帧识别到相同学号时，认为结果可靠

        Args:
            frame_results: 最近N帧的识别结果
            min_agreement: 至少需要多少帧一致

        Returns:
            稳定的学号，或None
        """
        if not frame_results:
            return None

        # 过滤空结果
        valid = [r for r in frame_results if r]
        if not valid:
            return None

        counter = Counter(valid)
        most_common, count = counter.most_common(1)[0]

        if count >= min_agreement:
            logger.info(f"多帧融合: '{most_common}' 出现 {count}/{len(frame_results)} 次")
            return most_common

        return None