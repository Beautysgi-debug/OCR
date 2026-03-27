"""
匹配决策引擎
- 精确匹配
- 模糊匹配 (编辑距离、序列相似度)
- 混淆字符校正
- 综合评分
"""
import re
import logging
from typing import Dict, Optional, Tuple
from difflib import SequenceMatcher
from enum import Enum

logger = logging.getLogger(__name__)


class VerifyResult(str, Enum):
    MATCH = "MATCH"
    PROBABLE_MATCH = "PROBABLE_MATCH"
    NO_MATCH = "NO_MATCH"
    ERROR = "ERROR"


class MatchingEngine:
    """匹配决策引擎"""

    def __init__(self, config):
        self.config = config

    def verify(self, asr_id: str, ocr_id: str,
               asr_confidence: float = 1.0,
               ocr_confidence: float = 1.0) -> Dict:
        """
        综合验证两个学号是否匹配

        Returns:
            dict: {
                'verdict': MATCH/PROBABLE_MATCH/NO_MATCH,
                'asr_id': ASR识别的学号,
                'ocr_id': OCR识别的学号,
                'corrected_asr_id': 校正后的ASR学号,
                'corrected_ocr_id': 校正后的OCR学号,
                'exact_match': 是否精确匹配,
                'similarity': 相似度分数,
                'edit_distance': 编辑距离,
                'overall_confidence': 综合置信度,
                'details': 详细分析,
                'need_llm': 是否需要LLM进一步判断
            }
        """
        logger.info(f"开始匹配: ASR='{asr_id}' vs OCR='{ocr_id}'")

        # 空值检查
        if not asr_id or not ocr_id:
            return self._build_result(
                asr_id, ocr_id, asr_id, ocr_id,
                VerifyResult.ERROR, 0.0,
                "ASR或OCR未能提取到学号"
            )

        # 标准化
        norm_asr = self._normalize(asr_id)
        norm_ocr = self._normalize(ocr_id)

        # === Level 1: 精确匹配 ===
        if norm_asr == norm_ocr:
            confidence = min(asr_confidence, ocr_confidence)
            return self._build_result(
                asr_id, ocr_id, norm_asr, norm_ocr,
                VerifyResult.MATCH, confidence,
                "精确匹配成功"
            )

        # === Level 2: 混淆字符校正后匹配 ===
        corrected_asr = self._fix_confusion(norm_asr)
        corrected_ocr = self._fix_confusion(norm_ocr)

        if corrected_asr == corrected_ocr:
            confidence = min(asr_confidence, ocr_confidence) * 0.95
            return self._build_result(
                asr_id, ocr_id, corrected_asr, corrected_ocr,
                VerifyResult.MATCH, confidence,
                f"混淆字符校正后匹配: '{norm_asr}'->'{corrected_asr}', '{norm_ocr}'->'{corrected_ocr}'"
            )

        # === Level 3: 模糊匹配 ===
        similarity = self._sequence_similarity(corrected_asr, corrected_ocr)
        edit_dist = self._levenshtein_distance(corrected_asr, corrected_ocr)
        partial_analysis = self._partial_match_analysis(corrected_asr, corrected_ocr)

        if (similarity >= self.config.fuzzy_match_threshold and
                edit_dist <= self.config.max_edit_distance):
            confidence = similarity * min(asr_confidence, ocr_confidence) * 0.85
            return self._build_result(
                asr_id, ocr_id, corrected_asr, corrected_ocr,
                VerifyResult.PROBABLE_MATCH, confidence,
                f"模糊匹配: 相似度={similarity:.4f}, 编辑距离={edit_dist}. {partial_analysis}",
                need_llm=True
            )

        # === Level 4: 不匹配，但可能需要LLM分析 ===
        need_llm = similarity > 0.5  # 有一定相似度但未达到阈值
        return self._build_result(
            asr_id, ocr_id, corrected_asr, corrected_ocr,
            VerifyResult.NO_MATCH,
            1 - similarity,
            f"不匹配: 相似度={similarity:.4f}, 编辑距离={edit_dist}. {partial_analysis}",
            need_llm=need_llm,
            similarity=similarity,
            edit_distance=edit_dist
        )

    def _normalize(self, id_str: str) -> str:
        """标准化学号字符串"""
        result = id_str.strip().upper()
        result = re.sub(r'[\s\-_\.]', '', result)
        return result

    def _fix_confusion(self, id_str: str) -> str:
        """修复常见的OCR/ASR混淆字符"""
        result = id_str
        for wrong, correct in self.config.confusion_pairs.items():
            # 只在数字上下文中替换
            # 如果字符串主要是数字，将混淆字母替换为数字
            if self._is_mostly_digits(result):
                result = result.replace(wrong, correct)
        return result

    def _is_mostly_digits(self, s: str) -> bool:
        """判断字符串是否主要由数字组成"""
        if not s:
            return False
        digit_count = sum(1 for c in s if c.isdigit())
        return digit_count / len(s) > 0.6

    def _sequence_similarity(self, s1: str, s2: str) -> float:
        """计算序列相似度"""
        return SequenceMatcher(None, s1, s2).ratio()

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """计算编辑距离"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)

        prev_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row
        return prev_row[-1]

    def _partial_match_analysis(self, s1: str, s2: str) -> str:
        """分析两个字符串的差异位置"""
        if len(s1) != len(s2):
            return f"长度不同: {len(s1)} vs {len(s2)}"

        diffs = []
        for i, (c1, c2) in enumerate(zip(s1, s2)):
            if c1 != c2:
                diffs.append(f"位置{i}: '{c1}' vs '{c2}'")

        if diffs:
            return f"差异位置: {'; '.join(diffs)}"
        return "无差异"

    def _build_result(self, asr_id, ocr_id, corrected_asr, corrected_ocr,
                      verdict, confidence, details,
                      need_llm=False, similarity=None, edit_distance=None) -> Dict:
        """构建结果字典"""
        if similarity is None:
            similarity = self._sequence_similarity(corrected_asr, corrected_ocr)
        if edit_distance is None:
            edit_distance = self._levenshtein_distance(corrected_asr, corrected_ocr)

        return {
            'verdict': verdict.value,
            'asr_id': asr_id,
            'ocr_id': ocr_id,
            'corrected_asr_id': corrected_asr,
            'corrected_ocr_id': corrected_ocr,
            'exact_match': corrected_asr == corrected_ocr,
            'similarity': round(similarity, 4),
            'edit_distance': edit_distance,
            'overall_confidence': round(confidence, 4),
            'details': details,
            'need_llm': need_llm,
        }