"""
LLM智能验证模块
- 智能纠错
- 不确定性推理
- 自然语言解释
- 多轮交互建议
"""
import json
import logging
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """你是一个专业的学生证身份验证AI助手。你的工作是分析语音识别(ASR)和光学字符识别(OCR)的结果，判断学生口述的学号是否与学生证上的学号一致。

你具备以下专业知识：
1. OCR常见错误：O/0混淆、l/1/I混淆、5/S混淆、8/B混淆、6/G混淆等
2. ASR常见错误：同音字混淆、数字读法差异（"一"/"幺"、"二"/"两"）、背景噪声影响
3. 学号格式规则：通常为8-12位数字，可能有字母前缀，通常以年份开头（如2021、2022）

请始终以JSON格式回复。"""


VERIFICATION_PROMPT = """请分析以下语音识别和OCR识别的结果，判断学号是否匹配。

## 语音识别(ASR)结果
- 原始转写文本: "{asr_raw_text}"
- 提取的学号: "{asr_id}"
- 识别置信度: {asr_confidence}

## OCR识别结果
- 卡片上所有文字: {ocr_all_texts}
- 提取的学号: "{ocr_id}"
- 识别置信度: {ocr_confidence}

## 初步匹配结果
- 相似度: {similarity}
- 编辑距离: {edit_distance}
- 初步判定: {preliminary_verdict}
- 差异分析: {details}

## 请完成以下分析:

1. **错误分析**: 分析ASR和OCR各自可能的识别错误
2. **学号校正**: 基于常见错误模式，给出你认为最可能的正确学号
3. **匹配判定**: 学生口述的学号与卡片上的学号是否实际匹配
4. **置信度**: 你对判定结果的置信度(0-100)
5. **建议**: 如果不确定，给出下一步建议

请以以下JSON格式回复:
{{
    "error_analysis": {{
        "asr_possible_errors": ["ASR可能的错误"],
        "ocr_possible_errors": ["OCR可能的错误"]
    }},
    "corrected_asr_id": "校正后的ASR学号",
    "corrected_ocr_id": "校正后的OCR学号",
    "is_match": true或false,
    "confidence": 0到100的整数,
    "reasoning": "详细的判断理由",
    "suggestion": "给用户的建议",
    "risk_level": "low/medium/high"
}}"""


class LLMService:
    """LLM智能验证服务"""

    def __init__(self, config):
        self.config = config
        self.client = None

    def _init_client(self):
        """初始化LLM客户端"""
        if self.client is not None:
            return

        if self.config.provider == "deepseek":
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.config.api_key,
                base_url="https://api.deepseek.com",  # DeepSeek的地址
                timeout=self.config.timeout,
            )
            logger.info("DeepSeek 客户端初始化成功")
            return


        if self.config.provider == "openai":
            try:
                from openai import OpenAI
                self.client = OpenAI(
                    api_key=self.config.api_key,
                    base_url=self.config.base_url,
                    timeout=self.config.timeout,
                )
                logger.info("OpenAI 客户端初始化成功")
            except ImportError:
                raise ImportError("请安装: pip install openai")

        elif self.config.provider == "dashscope":
            try:
                from openai import OpenAI
                self.client = OpenAI(
                    api_key=self.config.api_key,
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                    timeout=self.config.timeout,
                )
                logger.info("DashScope 客户端初始化成功")
            except ImportError:
                raise ImportError("请安装: pip install openai")

        elif self.config.provider == "local":
            try:
                from openai import OpenAI
                self.client = OpenAI(
                    api_key="not-needed",
                    base_url=self.config.base_url or "http://localhost:8080/v1",
                    timeout=self.config.timeout,
                )
                logger.info(f"本地LLM 客户端初始化成功: {self.config.base_url}")
            except ImportError:
                raise ImportError("请安装: pip install openai")

    def _extract_json(self, text: str) -> dict:
        """从LLM回复中提取JSON"""
        import json
        import re

        # 方法1：直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 方法2：提取 ```json ... ``` 代码块
        json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # 方法3：提取 { ... } 部分
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        # 方法4：解析失败，返回原始文本
        logger.warning(f"无法从LLM回复中提取JSON: {text[:200]}")
        return {
            "is_match": None,
            "confidence": 0,
            "reasoning": text,
            "suggestion": "LLM返回格式异常，请参考原始分析",
        }


    def intelligent_verify(self, asr_result: Dict, ocr_result: Dict,
                           matching_result: Dict) -> Dict:
        """
        LLM智能验证

        Args:
            asr_result: ASR模块的输出
            ocr_result: OCR模块的输出
            matching_result: 匹配引擎的输出

        Returns:
            LLM分析结果
        """
        self._init_client()

        # 构建OCR文本摘要（避免prompt过长）
        ocr_texts_summary = self._summarize_ocr_texts(ocr_result.get('all_texts', []))

        prompt = VERIFICATION_PROMPT.format(
            asr_raw_text=asr_result.get('raw_text', ''),
            asr_id=asr_result.get('extracted_id', ''),
            asr_confidence=asr_result.get('confidence', 0),
            ocr_all_texts=ocr_texts_summary,
            ocr_id=ocr_result.get('extracted_id', ''),
            ocr_confidence=ocr_result.get('confidence', 0),
            similarity=matching_result.get('similarity', 0),
            edit_distance=matching_result.get('edit_distance', 0),
            preliminary_verdict=matching_result.get('verdict', ''),
            details=matching_result.get('details', ''),
        )

        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                # response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            result = self._extract_json(content)
            # ====================

            result['llm_model'] = self.config.model
            result['llm_provider'] = self.config.provider
            result['tokens_used'] = {
                'prompt': response.usage.prompt_tokens,
                'completion': response.usage.completion_tokens,
                'total': response.usage.total_tokens,
            }

            logger.info(f"LLM验证结果: match={result.get('is_match')}, "
                        f"confidence={result.get('confidence')}")
            return result

        except json.JSONDecodeError as e:
            # 这个分支可以删掉了，因为 _extract_json 已经处理了
            logger.error(f"LLM返回的JSON解析失败: {e}")
            return self._fallback_result(f"JSON解析错误: {e}")
        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            return self._fallback_result(str(e))

    def multimodal_verify(self, image_path: str, asr_text: str) -> Dict:
        """
        多模态LLM验证 — 直接让LLM看图+分析语音文本
        (需要GPT-4o / Qwen-VL等支持视觉的模型)
        """
        self._init_client()

        import base64
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()

        prompt = f"""请完成以下学生证验证任务:

1. 仔细观察这张学生证/ID卡图片，找出学号(Student ID)
2. 学生口述的内容经语音识别转写为: "{asr_text}"
3. 请判断口述的学号是否与卡片上的学号一致

请以JSON格式回复:
{{
    "ocr_id_from_image": "你从图片中读到的学号",
    "asr_id_from_speech": "你从语音转写中提取的学号",
    "is_match": true或false,
    "confidence": 0-100,
    "reasoning": "判断理由"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_b64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            result = json.loads(content)
            result['method'] = 'multimodal'
            return result

        except Exception as e:
            logger.error(f"多模态LLM调用失败: {e}")
            return self._fallback_result(str(e))

    def _summarize_ocr_texts(self, texts: List[Dict]) -> str:
        """摘要OCR文本，避免prompt过长"""
        if not texts:
            return "[]"

        summary = []
        for item in texts[:15]:  # 最多15条
            summary.append({
                'text': item.get('text', ''),
                'confidence': round(item.get('confidence', 0), 3)
            })
        return json.dumps(summary, ensure_ascii=False)

    def _fallback_result(self, error_msg: str) -> Dict:
        """LLM调用失败时的兜底结果"""
        return {
            'is_match': None,
            'confidence': 0,
            'reasoning': f"LLM分析不可用: {error_msg}",
            'suggestion': "请依赖传统匹配结果或人工核验",
            'error': error_msg,
            'risk_level': 'high',
        }