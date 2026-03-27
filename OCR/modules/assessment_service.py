"""
英语口语评估模块
- 管理评估问题
- 调用多个LLM评分
- 对比分析结果
"""
import json
import re
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# 评分标准 Prompt
RUBRIC_PROMPT = """
You are grading a student's English oral response from its transcript.

Evaluate the answer using these criteria:
1. Relevance to the question
2. Does it make sense
3. Clarity
4. Basic grammatical quality

Give:
- relevance: one of ["yes", "partly", "no"]
- makes_sense: one of ["yes", "partly", "no"]
- score: integer from 0 to 10
- short_comment: one short sentence

Be reasonably strict but fair.
Must return a valid JSON object ONLY, with keys: "relevance", "makes_sense", "score", "short_comment".
"""

# 默认的5个测试问题
DEFAULT_QUESTIONS = [
    "What is your favourite food?",
    "Briefly discuss your favourite hobby.",
    "Describe a memorable trip you have taken.",
    "What do you usually do on weekends?",
    "Do you prefer reading books or watching movies? Why?",
]


@dataclass
class AssessmentConfig:
    """评估配置"""
    questions: List[str] = field(default_factory=lambda: DEFAULT_QUESTIONS)
    models: List[Dict] = field(default_factory=lambda: [
        {
            "name": "deepseek-chat",
            "provider": "deepseek",
            "api_key": "",
            "base_url": "https://api.deepseek.com",
        },
    ])
    temperature: float = 0.0
    max_tokens: int = 512


class AssessmentService:
    """英语口语评估服务"""

    def __init__(self, config: AssessmentConfig):
        self.config = config
        self.clients = {}

    def _get_client(self, model_config: Dict):
        """获取或创建LLM客户端"""
        name = model_config["name"]
        if name not in self.clients:
            from openai import OpenAI
            self.clients[name] = OpenAI(
                api_key=model_config["api_key"],
                base_url=model_config["base_url"],
                timeout=30,
            )
        return self.clients[name]

    def evaluate_single(self, question: str, answer: str,
                        model_config: Dict) -> Dict:
        """
        用单个模型评估一个回答

        Returns:
            {
                "model": "deepseek-chat",
                "relevance": "yes",
                "makes_sense": "yes",
                "score": 8,
                "short_comment": "...",
                "raw_output": "...",
                "json_ok": True
            }
        """
        client = self._get_client(model_config)
        model_name = model_config["name"]

        messages = [
            {"role": "system", "content": RUBRIC_PROMPT},
            {"role": "user", "content": f"Question: {question}\nStudent answer: {answer}"}
        ]

        try:
            kwargs = {
                "model": model_name,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            }

            # DeepSeek 支持 json_object 格式
            if model_config.get("provider") == "deepseek":
                kwargs["response_format"] = {"type": "json_object"}

            response = client.chat.completions.create(**kwargs)
            raw_text = response.choices[0].message.content

            # 解析JSON
            parsed = self._parse_json(raw_text)

            if parsed:
                return {
                    "model": model_name,
                    "relevance": parsed.get("relevance"),
                    "makes_sense": parsed.get("makes_sense"),
                    "score": parsed.get("score"),
                    "short_comment": parsed.get("short_comment"),
                    "raw_output": raw_text,
                    "json_ok": True,
                    "tokens_used": response.usage.total_tokens if response.usage else 0,
                }
            else:
                return {
                    "model": model_name,
                    "relevance": None, "makes_sense": None,
                    "score": None, "short_comment": None,
                    "raw_output": raw_text,
                    "json_ok": False,
                    "error": "JSON解析失败",
                }

        except Exception as e:
            logger.error(f"模型 {model_name} 评估失败: {e}")
            return {
                "model": model_name,
                "relevance": None, "makes_sense": None,
                "score": None, "short_comment": None,
                "raw_output": "",
                "json_ok": False,
                "error": str(e),
            }

    def evaluate_all_models(self, question: str, answer: str) -> List[Dict]:
        """用所有配置的模型评估同一个回答"""
        results = []
        for model_config in self.config.models:
            logger.info(f"使用模型 {model_config['name']} 评估...")
            result = self.evaluate_single(question, answer, model_config)
            results.append(result)
        return results

    def evaluate_student(self, student_id: str,
                         answers: List[str],
                         questions: Optional[List[str]] = None) -> Dict:
        """
        评估一个学生的所有回答

        Args:
            student_id: 学号
            answers: 学生回答列表（Whisper转写后的文本）
            questions: 问题列表（默认用配置中的问题）

        Returns:
            完整的评估报告
        """
        if questions is None:
            questions = self.config.questions

        all_evaluations = []

        for i, (question, answer) in enumerate(zip(questions, answers)):
            logger.info(f"评估问题 {i+1}/{len(questions)}: {question[:30]}...")

            model_results = self.evaluate_all_models(question, answer)

            all_evaluations.append({
                "question_number": i + 1,
                "question": question,
                "answer": answer,
                "evaluations": model_results,
            })

        # 生成汇总分析
        summary = self._generate_summary(all_evaluations)

        return {
            "student_id": student_id,
            "total_questions": len(questions),
            "models_used": [m["name"] for m in self.config.models],
            "evaluations": all_evaluations,
            "summary": summary,
        }

    def _generate_summary(self, evaluations: List[Dict]) -> Dict:
        """生成汇总分析"""
        model_scores = {}  # model_name -> [scores]

        for eval_item in evaluations:
            for model_result in eval_item["evaluations"]:
                model = model_result["model"]
                score = model_result.get("score")

                if model not in model_scores:
                    model_scores[model] = []

                if score is not None:
                    model_scores[model].append(score)

        # 计算每个模型的平均分
        model_averages = {}
        for model, scores in model_scores.items():
            if scores:
                model_averages[model] = {
                    "average_score": round(sum(scores) / len(scores), 2),
                    "min_score": min(scores),
                    "max_score": max(scores),
                    "scores": scores,
                }

        # 模型间一致性分析
        agreement = self._analyze_agreement(evaluations)

        return {
            "model_averages": model_averages,
            "agreement_analysis": agreement,
        }

    def _analyze_agreement(self, evaluations: List[Dict]) -> Dict:
        """分析模型间的一致性"""
        score_diffs = []
        relevance_agreements = 0
        total_comparisons = 0

        for eval_item in evaluations:
            results = eval_item["evaluations"]
            # 两两比较
            for i in range(len(results)):
                for j in range(i + 1, len(results)):
                    r1, r2 = results[i], results[j]
                    s1, s2 = r1.get("score"), r2.get("score")
                    rel1, rel2 = r1.get("relevance"), r2.get("relevance")

                    if s1 is not None and s2 is not None:
                        score_diffs.append(abs(s1 - s2))
                        total_comparisons += 1

                    if rel1 and rel2:
                        if rel1 == rel2:
                            relevance_agreements += 1

        return {
            "average_score_difference": round(sum(score_diffs) / max(len(score_diffs), 1), 2),
            "max_score_difference": max(score_diffs) if score_diffs else 0,
            "relevance_agreement_rate": round(
                relevance_agreements / max(total_comparisons, 1) * 100, 1
            ),
            "total_comparisons": total_comparisons,
        }

    def _parse_json(self, text: str) -> Optional[Dict]:
        """安全解析JSON"""
        text = text.strip()

        # 去掉 markdown 代码块
        if text.startswith("```"):
            lines = text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        try:
            return json.loads(text)
        except Exception:
            pass

        # 提取 { ... }
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass

        return None