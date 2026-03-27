"""
FastAPI 服务端（完整版）
包含：学号验证 + 实时OCR + 英语口语评估
"""
import os
import uuid
import time
import json
import base64
import logging
from datetime import datetime
from typing import Optional, List

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel as PydanticBaseModel

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import config
from modules.audio_preprocessor import AudioPreprocessor
from modules.asr_service import ASRService
from modules.image_preprocessor import ImagePreprocessor
from modules.ocr_service import OCRService
from modules.matching_engine import MatchingEngine
from modules.llm_service import LLMService

# ============================================================
# 日志配置
# ============================================================
logging.basicConfig(
    level=getattr(logging, config.log_level),
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(config.log_dir, f"app_{datetime.now():%Y%m%d}.log"),
            encoding='utf-8'
        ),
    ]
)
logger = logging.getLogger("api")

# ============================================================
# 初始化各模块
# ============================================================
audio_preprocessor = AudioPreprocessor(config.audio)
asr_service = ASRService(config.whisper)
image_preprocessor = ImagePreprocessor(config.image)
ocr_service = OCRService(config.ocr)
matching_engine = MatchingEngine(config.matching)
llm_service = LLMService(config.llm)

# ============================================================
# 实时OCR多帧融合缓冲
# ============================================================
realtime_frame_buffer = []
MAX_FRAME_BUFFER = 5

# ============================================================
# FastAPI 应用
# ============================================================
app = FastAPI(
    title="学生证验证 & 英语口语评估系统",
    description="Multimodal Assessment using OCR, Speech Recognition & LLM",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 初始化评估模块（可选）
# ============================================================
assessment_service = None
try:
    from modules.assessment_service import AssessmentService, AssessmentConfig

    assessment_config = AssessmentConfig()
    assessment_config.models = [
        {
            "name": "deepseek-chat",
            "provider": "deepseek",
            "api_key": config.llm.api_key,
            "base_url": "https://api.deepseek.com",
        },
    ]
    assessment_service = AssessmentService(assessment_config)
    logger.info("评估服务初始化成功")
except ImportError as e:
    logger.warning(f"评估模块未加载: {e}")
except Exception as e:
    logger.warning(f"评估模块初始化失败: {e}")


# ============================================================
# 数据模型
# ============================================================
class VerificationResponse(PydanticBaseModel):
    request_id: str
    status: str
    timestamp: str
    processing_time_ms: int
    asr_result: dict
    ocr_result: dict
    matching_result: dict
    llm_result: Optional[dict] = None
    final_verdict: str
    final_confidence: float
    message: str


class RealtimeOCRRequest(PydanticBaseModel):
    image: str  # base64图片


class AssessmentRequest(PydanticBaseModel):
    student_id: str
    answers: List[str]
    questions: Optional[List[str]] = None


class SingleEvalRequest(PydanticBaseModel):
    question: str
    answer: str


# ============================================================
# 辅助函数
# ============================================================
def _convert_numpy(obj):
    """递归地把 numpy 类型转成 Python 原生类型"""
    import numpy as np
    if isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_numpy(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


async def _save_upload(file: UploadFile, request_id: str, file_type: str) -> str:
    """保存上传文件"""
    ext = os.path.splitext(file.filename)[1] if file.filename else ".bin"
    filename = f"{request_id}_{file_type}{ext}"
    filepath = os.path.join(config.upload_dir, filename)
    content = await file.read()
    with open(filepath, "wb") as f:
        f.write(content)
    logger.info(f"保存文件: {filepath} ({len(content)} bytes)")
    return filepath


def _make_final_decision(matching_result: dict, llm_result: Optional[dict]):
    """综合匹配引擎和LLM结果做最终决策"""
    verdict = matching_result.get('verdict', 'ERROR')
    confidence = matching_result.get('overall_confidence', 0)

    if llm_result and llm_result.get('is_match') is not None:
        llm_match = llm_result['is_match']
        llm_confidence = llm_result.get('confidence', 0) / 100.0

        if verdict == "MATCH":
            message = f"✅ 验证通过 | {matching_result.get('details', '')}"
        elif verdict == "PROBABLE_MATCH":
            if llm_match:
                verdict = "MATCH"
                confidence = max(confidence, llm_confidence)
                message = f"✅ 验证通过（LLM确认） | {llm_result.get('reasoning', '')}"
            else:
                verdict = "NO_MATCH"
                confidence = llm_confidence
                message = f"❌ 验证失败（LLM否定） | {llm_result.get('reasoning', '')}"
        elif verdict == "NO_MATCH":
            if llm_match and llm_confidence > 0.8:
                verdict = "PROBABLE_MATCH"
                confidence = llm_confidence * 0.8
                message = f"⚠️ LLM认为可能匹配 | {llm_result.get('reasoning', '')}"
            else:
                message = f"❌ 验证失败 | {llm_result.get('reasoning', '')}"
        else:
            message = f"⚠️ 状态未知 | {matching_result.get('details', '')}"

        if llm_result.get('suggestion'):
            message += f" | 建议: {llm_result['suggestion']}"
    else:
        if verdict == "MATCH":
            message = f"✅ 验证通过 | {matching_result.get('details', '')}"
        elif verdict == "PROBABLE_MATCH":
            message = f"⚠️ 可能匹配，建议人工确认 | {matching_result.get('details', '')}"
        elif verdict == "NO_MATCH":
            message = f"❌ 验证失败 | {matching_result.get('details', '')}"
        else:
            message = f"⚠️ {matching_result.get('details', '处理异常')}"

    return verdict, round(confidence, 4), message


def _cleanup_temp_files(request_id: str):
    """清理请求相关的临时文件"""
    try:
        upload_dir = config.upload_dir
        for f in os.listdir(upload_dir):
            if f.startswith(request_id):
                os.remove(os.path.join(upload_dir, f))
    except Exception:
        pass


# ============================================================
# API接口：基础页面
# ============================================================
@app.get("/", response_class=HTMLResponse)
async def root():
    """返回前端页面"""
    frontend_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "frontend", "index.html"
    )
    if os.path.exists(frontend_path):
        with open(frontend_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Student ID Verification & Assessment API</h1>")


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "modules": {
            "whisper": asr_service.model is not None,
            "ocr": ocr_service.engine is not None,
            "assessment": assessment_service is not None,
        }
    }


# ============================================================
# API接口：学号验证（核心）
# ============================================================
@app.post("/api/verify", response_model=VerificationResponse)
async def verify_student_id(
    audio: UploadFile = File(..., description="学生朗读学号的音频"),
    image: UploadFile = File(..., description="学生证图片"),
    use_llm: bool = Form(default=True, description="是否启用LLM"),
    language: str = Form(default="zh", description="语言"),
):
    """
    主验证接口
    上传音频和学生证图片，自动进行验证
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    logger.info(f"[{request_id}] 收到验证请求: audio={audio.filename}, image={image.filename}")

    try:
        # ========== 1. 保存文件 ==========
        audio_path = await _save_upload(audio, request_id, "audio")
        image_path = await _save_upload(image, request_id, "image")

        # ========== 2. 音频处理 + ASR ==========
        logger.info(f"[{request_id}] 开始音频处理...")
        processed_audio_path, audio_metadata = audio_preprocessor.process(audio_path)

        logger.info(f"[{request_id}] 开始语音识别...")
        asr_result = asr_service.transcribe(processed_audio_path)
        asr_result['audio_metadata'] = audio_metadata

        # ========== 3. 图像处理 + OCR ==========
        logger.info(f"[{request_id}] 开始图像处理...")
        try:
            processed_image_path, image_metadata = image_preprocessor.process(image_path)
        except Exception as e:
            logger.warning(f"[{request_id}] 图像预处理失败，使用原图: {e}")
            processed_image_path = image_path
            image_metadata = {"original_path": image_path, "note": "使用原图"}

        logger.info(f"[{request_id}] 开始OCR识别...")
        try:
            ocr_result = ocr_service.recognize(processed_image_path)
        except Exception:
            logger.warning(f"[{request_id}] 处理后图片识别失败，用原图重试")
            ocr_result = ocr_service.recognize(image_path)

        ocr_result['image_metadata'] = image_metadata

        # ========== 4. 匹配引擎 ==========
        logger.info(f"[{request_id}] 开始匹配分析...")
        matching_result = matching_engine.verify(
            asr_id=asr_result.get('extracted_id', ''),
            ocr_id=ocr_result.get('extracted_id', ''),
            asr_confidence=asr_result.get('confidence', 0),
            ocr_confidence=ocr_result.get('confidence', 0),
        )

        # ========== 5. LLM智能分析 ==========
        llm_result = None
        if use_llm and config.llm.enable_llm and config.llm.api_key:
            logger.info(f"[{request_id}] 调用LLM智能分析...")
            try:
                llm_result = llm_service.intelligent_verify(
                    asr_result=asr_result,
                    ocr_result=ocr_result,
                    matching_result=matching_result,
                )
                logger.info(f"[{request_id}] LLM分析完成")
            except Exception as e:
                logger.warning(f"[{request_id}] LLM分析失败: {e}")
                llm_result = {"error": str(e), "is_match": None}

        # ========== 6. 最终决策 ==========
        final_verdict, final_confidence, message = _make_final_decision(
            matching_result, llm_result
        )

        # ========== 7. 构建响应 ==========
        processing_time = int((time.time() - start_time) * 1000)

        # 清理大字段
        if 'all_texts' in ocr_result:
            ocr_result['all_texts'] = [
                {'text': t['text'], 'confidence': round(float(t['confidence']), 3)}
                for t in ocr_result['all_texts'][:10]
            ]

        # numpy类型转换
        asr_result = _convert_numpy(asr_result)
        ocr_result = _convert_numpy(ocr_result)
        matching_result = _convert_numpy(matching_result)
        if llm_result:
            llm_result = _convert_numpy(llm_result)

        response = VerificationResponse(
            request_id=request_id,
            status="success",
            timestamp=datetime.now().isoformat(),
            processing_time_ms=processing_time,
            asr_result=asr_result,
            ocr_result=ocr_result,
            matching_result=matching_result,
            llm_result=llm_result,
            final_verdict=final_verdict,
            final_confidence=final_confidence,
            message=message,
        )

        logger.info(f"[{request_id}] 验证完成: verdict={final_verdict}, "
                     f"confidence={final_confidence:.4f}, time={processing_time}ms")
        return response

    except ValueError as e:
        logger.warning(f"[{request_id}] 输入错误: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[{request_id}] 处理失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")
    finally:
        _cleanup_temp_files(request_id)


# ============================================================
# API接口：单独ASR
# ============================================================
@app.post("/api/asr-only")
async def asr_only(
    audio: UploadFile = File(...),
    language: str = Form(default="zh"),
):
    """仅语音识别"""
    request_id = str(uuid.uuid4())[:8]
    audio_path = await _save_upload(audio, request_id, "audio")

    processed_path, metadata = audio_preprocessor.process(audio_path)
    result = asr_service.transcribe(processed_path)
    result['audio_metadata'] = metadata

    # numpy转换
    result = _convert_numpy(result)
    return result


# ============================================================
# API接口：单独OCR
# ============================================================
@app.post("/api/ocr-only")
async def ocr_only(image: UploadFile = File(...)):
    """仅OCR识别"""
    request_id = str(uuid.uuid4())[:8]
    image_path = await _save_upload(image, request_id, "image")

    try:
        processed_path, metadata = image_preprocessor.process(image_path)
    except Exception:
        processed_path = image_path
        metadata = {}

    result = ocr_service.recognize(processed_path)
    result['image_metadata'] = metadata

    # numpy转换
    result = _convert_numpy(result)
    return result


# ============================================================
# API接口：实时OCR（摄像头）
# ============================================================
@app.post("/api/realtime-ocr")
async def realtime_ocr(request: RealtimeOCRRequest):
    """
    实时OCR接口
    接收摄像头帧（base64），返回识别结果
    支持多帧融合提高稳定性
    """
    global realtime_frame_buffer

    try:
        import numpy as np
        import cv2

        # 1. 解码base64图片
        image_data = request.image
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"extracted_id": "", "confidence": 0, "error": "无法解码图片"}

        # 2. OCR识别（实时模式不用增强，保证速度）
        ocr_service.load_model()

        if ocr_service._engine_name == "EasyOCR":
            results = ocr_service.engine.readtext(img)
            texts = []
            for (bbox, text, confidence) in results:
                clean_bbox = [[int(p[0]), int(p[1])] for p in bbox]
                texts.append({
                    'text': str(text),
                    'confidence': float(confidence),
                    'bbox': clean_bbox,
                    'source': 'EasyOCR'
                })
        elif ocr_service._engine_name == "PaddleOCR":
            ocr_results = ocr_service.engine.ocr(img, cls=True)
            texts = []
            if ocr_results and ocr_results[0]:
                for line in ocr_results[0]:
                    texts.append({
                        'text': line[1][0],
                        'confidence': float(line[1][1]),
                        'bbox': line[0],
                        'source': 'PaddleOCR'
                    })
        else:
            texts = []

        # 3. 提取学号
        student_id, id_confidence, id_bbox = ocr_service._extract_student_id(texts)
        current_id = student_id or ""

        # 4. 多帧融合
        realtime_frame_buffer.append(current_id)
        if len(realtime_frame_buffer) > MAX_FRAME_BUFFER:
            realtime_frame_buffer = realtime_frame_buffer[-MAX_FRAME_BUFFER:]

        stable_id = None
        if ocr_service.enhancer:
            stable_id = ocr_service.enhancer.multi_frame_fusion(
                realtime_frame_buffer, min_agreement=2
            )

        final_id = stable_id or current_id

        # 5. 后处理校正
        if final_id and ocr_service.enhancer:
            corrected_id, _ = ocr_service.enhancer.post_correct(final_id)
            final_id = corrected_id

        # 6. 文字预览
        all_texts_preview = ' | '.join([t['text'] for t in texts[:5]])

        return {
            "extracted_id": final_id,
            "current_frame_id": current_id,
            "is_stable": stable_id is not None,
            "confidence": round(float(id_confidence), 4) if id_confidence else 0,
            "engine": ocr_service._engine_name,
            "all_texts_preview": all_texts_preview,
            "total_blocks": len(texts),
            "buffer_size": len(realtime_frame_buffer),
        }

    except Exception as e:
        logger.error(f"实时OCR失败: {e}")
        return {"extracted_id": "", "confidence": 0, "error": str(e)}


# ============================================================
# API接口：清除实时OCR缓冲
# ============================================================
@app.post("/api/realtime-ocr/reset")
async def reset_realtime_buffer():
    """清除实时OCR的帧缓冲"""
    global realtime_frame_buffer
    realtime_frame_buffer = []
    return {"status": "ok", "message": "帧缓冲已清除"}


# ============================================================
# API接口：英语口语评估
# ============================================================
@app.post("/api/assess")
async def assess_student(request: AssessmentRequest):
    """
    评估学生的英语口语回答
    输入: 学号 + 回答列表 + 可选问题列表
    输出: 多模型评分结果 + 汇总分析
    """
    if assessment_service is None:
        raise HTTPException(status_code=500, detail="评估模块未加载")
    try:
        logger.info(f"开始评估学生 {request.student_id}, {len(request.answers)} 个回答")
        result = assessment_service.evaluate_student(
            student_id=request.student_id,
            answers=request.answers,
            questions=request.questions,
        )
        return result
    except Exception as e:
        logger.error(f"评估失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/assess-single")
async def assess_single(request: SingleEvalRequest):
    """评估单个问答对（实时评估用）"""
    if assessment_service is None:
        raise HTTPException(status_code=500, detail="评估模块未加载")
    try:
        results = assessment_service.evaluate_all_models(
            question=request.question,
            answer=request.answer,
        )
        return {"evaluations": results}
    except Exception as e:
        logger.error(f"单题评估失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/assessment-questions")
async def get_questions():
    """获取评估问题列表"""
    if assessment_service is None:
        return {"questions": []}
    return {"questions": assessment_service.config.questions}


# ============================================================
# API接口：完整评估流程
# ============================================================
@app.post("/api/full-assessment")
async def full_assessment(
    audio_files: List[UploadFile] = File(..., description="每个问题的录音"),
    image: UploadFile = File(..., description="学生证图片"),
    questions: Optional[str] = Form(default=None, description="自定义问题JSON数组"),
):
    """
    完整评估流程：
    1. OCR提取学号（身份验证）
    2. Whisper转写所有回答
    3. LLM评估每个回答
    """
    if assessment_service is None:
        raise HTTPException(status_code=500, detail="评估模块未加载")

    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    logger.info(f"[{request_id}] 收到完整评估请求, {len(audio_files)} 个音频")

    try:
        # 1. OCR提取学号
        image_path = await _save_upload(image, request_id, "image")
        try:
            processed_image_path, _ = image_preprocessor.process(image_path)
        except Exception:
            processed_image_path = image_path

        ocr_result = ocr_service.recognize(processed_image_path)
        student_id = ocr_result.get('extracted_id', 'unknown')
        logger.info(f"[{request_id}] OCR学号: {student_id}")

        # 2. Whisper转写每个录音
        transcripts = []
        for i, audio_file in enumerate(audio_files):
            audio_path = await _save_upload(audio_file, request_id, f"audio_{i}")

            try:
                processed_path, _ = audio_preprocessor.process(audio_path)
            except Exception:
                processed_path = audio_path

            asr_result = asr_service.transcribe(processed_path)
            transcript = asr_result.get('raw_text', '')
            transcripts.append(transcript)
            logger.info(f"[{request_id}] Q{i+1} 转写: {transcript[:60]}...")

        # 3. 解析问题列表
        if questions:
            question_list = json.loads(questions)
        else:
            question_list = assessment_service.config.questions

        # 4. LLM评估
        logger.info(f"[{request_id}] 开始LLM评估...")
        eval_result = assessment_service.evaluate_student(
            student_id=student_id,
            answers=transcripts,
            questions=question_list[:len(transcripts)],
        )

        # 附加信息
        eval_result['ocr_student_id'] = student_id
        eval_result['transcripts'] = transcripts
        eval_result['processing_time_ms'] = int((time.time() - start_time) * 1000)

        logger.info(f"[{request_id}] 完整评估完成, 耗时 {eval_result['processing_time_ms']}ms")
        return eval_result

    except Exception as e:
        logger.error(f"[{request_id}] 完整评估失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _cleanup_temp_files(request_id)


# ============================================================
# 启动事件：预加载模型
# ============================================================
@app.on_event("startup")
async def startup_event():
    """应用启动时预加载模型"""
    logger.info("=" * 60)
    logger.info("学生证验证 & 英语口语评估系统启动中...")
    logger.info("=" * 60)

    try:
        logger.info("预加载 Whisper 模型...")
        asr_service.load_model()
        logger.info("Whisper 模型加载完成 ✓")
    except Exception as e:
        logger.error(f"Whisper 加载失败: {e}")

    try:
        logger.info("预加载 OCR 模型...")
        ocr_service.load_model()
        logger.info("OCR 模型加载完成 ✓")
    except Exception as e:
        logger.error(f"OCR 加载失败: {e}")

    if assessment_service:
        logger.info("评估模块就绪 ✓")
    else:
        logger.warning("评估模块未加载")

    logger.info("=" * 60)
    logger.info(f"服务就绪: http://{config.host}:{config.port}")
    logger.info(f"  - 学号验证: POST /api/verify")
    logger.info(f"  - 实时OCR:  POST /api/realtime-ocr")
    logger.info(f"  - 口语评估: POST /api/assess")
    logger.info(f"  - 完整评估: POST /api/full-assessment")
    logger.info("=" * 60)