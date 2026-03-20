"""
FastAPI 服务端
提供RESTful API接口
"""
import os
import uuid
import time
import logging
from datetime import datetime
from typing import Optional
import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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
# FastAPI 应用
# ============================================================
app = FastAPI(
    title="学生证语音验证系统",
    description="Verification of Spoken Student ID Using Speech Recognition and OCR",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 数据模型
# ============================================================
class VerificationResponse(BaseModel):
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


# ============================================================
# API端点
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
    return HTMLResponse(content="<h1>Student ID Verification API</h1><p>前端文件未找到</p>")


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.post("/api/verify", response_model=VerificationResponse)
async def verify_student_id(
    audio: UploadFile = File(..., description="学生朗读学号的音频文件"),
    image: UploadFile = File(..., description="学生证图片"),
    use_llm: bool = Form(default=True, description="是否启用LLM智能分析"),
    language: str = Form(default="zh", description="语音语言 (zh/en)"),
):
    """
    主验证接口

    上传音频和学生证图片，系统自动进行：
    1. 音频预处理 + 语音识别
    2. 图像预处理 + OCR识别
    3. 学号匹配
    4. (可选) LLM智能分析
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    logger.info(f"[{request_id}] 收到验证请求: audio={audio.filename}, image={image.filename}")

    try:
        # ========== 1. 保存上传文件 ==========
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

        # 尝试预处理，如果失败就用原图
        try:
            processed_image_path, image_metadata = image_preprocessor.process(image_path)
        except Exception as e:
            logger.warning(f"[{request_id}] 图像预处理失败，使用原图: {e}")
            processed_image_path = image_path
            image_metadata = {"original_path": image_path, "note": "使用原图"}

        logger.info(f"[{request_id}] 开始OCR识别...")

        # OCR也加保护
        try:
            ocr_result = ocr_service.recognize(processed_image_path)
        except FileNotFoundError:
            logger.warning(f"[{request_id}] 处理后图片不存在，用原图重试")
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

        # ========== 5. LLM智能分析 (按需) ==========
        llm_result = None
        if (use_llm and config.llm.enable_llm and config.llm.api_key and
                matching_result.get('need_llm', False)):
            logger.info(f"[{request_id}] 触发LLM智能分析...")
            try:
                llm_result = llm_service.intelligent_verify(
                    asr_result=asr_result,
                    ocr_result=ocr_result,
                    matching_result=matching_result,
                )
            except Exception as e:
                logger.warning(f"[{request_id}] LLM分析失败: {e}")
                llm_result = {"error": str(e), "is_match": None}

        # ========== 6. 最终决策 ==========
        final_verdict, final_confidence, message = _make_final_decision(
            matching_result, llm_result
        )

        # ========== 7. 构建响应 ==========
        processing_time = int((time.time() - start_time) * 1000)

        # 清理大字段，避免响应过大
        if 'all_texts' in ocr_result:
            ocr_result['all_texts'] = [
                {'text': t['text'], 'confidence': round(t['confidence'], 3)}
                for t in ocr_result['all_texts'][:10]
            ]
        # ====== 把所有numpy类型转成Python原生类型 ======
        asr_result = _convert_numpy(asr_result)
        ocr_result = _convert_numpy(ocr_result)
        matching_result = _convert_numpy(matching_result)
        if llm_result:
            llm_result = _convert_numpy(llm_result)
        # =============================================


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
        # 清理临时文件
        _cleanup_temp_files(request_id)


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
    return result


@app.post("/api/ocr-only")
async def ocr_only(image: UploadFile = File(...)):
    """仅OCR识别"""
    request_id = str(uuid.uuid4())[:8]
    image_path = await _save_upload(image, request_id, "image")
    processed_path, metadata = image_preprocessor.process(image_path)
    result = ocr_service.recognize(processed_path)
    result['image_metadata'] = metadata
    return result


# ============================================================
# 辅助函数
# ============================================================
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
            # 匹配引擎说匹配，以匹配引擎为准
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
        # 无LLM结果
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
# 预加载模型
# ============================================================
@app.on_event("startup")
async def startup_event():
    """应用启动时预加载模型"""
    logger.info("=" * 60)
    logger.info("学生证语音验证系统启动中...")
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

    logger.info("=" * 60)
    logger.info(f"服务就绪: http://{config.host}:{config.port}")
    logger.info("=" * 60)