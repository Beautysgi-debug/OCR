"""
应用主入口
"""
import uvicorn
from config.settings import config


def main():
    """启动服务"""
    print("=" * 60)
    print("  学生证语音验证系统")
    print("  Verification of Spoken Student ID")
    print("  Using Speech Recognition and OCR")
    print("=" * 60)
    print(f"  Whisper Model : {config.whisper.model_size}")
    print(f"  OCR Engine    : PaddleOCR ({config.ocr.lang})")
    print(f"  LLM Provider  : {config.llm.provider} ({config.llm.model})")
    print(f"  LLM Enabled   : {config.llm.enable_llm}")
    print(f"  Device        : {config.whisper.device}")
    print(f"  Server        : http://{config.host}:{config.port}")
    print("=" * 60)

    uvicorn.run(
        "api.server:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        log_level=config.log_level.lower(),
        workers=1,  # 模型加载需要单worker
    )


if __name__ == "__main__":
    main()