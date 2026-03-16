import os
from .xtts_service import XTTSService
from .qwen_service import QwenTTSService

def create_tts_service():
    model_type = os.getenv("USE_TTS_MODEL", "xtts").lower()

    if model_type == "xtts":
        print("🎭 Выбрана модель → XTTS v2")
        return XTTSService(model_path="/app/models/xtts_v2")

    else:
        print("🎭 Выбрана модель → Qwen3-TTS")
        return QwenTTSService()