from pathlib import Path


ASR_MODEL = "whisper"
TTS_MODEL = "qwen"
VOICE_STORAGE = "voices"

DATA_AUDIO_DIR = Path("data/audio")
VOICE_CLONE_DIR = Path("voices")
TTS_OUTPUT_DIR = Path("generated_speech_audio")

DEFAULT_PERSON = "julia"

SUPPORTED_AUDIO_EXT = [".wav", ".ogg", ".mp3"]

AVATAR_NAME = "basic_avatar"