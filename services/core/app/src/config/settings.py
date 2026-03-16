from pathlib import Path

# Docker-пути (volumes)
DATA_AUDIO_DIR = Path("/app/volumes/audio_samples")
TTS_OUTPUT_DIR = Path("/app/volumes/generated_audio")
VOICE_CLONE_DIR = Path("/app/volumes/voices")
MODELS_DIR = Path("/app/models")

SUPPORTED_AUDIO_EXT = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}