import torch
import numpy as np
from qwen_tts import Qwen3TTSModel
from .base import BaseTTSService
from typing import Any

class QwenTTSService(BaseTTSService):
    def __init__(self, model_id="Qwen/Qwen3-TTS-12Hz-0.6B-Base", device="cuda:0"):
        self.model_id = model_id
        self.device = device
        self.model = None
        self._sample_rate = 16000 # Обычно 16k для Whisper-based или других, уточните у модели

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def load_model(self):
        print(f"🚀 Загрузка Qwen TTS ({self.model_id})...")
        self.model = Qwen3TTSModel.from_pretrained(
            self.model_id,
            device_map=self.device,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

    def get_speaker_embedding(self, audio_path: str):
        print(f"🧬 Создание промпта Qwen из {audio_path}...")
        # Логика создания клона для Qwen
        prompt_items = self.model.create_voice_clone_prompt(
            ref_audio=str(audio_path),
            # Здесь можно добавить логику транскрипции референса, если нужно
            # ref_text="...",
            x_vector_only_mode=True,
        )
        return prompt_items

    def generate_stream(self, text: str, speaker_embedding: Any):
        # Qwen пока не умеет стримить чанками нативно в public API, возвращаем всё сразу
        wavs, sr = self.model.generate_voice_clone(
            text=text,
            language="Russian",
            voice_clone_prompt=speaker_embedding,
        )
        # self._sample_rate = sr # Можно обновить рейт, если он динамический
        yield wavs[0].cpu().numpy()