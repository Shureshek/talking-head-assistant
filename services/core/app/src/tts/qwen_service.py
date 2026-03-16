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
        self._sample_rate = 24000

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def load_model(self):
        print(f"🚀 Загрузка Qwen TTS ({self.model_id})...")
        self.model = Qwen3TTSModel.from_pretrained(
            self.model_id,
            device_map=self.device,
            dtype=torch.bfloat16,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

    def get_speaker_embedding(self, audio_path: str, ref_text: str = None):
        print(f"🧬 Создание промпта Qwen из {audio_path}...")
        # Логика создания клона для Qwen
        use_text_mode = ref_text is not None and len(ref_text) > 0

        if use_text_mode:
            print(f"📝 Использую текст референса: {ref_text[:50]}...")
        else:
            print("⚠️ Текст референса не передан, использую x-vector (только тембр).")

        prompt_items = self.model.create_voice_clone_prompt(
            ref_audio=str(audio_path),
            ref_text=ref_text if use_text_mode else None,
            x_vector_only_mode=not use_text_mode,
        )
        return prompt_items

    def generate_stream(self, text: str, speaker_embedding: Any):
        wavs, sr = self.model.generate_voice_clone(
            text=text,
            language="Russian",
            voice_clone_prompt=speaker_embedding,
        )
        if sr != self._sample_rate:
            print(f"📢 Внимание: SR модели изменился на {sr}")
            self._sample_rate = sr

        if not wavs: return

        audio_chunk = wavs[0]

        # Безопасная конвертация в float32 для плеера (чтобы не было ошибок numpy)
        if hasattr(audio_chunk, 'cpu'):
            audio_chunk = audio_chunk.cpu()
        if hasattr(audio_chunk, 'numpy'):
            audio_chunk = audio_chunk.numpy()

        yield audio_chunk.astype(np.float32)