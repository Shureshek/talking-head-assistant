import torch
import numpy as np
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from .base import BaseTTSService

class XTTSService(BaseTTSService):
    def __init__(self, model_path: str, use_deepspeed=False):
        self.model_path = model_path
        self.use_deepspeed = use_deepspeed
        self.model = None
        self._sample_rate = 24000

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def load_model(self):
        print("🚀 Загрузка XTTS v2...")
        config = XttsConfig()
        config.load_json(f"{self.model_path}/config.json")
        self.model = Xtts.init_from_config(config)
        self.model.load_checkpoint(config, checkpoint_dir=self.model_path, use_deepspeed=self.use_deepspeed)
        self.model.cuda()

    def get_speaker_embedding(self, audio_path: str, ref_text: str = None):
        print(f"🧬 Создание латентов XTTS из {audio_path}...")
        gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(audio_path=[str(audio_path)])
        return {
            "gpt_cond_latent": gpt_cond_latent,
            "speaker_embedding": speaker_embedding
        }

    def generate_stream(self, text: str, speaker_embedding: dict):
        chunks = self.model.inference_stream(
            text,
            "ru",
            speaker_embedding['gpt_cond_latent'],
            speaker_embedding['speaker_embedding'],
            stream_chunk_size=20
        )
        for chunk in chunks:
            yield chunk.cpu().numpy()