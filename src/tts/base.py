from abc import ABC, abstractmethod
import numpy as np
from typing import Generator, Any

class BaseTTSService(ABC):
    @property
    @abstractmethod
    def sample_rate(self) -> int:
        pass

    @abstractmethod
    def load_model(self):
        """Загрузка весов модели в память"""
        pass

    @abstractmethod
    def get_speaker_embedding(self, audio_path: str) -> Any:
        """
        Превращает путь к аудио в формат, понятный модели
        (latents для XTTS или prompt_struct для Qwen)
        """
        pass

    @abstractmethod
    def generate_stream(self, text: str, speaker_embedding: Any) -> Generator[np.ndarray, None, None]:
        """Генерация аудио (потоковая или цельная)"""
        pass