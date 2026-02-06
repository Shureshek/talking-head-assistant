import torch
from qwen_tts import Qwen3TTSModel
import soundfile as sf
from typing import List, Union, Optional, Tuple, Any
from typing import Any


class QwenTTSWrapper:
    def __init__(self, device: str = "cuda:0"):
        self.model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            device_map=device,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

    def create_voice_prompt(
        self,
        ref_audio: str,
        ref_text: Optional[str] = None,
        x_vector_only_mode: bool = False,
    ) -> Any:
        """
        Создаёт переиспользуемый голосовой шаблон (prompt_items).
        Возвращает словарь с тензорами, готовый к сериализации.
        """
        prompt_items = self.model.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only_mode,
        )
        # Убеждаемся, что все тензоры на CPU для безопасной сериализации
        if isinstance(prompt_items, dict):
            for k, v in prompt_items.items():
                if isinstance(v, torch.Tensor):
                    prompt_items[k] = v.cpu()
        return prompt_items

    def generate_voice_clone(
        self,
        text: List[str],
        language: Union[str, List[str]],
        voice_clone_prompt: Any,
        output_path: Optional[str] = None,
    ) -> Tuple[List[torch.Tensor], int]:
        """
        Генерирует речь. Приоритет: voice_clone_prompt > (ref_audio + ref_text)
        """
        if voice_clone_prompt is not None:
            # Перемещаем тензоры обратно на устройство
            device_prompt = {}
            for k, v in voice_clone_prompt.items():
                if isinstance(v, torch.Tensor):
                    device_prompt[k] = v.to(self.device)
                else:
                    device_prompt[k] = v
            voice_clone_prompt = device_prompt

        wavs, sr = self.model.generate_voice_clone(
            text=text,
            language=language,
            voice_clone_prompt=voice_clone_prompt
        )

        if output_path:
            sf.write(output_path, wavs[0].cpu().numpy(), sr)

        return wavs, sr