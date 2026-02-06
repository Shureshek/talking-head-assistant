from pathlib import Path
from typing import Any
from src.config.settings import (
    DATA_AUDIO_DIR,
    VOICE_CLONE_DIR,
    DEFAULT_PERSON,
    SUPPORTED_AUDIO_EXT
)

from src.asr.whisper_asr import WhisperTranscriber
import pickle

class VoiceCloneManager:

    def __init__(self, model):
        VOICE_CLONE_DIR.mkdir(parents=True, exist_ok=True)
        self.transcriber = WhisperTranscriber()
        self.model = model

    # ------------------------------
    # публичный метод
    # ------------------------------
    def load_or_create_clone(self, person_name: str = DEFAULT_PERSON) -> Path:

        clone_path = VOICE_CLONE_DIR / f"{person_name}.pkl"
        print("clone_path:", clone_path)

        if clone_path.exists():
            with open(clone_path, "rb") as f:
                prompt_items = pickle.load(f)
            return prompt_items

        audio_path = self._find_person_audio(person_name)
        ref_text = self._transcribe_audio(audio_path)
        print(ref_text)

        return self._create_clone(person_name, audio_path, ref_text, clone_path)

    # ------------------------------
    # поиск аудио личности
    # ------------------------------
    def _find_person_audio(self, person_name: str) -> Path:
            person_dir = DATA_AUDIO_DIR / person_name
    
            if not person_dir.exists():
                raise FileNotFoundError(f"No person directory: {person_dir}")
    
            files = sorted(person_dir.iterdir())
            for file in files:
                if file.suffix.lower() in SUPPORTED_AUDIO_EXT:
                    return file
    
            raise FileNotFoundError(f"No audio files in {person_dir}")

    # ------------------------------
    # Transcribe audio to get ref_text using Whisper
    # ------------------------------
    def _transcribe_audio(self, audio_path: Path) -> str:
        text, _ = self.transcriber.transcribe(str(audio_path), language="ru")
        return text
        
    # ------------------------------
    # создание клона
    # ------------------------------
    def _create_clone(self, person_name: str, audio_path: Path, ref_text: str, clone_path: Path) -> Path:
        print(f"Creating clone for {person_name} from {audio_path}")

        # Create voice clone prompt
        prompt_items = self.model.create_voice_clone_prompt(
            ref_audio=str(audio_path),
            ref_text="Ой, кстати, из забавного, я не умею готовить пюре. Оно у меня всегда получается отвратительным, поэтому я его не готовлю и доверяю всем. Вот кому угодно, лишь бы не я это делала. Это вот что-то волшебное. Я всё, что угодно могу, но пюре и ещё пару блюд, которые вот простейшие, у меня не получаются. А, у меня омлет не получается. Звучит, конечно, странно, но то ли он пригорает, то ли ещё что-то, то ли ещё. И, короче, всегда у меня неудачные омлет и пюре, поэтому я подумала, потом как-нибудь научусь. Пока буду доверять всем подряд. У меня так есть история с подругой, с её мамой. Мы дружим с Машей с пяти лет, и она живёт прямо возле школы, и постоянно перед школой я к ней заходила. У меня была музыкалка, черчение, и потом после музыкалки, черчения я шла к Маше на часа два. Мы обычно завтракали у неё. И вот её мать готовит просто отвратительно. Вот всё, что только можно, она готовит отвратительно. Она ещё какой-то сумасшедший экспериментатор, у неё 500 новых рецептов, 500 новых вариаций, и всегда это какой-то просто дичайший отстой, я не знаю, это всегда невкусно. Вот всегда, когда прихожу к тёте Марине, бля, это означает, что у неё будет тяжело жить.",
            x_vector_only_mode=False,
        )

        clone_path.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = clone_path.with_suffix(".tmp")
    
        # Save the prompt_items
        with open(tmp_path, "wb") as f:
                pickle.dump(prompt_items, f)
        
        tmp_path.replace(clone_path)
        
        return prompt_items