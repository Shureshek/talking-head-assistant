import pickle
from pathlib import Path
from src.config.settings import DATA_AUDIO_DIR, SUPPORTED_AUDIO_EXT, VOICE_CLONE_DIR, DEFAULT_PERSON


class VoiceCloneManager:
    def __init__(self):
        # Создаем папку для клонов, если нет
        VOICE_CLONE_DIR.mkdir(parents=True, exist_ok=True)

    def load_or_create_embedding(self, person_name: str, tts_service, transcriber=None) -> any:
        """
        Универсальный метод:
        1. Проверяет, есть ли сохраненный файл клона для текущей модели.
        2. Если есть -> загружает.
        3. Если нет -> находит аудио, генерирует через сервис, сохраняет.
        """

        # 1. Формируем имя файла на основе имени модели (чтобы Qwen и XTTS не перезатирали друг друга)
        # Получим что-то типа: "Julia_XTTSService.pkl" или "Julia_QwenTTSService.pkl"
        model_tag = tts_service.__class__.__name__
        clone_filename = f"{person_name}_{model_tag}.pkl"
        clone_path = VOICE_CLONE_DIR / clone_filename
        self.transcriber = transcriber

        # 2. Попытка загрузки из кэша
        if clone_path.exists():
            print(f"📂 Загрузка кэшированного голоса: {clone_path}")
            try:
                with open(clone_path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"⚠️ Ошибка чтения кэша ({e}), пересоздаем...")

        # 3. Если кэша нет - создаем
        print(f"🧬 Генерирую новый слепок голоса для {model_tag}...")

        # Находим аудиофайл
        audio_path = self._find_person_audio(person_name)
        print(f"🎙 Используем референс: {audio_path.name}")

        # ТРАНСКРИБАЦИЯ WHISPER
        ref_text = None
        # Проверяем, есть ли транскрайбер (он инициализируется в __init__)
        if hasattr(self, 'transcriber') and self.transcriber:
            print("✍️ Транскрибирую референс для улучшения качества клона...")
            try:
                # transcribe возвращает (text, info)
                text, _ = self.transcriber.transcribe(str(audio_path))
                ref_text = text.strip()
                print(f"📜 Распознанный текст: \"{ref_text}\"")
            except Exception as e:
                print(f"⚠️ Ошибка транскрибации референса: {e}")

        # Просим сервис создать эмбеддинг (Service сам знает, как это делать: latents или prompt)
        embedding = tts_service.get_speaker_embedding(audio_path, ref_text)

        # 4. Сохраняем на диск
        try:
            with open(clone_path, "wb") as f:
                pickle.dump(embedding, f)
            print(f"💾 Голос сохранен: {clone_filename}")
        except Exception as e:
            print(f"⚠️ Не удалось сохранить кэш голоса: {e}")

        return embedding

    def _find_person_audio(self, person_name: str) -> Path:
        """Поиск первого подходящего аудиофайла в папке персонажа"""
        person_dir = DATA_AUDIO_DIR / person_name

        if not person_dir.exists():
            raise FileNotFoundError(f"Папка персонажа не найдена: {person_dir}")

        # Ищем файлы с поддерживаемыми расширениями
        for file in sorted(person_dir.iterdir()):
            if file.suffix.lower() in SUPPORTED_AUDIO_EXT:
                return file

        raise FileNotFoundError(f"В папке {person_name} нет аудиофайлов {SUPPORTED_AUDIO_EXT}")

    # Публичный метод для явного получения пути (если нужно в main)
    def get_reference_audio_path(self, person_name: str) -> Path:
        return self._find_person_audio(person_name)