# Talking Head Assistant

Локальный голосовой ассистент с автоматическим определением речи, клонированием голоса и синхронизацией губ (lip-sync) 
для аватара.
Проект разделён на микросервисы: клиент на Windows + два Docker-контейнера (core + musetalk).

## Предполагаемый функционал проекта:
- Автоматическое определение речи (VAD)
- Push-to-Talk режим (переключается клавишей F8)
- Распознавание речи (Whisper)
- Генерация ответа LLM (Qwen2.5)
- Клонирование голоса (XTTS / Qwen3-TTS)
- Синхронизация губ (MuseTalk)
- Потоковое воспроизведение видео + аудио

## Архитектура

- **client/** — клиент на Windows (микрофон, звук, клавиатура)
- **services/core/** — ASR + LLM + TTS (XTTS / Qwen)
- **services/musetalk/** — Lip-Sync сервис
- **volumes/** — данные (модели, аудио, результат видео)

## Возможности

- Автоопределение речи (VAD) — запись начинается и заканчивается автоматически  
- Push-to-Talk режим (по кнопке, переключаемый)
- Распознавание речи
- Клонирование голоса
- Синхронизация губ
- Проигрывание видео с аудио


## "Быстрый" запуск

### 1. Клонирование

```bash
git clone https://github.com/Shureshek/talking-head-assistant.git
cd talking-head-assistant
```
### 2. Подготовка volumes
```bash
mkdir -p volumes/models/core/xtts_v2
mkdir -p volumes/models/musetalk
mkdir -p volumes/result_video volumes/temp volumes/voices volumes/generated_audio
```
### 3. Скачивание моделей
**MuseTalk модели** — запусти скрипт download_musetalk_weights.sh  
**XTTS v2** — скачай с [Hugging Face](https://huggingface.co/coqui/XTTS-v2) и положи в volumes/models/core/xtts_v2


### 4. Запуск сервисов
```bash
docker compose up --build -d
```

### 5. Запуск клиента
Создайте новое окружение и установите все зависимости:
```bash
pip install -r requirements_client.txt
python main.py
```
Также клиент запускается на хосте через run_talking_head_assistant.bat


## Структура проекта
```
talking-head-assistant-docker/
├── client/                    # Клиент (микрофон, плеер, клавиатура)
│   ├── main.py
│   ├── config.py
│   ├── core_client.py
│   ├── musetalk_client.py
│   ├── player.py
│   └── requirements_client.txt
│
├── services/
│   ├── core/                  # ASR + LLM + TTS
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── app/
│   │       ├── main.py
│   │       └── src/
│   │
│   └── musetalk/              # Lip-Sync сервис
│       ├── Dockerfile
│       ├── requirements_musetalk.txt
│       └── musetalk_core/
│
├── volumes/                   # Модели, аудио, результаты (игнорируется в git)
│   ├── audio_samples/
│   ├── generated_audio/
│   ├── models/
│   ├── result_video/
│   ├── temp/
│   ├── voices/
│   └── download_musetalk_weights.sh
│
├── docker-compose.yml
├── .gitignore
├── .dockerignore
└── README.md
```

