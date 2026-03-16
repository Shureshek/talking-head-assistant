# Talking Head Assistant

Локальный голосовой ассистент с автоматическим определением речи, клонированием голоса и синхронизацией губ (lip-sync) 
для аватара.

Предполагаемый функционал проекта:

- распознавание речи (ASR),
- определение активности речи (VAD),
- генерацию ответа голосом с клоном,
- синхронизацию губ через Wav2Lip,
- воспроизведение видео с аудио в реальном времени.

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
### 3. Запуск
```bash
docker compose up --build -d
```
Клиент запускается на хосте через run_talking_head_assistant.bat


## Структура проекта
```
talking-head-assistant/
├── client/                 # клиент (Windows)
├── services/
│   ├── core/               # ASR + LLM + TTS
│   └── musetalk/           # Lip-Sync
├── volumes/                # данные (игнорируется)
├── docker-compose.yml
├── README.md
└── .gitignore
```

