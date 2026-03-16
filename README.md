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
### 3. Скачивание весов
#### MuseTalk веса
**Вариант 1** 
```
# Запусти скрипт
download_musetalk_weights.sh  
```
**Вариант 2 (ручное скачивание)**  
Вы также можете скачать веса по следующим ссылкам:

[musetalk](https://huggingface.co/TMElyralab/MuseTalk/tree/main)
[sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse/tree/main)
[whisper](https://huggingface.co/openai/whisper-tiny/tree/main)
[dwpose](https://huggingface.co/yzd-v/DWPose/tree/main)
[syncnet](https://huggingface.co/ByteDance/LatentSync/tree/main)
[face-parse-bisent](https://drive.google.com/file/d/154JgKpzCPW82qINcVieuPH3fZ2e0P812/view?pli=1)
[resnet18](https://download.pytorch.org/models/resnet18-5c106cde.pth)

В итоге, эти веса должны храниться в следующем виде:
```
./volumes/models/musetalk
├── musetalk
│   └── musetalk.json
│   └── pytorch_model.bin
├── musetalkV15
│   └── musetalk.json
│   └── unet.pth
├── syncnet
│   └── latentsync_syncnet.pt
├── dwpose
│   └── dw-ll_ucoco_384.pth
├── face-parse-bisent
│   ├── 79999_iter.pth
│   └── resnet18-5c106cde.pth
├── sd-vae
│   ├── config.json
│   └── diffusion_pytorch_model.bin
└── whisper
    ├── config.json
    ├── pytorch_model.bin
    └── preprocessor_config.json
    
```

#### XTTS v2
Скачай с [Hugging Face](https://huggingface.co/coqui/XTTS-v2) и положи в volumes/models/core/xtts_v2

### 4. Запуск сервисов
```bash
docker compose up --build -d
```

### 4. Запуск клиента
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

