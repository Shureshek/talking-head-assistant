import asyncio
import queue
from queue import Queue, Empty
import threading
import sounddevice as sd
import numpy as np
import soundfile as sf
import subprocess
from pathlib import Path
import cv2
from datetime import datetime
import torch
import webrtcvad
from collections import deque
from pynput import keyboard as pynput_keyboard  # Замена на pynput для надёжного key listening
from src.asr.whisper_asr import WhisperTranscriber
from src.voice_clone.clone_manager import VoiceCloneManager
from qwen_tts import Qwen3TTSModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

# Параметры
SAMPLE_RATE = 16000
CHUNK_SIZE = 480  # 30ms для VAD (16000 * 0.03)
VAD_AGGRESSIVENESS = 2  # 0-3, где 3 самый агрессивный
SILENCE_TIMEOUT = 1.5  # секунд тишины для завершения фразы
MIN_SPEECH_DURATION = 0.5  # минимальная длительность речи для обработки
USE_PUSH_TO_TALK = False
PUSH_TO_TALK_KEY = pynput_keyboard.Key.space  # Теперь используем pynput Key
push_to_talk_active = threading.Event()
audio_data_queue = Queue()  # thread-safe очередь из stdlib
stop_event = threading.Event()

# Глобальная история разговора
conversation_history = [
    {"role": "system", "content": "Ты кокетливая флиртующая девушка по имени Юля. Говори на русском языке. Отвечай кратко и по делу."}
]
MAX_HISTORY_LENGTH = 20  # Ограничение для памяти

# Инициализация TTS модели
tts_model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

# Инициализация LLM (Qwen1.5-7B-Chat локально)
model_id = "Qwen/Qwen1.5-7B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda:0",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

"""Основной цикл обработки диалога"""
transcriber = WhisperTranscriber()
manager = VoiceCloneManager(model=tts_model)
person_name = "Julia"
print("🎭 Инициализация системы...")
prompt_items = manager.load_or_create_clone(person_name)
print("✅ Система готова к работе")

# ==================== VoiceActivityDetector ====================
class VoiceActivityDetector:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.chunk_size = int(sample_rate * 0.03)
        self.silence_frames_threshold = int(SILENCE_TIMEOUT / 0.03)
        self.silence_counter = 0
        self.is_speaking = False
        self.audio_buffer = []

        # Для корректного переключения режимов и edge-detection PTT
        self.last_use_ptt = False          # последний режим (VAD / PTT)
        self.last_ptt_state = False        # последнее состояние кнопки PTT

    def process_chunk(self, audio_chunk):
        """VAD-режим — возвращает True, если фраза закончилась по тишине"""
        try:
            if len(audio_chunk) != self.chunk_size:
                if len(audio_chunk) < self.chunk_size:
                    padding = np.zeros(self.chunk_size - len(audio_chunk), dtype=np.int16)
                    audio_chunk = np.concatenate([audio_chunk, padding])
                else:
                    audio_chunk = audio_chunk[:self.chunk_size]

            audio_bytes = audio_chunk.tobytes()
            is_speech = self.vad.is_speech(audio_bytes, self.sample_rate)

            if is_speech:
                self.silence_counter = 0
                if not self.is_speaking:
                    self.is_speaking = True
                    print("🎤 Речь обнаружена, начинаю запись...")
            else:
                if self.is_speaking:
                    self.silence_counter += 1
                    if self.silence_counter >= self.silence_frames_threshold:
                        self.is_speaking = False
                        print("⏸️ Речь закончилась (VAD), обрабатываю...")
                        return True

            # Сохраняем в буфер
            audio_float = audio_chunk.astype(np.float32) / 32768.0
            self.audio_buffer.extend(audio_float)
            return False

        except Exception as e:
            print(f"Ошибка VAD: {e}")
            return False

    def get_audio(self):
        """Извлекает накопленное аудио и полностью очищает буфер"""
        if not self.audio_buffer:
            return np.array([], dtype=np.float32)
        audio = np.array(self.audio_buffer, dtype=np.float32)
        self.audio_buffer.clear()
        self.silence_counter = 0
        self.is_speaking = False
        return audio

# ==================== record_audio_with_vad ====================
async def record_audio_with_vad():
    """Асинхронная запись с VAD + PTT. Теперь кладём готовое аудио в очередь."""
    vad_detector = VoiceActivityDetector()
    stop_recording = threading.Event()

    def audio_callback(indata, frames, time, status):
        if status:
            print(f"Аудио статус: {status}")

        audio_chunk = np.frombuffer(indata, dtype=np.int16)
        phrase_completed = False

        current_use_ptt = USE_PUSH_TO_TALK
        current_ptt = push_to_talk_active.is_set()

        # 1. Обработка смены режима
        if current_use_ptt != vad_detector.last_use_ptt:
            print(f"🎛 Режим изменён → {'Push-to-talk' if current_use_ptt else 'VAD'}")
            if current_use_ptt and vad_detector.is_speaking and not current_ptt:
                # VAD → PTT и кнопка НЕ зажата → сразу завершаем текущую фразу
                vad_detector.is_speaking = False
                phrase_completed = True
                print("⏸️ Завершаю фразу из-за смены режима (кнопка не зажата)")
            vad_detector.last_use_ptt = current_use_ptt

        # 2. Логика текущего режима
        if current_use_ptt:  # Push-to-talk
            # Edge-detection нажатия/отпускания
            if current_ptt != vad_detector.last_ptt_state:
                if current_ptt:  # нажали
                    vad_detector.is_speaking = True
                    print("🎤 Push-to-talk запись началась")
                else:  # отпустили
                    if vad_detector.is_speaking:
                        vad_detector.is_speaking = False
                        phrase_completed = True
                        print("⏸️ Push-to-talk запись завершена")
                vad_detector.last_ptt_state = current_ptt

            # Копируем аудио только пока кнопка зажата
            if current_ptt:
                audio_float = audio_chunk.astype(np.float32) / 32768.0
                vad_detector.audio_buffer.extend(audio_float)

        else:  # VAD-режим
            phrase_completed = vad_detector.process_chunk(audio_chunk)

        # 3. Если фраза завершилась — сразу извлекаем аудио и кладём в очередь
        if phrase_completed:
            try:
                audio_to_yield = vad_detector.get_audio()
                if len(audio_to_yield) >= SAMPLE_RATE * MIN_SPEECH_DURATION:
                    audio_data_queue.put(audio_to_yield)          # ← теперь кладём само аудио!
                else:
                    print("Слишком короткий фрагмент, игнорируем")
            except Exception as e:
                print(f"Ошибка при добавлении аудио в очередь: {e}")

    # Запуск потока записи
    def run_recording():
        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=CHUNK_SIZE,
            dtype='int16',
            channels=1,
            callback=audio_callback
        ):
            while not stop_recording.is_set():
                sd.sleep(200)

    recording_thread = threading.Thread(target=run_recording, daemon=True)
    recording_thread.start()

    try:
        while not stop_event.is_set():
            await asyncio.sleep(0.1)
            try:
                # Теперь в очереди лежит готовое np.array аудио
                audio_float = audio_data_queue.get_nowait()
                yield audio_float
            except Empty:
                continue
    finally:
        stop_recording.set()
        if recording_thread.is_alive():
            recording_thread.join(timeout=1)
        print("Запись аудио остановлена")

async def transcribe_audio_stream(transcriber):
    """Потоковое распознавание речи"""
    print("🎧 Слушаю... Говорите что-нибудь")
   
    async for audio_data in record_audio_with_vad():
        if len(audio_data) == 0:
            continue
           
        print(f"📊 Получено аудио: {len(audio_data)/SAMPLE_RATE:.2f} сек")
       
        # Распознаем речь
        try:
            text, info = transcriber.transcribe(audio_data)
            if text and text.strip():
                print(f"📝 Распознано: {text.strip()}")
                yield text.strip()
        except Exception as e:
            print(f"❌ Ошибка распознавания: {e}")

async def process_conversation():
    global conversation_history  # ← Добавьте это, чтобы модифицировать глобальную переменную
    
    # Основной цикл диалога
    async for recognized_text in transcribe_audio_stream(transcriber):
        if not recognized_text:
            continue
        
        # Добавляем в историю
        conversation_history.append({"role": "user", "content": recognized_text})
        
        # Ограничиваем историю
        if len(conversation_history) > MAX_HISTORY_LENGTH:
            conversation_history = [conversation_history[0]] + conversation_history[-MAX_HISTORY_LENGTH + 1:]  # Сохраняем system prompt
        
        print("🤖 Генерирую ответ с LLM...")
        
        # Подготовка промпта для Qwen с attention_mask
        tokenized = tokenizer.apply_chat_template(
            conversation_history,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True
        ).to("cuda:0")
        
        # Streaming генерация
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        generation_kwargs = {
            "input_ids": tokenized.input_ids,
            "attention_mask": tokenized.attention_mask,
            "streamer": streamer,
            "max_new_tokens": 200,
            "do_sample": True,
            "temperature": 0.7,
        }
        
        # Запускаем генерацию в отдельном thread
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Собираем ответ по частям и стрим в TTS
        response_text = ""
        chunk = ""
        
        for new_token in streamer:
            if new_token:
                response_text += new_token
                chunk += new_token
                # Проверяем на конец предложения
                if chunk.endswith(('.', '!', '?')):
                    print(f"🤖 Часть ответа: {chunk}")
                    # Асинхронно обрабатываем чанк (TTS + play)
                    await generate_and_play_tts(chunk.strip())
                    chunk = ""
        
        # Если остался незавершённый чанк в конце
        if chunk.strip():
            print(f"🤖 Последняя часть: {chunk}")
            await generate_and_play_tts(chunk.strip())
        
        # Добавляем полный ответ в историю
        conversation_history.append({"role": "assistant", "content": response_text.strip()})
        
        # Ждём завершения thread
        thread.join()

async def generate_and_play_tts(text_chunk):
    if not text_chunk:
        return
    
    print(f"🎵 Генерирую речь для: {text_chunk}")
    wavs, sr = tts_model.generate_voice_clone(
        text=text_chunk,
        language="Russian",
        voice_clone_prompt=prompt_items,
    )
    
    # Сохраняем аудио
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    response_wav = Path("E:/Coding/talking-head-assistant/generated_speech_audio") / f"{person_name}_chunk_{stamp}.wav"
    sf.write(response_wav, wavs[0], sr)
    print(f"💾 Аудио сохранено: {response_wav}")
    
    # Синхронизация губ
    video_path = await run_wav2lip(response_wav, person_name)
    
    # Воспроизведение (sequential, т.к. await)
    if video_path:
        await play_video_with_audio(video_path, response_wav)

async def run_wav2lip(audio_path, person_name):
    """Запуск Wav2Lip в отдельном процессе"""
    WAV2LIP_PYTHON = r"C:\Users\Shurik\anaconda3\envs\wav2lip\python.exe"
    WAV2LIP_DIR = Path(r"E:\Coding\talking-head-assistant\third_party\Wav2Lip")
    RESULT_VIDEO_DIR = Path(r"E:\Coding\talking-head-assistant\result_video")
   
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    result_video_path = RESULT_VIDEO_DIR / f"{person_name}_{stamp}.mp4"
   
    print("👄 Запускаю синхронизацию губ...")
   
    # Запускаем в отдельном потоке, чтобы не блокировать asyncio
    def run_command():
        wav2lip_cmd = [
            WAV2LIP_PYTHON,
            "inference.py",
            "--checkpoint_path", "checkpoints/wav2lip_gan.pth",
            "--face", "face_video.mp4",
            "--audio", str(audio_path),
            "--outfile", str(result_video_path),
            "--nosmooth"
        ]
       
        try:
            result = subprocess.run(
                wav2lip_cmd,
                cwd=WAV2LIP_DIR,
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            if result.returncode == 0:
                print("✅ Синхронизация завершена")
                return result_video_path
            else:
                print(f"❌ Ошибка Wav2Lip: {result.stderr}")
                return None
        except Exception as e:
            print(f"❌ Ошибка запуска Wav2Lip: {e}")
            return None
   
    # Запускаем в пуле потоков
    loop = asyncio.get_event_loop()
    try:
        video_path = await loop.run_in_executor(None, run_command)
        return video_path
    except Exception as e:
        print(f"❌ Ошибка при выполнении Wav2Lip: {e}")
        return None

async def play_video_with_audio(video_path, audio_path):
    """Воспроизведение видео с аудио"""
    if not video_path.exists():
        print("❌ Видеофайл не найден")
        return
   
    print(f"🎬 Воспроизведение: {video_path.name}")
   
    # Функция для воспроизведения аудио
    def play_audio():
        try:
            data, sr = sf.read(audio_path)
            sd.play(data, sr)
            sd.wait()
        except Exception as e:
            print(f"❌ Ошибка воспроизведения аудио: {e}")
   
    # Запускаем аудио в отдельном потоке
    audio_thread = threading.Thread(target=play_audio, daemon=True)
    audio_thread.start()
   
    # Воспроизводим видео через OpenCV
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("❌ Не удалось открыть видео")
        return
   
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps > 0 else 30
   
    print("▶️ Воспроизведение (нажмите 'q' для выхода)...")
   
    while True:
        ret, frame = cap.read()
        if not ret:
            break
       
        cv2.imshow("Avatar", frame)
       
        # Проверяем нажатие клавиши 'q'
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
       
        # Также проверяем, закончилось ли аудио
        if not audio_thread.is_alive():
            break
   
    cap.release()
    cv2.destroyAllWindows()

def toggle_record_mode():
    global USE_PUSH_TO_TALK
    USE_PUSH_TO_TALK = not USE_PUSH_TO_TALK
    print("🎛 Режим записи:",
          "Push-to-talk" if USE_PUSH_TO_TALK else "VAD")

def start_keyboard_listener():
    def on_press(key):
        if key == PUSH_TO_TALK_KEY:
            push_to_talk_active.set()
        elif key == pynput_keyboard.Key.f8:
            toggle_record_mode()

    def on_release(key):
        if key == PUSH_TO_TALK_KEY:
            push_to_talk_active.clear()

    listener = pynput_keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()  # Запускает в отдельном потоке
    listener.wait()   # Ждёт, пока не остановится (но поскольку daemon, ок)

async def main():
    """Основная функция"""
    print("=" * 50)
    print("🤖 Голосовой ассистент с автоопределением речи")
    print("=" * 50)
    print("\nКоманды:")
    print(" • Начните говорить - система автоматически определит речь")
    print(" • Нажмите Ctrl+C для выхода")
    print()
   
    # ⌨️ запускаем обработку клавиатуры в фоне
    keyboard_thread = threading.Thread(target=start_keyboard_listener, daemon=True)
    keyboard_thread.start()
   
    try:
        await process_conversation()
    except KeyboardInterrupt:
        print("\n\n👋 Завершение работы...")
        stop_event.set()
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        stop_event.set()

if __name__ == "__main__":
    # Проверка зависимостей
    try:
        import webrtcvad
        print("✅ VAD библиотека загружена")
    except ImportError:
        print("❌ Установите webrtcvad: pip install webrtcvad")
        exit(1)
   
    # Запуск асинхронного приложения
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nПрограмма завершена пользователем")
    except Exception as e:
        print(f"Ошибка при запуске: {e}")