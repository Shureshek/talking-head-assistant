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
import keyboard

from src.asr.whisper_asr import WhisperTranscriber
from src.voice_clone.clone_manager import VoiceCloneManager
from qwen_tts import Qwen3TTSModel

# Параметры
SAMPLE_RATE = 16000
CHUNK_SIZE = 480  # 30ms для VAD (16000 * 0.03)
VAD_AGGRESSIVENESS = 2  # 0-3, где 3 самый агрессивный
SILENCE_TIMEOUT = 1.5  # секунд тишины для завершения фразы
MIN_SPEECH_DURATION = 0.5  # минимальная длительность речи для обработки

USE_PUSH_TO_TALK = False
PUSH_TO_TALK_KEY = 'space' # клавиша активации
push_to_talk_active = threading.Event()

audio_data_queue = Queue()          # thread-safe очередь из stdlib
stop_event = threading.Event()

# Инициализация TTS модели
tts_model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

class VoiceActivityDetector:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.chunk_size = int(sample_rate * 0.03)  # 30ms
        self.silence_frames_threshold = int(SILENCE_TIMEOUT / 0.03)  # 1.5 секунд тишины
        self.silence_counter = 0
        self.is_speaking = False
        self.audio_buffer = []
        
    def process_chunk(self, audio_chunk):
        """Обработка аудио-чанка через VAD"""
        try:
            # audio_chunk уже должен быть int16
            # Проверяем размер чанка
            if len(audio_chunk) != self.chunk_size:
                # Дополняем или обрезаем до нужного размера
                if len(audio_chunk) < self.chunk_size:
                    # Дополняем нулями
                    padding = np.zeros(self.chunk_size - len(audio_chunk), dtype=np.int16)
                    audio_chunk = np.concatenate([audio_chunk, padding])
                else:
                    audio_chunk = audio_chunk[:self.chunk_size]
            
            # Конвертируем в bytes для VAD
            audio_bytes = audio_chunk.tobytes()
            
            # Проверяем, содержит ли чанк речь
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
                        print("⏸️  Речь закончилась, обрабатываю...")
                        return True  # Сигнал о завершении фразы
            
            # Сохраняем аудио в буфер (конвертируем в float32 для дальнейшей обработки)
            audio_float = audio_chunk.astype(np.float32) / 32768.0
            self.audio_buffer.extend(audio_float)
            return False
            
        except Exception as e:
            print(f"Ошибка VAD: {e}")
            return False
    
    def get_audio(self):
        """Получить накопленное аудио и очистить буфер"""
        if len(self.audio_buffer) == 0:
            return np.array([], dtype=np.float32)
        
        audio = np.array(self.audio_buffer, dtype=np.float32)
        self.audio_buffer = []
        self.silence_counter = 0
        self.is_speaking = False
        return audio

async def record_audio_with_vad():
    """Асинхронная запись аудио с VAD"""
    vad_detector = VoiceActivityDetector()
    stop_recording = threading.Event()
    
    def audio_callback(indata, frames, time, status):
        """Callback для записи аудио"""
        if status:
            print(f"Аудио статус: {status}")
        
        # indata - это буфер байтов, конвертируем в numpy int16
        # frames указывает количество кадров (сэмплов)
        audio_chunk = np.frombuffer(indata, dtype=np.int16)
        
        # Обрабатываем через VAD
        phrase_completed = False

        if USE_PUSH_TO_TALK:
            # запись идёт только если нажата кнопка
            if push_to_talk_active.is_set():
        
                if not vad_detector.is_speaking:
                    vad_detector.is_speaking = True
                    print("🎤 Push-to-talk запись началась")
        
                # вручную копим аудио
                audio_float = audio_chunk.astype(np.float32) / 32768.0
                vad_detector.audio_buffer.extend(audio_float)
        
            else:
                # кнопка отпущена → завершаем фразу
                if vad_detector.is_speaking:
                    vad_detector.is_speaking = False
                    phrase_completed = True
                    print("⏸️ Push-to-talk запись завершена")
        
        else:
            # стандартный VAD режим
            phrase_completed = vad_detector.process_chunk(audio_chunk)
        
        # Если фраза завершена, сигнализируем об этом через очередь
        if phrase_completed:
            try:
                audio_data_queue.put("PHRASE_END")
            except Exception as e:
                print(f"Ошибка при добавлении в очередь: {e}")
    
    # Запускаем поток записи
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
            try:
                # Используем asyncio.sleep для неблокирующего ожидания
                await asyncio.sleep(0.1)
                
                # Проверяем, есть ли сообщение о завершении фразы
                try:
                    item = audio_data_queue.get_nowait()
                    if item == "PHRASE_END":
                        audio_float = vad_detector.get_audio()
                        if len(audio_float) >= SAMPLE_RATE * MIN_SPEECH_DURATION:
                            yield audio_float
                        else:
                            print("Слишком короткий фрагмент, игнорируем")
                except Empty:
                    continue
                    
            except Exception as e:
                print("Ошибка в обработке очереди:", e)
                break
                
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
    """Основной цикл обработки диалога"""
    transcriber = WhisperTranscriber()
    manager = VoiceCloneManager(model=tts_model)
    person_name = "Julia"
    
    print("🎭 Инициализация системы...")
    prompt_items = manager.load_or_create_clone(person_name)
    print("✅ Система готова к работе")
    
    # Основной цикл диалога
    async for recognized_text in transcribe_audio_stream(transcriber):
        if not recognized_text:
            continue
            
        # Генерация ответа
        response_text = f"Вы сказали: {recognized_text}"
        print(f"🤖 Ответ: {response_text}")
        
        # Генерация аудио
        print("🎵 Генерирую речь...")
        wavs, sr = tts_model.generate_voice_clone(
            text=response_text,
            language="Russian",
            voice_clone_prompt=prompt_items,
        )
        
        # Сохраняем аудио
        response_wav = Path("E:/Coding/talking-head-assistant/generated_speech_audio") / f"{person_name}_response.wav"
        sf.write(response_wav, wavs[0], sr)
        print(f"💾 Аудио сохранено: {response_wav}")
        
        # Синхронизация губ
        await run_wav2lip(response_wav, person_name)
        
        # Воспроизведение
        await play_video_with_audio(response_wav)

async def run_wav2lip(audio_path, person_name):
    """Запуск Wav2Lip в отдельном процессе"""
    WAV2LIP_PYTHON = r"C:\Users\Shurik\anaconda3\envs\wav2lip\python.exe"
    WAV2LIP_DIR = Path(r"E:\Coding\talking-head-assistant\third_party\Wav2Lip")
    RESULT_VIDEO_DIR = Path(r"E:\Coding\talking-head-assistant\result_video")
    
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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

async def play_video_with_audio(audio_path):
    """Воспроизведение видео с аудио"""
    # Находим последнее созданное видео
    video_dir = Path(r"E:\Coding\talking-head-assistant\result_video")
    video_files = list(video_dir.glob("*.mp4"))
    if not video_files:
        print("❌ Видеофайл не найден")
        return
    
    latest_video = max(video_files, key=lambda x: x.stat().st_mtime)
    
    print(f"🎬 Воспроизведение: {latest_video.name}")
    
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
    cap = cv2.VideoCapture(str(latest_video))
    if not cap.isOpened():
        print("❌ Не удалось открыть видео")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps > 0 else 30
    
    print("▶️  Воспроизведение (нажмите 'q' для выхода)...")
    
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

def start_keyboard_listener():
    def on_press(event):
        if event.name == PUSH_TO_TALK_KEY:
            push_to_talk_active.set()

    def on_release(event):
        if event.name == PUSH_TO_TALK_KEY:
            push_to_talk_active.clear()

    keyboard.on_press(on_press)
    keyboard.on_release(on_release)

    keyboard.add_hotkey('f8', toggle_record_mode)

def toggle_record_mode():
    global USE_PUSH_TO_TALK
    USE_PUSH_TO_TALK = not USE_PUSH_TO_TALK
    print("🎛 Режим записи:", 
          "Push-to-talk" if USE_PUSH_TO_TALK else "VAD")

async def main():
    """Основная функция"""
    print("=" * 50)
    print("🤖 Голосовой ассистент с автоопределением речи")
    print("=" * 50)
    print("\nКоманды:")
    print("  • Начните говорить - система автоматически определит речь")
    print("  • Нажмите Ctrl+C для выхода")
    print()
    
    # ⌨️ запускаем обработку клавиатуры
    start_keyboard_listener()
    
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