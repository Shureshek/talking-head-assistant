import asyncio
import queue
from queue import Queue, Empty
import threading
import httpx
import sounddevice as sd
import numpy as np
import soundfile as sf
from pathlib import Path
import cv2
from datetime import datetime
import webrtcvad
from pynput import keyboard as pynput_keyboard
from threading import Thread
import time

# ==================== НОВЫЕ ИМПОРТЫ ====================
from config import CORE_URL, PERSON_NAME, ENABLE_LIP_SYNC
from player import AudioStreamPlayer, play_video_sync
from core_client import CoreClient
from musetalk_client import MuseTalkClient

# ==================== КОНФИГУРАЦИЯ ====================
SAMPLE_RATE = 24000                     # будет обновлено из /info
CHUNK_SIZE = 480
VAD_AGGRESSIVENESS = 2
SILENCE_TIMEOUT = 1.5
MIN_SPEECH_DURATION = 0.5
USE_PUSH_TO_TALK = False
PUSH_TO_TALK_KEY = pynput_keyboard.Key.space

# Глобальные объекты
core_client = CoreClient()
musetalk_client = MuseTalkClient()
global_player = None

push_to_talk_active = threading.Event()
audio_data_queue = Queue()
stop_event = threading.Event()

conversation_history = [
    {"role": "system", "content": "Ты директор web-студии. При любой возможности стараешься предложить и продать сайт на обсуждаемую тематику. Отвечай кратко и по делу."}
]
MAX_HISTORY_LENGTH = 20

TTS_OUTPUT_DIR = Path("volumes/generated_audio")   # можно вынести в config.py
TTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================== VoiceActivityDetector (без изменений) ====================
class VoiceActivityDetector:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.chunk_size = int(sample_rate * 0.03)
        self.silence_frames_threshold = int(SILENCE_TIMEOUT / 0.03)
        self.silence_counter = 0
        self.is_speaking = False
        self.audio_buffer = []

        self.last_use_ptt = False
        self.last_ptt_state = False

    def process_chunk(self, audio_chunk):
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
                    print("🎤 Речь обнаружена...")
            else:
                if self.is_speaking:
                    self.silence_counter += 1
                    if self.silence_counter >= self.silence_frames_threshold:
                        self.is_speaking = False
                        print("⏸️ Речь закончилась, обрабатываю...")
                        return True

            audio_float = audio_chunk.astype(np.float32) / 32768.0
            self.audio_buffer.extend(audio_float)
            return False
        except Exception as e:
            print(f"Ошибка VAD: {e}")
            return False

    def get_audio(self):
        if not self.audio_buffer:
            return np.array([], dtype=np.float32)
        audio = np.array(self.audio_buffer, dtype=np.float32)
        self.audio_buffer.clear()
        self.silence_counter = 0
        self.is_speaking = False
        return audio


# ==================== record_audio_with_vad (без изменений) ====================
async def record_audio_with_vad():
    vad_detector = VoiceActivityDetector()
    stop_recording = threading.Event()

    def audio_callback(indata, frames, time, status):
        if status:
            print(f"Аудио статус: {status}")

        audio_chunk = np.frombuffer(indata, dtype=np.int16)
        phrase_completed = False

        current_use_ptt = USE_PUSH_TO_TALK
        current_ptt = push_to_talk_active.is_set()

        if current_use_ptt != vad_detector.last_use_ptt:
            print(f"🎛 Режим изменён → {'Push-to-talk' if current_use_ptt else 'VAD'}")
            vad_detector.last_use_ptt = current_use_ptt

        if current_use_ptt:
            if current_ptt != vad_detector.last_ptt_state:
                if current_ptt:
                    vad_detector.is_speaking = True
                    print("🎤 Push-to-talk запись началась")
                else:
                    if vad_detector.is_speaking:
                        vad_detector.is_speaking = False
                        phrase_completed = True
                        print("⏸️ Push-to-talk запись завершена")
                vad_detector.last_ptt_state = current_ptt

            if current_ptt:
                audio_float = audio_chunk.astype(np.float32) / 32768.0
                vad_detector.audio_buffer.extend(audio_float)
        else:
            phrase_completed = vad_detector.process_chunk(audio_chunk)

        if phrase_completed:
            try:
                audio_to_yield = vad_detector.get_audio()
                if len(audio_to_yield) >= SAMPLE_RATE * MIN_SPEECH_DURATION:
                    audio_data_queue.put(audio_to_yield)
                else:
                    print("Слишком короткий фрагмент, игнорируем")
            except Exception as e:
                print(f"Ошибка при добавлении аудио в очередь: {e}")

    def run_recording():
        with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=CHUNK_SIZE,
                               dtype='int16', channels=1, callback=audio_callback):
            while not stop_recording.is_set():
                sd.sleep(200)

    recording_thread = threading.Thread(target=run_recording, daemon=True)
    recording_thread.start()

    try:
        while not stop_event.is_set():
            await asyncio.sleep(0.1)
            try:
                audio_float = audio_data_queue.get_nowait()
                yield audio_float
            except Empty:
                continue
    finally:
        stop_recording.set()
        if recording_thread.is_alive():
            recording_thread.join(timeout=1)
        print("Запись аудио остановлена")


# ==================== transcribe_audio_stream (новая версия) ====================
async def transcribe_audio_stream():
    print("🎧 Слушаю... Говорите что-нибудь")
    async for audio_data in record_audio_with_vad():
        if len(audio_data) == 0:
            continue
        print(f"📊 Получено аудио: {len(audio_data) / SAMPLE_RATE:.2f} сек")

        try:
            text = await core_client.transcribe(audio_data)
            if text and text.strip():
                print(f"📝 Распознано: {text.strip()}")
                yield text.strip()
        except Exception as e:
            print(f"❌ Ошибка распознавания: {e}")


# ==================== process_conversation (новая версия) ====================
async def process_conversation():
    global conversation_history

    async for recognized_text in transcribe_audio_stream():
        if not recognized_text:
            continue

        conversation_history.append({"role": "user", "content": recognized_text})

        if len(conversation_history) > MAX_HISTORY_LENGTH:
            conversation_history = [conversation_history[0]] + conversation_history[-MAX_HISTORY_LENGTH + 1:]

        print("🤖 Генерирую ответ с LLM...")

        response_text = await core_client.generate_llm(conversation_history)

        # Разбиваем на предложения и сразу отправляем в TTS (низкая задержка)
        sentences = [s.strip() + "." for s in response_text.split(". ") if s.strip()]
        for sentence in sentences:
            print(f"🤖 Фраза для TTS: {sentence}")
            await stream_and_play(sentence)

        conversation_history.append({"role": "assistant", "content": response_text.strip()})


# ==================== stream_and_play (новая версия) ====================
async def stream_and_play(text_chunk: str):
    if not text_chunk: return
    print(f"🎵 TTS: {text_chunk}")

    try:
        audio_np = await core_client.tts(text_chunk)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = TTS_OUTPUT_DIR / f"response_{PERSON_NAME}_{timestamp}.wav"
        sf.write(str(file_path), audio_np, SAMPLE_RATE)

        if ENABLE_LIP_SYNC:
            print("🎬 Запуск видео + аудио...")

            # 1. Сохраняем аудио во временный файл для MuseTalk
            temp_wav = Path("volumes/temp/tts_out.wav")
            temp_wav.parent.mkdir(parents=True, exist_ok=True)
            sf.write(temp_wav, audio_np, SAMPLE_RATE)

            # 2. Отправляем на MuseTalk
            video_path = await musetalk_client.generate_video(temp_wav)

            # 3. Воспроизводим видео + звук синхронно (ОДИН РАЗ)
            if video_path and video_path.exists():
                play_video_sync(video_path, global_player, audio_np)
            else:
                print("❌ Видео не получено, играем только звук")
                global_player.play(audio_np)
        else:
            # Без липсинка — просто звук
            global_player.play(audio_np)

    except Exception as e:
        print(f"❌ Ошибка стриминга: {e}")


# ==================== play_video_only (без изменений) ====================
async def play_video_only(video_path: Path, original_audio_path: Path = None):
    if not video_path.exists():
        print("❌ Видеофайл не найден")
        return

    print(f"🎬 Воспроизведение: {video_path.name}")

    wav_path = original_audio_path if original_audio_path and original_audio_path.exists() else video_path.with_suffix(".wav")

    def play_audio():
        try:
            data, sr = sf.read(str(wav_path))
            sd.play(data, sr)
            sd.wait()
        except Exception as e:
            print(f"❌ Ошибка воспроизведения аудио: {e}")

    audio_thread = threading.Thread(target=play_audio, daemon=True)
    audio_thread.start()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("❌ Не удалось открыть видео")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps > 0 else 30

    print("▶️ Воспроизведение (q — выход)...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Avatar", frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
        if not audio_thread.is_alive():
            break

    cap.release()
    cv2.destroyAllWindows()


# ==================== Клавиатура (без изменений) ====================
def toggle_record_mode():
    global USE_PUSH_TO_TALK
    USE_PUSH_TO_TALK = not USE_PUSH_TO_TALK
    print("🎛 Режим записи:", "Push-to-talk" if USE_PUSH_TO_TALK else "VAD")


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
    listener.start()
    listener.wait()



# ==================== main ====================
async def main():
    global global_player, SAMPLE_RATE
    print("=" * 50)
    print("🤖 Голосовой ассистент (Docker версия)")
    print("=" * 50)

    # Получаем sample_rate из core-сервиса
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{CORE_URL}/info")
        SAMPLE_RATE = r.json()["sample_rate"]

    global_player = AudioStreamPlayer(SAMPLE_RATE)

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
    try:
        import webrtcvad
        print("✅ VAD загружен")
    except ImportError:
        print("❌ Установите webrtcvad: pip install webrtcvad")
        exit(1)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nПрограмма завершена пользователем")
        stop_event.set()
        if global_player:
            global_player.stop()
    except Exception as e:
        print(f"Ошибка запуска: {e}")