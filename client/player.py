import sounddevice as sd
import numpy as np
import threading
import queue
import time
import cv2
from pathlib import Path


class AudioStreamPlayer:
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.q = queue.Queue()
        self.stop_event = threading.Event()
        self.stream_thread = threading.Thread(target=self._play_loop, daemon=True)
        self.stream_thread.start()
        print(f"🔈 Аудио-плеер запущен (SR: {sample_rate})")

    def _play_loop(self):
        try:
            with sd.OutputStream(samplerate=self.sample_rate, channels=1, dtype='float32') as stream:
                while not self.stop_event.is_set():
                    try:
                        audio_chunk = self.q.get(timeout=0.1)
                        if audio_chunk is None:
                            break
                        # write блокирует этот фоновый поток, пока звук играет
                        stream.write(audio_chunk)
                    except queue.Empty:
                        continue
        except Exception as e:
            print(f"❌ Не удалось открыть аудиопоток: {e}")

    def play(self, audio_chunk: np.ndarray):
        if audio_chunk is None or len(audio_chunk) == 0: return
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)
        self.q.put(audio_chunk)

    def stop(self):
        self.stop_event.set()
        self.q.put(None)

def play_video_sync(video_path: Path, audio_player: AudioStreamPlayer, audio_data: np.ndarray):
    """Синхронно воспроизводит видео + звук БЕЗ ДВОЙНОГО АУДИО"""
    if not video_path or not video_path.exists():
        print(f"❌ Видеофайл не найден: {video_path}")
        audio_player.play(audio_data)
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Не удалось открыть видео: {video_path}")
        audio_player.play(audio_data)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_duration = 1.0 / fps

    print(f"🎬 Воспроизведение: {video_path.name}")

    # Запускаем звук ОДИН РАЗ
    audio_player.play(audio_data)

    start_time = time.time()
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Avatar", frame)

        frame_idx += 1
        expected = frame_idx * frame_duration
        elapsed = time.time() - start_time

        sleep_time = expected - elapsed
        if sleep_time > 0:
            key = cv2.waitKey(max(1, int(sleep_time * 1000)))
        else:
            key = cv2.waitKey(1)

        if key & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


class VideoStreamPlayer:
    def __init__(self, sample_rate: int = 24000):
        self.audio_player = AudioStreamPlayer(sample_rate)
        self.stop_event = threading.Event()
        # УБИРАЕМ maxsize. Кадров немного (сотни), памяти хватит с запасом.
        self.frame_queue = queue.Queue()
        self.play_thread = threading.Thread(target=self._video_play_loop, daemon=True)
        self.play_thread.start()

    def _video_play_loop(self):
        window_name = "Talking Head"
        cv2.namedWindow(window_name)

        fps = 25.0
        frame_duration = 1.0 / fps
        start_time = None
        frames_played = 0

        while not self.stop_event.is_set():
            try:
                # Таймаут очень маленький, чтобы цикл быстро возвращался к waitKey
                frame = self.frame_queue.get(timeout=0.01)

                if isinstance(frame, str) and frame == "RESET":
                    start_time = None
                    frames_played = 0
                    continue

                if frame is None:
                    break

                if start_time is None:
                    start_time = time.time()

                cv2.imshow(window_name, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

                # Тайминг
                frames_played += 1
                expected_time = frames_played * frame_duration
                elapsed_time = time.time() - start_time
                sleep_time = expected_time - elapsed_time

                if sleep_time > 0:
                    wait_ms = max(1, int(sleep_time * 1000))
                    if cv2.waitKey(wait_ms) & 0xFF == 27:  # Нажатие ESC
                        break
                else:
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

            except queue.Empty:
                # САМОЕ ВАЖНОЕ: Не даем окну зависнуть на Windows!
                if cv2.waitKey(10) & 0xFF == 27:
                    break

        cv2.destroyAllWindows()

    def play_frame(self, frame):
        # Гарантированно добавляем все кадры, которые отдает сеть
        if frame is not None:
            self.frame_queue.put(frame)

    def stop(self):
        self.stop_event.set()
        self.frame_queue.put(None)
        self.audio_player.stop()
        if self.play_thread.is_alive():
            self.play_thread.join(timeout=1)