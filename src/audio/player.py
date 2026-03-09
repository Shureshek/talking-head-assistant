import sounddevice as sd
import numpy as np
import threading
from queue import Queue, Empty


class AudioStreamPlayer:
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.q = Queue()
        self.stop_event = threading.Event()
        self.stream_thread = threading.Thread(target=self._play_loop, daemon=True)
        self.stream_thread.start()
        print(f"🔈 Аудио-плеер запущен (SR: {sample_rate})")

    def _play_loop(self):
        """Фоновый поток, который кормит аудиокарту данными"""
        try:
            # Открываем поток на вывод один раз и держим его
            with sd.OutputStream(samplerate=self.sample_rate, channels=1, dtype='float32') as stream:
                while not self.stop_event.is_set():
                    try:
                        # Ждем данные из очереди (timeout чтобы проверять stop_event)
                        audio_chunk = self.q.get(timeout=0.5)

                        # Если пришел сигнал остановки (None)
                        if audio_chunk is None:
                            break

                        # write блокирует выполнение ЭТОГО потока, пока звук не проиграется,
                        # но основной поток программы (main) в это время свободен!
                        stream.write(audio_chunk)

                    except Empty:
                        continue
                    except Exception as e:
                        print(f"❌ Ошибка воспроизведения: {e}")
        except Exception as e:
            print(f"❌ Не удалось открыть аудиопоток: {e}")

    def play(self, audio_chunk: np.ndarray):
        """Добавить кусок аудио в очередь (не блокирует основной поток)"""
        if audio_chunk is None or len(audio_chunk) == 0:
            return
        # Приводим к float32, если пришло что-то другое
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)

        self.q.put(audio_chunk)

    def stop(self):
        """Корректная остановка плеера"""
        self.stop_event.set()
        self.q.put(None)  # Разблокируем get, если он ждет
        if self.stream_thread.is_alive():
            self.stream_thread.join()