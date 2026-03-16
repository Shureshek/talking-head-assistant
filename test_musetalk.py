# test_musetalk.py
import httpx
import time
from pathlib import Path

# Конфигурация
AUDIO_FILE = Path("volumes/temp/response_julia_20260315_222811.wav")
MUSE_URL = "http://localhost:8001/generate_video"


def main():
    if not AUDIO_FILE.exists():
        print(f"❌ Аудиофайл не найден: {AUDIO_FILE.absolute()}")
        return

    print(f"Тест MuseTalk через FastAPI")
    print(f"Входной аудио: {AUDIO_FILE}")
    print(f"URL: {MUSE_URL}\n")

    start_time = time.time()

    try:
        with open(AUDIO_FILE, "rb") as f:
            files = {"audio": (AUDIO_FILE.name, f, "audio/wav")}
            response = httpx.post(MUSE_URL, files=files, timeout=600.0)

        if response.status_code == 200:
            data = response.json()
            video_path = Path("volumes") / data.get("video_path", "")

            if video_path.exists():
                duration = time.time() - start_time
                print(f"\n✅ Успех!")
                print(f"Видео сохранено: {video_path.absolute()}")
                print(f"Время выполнения: {duration:.2f} сек")
            else:
                print("❌ Путь к видео получен, но файл не существует")
                print(f"Полученный путь: {video_path}")
        else:
            print(f"❌ Ошибка сервера: {response.status_code}")
            print(response.text)

    except Exception as e:
        print(f"❌ Ошибка запроса:")
        print(e)


if __name__ == "__main__":
    main()