import httpx
from pathlib import Path


class MuseTalkClient:
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url.rstrip("/")

    async def generate_video(self, audio_path: Path) -> Path | None:
        url = f"{self.base_url}/generate_video"
        print(f"DEBUG: [Client] Отправка {audio_path.name} на генерацию видео...")

        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                with open(audio_path, "rb") as f:
                    r = await client.post(url, files={"audio": f})

                if r.status_code == 200:
                    data = r.json()
                    # Возвращаем локальный путь к файлу (предполагается общая папка volumes)
                    return Path("volumes") / data.get("video_path", "")
                else:
                    print(f"❌ Ошибка сервера: {r.status_code} - {r.text}")
        except Exception as e:
            print(f"❌ Ошибка соединения с MuseTalk: {e}")

        return None