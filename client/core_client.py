import httpx
import numpy as np
import soundfile as sf
import io

class CoreClient:
    def __init__(self):
        self.url = "http://localhost:8000"

    async def transcribe(self, audio_np: np.ndarray) -> str:
        async with httpx.AsyncClient(timeout=30.0) as client:
            files = {"audio": ("audio.raw", audio_np.tobytes(), "application/octet-stream")}
            r = await client.post(f"{self.url}/asr", files=files)
            r.raise_for_status()
            return r.json().get("text", "")

    async def generate_llm(self, history: list) -> str:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(f"{self.url}/llm", json={"messages": history})
            r.raise_for_status()
            return r.json().get("text", "")

    async def tts(self, text: str) -> np.ndarray:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(f"{self.url}/tts", data={"text": text})
            r.raise_for_status()
            buffer = io.BytesIO(r.content)
            data, _ = sf.read(buffer)
            return data.astype(np.float32)