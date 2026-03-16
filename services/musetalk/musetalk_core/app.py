from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pathlib import Path
import sys

sys.path.append("/app/musetalk_core")
from musetalk_service import MuseTalkService

app = FastAPI(title="MuseTalk Service")
musetalk = MuseTalkService(avatar_id="basic_avatar", batch_size=64)

@app.post("/generate_video")
async def generate_video(audio: UploadFile = File(...)):
    audio_path = Path("/app/volumes/temp") / audio.filename
    audio_path.parent.mkdir(parents=True, exist_ok=True)

    with open(audio_path, "wb") as f:
        f.write(await audio.read())

    video_path = await musetalk.generate(
        audio_path=audio_path,
        output_dir=Path("/app/volumes/result_video"),
        result_name=f"muse_{audio.filename.split('.')[0]}"
    )

    return JSONResponse({"video_path": str(video_path.relative_to("/app/volumes"))})