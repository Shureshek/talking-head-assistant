from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse
import numpy as np
import soundfile as sf
import io
import torch
from pathlib import Path
from src.asr.whisper_asr import WhisperTranscriber
from src.tts import create_tts_service
from src.clone_manager import VoiceCloneManager
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.config.settings import TTS_OUTPUT_DIR, VOICE_CLONE_DIR

app = FastAPI(title="Core Service — ASR + LLM + TTS")

# ==================== ИНИЦИАЛИЗАЦИЯ (один раз при старте контейнера) ====================
print("🚀 Запуск Core Service...")

tts_service = create_tts_service()          # выбирает XTTS или Qwen по env
tts_service.load_model()

transcriber = WhisperTranscriber()

manager = VoiceCloneManager()
person_name = "david"                       # можно вынести в .env позже

speaker_embedding = manager.load_or_create_embedding(
    person_name, tts_service, transcriber=transcriber
)
print("✅ Голос клонирован и загружен")

# LLM
model_id = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda:0",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

@app.get("/info")
async def info():
    return {"sample_rate": tts_service.sample_rate, "person_name": person_name}

@app.post("/asr")
async def asr(audio: UploadFile = File(...)):
    content = await audio.read()
    audio_np = np.frombuffer(content, dtype=np.float32)
    text, _ = transcriber.transcribe(audio_np)
    return {"text": text.strip() if text else ""}

@app.post("/llm")
async def llm(history: dict):
    tokenized = tokenizer.apply_chat_template(
        history["messages"], add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to("cuda:0")

    with torch.no_grad():
        output = model.generate(
            input_ids=tokenized.input_ids,
            attention_mask=tokenized.attention_mask,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(output[0][tokenized.input_ids.shape[1]:], skip_special_tokens=True)
    return {"text": response.strip()}

@app.post("/tts")
async def tts(text: str = Form(...)):
    if not text.strip():
        return JSONResponse({"error": "empty"}, status_code=400)

    full_audio = []
    for chunk in tts_service.generate_stream(text, speaker_embedding):
        full_audio.append(chunk)

    if full_audio:
        wav = np.concatenate(full_audio)
        buffer = io.BytesIO()
        sf.write(buffer, wav, tts_service.sample_rate, format="WAV")
        buffer.seek(0)
        return StreamingResponse(buffer, media_type="audio/wav")
    return JSONResponse({"error": "no audio"}, status_code=500)