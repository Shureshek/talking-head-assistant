import sys
import os
import time
import queue
import threading
import copy
import pickle
import glob
import shutil
import cv2
import numpy as np
import torch
import asyncio
from pathlib import Path
from tqdm import tqdm
import subprocess

# Настраиваем импорты MuseTalk
MUSE_DIR = Path("/app/musetalk_core")
if str(MUSE_DIR) not in sys.path:
    sys.path.append(str(MUSE_DIR))

from musetalk.utils.utils import load_all_model, datagen
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.blending import get_image_blending
from musetalk.utils.preprocessing import read_imgs
from transformers import WhisperModel


class MuseTalkService:
    def __init__(self, avatar_id="basic_avatar", batch_size=16):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.avatar_id = avatar_id
        self.fps = 25

        # Оптимизации CUDA для RTX 4090
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

        print("🚀 [MuseTalk] Загрузка моделей в VRAM (FP16)...")
        self._load_models()

        print(f"👤 [MuseTalk] Кэширование аватара '{avatar_id}' в RAM...")
        self._load_avatar()

        print("✅ [MuseTalk] Сервис полностью готов к Real-Time!")

    def _load_models(self):
        unet_path = MUSE_DIR / "models/musetalkV15/unet.pth"
        unet_config = MUSE_DIR / "models/musetalkV15/musetalk.json"
        whisper_dir = MUSE_DIR / "models/whisper"

        # 1. Загрузка VAE, UNet, Positional Encoding
        self.vae, self.unet, self.pe = load_all_model(
            unet_model_path=str(unet_path),
            vae_type="sd-vae",
            unet_config=str(unet_config),
            device=self.device
        )
        self.timesteps = torch.tensor([0], device=self.device)

        # Перевод в Half Precision (FP16)
        self.pe = self.pe.half().to(self.device)
        self.vae.vae = self.vae.vae.half().to(self.device)
        self.unet.model = self.unet.model.half().to(self.device)
        self.weight_dtype = self.unet.model.dtype

        # 2. Загрузка Whisper (для извлечения аудио фич)
        self.audio_processor = AudioProcessor(feature_extractor_path=str(whisper_dir))
        self.whisper = WhisperModel.from_pretrained(str(whisper_dir))
        self.whisper = self.whisper.to(device=self.device, dtype=self.weight_dtype).eval()
        self.whisper.requires_grad_(False)

    def _load_avatar(self):
        """Загружает все данные аватара в оперативную память один раз"""
        avatar_path = MUSE_DIR / "results/v15/avatars" / self.avatar_id
        if not avatar_path.exists():
            raise FileNotFoundError(f"Аватар не найден: {avatar_path}. Запустите preparation: True")

        # Латенты
        self.input_latent_list_cycle = torch.load(str(avatar_path / "latents.pt"))

        # Координаты лица
        with open(avatar_path / "coords.pkl", 'rb') as f:
            self.coord_list_cycle = pickle.load(f)

        # Маски и координаты масок
        with open(avatar_path / "mask_coords.pkl", 'rb') as f:
            self.mask_coords_list_cycle = pickle.load(f)

        # Кадры (полные картинки)
        imgs_path = glob.glob(str(avatar_path / "full_imgs/*.[jpJP][pnPN]*[gG]"))
        imgs_path = sorted(imgs_path, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.frame_list_cycle = read_imgs(imgs_path)

        # Кадры (маски)
        masks_path = glob.glob(str(avatar_path / "mask/*.[jpJP][pnPN]*[gG]"))
        masks_path = sorted(masks_path, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.mask_list_cycle = read_imgs(masks_path)

    def _process_frames_thread(self, res_frame_queue, video_len, tmp_dir):
        """Фоновый поток для блендинга сгенерированных губ с оригинальным кадром"""
        idx = 0
        while idx < video_len:
            try:
                res_frame = res_frame_queue.get(block=True, timeout=2)
                if res_frame is None:
                    print("DEBUG: [Service] Получен сигнал конца (None)")
                    break
            except queue.Empty:
                continue

            cycle_idx = idx % len(self.coord_list_cycle)
            bbox = self.coord_list_cycle[cycle_idx]
            ori_frame = copy.deepcopy(self.frame_list_cycle[cycle_idx])
            x1, y1, x2, y2 = bbox

            try:
                res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
            except Exception:
                idx += 1
                continue

            mask = self.mask_list_cycle[cycle_idx]
            mask_crop_box = self.mask_coords_list_cycle[cycle_idx]

            # Накладываем губы
            combine_frame = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)

            # Сохраняем во временную папку для FFmpeg
            cv2.imwrite(f"{tmp_dir}/{str(idx).zfill(8)}.png", combine_frame)
            idx += 1

    @torch.no_grad()
    def _generate_sync(self, audio_path: str, output_vid_path: str):
        """Синхронный метод инференса (блокирующий)"""
        start_time = time.time()

        # 1. Извлечение фич из аудио (Whisper)
        whisper_input_features, librosa_length = self.audio_processor.get_audio_feature(audio_path,
                                                                                        weight_dtype=self.weight_dtype)
        whisper_chunks = self.audio_processor.get_whisper_chunk(
            whisper_input_features, self.device, self.weight_dtype, self.whisper, librosa_length,
            fps=self.fps, audio_padding_length_left=2, audio_padding_length_right=2
        )

        video_num = len(whisper_chunks)
        if video_num == 0:
            print("❌ Аудио слишком короткое")
            return None

        # Временная папка для сборки
        tmp_dir = str(Path(output_vid_path).parent / "tmp_muse")
        os.makedirs(tmp_dir, exist_ok=True)

        # 2. Запуск потока сборки (блендинга)
        res_frame_queue = queue.Queue()
        process_thread = threading.Thread(target=self._process_frames_thread,
                                          args=(res_frame_queue, video_num, tmp_dir))
        process_thread.start()

        # 3. Инференс (UNet + VAE)
        gen = datagen(whisper_chunks, self.input_latent_list_cycle, self.batch_size)
        total_batches = int(np.ceil(float(video_num) / self.batch_size))

        for whisper_batch, latent_batch in tqdm(gen, total=total_batches, desc="Lipsync Inference"):
            audio_feature_batch = self.pe(whisper_batch.to(self.device))
            latent_batch = latent_batch.to(device=self.device, dtype=self.unet.model.dtype)

            # Генерация латентов
            pred_latents = self.unet.model(latent_batch, self.timesteps,
                                           encoder_hidden_states=audio_feature_batch).sample
            pred_latents = pred_latents.to(device=self.device, dtype=self.vae.vae.dtype)

            # Декодирование в картинки
            recon = self.vae.decode_latents(pred_latents)
            for res_frame in recon:
                res_frame_queue.put(res_frame)

        process_thread.join()

        print(f"⚡ Инференс завершен за {time.time() - start_time:.2f}s. Сборка видео...")

        # 4. Сборка видео через FFmpeg
        temp_mp4 = f"{tmp_dir}/temp_video.mp4"


        cmd_img2video = (f'ffmpeg -y -v warning -r {self.fps} -f image2 -i "{tmp_dir}/%08d.png" '
                         f'-c:v libx264 -preset fast -crf 20 -pix_fmt yuv420p "{temp_mp4}"')

        subprocess.run(cmd_img2video, shell=True, check=True)

        cmd_combine = (
            f'ffmpeg -y -v warning -i "{audio_path}" -i "{temp_mp4}" '
            f'-c:v copy -c:a aac "{output_vid_path}"'
        )
        subprocess.run(cmd_combine, shell=True, check=True)

        # Очистка
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return output_vid_path

    async def generate(self, audio_path: Path, output_dir: Path, result_name: str):
        """Асинхронная обертка для неблокирующего вызова в Event Loop"""
        output_vid_path = str(output_dir / f"{result_name}.mp4")

        loop = asyncio.get_running_loop()
        # Запускаем тяжелую сборку в пуле потоков
        result = await loop.run_in_executor(
            None, self._generate_sync, audio_path, output_vid_path
        )
        return Path(result)


