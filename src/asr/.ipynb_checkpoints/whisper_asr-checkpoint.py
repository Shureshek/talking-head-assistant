from faster_whisper import WhisperModel


class WhisperTranscriber:
    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
    ):
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )

    def transcribe(
        self,
        audio_path: str,
        language: str = "ru",
        vad_filter: bool = True,
        beam_size: int = 10,
    ):
        segments, info = self.model.transcribe(
            audio_path,
            language=language,
            vad_filter=vad_filter,
            beam_size=beam_size
        )
        
        text = ".".join([s.text for s in segments])
        return text, info
