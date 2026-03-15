import numpy as np
from faster_whisper import WhisperModel

from config import STTConfig
from utils.logger import get_logger

logger = get_logger(__name__)


class FasterWhisperBackend:
    def __init__(self, config: STTConfig):
        self.config = config
        logger.info(
            "Loading Whisper model '%s' on device '%s'...",
            config.model_size,
            config.device,
        )
        self.model = WhisperModel(config.model_size, device=config.device)
        logger.info("Whisper model loaded.")

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(audio, language=self.config.language)
        return " ".join(seg.text for seg in segments).strip()
