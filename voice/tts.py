import sounddevice as sd
import numpy as np

from config import TTSConfig
from utils.logger import get_logger

logger = get_logger(__name__)


class PiperBackend:
    def __init__(self, config: TTSConfig):
        import piper as piper_tts

        self.config = config
        logger.info("Loading Piper voice model '%s'...", config.voice_model_path)
        self.voice = piper_tts.PiperVoice.load(config.voice_model_path)
        logger.info("Piper voice loaded.")

    def synthesize(self, text: str) -> bytes:
        audio = b""
        for chunk in self.voice.synthesize(text):
            audio += chunk.audio_int16_bytes
        return audio

    def speak(self, text: str, voice_override: str | None = None) -> None:
        if voice_override and voice_override != self.config.voice:
            logger.debug("Voice override '%s' requested but runtime voice switching is not supported; using loaded voice.", voice_override)
        chunks = list(self.voice.synthesize(text))
        if not chunks:
            return
        sample_rate = chunks[0].sample_rate
        audio = np.concatenate([c.audio_float_array for c in chunks])
        sd.play(audio, samplerate=sample_rate)
        sd.wait()
