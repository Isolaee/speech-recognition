import numpy as np
import sounddevice as sd
import torch
from typing import Generator

from config import VADConfig
from utils.logger import get_logger

logger = get_logger(__name__)

SAMPLE_RATE = 16000
CHUNK_SIZE = 512  # 32ms at 16kHz — silero-vad minimum chunk size


class AudioInput:
    def __init__(self, config: VADConfig):
        self.config = config
        logger.info("Loading silero-vad model...")
        self.model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )
        self.model.eval()
        logger.info("silero-vad loaded.")

    def stream_utterances(self) -> Generator[np.ndarray, None, None]:
        # Number of silent chunks required before yielding an utterance.
        # Each chunk is CHUNK_SIZE / SAMPLE_RATE seconds long.
        chunk_ms = CHUNK_SIZE / SAMPLE_RATE * 1000
        silence_limit = max(1, int(self.config.silence_duration_ms / chunk_ms))

        speech_frames: list[np.ndarray] = []
        in_speech = False
        silence_count = 0

        logger.info("Mic open — waiting for speech...")

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SIZE,
        ) as stream:
            while True:
                data, _ = stream.read(CHUNK_SIZE)
                audio = data[:, 0]  # (CHUNK_SIZE,) float32

                tensor = torch.from_numpy(audio)
                prob: float = self.model(tensor, SAMPLE_RATE).item()

                if prob >= self.config.threshold:
                    if not in_speech:
                        logger.debug("Speech started (prob=%.2f)", prob)
                    in_speech = True
                    silence_count = 0
                    speech_frames.append(audio.copy())
                elif in_speech:
                    speech_frames.append(audio.copy())
                    silence_count += 1
                    if silence_count >= silence_limit:
                        utterance = np.concatenate(speech_frames)
                        logger.debug(
                            "Utterance ready: %d samples (%.2fs)",
                            len(utterance),
                            len(utterance) / SAMPLE_RATE,
                        )
                        yield utterance
                        speech_frames = []
                        in_speech = False
                        silence_count = 0
                        self.model.reset_states()
