"""
Voice round-trip test (Step 7).

Speak a sentence → transcribed via Whisper → spoken back via Piper.

Usage:
    python voice_test.py
    python voice_test.py --config path/to/config.yaml
"""

import argparse

from config import load_config
from voice.input import AudioInput
from voice.stt import FasterWhisperBackend
from voice.tts import PiperBackend
from utils.logger import get_logger

logger = get_logger(__name__)


def main(config_path: str) -> None:
    config = load_config(config_path)

    audio_input = AudioInput(config.vad)
    stt = FasterWhisperBackend(config.stt)
    tts = PiperBackend(config.tts)

    print("Listening... speak a sentence and hear it repeated back. Ctrl+C to stop.\n")

    for utterance in audio_input.stream_utterances():
        print(f"Utterance: {len(utterance)} samples ({len(utterance) / 16000:.2f}s)")
        text = stt.transcribe(utterance)
        print(f"Transcribed: {text!r}")
        if text:
            tts.speak(text)
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice round-trip test")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    try:
        main(args.config)
    except KeyboardInterrupt:
        print("\nStopped.")
