# Speech Recognition — Local Voice Agent

## Overview
A fully offline-capable voice agent that listens, understands, and responds using local speech recognition (Whisper), local TTS (Piper), and a local LLM (Ollama). Complex tasks are optionally escalated to Claude for higher accuracy. Skills define what the agent can do and are loaded from YAML definitions at runtime.

## Problem It Solves
- Cloud voice assistants (Alexa, Google Assistant) send audio and queries to remote servers — no data stays local
- Switching LLM backends or adding new capabilities requires code changes in most voice agent frameworks
- Target users: developers and privacy-conscious users who want a customisable, offline-first voice assistant they can extend with their own tools

## Use Cases
1. A developer runs the agent on a home server — it answers questions, runs web searches, and reads back results entirely on-device using Ollama + Whisper + Piper, with no internet required for basic tasks
2. A user asks a complex coding question — the agent automatically escalates to Claude via the `escalate_to_claude` tool and streams the response sentence-by-sentence through the local TTS as the LLM generates it
3. A new "reminder" skill is added by dropping a YAML definition file into `skills/definitions/` — no code changes needed

## Key Features
- **Offline-first STT** — faster-whisper with configurable model size (tiny → large-v3) and device (CPU / CUDA)
- **Offline TTS** — Piper ONNX-based neural TTS with natural-sounding voices
- **Dual LLM backends** — Ollama for local inference, Claude API for escalation; tools are split between backends
- **Skill system** — YAML-defined skills loaded at startup; list available skills with `--list-skills`
- **Voice activity detection** — Silero VAD with configurable silence threshold to avoid spurious triggers
- **Wake word support** — optional wake word gate before each utterance
- **Text-only mode** — `--no-voice` flag for stdin/stdout testing without a microphone

## Tech Stack
| Component | Technology |
|---|---|
| STT | faster-whisper (CTranslate2) |
| TTS | Piper (ONNX) |
| VAD | Silero VAD (PyTorch) |
| Local LLM | Ollama (`qwen2.5:7b` default) |
| Cloud LLM | Anthropic Claude API |
| Audio I/O | sounddevice + numpy |
| Config | PyYAML + Pydantic |

### Built-in tools
`web_search`, `file_ops`, `time_weather`, `calculator`, `system_info`, `stock_market`, `escalate_to_claude`, `code_executor`, `web_scraper`, `api_caller`

## Getting Started

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- A Piper voice model (`.onnx` + `.json`) placed in `models/`

```bash
# 1. Clone and install dependencies
git clone https://github.com/Isolaee/speech-recognition.git
cd speech-recognition
pip install -r requirements.txt

# 2. Pull an Ollama model
ollama pull qwen2.5:7b

# 3. Download a Piper voice model
# https://huggingface.co/rhasspy/piper-voices
# Place .onnx and .onnx.json in models/

# 4. Configure
cp .env.example .env           # add ANTHROPIC_API_KEY if using Claude escalation
# Edit config.yaml to set model paths, VAD thresholds, etc.

# 5. Run
python main.py                 # voice mode
python main.py --no-voice      # text-only mode
python main.py --list-skills   # list available skills
```

### Key config options (`config.yaml`)

| Setting | Default | Description |
|---|---|---|
| `stt.model_size` | `base` | Whisper model size (`tiny` → `large-v3`) |
| `tts.voice_model_path` | — | Path to Piper `.onnx` file |
| `agent.active_skill` | `default` | Skill to load on startup |
| `agent.wake_word_enabled` | `true` | Require wake word before each query |
| `agent.speak_while_streaming` | `true` | Stream TTS sentence-by-sentence |
| `escalation.enabled` | `true` | Allow escalation to Claude |
