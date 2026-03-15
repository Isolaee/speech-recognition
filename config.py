import os
from typing import List, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


class OllamaConfig(BaseModel):
    base_url: str = "http://localhost:11434"
    model: str = "llama3.1:8b"
    temperature: float = 0.7
    context_window: int = 8192


class ClaudeConfig(BaseModel):
    model: str = "claude-sonnet-4-6"
    api_key_env: str = "ANTHROPIC_API_KEY"
    temperature: float = 0.7
    max_tokens: int = 4096


class STTConfig(BaseModel):
    backend: str = "faster_whisper"
    model_size: str = "base"
    language: str = "en"
    device: str = "cpu"


class TTSConfig(BaseModel):
    backend: str = "piper"
    voice: str = "en_US-lessac-medium"
    voice_model_path: str = "models/en_US-lessac-medium.onnx"
    rate: float = 1.0
    volume: float = 1.0


class VADConfig(BaseModel):
    threshold: float = 0.5
    silence_duration_ms: int = 800


class AgentConfig(BaseModel):
    active_skill: str = "default"
    history_max_turns: int = 20
    tool_timeout_seconds: int = 15
    speak_while_streaming: bool = True


class EscalationConfig(BaseModel):
    enabled: bool = True


class ToolsConfig(BaseModel):
    ollama_enabled: List[str] = Field(default_factory=lambda: [
        "web_search", "file_ops", "time_weather", "calculator", "system_info", "escalate_to_claude"
    ])
    claude_enabled: List[str] = Field(default_factory=lambda: [
        "code_executor", "web_scraper", "api_caller"
    ])


class Config(BaseModel):
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    claude: ClaudeConfig = Field(default_factory=ClaudeConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    vad: VADConfig = Field(default_factory=VADConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    escalation: EscalationConfig = Field(default_factory=EscalationConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)


def load_config(path: str = "config.yaml") -> Config:
    load_dotenv()
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return Config.model_validate(data)
