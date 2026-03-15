# Voice Chat Agent - Architecture Plan

## Overview

A modular, Python-based voice chat agent that uses a local Ollama model as its primary brain, with an optional escalation path to the Claude API for complex tasks. Voice input is processed via STT (faster-whisper), responses are synthesized via TTS (piper), and both LLM backends can invoke tools. Skills are reusable prompt-template modules loaded dynamically.

---

## Project Structure

```
voice-agent/
├── main.py                        # Entry point, event loop, CLI args
├── config.py                      # Centralized config (loaded from config.yaml)
├── config.yaml                    # User-editable settings (models, voices, keys, thresholds)
├── requirements.txt
├── plan.md                        # This document
│
├── voice/
│   ├── __init__.py
│   ├── input.py                   # Mic capture + VAD combined (sounddevice + silero-vad)
│   ├── stt.py                     # STT via faster-whisper
│   └── tts.py                     # TTS via piper (additional backends added if needed)
│
├── llm/
│   ├── __init__.py
│   ├── base.py                    # Abstract LLMBackend interface + LLMResponse dataclass
│   ├── ollama_backend.py          # Ollama integration (ollama SDK)
│   ├── claude_backend.py          # Anthropic Claude integration
│   └── router.py                  # Escalation logic: decides Ollama vs Claude
│
├── tools/
│   ├── __init__.py
│   ├── base.py                    # BaseTool abstract class + ToolResult dataclass
│   ├── registry.py                # ToolRegistry: maps name → callable + schema + backend flags
│   ├── web_search.py              # DuckDuckGo search (ollama)
│   ├── file_ops.py                # Read/write/list local files (ollama)
│   ├── time_weather.py            # Current time, weather via Open-Meteo (ollama)
│   ├── calculator.py              # Safe math expression eval (ollama)
│   ├── system_info.py             # CPU, RAM, battery, OS info (ollama)
│   ├── code_executor.py           # Sandboxed Python execution (claude-only)
│   ├── web_scraper.py             # Full page scraping + extraction (claude-only)
│   └── api_caller.py              # Generic HTTP API caller (claude-only)
│
├── skills/
│   ├── __init__.py
│   ├── loader.py                  # Dynamic skill loading from .yaml files
│   ├── base.py                    # Skill dataclass (name, system_prompt, tools, config)
│   └── definitions/               # YAML skill definitions
│       ├── default.yaml           # Base conversational assistant
│       ├── coding_assistant.yaml  # Code-focused persona + tools
│       ├── research_assistant.yaml# Web search heavy persona
│       └── home_assistant.yaml    # IoT / home automation persona
│
├── agent/
│   ├── __init__.py
│   ├── context.py                 # ConversationContext: message history, metadata
│   ├── orchestrator.py            # Main agent loop: coordinates voice→LLM→tools→voice
│   └── tool_executor.py           # Executes tool calls, handles retries, formats results
│
└── utils/
    ├── __init__.py
    └── logger.py                  # Structured logging setup (rich)
```

---

## Key Components and Responsibilities

### `config.yaml` / `config.py`
- Holds all tunable settings: STT/TTS config, Ollama model, Claude API key, VAD sensitivity, escalation thresholds, active skill, tool enable/disable flags.
- `config.py` loads YAML into a typed `Config` dataclass with sensible defaults. No hard-coded magic values elsewhere.

### `voice/input.py`
- Wraps `sounddevice` to stream raw PCM audio frames.
- Integrates `silero-vad` to detect speech segments.
- Buffers audio until a complete utterance is detected (silence after speech), then emits a complete audio segment.
- Exposes a generator or async queue of audio chunks ready for STT.

### `voice/stt.py`
- `FasterWhisperBackend`: uses `faster-whisper` (CTranslate2) — fast on CPU, practical for real-time without a GPU.
- Exposes `transcribe(audio_bytes) -> str`.
- Additional backends (standard Whisper) added only if faster-whisper proves insufficient.

### `voice/tts.py`
- `PiperBackend`: ONNX-based neural TTS, fast on CPU (~50ms), fully offline, ~50MB voices, voice-assistant optimized.
- Exposes `synthesize(text) -> audio_bytes` and `speak(text)` convenience method.
- Additional backends (`kokoro`, `pyttsx3`, `edge-tts`) added only if piper proves insufficient.

### `llm/base.py`
- `LLMBackend` abstract class:
  - `chat(messages, tools, skill) -> LLMResponse`
  - `LLMResponse` dataclass: `text`, `tool_calls`, `usage`

### `llm/ollama_backend.py`
- Connects to local Ollama via the `ollama` Python SDK.
- Formats tool definitions as JSON schemas in Ollama's function-calling format.
- Supports streaming for lower latency (streams text, then speaks sentence-by-sentence).

### `llm/claude_backend.py`
- Uses `anthropic` Python SDK.
- Formats tools in Anthropic's tool schema format.
- Handles multi-turn tool_use / tool_result cycles until a final text response is produced.

### `llm/router.py`
- `LLMRouter`: decides whether a given request should go to Ollama or Claude.
- Two escalation layers (see Escalation section).
- Returns `RoutingDecision(backend, prompt_override, context_summary)`.

### `tools/registry.py`
- `ToolRegistry`: maps tool name → `(callable, JSON_schema, backends)`.
- Each tool declares which backends it supports: `backends=["ollama"]`, `backends=["claude"]`, or `backends=["ollama", "claude"]`.
- Single registry; `get_schemas(backend)` filters by backend flag and returns appropriately formatted tool definitions.
- Tools self-register via a `@register_tool(backends=["ollama"])` decorator.

### `tools/base.py`
- `BaseTool`: abstract base with `name`, `description`, `parameters_schema`, `backends`, `execute(**kwargs) -> ToolResult`.
- `ToolResult`: `success: bool`, `output: str`, `error: str | None`, `metadata: dict`.

### `skills/loader.py`
- Scans `skills/definitions/*.yaml` on startup.
- Each YAML defines: `name`, `system_prompt`, `enabled_tools` (list), `tts_voice` (optional), `ollama_model_override` (optional), `always_escalate: bool`.
- `SkillLoader.load(name) -> Skill` merges skill config with base config.
- Active skill can be changed at runtime via voice command ("switch to coding mode").

### `agent/orchestrator.py`
- The main coordination loop:
  1. Listen for wake word (optional, via keyword matching on STT output).
  2. Capture utterance via `voice/input.py`.
  3. Transcribe via STT.
  4. Prepend to `ConversationContext`.
  5. Route via `LLMRouter`.
  6. Call chosen backend with messages + tools + skill.
  7. If tool calls returned, execute via `ToolExecutor`, append results, loop back to LLM.
  8. When final text response obtained, clean up for TTS, synthesize, play audio.
  9. Update context, log turn.

### `agent/tool_executor.py`
- Receives a list of tool call requests from an LLM response.
- Looks up each tool in the registry (filtered by active backend).
- Executes sequentially or in parallel (configurable).
- Handles exceptions, timeouts, and formats results back into the message format expected by each backend.

### `agent/context.py`
- `ConversationContext`: sliding window message history (configurable max tokens or turn count).
- Stores metadata: current skill, active backend, session start time.
- Provides `to_messages() -> list[dict]` for each backend format.
- Supports context summarization (calls Ollama to compress old history when window fills).

---

## Data Flow

```
Microphone
    |
    v
[voice/input.py]       -- raw PCM + VAD → speech segment -->
[voice/stt.py]         -- transcribed text -->
[agent/context.py]     -- append user message
    |
    v
[llm/router.py]
    |
    +--> [llm/ollama_backend.py]
    |           |
    |     [tool calls?] --> [agent/tool_executor.py] --> [tools/*.py (ollama)]
    |           |                                              |
    |           +<-----------  tool results <-----------------+
    |           |
    |     [escalate?] --> [llm/router.py] --> [llm/claude_backend.py]
    |                                               |
    |                                       [tool calls?] --> [tools/*.py (claude)]
    |                                               |
    |                                       final text response
    |
    +--> final text response
    |
    v
[voice/tts.py]         -- audio bytes
[sounddevice playback] -- spoken response
    |
    v
[agent/context.py]     -- append assistant message
```

---

## Escalation Architecture (Ollama → Claude)

Two escalation layers are supported:

### Layer 1: Explicit Escalation Tool (Preferred)

Ollama is given a tool named `escalate_to_claude` in its tool list:

```
Tool: escalate_to_claude
Parameters:
  - reason: string (why Ollama is escalating)
  - refined_prompt: string (reworded/clarified version of the user's request)
  - context_summary: string (brief summary of conversation so far)
```

When Ollama calls this tool, the router intercepts it and:
1. Takes `refined_prompt` as the new user message to Claude.
2. Prepends `context_summary` as a system note.
3. Sends the request to `ClaudeBackend` with the Claude-specific tool set.
4. Returns Claude's response as if it were Ollama's own response.
5. Speaks a brief bridging phrase like "Let me use a more powerful model for this one."

### Layer 2: Skill-Level Hard Routing

Certain skills (e.g., `coding_assistant`) can set `always_escalate: true` in their YAML, causing the router to skip Ollama entirely for those skill contexts.

---

## Tool Architecture

### Tool Definition (decorator pattern)

```python
# tools/web_search.py example:

@register_tool(backends=["ollama"])
class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Search the web for current information"
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "num_results": {"type": "integer", "default": 5}
        },
        "required": ["query"]
    }

    def execute(self, query: str, num_results: int = 5) -> ToolResult:
        # Implementation using duckduckgo_search
        ...
```

### Tool Schema Generation

`ToolRegistry.get_schemas("ollama")` returns tool definitions filtered to Ollama-compatible tools, formatted for Ollama's API.
`ToolRegistry.get_schemas("claude")` returns Anthropic-formatted definitions for Claude-only and shared tools.
Each backend translates schemas internally, keeping tools backend-agnostic.

### Tool Execution Loop

```
LLM returns tool_calls
    |
    v
tool_executor.execute_batch(tool_calls, backend)
    |
    +--> for each call: registry.lookup(name).execute(**args)
    |
    v
List[ToolResult]  -->  formatted as tool_result messages
    |
    v
Appended to message history, re-sent to LLM
    |
    v
LLM returns next response (may have more tool calls or final text)
```

---

## Skills Architecture

### Skill YAML Format

```yaml
# skills/definitions/research_assistant.yaml
name: research_assistant
description: "Deep research with web search and summarization"
system_prompt: |
  You are a thorough research assistant. When answering questions,
  always search for current information before responding.
  Cite sources when possible. Be concise but complete.
enabled_tools:
  - web_search
  - web_scraper
  - calculator
tts_voice: en_US-lessac-medium  # piper voice model name
ollama_model_override: null     # use default from config.yaml
always_escalate: false
escalation_threshold: 0.6       # lower = escalate more readily
```

### Skill Loading and Switching

- On startup, `SkillLoader` scans `skills/definitions/` and indexes all skills by name.
- Default skill is set in `config.yaml`.
- Skills can be switched via:
  - Voice: "switch to coding mode" / "use research assistant"
  - CLI arg: `--skill coding_assistant`
  - Programmatic: `orchestrator.set_skill("research_assistant")`
- When a skill is activated, its `system_prompt` replaces the system message in `ConversationContext`, and its `enabled_tools` filter the tool registry passed to the LLM.

---

## Python Package Requirements

```
# requirements.txt

# Core audio
sounddevice>=0.4.6
numpy>=1.24.0

# Voice Activity Detection
silero-vad>=4.0.0        # VAD (torch-based)

# STT
faster-whisper>=1.0.0    # fast CPU/GPU whisper

# TTS
piper-tts>=1.2.0         # offline neural TTS (ONNX-based)

# LLM backends
ollama>=0.3.0            # Ollama Python SDK
anthropic>=0.30.0        # Claude API SDK

# Tools
duckduckgo-search>=6.0.0 # web search
httpx>=0.27.0            # HTTP client
beautifulsoup4>=4.12.0   # web scraping
simpleeval>=0.9.13       # safe math expression evaluator
psutil>=5.9.0            # system info

# Utilities
pyyaml>=6.0.1            # config and skill YAML
pydantic>=2.0.0          # typed configs
python-dotenv>=1.0.0     # .env file support
rich>=13.0.0             # pretty terminal output
torch>=2.0.0             # required by silero-vad
```

---

## Configuration Reference (config.yaml skeleton)

```yaml
# LLM settings
ollama:
  base_url: "http://localhost:11434"
  model: "llama3.1:8b"
  temperature: 0.7
  context_window: 8192

claude:
  model: "claude-sonnet-4-6"
  api_key_env: "ANTHROPIC_API_KEY"
  temperature: 0.7
  max_tokens: 4096

# Voice settings
stt:
  backend: "faster_whisper"
  model_size: "base"               # tiny | base | small | medium | large-v3
  language: "en"
  device: "cpu"                    # "cpu" | "cuda"

tts:
  backend: "piper"
  voice: "en_US-lessac-medium"
  rate: 1.0
  volume: 1.0

vad:
  threshold: 0.5
  silence_duration_ms: 800

# Agent settings
agent:
  active_skill: "default"
  history_max_turns: 20
  tool_timeout_seconds: 15
  speak_while_streaming: true      # TTS sentence by sentence as LLM streams

# Escalation settings
escalation:
  enabled: true

# Tool settings
tools:
  ollama_enabled:
    - web_search
    - file_ops
    - time_weather
    - calculator
    - system_info
    - escalate_to_claude
  claude_enabled:
    - code_executor
    - web_scraper
    - api_caller
```

---

## Step-by-Step Build Order

### Phase 1: Foundation
1. Set up `config.yaml` and `config.py` with typed `Config` dataclass.
2. Implement `utils/logger.py` (structured logging with `rich`).

### Phase 2: Voice Pipeline
3. Implement `voice/input.py` — mic capture + VAD in one module.
4. Implement `voice/stt.py` — `FasterWhisperBackend`.
5. Implement `voice/tts.py` — `PiperBackend`.
6. Build `voice_test.py`: mic → VAD → STT → print → TTS → speak. Verify full round trip.

### Phase 3: LLM Backends
7. Implement `llm/base.py` abstract classes and `LLMResponse` dataclass.
8. Implement `llm/ollama_backend.py` — basic chat (no tools yet).
9. Implement `llm/claude_backend.py` — basic chat (no tools yet).
10. Implement `agent/context.py` — message history with sliding window.

### Phase 4: Tool System
11. Implement `tools/base.py` and `tools/registry.py` with `@register_tool(backends=[...])` decorator.
12. Implement tools: `calculator.py`, `time_weather.py`, `web_search.py`, `file_ops.py`, `system_info.py`.
13. Implement `agent/tool_executor.py` — execution loop with error handling.
14. Wire tools into `ollama_backend.py`.

### Phase 5: Escalation and Routing
15. Implement `escalate_to_claude` as a pseudo-tool in Ollama's tool list.
16. Implement `llm/router.py` with both escalation layers.
17. Implement Claude tools: `web_scraper.py`, `code_executor.py`, `api_caller.py`.
18. Wire Claude tools into `claude_backend.py`.
19. Test all escalation paths.

### Phase 6: Skills System
20. Implement `skills/base.py` `Skill` dataclass.
21. Implement `skills/loader.py` YAML parser and skill indexer.
22. Write `skills/definitions/default.yaml` and 2-3 additional skill YAMLs.
23. Integrate skill loading into context and tool registry filtering.

### Phase 7: Orchestrator and Integration
24. Implement `agent/orchestrator.py` — wire together the full pipeline.
25. Implement `main.py` — argument parsing, startup, clean shutdown.
26. Add `--no-voice` text-only mode for debugging (stdin/stdout).
27. End-to-end integration tests.

### Phase 8: Polish
28. Add wake word detection (keyword matching on STT output).
29. Add context summarization when history window fills.
30. Add alternative TTS/STT backends if piper or faster-whisper prove insufficient.

---

## Key Design Decisions

**`sounddevice` over `pyaudio`**: Cleaner API, better NumPy integration, no PortAudio compilation issues on Linux, native async streaming.

**`faster-whisper` as default STT**: 4-8x faster than standard Whisper on CPU, practical for real-time without a GPU.

**`piper-tts` as default TTS**: Fast ONNX-based neural TTS (~50ms on CPU), fully offline, voice-assistant optimized. Additional backends deferred until actually needed.

**Explicit `escalate_to_claude` tool**: Gives Ollama agency to provide a refined prompt and context summary to Claude, resulting in better responses. Combined with skill-level hard routing, this covers all practical escalation needs without passive pattern detection overhead.

**Single tool registry with backend flags**: All tools live in `tools/`, each declaring which backends support them. Simpler than two parallel directory trees, easier to add cross-backend tools.

**YAML for skills**: Editable by non-developers, hot-reloadable, keeps behavior config out of code.