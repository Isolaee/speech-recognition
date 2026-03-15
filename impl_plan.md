# Voice Agent — Concrete Implementation Plan

Each step has a clear deliverable and a manual verification method. Complete steps in order; each builds on the previous.

---

## Phase 1: Foundation

### Step 1 — Project scaffold
- Create directory structure as defined in `plan.md`
- Create empty `__init__.py` files in every package directory
- Create `requirements.txt` with all dependencies from plan
- Create `.env.example` with `ANTHROPIC_API_KEY=`
- **Verify**: `python -c "import voice, llm, tools, skills, agent"` runs without error

### Step 2 — Config system
- Write `config.yaml` skeleton (full example from plan)
- Write `config.py`:
  - Define nested Pydantic `BaseModel` classes: `OllamaConfig`, `ClaudeConfig`, `STTConfig`, `TTSConfig`, `VADConfig`, `AgentConfig`, `EscalationConfig`, `ToolsConfig`, `Config`
  - `load_config(path="config.yaml") -> Config` reads YAML, validates, returns typed object
  - Load `.env` via `python-dotenv` before reading config (for API key injection)
- **Verify**: `python -c "from config import load_config; c = load_config(); print(c.ollama.model)"` prints model name

### Step 3 — Logger
- Write `utils/logger.py`:
  - `get_logger(name) -> logging.Logger` returns a logger with a `rich.logging.RichHandler`
  - Log levels controlled by env var `LOG_LEVEL` (default `INFO`)
- **Verify**: `python -c "from utils.logger import get_logger; get_logger('test').info('ok')"` prints formatted line

---

## Phase 2: Voice Pipeline

### Step 4 — Mic capture + VAD (`voice/input.py`)
- Implement `AudioInput` class:
  - `__init__(config: VADConfig)`: load silero-vad model via `torch.hub.load`
  - `stream_utterances() -> Generator[np.ndarray, None, None]`:
    - Open `sounddevice.InputStream` at 16kHz mono
    - Buffer incoming frames
    - Run silero-vad on each chunk (30ms frames)
    - When speech detected: accumulate frames
    - When silence detected after speech (>= `silence_duration_ms`): yield full utterance as `np.ndarray`
- **Verify**: Run script — speak into mic, see utterance lengths printed

### Step 5 — STT (`voice/stt.py`)
- Implement `FasterWhisperBackend`:
  - `__init__(config: STTConfig)`: load `faster_whisper.WhisperModel(model_size, device=device)`
  - `transcribe(audio: np.ndarray) -> str`: call `model.transcribe(audio, language=language)`, join segment texts, strip whitespace
- **Verify**: Record a short WAV file, call `transcribe` on it, check output text

### Step 6 — TTS (`voice/tts.py`)
- Implement `PiperBackend`:
  - `__init__(config: TTSConfig)`: instantiate `piper.PiperVoice.load(voice_model_path)`
  - `synthesize(text: str) -> bytes`: synthesize to raw PCM bytes
  - `speak(text: str)`: synthesize then play via `sounddevice.play` + `sounddevice.wait`
- **Verify**: `python -c "from voice.tts import PiperBackend; ..."` — hear audio output

### Step 7 — Voice round-trip test (`voice_test.py`)
- Write standalone script:
  1. Instantiate `AudioInput`, `FasterWhisperBackend`, `PiperBackend`
  2. Loop: get utterance → transcribe → print → speak transcription back
- **Verify**: Speak a sentence, hear it repeated back

---

## Phase 3: LLM Backends

### Step 8 — LLM abstractions (`llm/base.py`)
- Define dataclasses:
  ```python
  @dataclass
  class ToolCall:
      id: str
      name: str
      arguments: dict

  @dataclass
  class LLMResponse:
      text: str | None
      tool_calls: list[ToolCall]
      usage: dict
  ```
- Define abstract class `LLMBackend`:
  ```python
  class LLMBackend(ABC):
      @abstractmethod
      def chat(self, messages: list[dict], tools: list[dict], system: str) -> LLMResponse: ...
  ```

### Step 9 — Ollama backend (`llm/ollama_backend.py`)
- Implement `OllamaBackend(LLMBackend)`:
  - `__init__(config: OllamaConfig)`: instantiate `ollama.Client(host=base_url)`
  - `chat(messages, tools, system) -> LLMResponse`:
    - Prepend system message if provided
    - Call `client.chat(model=model, messages=messages, tools=tools, stream=False)`
    - Parse response: extract `message.content` as text, `message.tool_calls` as `ToolCall` list
  - `stream_chat(messages, tools, system) -> Generator[str, None, None]`:
    - Same but `stream=True`, yield text chunks as they arrive
- **Verify**: Call backend with a simple message, confirm Ollama responds

### Step 10 — Claude backend (`llm/claude_backend.py`)
- Implement `ClaudeBackend(LLMBackend)`:
  - `__init__(config: ClaudeConfig)`: instantiate `anthropic.Anthropic(api_key=...)`
  - `chat(messages, tools, system) -> LLMResponse`:
    - Convert tool schemas to Anthropic format (`input_schema` instead of `parameters`)
    - Call `client.messages.create(...)`
    - Handle multi-turn tool loop: if response contains `tool_use` blocks, format as `tool_result` and re-call until final `text` block
    - Return `LLMResponse` with final text
- **Verify**: Set `ANTHROPIC_API_KEY`, call backend with a simple message, confirm response

### Step 11 — Conversation context (`agent/context.py`)
- Implement `ConversationContext`:
  - `__init__(max_turns: int, skill: Skill | None)`: initialize empty history
  - `add_user(text: str)`: append `{"role": "user", "content": text}`
  - `add_assistant(text: str)`: append `{"role": "assistant", "content": text}`
  - `add_tool_call(tool_call: ToolCall)`: append in backend-appropriate format
  - `add_tool_result(tool_id: str, result: str)`: append tool result message
  - `to_messages() -> list[dict]`: return history, apply sliding window by turn count
  - `system_prompt -> str`: return active skill's system prompt or default
  - `set_skill(skill: Skill)`: update active skill, reset system prompt
- **Verify**: Add messages, call `to_messages()`, confirm correct list structure

---

## Phase 4: Tool System

### Step 12 — Tool base + registry (`tools/base.py`, `tools/registry.py`)
- `tools/base.py`:
  ```python
  @dataclass
  class ToolResult:
      success: bool
      output: str
      error: str | None = None
      metadata: dict = field(default_factory=dict)

  class EscalationRequested(Exception):
      def __init__(self, reason: str, refined_prompt: str, context_summary: str): ...

  class BaseTool(ABC):
      name: str
      description: str
      parameters_schema: dict
      backends: list[str]  # ["ollama"], ["claude"], or ["ollama", "claude"]

      @abstractmethod
      def execute(self, **kwargs) -> ToolResult: ...
  ```
- `tools/registry.py`:
  - `ToolRegistry` with internal `dict[str, BaseTool]`
  - `register(tool: BaseTool)`: add to registry
  - `get_schemas(backend: str) -> list[dict]`: filter by backend, return formatted schema list
  - `lookup(name: str) -> BaseTool`: raise `KeyError` if not found
  - `@register_tool(backends: list[str])` decorator: instantiates class, registers it
  - Module-level singleton `registry = ToolRegistry()`
- **Verify**: Create a dummy tool with the decorator, call `registry.get_schemas("ollama")`, confirm schema appears

### Step 13 — Ollama tools
Implement each as a class using `@register_tool(backends=["ollama"])`:

- **`tools/calculator.py`**: use `simpleeval.simple_eval(expr)` to safely evaluate math strings
- **`tools/time_weather.py`**:
  - `get_current_time`: return `datetime.now().isoformat()`
  - `get_weather`: GET `https://api.open-meteo.com/v1/forecast` with lat/lon params, return temperature + description
- **`tools/system_info.py`**: use `psutil` to return CPU%, RAM usage, battery %, OS name
- **`tools/file_ops.py`**: `read_file(path)`, `write_file(path, content)`, `list_directory(path)` — validate paths are within a configured safe root
- **`tools/web_search.py`**: use `duckduckgo_search.DDGS().text(query, max_results=n)`, return formatted title + URL + snippet list

- **Verify each**: Call `tool.execute(...)` directly in a REPL, confirm `ToolResult.success == True`

### Step 14 — Tool executor (`agent/tool_executor.py`)
- Implement `ToolExecutor`:
  - `__init__(registry: ToolRegistry, timeout: int)`
  - `execute_batch(tool_calls: list[ToolCall], backend: str) -> list[tuple[ToolCall, ToolResult]]`:
    - For each call: `registry.lookup(name).execute(**arguments)`
    - Wrap in try/except; on exception return `ToolResult(success=False, output="", error=str(e))`
    - Enforce timeout using `concurrent.futures.ThreadPoolExecutor`
    - Return list of `(tool_call, result)` pairs
- **Verify**: Execute a batch with one valid and one invalid tool name, confirm error handling works

### Step 15 — Wire tools into Ollama backend
- Update `OllamaBackend.chat` to loop on tool calls:
  ```
  while True:
      response = client.chat(model, messages, tools)
      if response has tool_calls:
          results = tool_executor.execute_batch(response.tool_calls)
          append tool calls + results to messages
          continue
      else:
          return LLMResponse(text=response.text)
  ```
- **Verify**: Ask Ollama "what time is it?" with `time_weather` tool enabled — confirm it calls the tool and returns a real answer

---

## Phase 5: Escalation and Routing

### Step 16 — Escalation pseudo-tool (`tools/escalate.py`)
- Implement `EscalateTool` with `backends=["ollama"]`:
  - `parameters_schema`: `reason: str`, `refined_prompt: str`, `context_summary: str`
  - `execute()`: raise `EscalationRequested(reason, refined_prompt, context_summary)` — never returns normally
- **Verify**: Calling `EscalateTool.execute(...)` raises `EscalationRequested`

### Step 17 — Router (`llm/router.py`)
- Implement `LLMRouter`:
  - `__init__(ollama: OllamaBackend, claude: ClaudeBackend, config: EscalationConfig)`
  - `chat(messages, ollama_tools, claude_tools, skill, tts) -> LLMResponse`:
    1. If `skill.always_escalate` → go directly to Claude
    2. Otherwise call `OllamaBackend.chat` with tool executor
    3. Catch `EscalationRequested`:
       - Call `tts.speak("Let me use a more powerful model for this one.")`
       - Build Claude message list from `refined_prompt` + `context_summary`
       - Call `ClaudeBackend.chat` with Claude tools
       - Return Claude's response
    4. Otherwise return Ollama's response
- **Verify**: Set `always_escalate: true` in a skill, confirm Claude responds

### Step 18 — Claude tools
Implement each with `@register_tool(backends=["claude"])`:

- **`tools/web_scraper.py`**: GET URL via `httpx`, parse with `BeautifulSoup`, extract main text content, truncate to 4000 chars
- **`tools/code_executor.py`**: run Python code string in a `subprocess` with timeout; return stdout/stderr; never use `shell=True`
- **`tools/api_caller.py`**: `httpx.request(method, url, headers, json_body)` with configurable timeout; return status + response body

- **Verify each**: Call directly in REPL, confirm output

### Step 19 — Wire Claude tools into Claude backend
- Update `ClaudeBackend.chat` to use the same tool executor loop pattern as Ollama backend
- **Verify**: Route a code execution request through escalation, confirm Claude calls `code_executor` and returns result

---

## Phase 6: Skills System

### Step 20 — Skill dataclass (`skills/base.py`)
```python
@dataclass
class Skill:
    name: str
    description: str
    system_prompt: str
    enabled_tools: list[str]
    tts_voice: str | None = None
    ollama_model_override: str | None = None
    always_escalate: bool = False
    escalation_threshold: float = 0.8
```

### Step 21 — Skill loader (`skills/loader.py`)
- Implement `SkillLoader`:
  - `__init__(definitions_dir: str)`: scan `*.yaml` files, parse each into `Skill` dataclass
  - `load(name: str) -> Skill`: look up by name, raise `ValueError` if not found
  - `list_skills() -> list[str]`: return all skill names
- **Verify**: `SkillLoader("skills/definitions").list_skills()` returns expected names

### Step 22 — Write skill YAML files
Create the following in `skills/definitions/`:

- **`default.yaml`**: general assistant, all ollama tools enabled, `always_escalate: false`
- **`coding_assistant.yaml`**: code-focused prompt, `always_escalate: true`, code_executor + file_ops tools
- **`research_assistant.yaml`**: research prompt, web_search + web_scraper, `escalation_threshold: 0.5`
- **`home_assistant.yaml`**: home/IoT prompt, system_info + time_weather, ollama only

### Step 23 — Integrate skills into context and registry
- `ConversationContext.set_skill(skill)`: update system prompt, store active skill
- Router and executor accept `skill.enabled_tools` as a filter when calling `registry.get_schemas`
- `OllamaBackend`: use `skill.ollama_model_override` if set
- `PiperBackend.speak`: accept optional voice override per call (for `skill.tts_voice`)
- **Verify**: Switch to `coding_assistant` skill, confirm only its tools are passed to LLM

---

## Phase 7: Orchestrator and Integration

### Step 24 — Orchestrator (`agent/orchestrator.py`)
- Implement `Orchestrator`:
  - `__init__(config: Config)`: instantiate all components (AudioInput, STT, TTS, backends, router, registry, executor, context, skill_loader)
  - `run()`: main loop:
    ```
    for utterance in audio_input.stream_utterances():
        text = stt.transcribe(utterance)
        if is_skill_switch(text): handle_skill_switch(text); continue
        context.add_user(text)
        response = router.chat(context.to_messages(), ...)
        context.add_assistant(response.text)
        tts.speak(response.text)
    ```
  - `set_skill(name: str)`: load skill, update context
  - `is_skill_switch(text: str) -> bool`: regex match "switch to X" / "use X assistant"
  - `handle_skill_switch(text: str)`: extract skill name, call `set_skill`

### Step 25 — Entry point (`main.py`)
- Use `argparse`:
  - `--skill NAME`: override default skill
  - `--no-voice`: text-only mode (stdin/stdout instead of mic/speaker)
  - `--config PATH`: alternate config file path
  - `--list-skills`: print available skills and exit
- Instantiate `Orchestrator`, call `orchestrator.run()`
- Handle `KeyboardInterrupt` cleanly
- **Verify**: `python main.py --list-skills` prints skill names; `python main.py --no-voice` accepts typed input

### Step 26 — Text-only debug mode
- In `Orchestrator`, detect `no_voice=True`:
  - Replace `audio_input.stream_utterances()` with a generator that reads `input()` lines
  - Replace `tts.speak(text)` with `print(text)`
- **Verify**: Full conversation loop works via keyboard without any audio hardware

### Step 27 — End-to-end integration test (`test_integration.py`)
- Mock `OllamaBackend.chat` to return a tool call then a final text response
- Confirm `ToolExecutor` runs the tool and loops back to LLM
- Confirm final text is returned from `Orchestrator`
- Test escalation path: mock Ollama to raise `EscalationRequested`, confirm Claude backend is called
- **Verify**: `python test_integration.py` passes all assertions

---

## Phase 8: Polish

### Step 28 — Wake word detection
- After STT in `Orchestrator.run()`: check if text matches configured wake word (e.g., "hey agent")
- Only process utterance if wake word detected (or wake word disabled in config)
- Add `wake_word: "hey agent"` and `wake_word_enabled: false` to `config.yaml`

### Step 29 — Context summarization
- In `ConversationContext.to_messages()`: if history exceeds `max_turns`, call `OllamaBackend.chat` with a summarization prompt on the oldest half of messages, replace them with a single summary message
- **Verify**: Fill context beyond max_turns, confirm it compresses without losing current topic

### Step 30 — Alternative backend support (if needed)
- Add `WhisperBackend` to `voice/stt.py` behind a config switch
- Add `KokoroBackend`, `Pyttsx3Backend` to `voice/tts.py` behind config switch
- Add `webrtcvad` fallback to `voice/input.py` if silero-vad is unavailable

---

## Milestone Checkpoints

| After step | You should have |
|------------|----------------|
| 3  | Config loads, logger works |
| 7  | Speak → hear yourself repeated back |
| 10 | Text chat works with both Ollama and Claude |
| 15 | Ollama uses tools to answer real questions |
| 19 | Hard questions escalate to Claude with Claude tools |
| 23 | Skill switching changes behavior and available tools |
| 26 | Full voice agent running end-to-end |
| 27 | Integration tests pass |
| 30 | Production-ready with fallbacks |
