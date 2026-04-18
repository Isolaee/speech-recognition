# Speech Recognition Tool

## Overview
This project is a speech recognition tool that converts spoken language into text and can intelligently invoke other tools or functions based on the recognized speech. It combines automatic speech recognition (ASR) with tool integration capabilities.

## Features
- Speech Recognition - Convert voice input to text with high accuracy
- Tool Invocation - Execute tools and commands based on recognized speech
- Natural Language Processing - Understand intent from user speech
- Real-time Processing - Fast speech-to-text conversion
- Plugin System - Extend with custom tools and actions
- Command Recognition - Identify and execute specific commands
- Transcription - Save and manage transcripts

## Getting Started

### Prerequisites
- Python 3.8 or later
- Microphone or audio input device
- pip and virtualenv

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Isolaee/speech-recognition.git
   cd speech-recognition
   ```
2. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Speech Recognition
```python
from speech_recognition import SpeechRecognizer

recognizer = SpeechRecognizer()
text = recognizer.listen()
print(f"You said: {text}")
```

### Invoke Tools Based on Speech
```python
from speech_recognition import ToolInvoker

invoker = ToolInvoker()
invoker.process_speech("open calculator")
```

### Command Line Usage
```bash
python speech_tool.py --listen
python speech_tool.py --text "command to execute"
```

## Supported Tools
- File operations (open, create, delete files)
- Application launching
- Web searches
- System commands
- Custom registered tools

## Registering Custom Tools
```python
from speech_recognition import ToolInvoker

invoker = ToolInvoker()
voker.register_tool('my_tool', my_function, keywords=['trigger', 'words'])
```

## Configuration
Configure the tool through config.yml for speech recognition engine, language, confidence threshold, and enabled tools.

## Project Structure
- /speech_recognition - Core speech recognition module
- /tools - Available tools and integrations
- /config - Configuration files
- /tests - Unit tests
- /examples - Example scripts

## Contributing
Contributions are welcome! Please submit pull requests or open issues.

## License
See LICENSE file for details.

## Author
Isolaee