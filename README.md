# SkyNet

A powerful local AI coding assistant CLI powered by Ollama. Like Claude Code, but running entirely on your own hardware.

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Ollama](https://img.shields.io/badge/ollama-local%20LLM-orange.svg)

## Features

- **Local LLM Inference** - Runs on your GPU via Ollama (no API keys needed)
- **Multi-Model Support** - Switch between models on the fly (`/models`, `/model`)
- **File Operations** - Read, write, and edit files with intelligent search/replace
- **Code Search** - Grep and glob through your codebase
- **Shell Execution** - Run commands with safety guards
- **Git Integration** - Status, diff, commit, log, and branch management
- **Web Search** - Search the web and fetch documentation
- **Session Persistence** - Save and resume conversations
- **Rich Terminal UI** - Markdown rendering, syntax highlighting, and more

## Requirements

- Python 3.11+
- [Ollama](https://ollama.ai) installed and running
- A GPU with sufficient VRAM for your chosen model (e.g., RTX 3090/4090/5090 for 32B models)

## Installation

```bash
# Clone the repository
git clone https://github.com/PySecNinja/skynet.git
cd skynet

# Install in editable mode
pip install -e .

# Or install directly
pip install .
```

## Quick Start

```bash
# Make sure Ollama is running
ollama serve

# Pull a coding model (if you haven't already)
ollama pull qwen2.5-coder:32b

# Start SkyNet
skynet
```

## Usage

### Basic Commands

```bash
# Start interactive session
skynet

# Use a specific model
skynet --model qwen2.5-coder:7b

# Resume last session
skynet --resume
skynet -r

# Resume specific session
skynet --session 20231231_143022

# Skip confirmation prompts
skynet --no-confirm
```

### In-Session Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/models` | List installed Ollama models |
| `/model <name\|number>` | Switch to a different model |
| `/sessions` | List saved sessions |
| `/resume <id>` | Resume a specific session |
| `/save` | Manually save current session |
| `/clear` | Clear conversation history |
| `/quit`, `/exit`, `/q` | Save and exit |

### Available Tools

SkyNet has access to 12 tools:

#### File Operations
- `read_file` - Read file contents with line numbers
- `write_file` - Create or overwrite files
- `edit_file` - Search and replace within files

#### Code Search
- `grep` - Search file contents with regex
- `glob` - Find files by pattern

#### Shell
- `bash` - Execute shell commands (with safety guards)

#### Git
- `git_status` - Show repository status
- `git_diff` - Show changes
- `git_commit` - Create commits
- `git_log` - View commit history
- `git_branch` - Manage branches

#### Web
- `web_search` - Search the web via DuckDuckGo
- `web_fetch` - Fetch and extract text from URLs

## Configuration

SkyNet can be configured via environment variables:

```bash
# Default model
export CLAUDE_CLONE_MODEL="qwen2.5-coder:32b"

# Ollama host
export CLAUDE_CLONE_OLLAMA_HOST="http://localhost:11434"

# Context window size
export CLAUDE_CLONE_NUM_CTX=32768

# Temperature
export CLAUDE_CLONE_TEMPERATURE=0.3
```

## Recommended Models

| Model | VRAM | Use Case |
|-------|------|----------|
| `qwen2.5-coder:32b` | ~20GB | Best coding performance |
| `qwen2.5-coder:14b` | ~10GB | Good balance |
| `qwen2.5-coder:7b` | ~5GB | Fast, lightweight |
| `deepseek-coder:33b` | ~20GB | Alternative coding model |
| `llama3.1:8b` | ~5GB | General purpose |

## Project Structure

```
skynet/
├── pyproject.toml          # Project metadata and dependencies
├── README.md               # This file
├── LICENSE                 # MIT License
└── src/
    └── claude_clone/
        ├── cli.py          # CLI entry point and REPL
        ├── config.py       # Configuration management
        ├── core/
        │   ├── agent.py    # Main orchestration loop
        │   └── session.py  # Session persistence
        ├── llm/
        │   └── ollama_provider.py  # Ollama integration
        ├── tools/
        │   ├── base.py     # Tool base class
        │   ├── registry.py # Tool registration
        │   ├── file_ops.py # File tools
        │   ├── search.py   # Grep/glob tools
        │   ├── shell.py    # Bash tool
        │   ├── git.py      # Git tools
        │   └── web.py      # Web search/fetch
        └── ui/
            └── console.py  # Rich terminal UI
```

## Safety Features

- **Dangerous Command Blocking** - Prevents destructive commands like `rm -rf /`
- **Confirmation Prompts** - Asks before writing files or running risky commands
- **Path Validation** - Restricts access to sensitive directories
- **Timeout Protection** - Commands timeout after 2 minutes by default

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run linting
ruff check src/

# Run type checking
mypy src/

# Run tests
pytest
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Ollama](https://ollama.ai) for making local LLM inference easy
- [Rich](https://github.com/Textualize/rich) for beautiful terminal output
- [Click](https://click.palletsprojects.com/) for CLI framework
- Inspired by [Claude Code](https://claude.ai/code) by Anthropic

---

**Note:** This project is not affiliated with Anthropic or Claude. "SkyNet" is a playful name - this AI is here to help, not to take over the world!
