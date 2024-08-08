# TLDR AI CLI Tool

[![PyPI version](https://badge.fury.io/py/tldrai.svg)](https://badge.fury.io/py/tldrai)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


## Overview

TLDR AI CLI Tool is a command-line interface (CLI) designed to provide quick and efficient solutions to complex queries using Large Language Models (LLMs). The tool integrates various AI models to fetch, process, and summarize information from sources like Stack Overflow.

The TLDR AI CLI Tool is inspired by [howdoi](https://github.com/gleitz/howdoi). Unlike howdoi, this tool leverages the power of LLMs and can even work offline if Stack Overflow is not used.


## Features

- Summarizes complex topics into concise, understandable summaries.
- Fetches and processes Stack Overflow questions and answers.
- Utilizes multiple LLMs, including Ollama and other transformers.
- Configurable via YAML files.
- Supports real-time response streaming.

## Installation

### Prerequisites

- Python 3.8+
- Ollama

### Install Ollama

Ollama is required for some functionalities of this tool. Please install Ollama according to your operating system by following the instructions provided in their [Documentation](https://github.com/ollama/ollama/blob/main/README.md#ollama).

### Install TLDR AI CLI Tool

You can install the TLDR AI CLI Tool via `pip`:

```bash
pip install tldrai
```

## Usage

To use the tool, simply run:

```bash
tldrai apply function to pandas column
```

You can also override specific configuration settings using the --set argument:

```bash
tldrai apply function to pandas column --set summarization_pipeline.model=stable-code precision=float32 -v
```

### CLI Options

- `question`: The question or query to process.
- `--config`: Path to the configuration file (default: config/config.yaml).
- `--set`: Override configuration key-value pairs (e.g., key=value).
- `-v`, `--verbose`: Output all logs verbose.
- `-s`, `--search`: Search for similar quetions on Stack Overflow and feed it to the model as suggestions

## Configuration

The TLDR AI CLI Tool is highly configurable. Configuration settings are managed through YAML files. Currently, the default configuration file used is `config/ollama_stable_code.yml`.

## Forming questions
I found two distinct ways to form the questions, depending on if you use StackOverflow search or not:
- Without searching StackOverflow I found it more useful to form a full sentence (without the 'how to' part), for example `tldrai solve \'ModuleNotFound\' error`
- To search StackOverflow (using the `-s` or `--search` flag) it's better to form a question as if you wanted to google it, for example `tldrai \'ModuleNotFound\' -s`

The examples are simply for illustrative purpose and sometimes the question can be formed that simultaneously satisfies the both ways.
## Contributing

We welcome contributions to the TLDR AI CLI Tool! If you have any suggestions or find any issues, please open an issue or subApache 2.0 a pull request on our [GitHub repository](https://github.com/yourusername/tldrai).

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Ollama](https://github.com/ollama/ollama)
- [Hydra](https://github.com/facebookresearch/hydra)
- [StackAPI](https://github.com/lukasz-madon/stackapi)
- [WandB](https://www.wandb.com/)

---

Thank you for using TLDR AI CLI Tool. We hope it enhances your productivity and makes working with complex information easier and more efficient.
