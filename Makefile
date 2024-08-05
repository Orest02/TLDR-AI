# Makefile

.PHONY: check-ollama install-ollama check-poetry install-poetry setup

check-ollama:
	@which ollama > /dev/null 2>&1 || $(MAKE) install-ollama

install-ollama:
	@echo "Ollama is not installed. Installing..."
	@# Add the command to install Ollama, for example:
	@# curl -sSL https://ollama/install.sh | sh
	@# Replace the above line with the actual installation command for Ollama

check-poetry:
	@which poetry > /dev/null 2>&1 || $(MAKE) install-poetry

install-poetry:
	@echo "Poetry is not installed. Installing..."
	@curl -sSL https://install.python-poetry.org | python3 -

setup: check-ollama check-poetry
	@poetry install

install: check-ollama check-poetry
	@poetry install --no-dev
