import subprocess
import sys


def check_ollama_installed():
    try:
        subprocess.run(["ollama", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Ollama is already installed.")
    except subprocess.CalledProcessError:
        print("Ollama is not installed. Installing now...")
        install_ollama()


def install_ollama():
    # Replace the following line with the actual installation command for Ollama
    install_command = "curl -sSL https://ollama/install.sh | sh"
    result = subprocess.run(install_command, shell=True)
    if result.returncode != 0:
        print("Failed to install Ollama. Please install it manually.")
        sys.exit(1)
    else:
        print("Ollama installed successfully.")


if __name__ == "__main__":
    check_ollama_installed()