set shell := ["bash", "-c"]

# Default task
default:
    @just --list

# Full setup: system dependencies, uv installation, and venv creation
setup: install-sys-deps install-uv create-env

# Install uv (The fastest Python package manager) if missing
install-uv:
    @if ! command -v uv &> /dev/null; then \
        echo "uv not found. Installing uv..."; \
        curl -LsSf https://astral.sh/uv/install.sh | sh; \
        source $HOME/.cargo/env; \
    else \
        echo "uv is already installed."; \
    fi

# Create virtual environment and install dependencies using uv
create-env:
    @echo "Creating virtual environment with Python 3.14 (managed by uv)..."
    @uv venv --python 3.14
    @echo "Installing dependencies via uv (lightning fast)..."
    @uv pip install -r requirements.txt
    @echo "Environment ready."

# Run the application
run:
    @echo "Launching Atelier of Synthetic Voice..."
    @source .venv/bin/activate && python main.py

# Install system dependencies via Homebrew (macOS only)
install-sys-deps:
    @if [[ "$OSTYPE" == "darwin"* ]]; then \
        if ! command -v brew &> /dev/null; then \
            echo "Homebrew not found. Please install it first: https://brew.sh/"; \
            exit 1; \
        fi; \
        if ! command -v ffmpeg &> /dev/null; then \
            echo "Installing ffmpeg..."; \
            brew install ffmpeg; \
        else \
            echo "ffmpeg already installed."; \
        fi; \
    fi

# Clean up environment
clean:
    rm -rf .venv
    find . -type d -name "__pycache__" -exec rm -rf {} +
