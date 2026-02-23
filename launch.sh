#!/bin/bash
set -e

# Check for Just and install if missing (failsafe)
if ! command -v just &> /dev/null; then
    if command -v brew &> /dev/null; then
        echo "Installing 'just' via Homebrew..."
        brew install just
    else
        echo "Error: 'just' tool is missing and Homebrew is not available."
        exit 1
    fi
fi

just run
