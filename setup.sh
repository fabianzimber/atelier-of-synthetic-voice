#!/bin/bash
set -e

# Colors for output
BLUE='\033[0;34m'
GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${BLUE}Starting Atelier of Synthetic Voice Expert Setup...${NC}"

# 1. Install uv (The gold standard for Python package management)
if ! command -v uv &> /dev/null; then
    echo -e "${BLUE}Installing 'uv' for high-performance environment management...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Source for immediate use in the current script
    source $HOME/.cargo/env
fi

# 2. Check for Homebrew and install 'just' task runner
if ! command -v just &> /dev/null; then
    echo -e "${BLUE}Installing 'just' task runner...${NC}"
    if command -v brew &> /dev/null; then
        brew install just
    else
        # Fallback to uv tool installation if brew is missing
        uv tool install just
    fi
fi

# 3. Run the optimized setup via justfile
just setup

echo -e "${GREEN}Expert setup complete. Environment is isolated and optimized.${NC}"
echo -e "Launch the studio with: ${BLUE}./launch.sh${NC}"
