#!/bin/bash
# Start LangGraph Studio for visualizing the agent

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting LangGraph Studio...${NC}"
echo ""
echo "LangGraph Studio will open in your browser"
echo "You'll be able to visualize and debug the agent workflow"
echo ""
echo -e "${YELLOW}Make sure you have langgraph-cli installed:${NC}"
echo "  pip install langgraph-cli"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop LangGraph Studio${NC}"
echo ""

# Check if langgraph is installed
if ! command -v langgraph &> /dev/null; then
    echo "Error: langgraph-cli not found. Installing..."
    pip install langgraph-cli
fi

# Start LangGraph Studio
langgraph dev

