#!/bin/bash
# Convenience script for running Docker commands

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# .env is optional: with no OPENAI_API_KEY the app runs on its local
# providers. Refusing to start without one made the offline path unreachable
# through this script.
if [ ! -f .env ]; then
    echo -e "${YELLOW}No .env found - running in local mode (no OpenAI calls).${NC}"
    echo -e "${YELLOW}Copy .env.example to .env and set OPENAI_API_KEY to use OpenAI.${NC}"
fi

# Function to display usage
usage() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  build         - Build Docker image"
    echo "  ingest <path> - Ingest data from directory"
    echo "  query <text>  - Generate report from query"
    echo "  stats         - Show collection statistics"
    echo "  clear         - Clear all data"
    echo "  shell         - Start interactive shell in container"
    echo "  logs          - Show container logs"
    echo "  down          - Stop and remove containers"
    echo ""
    echo "Examples:"
    echo "  $0 build"
    echo "  $0 ingest ./data/raw_data"
    echo "  $0 query 'What are the revenue trends?'"
}

# Build image
build() {
    echo -e "${GREEN}Building Docker image...${NC}"
    docker compose build
}

# Ingest data
ingest() {
    if [ -z "$1" ]; then
        echo -e "${RED}Error: Please provide a directory path${NC}"
        exit 1
    fi
    echo -e "${GREEN}Ingesting data from $1...${NC}"
    docker compose run --rm api python main.py --ingest-dir "$1"
}

# Generate report
query() {
    if [ -z "$1" ]; then
        echo -e "${RED}Error: Please provide a query${NC}"
        exit 1
    fi
    echo -e "${GREEN}Generating report...${NC}"
    docker compose run --rm api python main.py --query "$1" --output --format markdown
}

# Show stats
stats() {
    echo -e "${GREEN}Fetching collection statistics...${NC}"
    docker compose run --rm api python main.py --stats
}

# Clear data
clear_data() {
    echo -e "${YELLOW}Clearing all data...${NC}"
    docker compose run --rm api python main.py --clear
}

# Interactive shell
shell() {
    echo -e "${GREEN}Starting interactive shell...${NC}"
    docker compose run --rm api /bin/bash
}

# Show logs
logs() {
    docker compose logs -f api
}

# Stop containers
down() {
    echo -e "${GREEN}Stopping containers...${NC}"
    docker compose down
}

# Main command handler
case "$1" in
    build)
        build
        ;;
    ingest)
        ingest "$2"
        ;;
    query)
        query "$2"
        ;;
    stats)
        stats
        ;;
    clear)
        clear_data
        ;;
    shell)
        shell
        ;;
    logs)
        logs
        ;;
    down)
        down
        ;;
    *)
        usage
        exit 1
        ;;
esac

