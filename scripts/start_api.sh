#!/bin/bash
# Start the FastAPI server locally.
#
# Port 8350 rather than 8000: 8000 is the single most contested port on a
# developer machine, and a server that silently binds to someone else's is a
# bad first five minutes. Override with PORT=... if 8350 is taken.

set -e

PORT="${PORT:-8350}"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Starting RAG API Server...${NC}"
echo ""
echo "API will be available at:"
echo "  • Main API: http://localhost:${PORT}"
echo "  • Interactive Docs (Swagger): http://localhost:${PORT}/docs"
echo "  • Alternative Docs (ReDoc): http://localhost:${PORT}/redoc"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
echo ""

exec python -m uvicorn src.api.main:app --host 0.0.0.0 --port "${PORT}" --reload
