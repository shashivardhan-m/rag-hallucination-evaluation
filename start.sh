#!/bin/bash
# ============================================================
# RAG Hallucination Evaluator — Full Stack Startup Script
# ============================================================
set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend"

echo "🚀 RAG Hallucination Evaluator - Starting Up"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# --- Check .env ---
if [ ! -f "$BACKEND_DIR/.env" ]; then
  echo "⚠️  No .env found. Copying from .env.example..."
  cp "$BACKEND_DIR/.env.example" "$BACKEND_DIR/.env"
  echo "   → Please edit backend/.env and add your GROQ_API_KEY"
fi

# --- Python venv ---
if [ ! -d "$BACKEND_DIR/venv" ]; then
  echo "🐍 Creating Python virtual environment..."
  python3 -m venv "$BACKEND_DIR/venv"
fi

echo "📦 Installing/verifying backend dependencies..."
source "$BACKEND_DIR/venv/bin/activate"
pip install -q -r "$BACKEND_DIR/requirements.txt"

# --- Node modules ---
if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
  echo "📦 Installing frontend dependencies..."
  cd "$FRONTEND_DIR" && npm install
fi

# --- Start Backend ---
echo ""
echo "⚡ Starting FastAPI backend on http://localhost:8000 ..."
cd "$BACKEND_DIR"
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
echo "   Backend PID: $BACKEND_PID"

# --- Start Frontend ---
echo "🎨 Starting Vite frontend on http://localhost:5173 ..."
cd "$FRONTEND_DIR"
npm run dev &
FRONTEND_PID=$!
echo "   Frontend PID: $FRONTEND_PID"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Both servers started!"
echo "   Frontend →  http://localhost:5173"
echo "   Backend  →  http://localhost:8000"
echo "   API Docs →  http://localhost:8000/docs"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Press Ctrl+C to stop both servers."

# --- Wait and cleanup ---
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo '👋 Stopped'; exit 0" INT TERM

wait $BACKEND_PID
wait $FRONTEND_PID
