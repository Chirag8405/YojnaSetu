#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend"
BACKEND_PID=""
FRONTEND_PID=""

port_in_use() {
  local port="$1"
  ss -ltn 2>/dev/null | grep -qE "[\.:]${port}[[:space:]]"
}

cleanup() {
  if [[ -n "${BACKEND_PID:-}" ]] && kill -0 "$BACKEND_PID" 2>/dev/null; then
    kill "$BACKEND_PID" 2>/dev/null || true
  fi
  if [[ -n "${FRONTEND_PID:-}" ]] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
    kill "$FRONTEND_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv is required for backend setup. Install from https://docs.astral.sh/uv/"
  exit 1
fi

if [[ ! -f "$ROOT_DIR/.env" ]] && [[ -f "$ROOT_DIR/.env.example" ]]; then
  cp "$ROOT_DIR/.env.example" "$ROOT_DIR/.env"
  echo "Created .env from .env.example"
fi

if [[ ! -x "$BACKEND_DIR/.venv311/bin/python" ]]; then
  echo "Preparing backend Python 3.11 environment..."
  (
    cd "$BACKEND_DIR"
    uv python install 3.11
    uv venv --python 3.11 .venv311
    uv pip install --python .venv311/bin/python -r requirements.txt
  )
fi

if port_in_use 8000; then
  echo "Backend port 8000 is already in use; reusing existing backend process."
else
  echo "Starting backend on http://localhost:8000"
  (
    cd "$BACKEND_DIR"
    exec .venv311/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
  ) &
  BACKEND_PID=$!
fi

if port_in_use 3000; then
  echo "Frontend port 3000 is already in use; reusing existing frontend process."
else
  echo "Starting frontend on http://localhost:3000"
  (
    cd "$FRONTEND_DIR"
    exec npm run dev
  ) &
  FRONTEND_PID=$!
fi

if [[ -n "$BACKEND_PID" && -n "$FRONTEND_PID" ]]; then
  wait -n "$BACKEND_PID" "$FRONTEND_PID"
elif [[ -n "$BACKEND_PID" ]]; then
  wait "$BACKEND_PID"
elif [[ -n "$FRONTEND_PID" ]]; then
  wait "$FRONTEND_PID"
else
  echo "Both backend and frontend are already running."
fi
