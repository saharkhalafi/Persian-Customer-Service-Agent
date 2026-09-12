#!/bin/sh
set -eu

PORT="${PORT:-8000}"
WORKERS="${UVICORN_WORKERS:-1}"

if [ "${1:-api}" = "api" ]; then
  exec uvicorn app.main:app --host 0.0.0.0 --port "$PORT" --workers "$WORKERS"
fi

if [ "$1" = "migrate" ]; then
  exec alembic upgrade head
fi

exec "$@"
