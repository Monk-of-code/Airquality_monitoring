#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load .env
set -a
source "$ROOT_DIR/.env"
set +a

echo "==> Creating database (if not exists): ${POSTGRES_DB}"
psql "host=${POSTGRES_HOST} port=${POSTGRES_PORT} dbname=${POSTGRES_ADMIN_DB} user=${POSTGRES_USER} password=${POSTGRES_PASSWORD}" \
  -v ON_ERROR_STOP=1 \
  -f "$ROOT_DIR/db/create_db.sql"

echo "==> Setting up Bronze schema + hypertables in ${POSTGRES_DB}"
psql "host=${POSTGRES_HOST} port=${POSTGRES_PORT} dbname=${POSTGRES_DB} user=${POSTGRES_USER} password=${POSTGRES_PASSWORD}" \
  -v ON_ERROR_STOP=1 \
  -f "$ROOT_DIR/db/bronze_setup.sql"

echo "DB setup complete."
