#!/usr/bin/env bash
# Pull the images for the current commit from GHCR and restart the stack.
# Called over SSH by the `cd` workflow (after it has checked out GIT_SHA);
# can also be run by hand on the server:  bash deploy/deploy.sh
set -euo pipefail
cd "$(dirname "$0")/.."

GIT_SHA="${GIT_SHA:-$(git rev-parse HEAD)}"
REGISTRY="${REGISTRY:-ghcr.io/anasnaveed69}"
PORT="${HTTP_PORT:-80}"

# keep the deployed tag in .env so later manual `docker compose` commands use it too
touch .env
set_env() {
  if grep -q "^$1=" .env; then sed -i "s|^$1=.*|$1=$2|" .env; else echo "$1=$2" >> .env; fi
}
set_env IMAGE_TAG "$GIT_SHA"
set_env REGISTRY "$REGISTRY"
set_env GIT_COMMIT "$GIT_SHA"

if [ -n "${GHCR_TOKEN:-}" ]; then
  echo "$GHCR_TOKEN" | docker login ghcr.io -u "${GHCR_USER:-github}" --password-stdin
  trap 'docker logout ghcr.io >/dev/null 2>&1 || true' EXIT
fi

mkdir -p data/raw data/processed models/serving
docker compose pull --quiet
docker compose up -d --remove-orphans
docker image prune -f > /dev/null

for _ in $(seq 1 36); do
  if curl -fsS "http://localhost:${PORT}/api/health" > /dev/null; then
    echo "deployed ${GIT_SHA:0:7}"
    docker compose ps
    curl -sS "http://localhost:${PORT}/api/ready" || true
    exit 0
  fi
  sleep 5
done
echo "API did not become healthy" >&2
docker compose logs --tail 80 api nginx >&2
exit 1
