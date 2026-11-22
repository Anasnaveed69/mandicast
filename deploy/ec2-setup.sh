#!/usr/bin/env bash
# One-time setup of a fresh Ubuntu 22.04/24.04 EC2 instance for MandiCast.
#   ssh ubuntu@<ip> 'bash -s' < deploy/ec2-setup.sh
# Recommended: t3.medium (4 GB). t3.small works with the swap file below.
# Security group: allow 22 (your IP) and 80 (anyone).
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/Anasnaveed69/mandicast.git}"
APP_DIR="$HOME/mandicast"

# docker engine + compose plugin
if ! command -v docker > /dev/null; then
  curl -fsSL https://get.docker.com | sudo sh
  sudo usermod -aG docker "$USER"
fi
sudo apt-get install -y git curl > /dev/null

# swap: torch + the pipeline need more than 2 GB at peak
if ! swapon --show | grep -q swapfile; then
  sudo fallocate -l 4G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile > /dev/null
  sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab > /dev/null
fi

[ -d "$APP_DIR/.git" ] || git clone "$REPO_URL" "$APP_DIR"
cd "$APP_DIR"
mkdir -p data/raw data/processed models/serving

if [ ! -f .env ]; then
  cp .env.example .env
  sed -i "s|^ADMIN_TOKEN=.*|ADMIN_TOKEN=$(openssl rand -hex 24)|" .env
  sed -i "s|^POSTGRES_PASSWORD=.*|POSTGRES_PASSWORD=$(openssl rand -hex 24)|" .env
fi

cat <<EOF

Server ready. Next:
  1. log out and back in (docker group), then: cd ~/mandicast
  2. data: set DVC_REMOTE_URL in .env (+ an IAM role / AWS keys for the bucket),
     or copy the csv:  scp data/raw/amis_lahore.csv ubuntu@<ip>:~/mandicast/data/raw/
  3. first deploy: push to main (GitHub Actions) or run  bash deploy/deploy.sh
  4. train + register into MLflow:  docker compose --profile train run --rm pipeline
EOF
