#!/usr/bin/env bash
set -euo pipefail

# Idempotent Ubuntu EC2 setup for Intelligent OSINT (systemctl + nginx + certbot)
# Usage:
#   sudo bash setup_ec2.sh [DOMAIN] [REPO_URL] [BRANCH]

DOMAIN="${1:-_}"
REPO_URL="${2:-https://github.com/alaotach/austint.git}"
BRANCH="${3:-ui-redesign}"
APP_DIR="/opt/austint"
RUN_USER="${SUDO_USER:-ubuntu}"

echo "[1/6] Installing system packages..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y python3-venv python3-pip git nginx certbot python3-certbot-nginx

echo "[2/6] Cloning or updating repository in ${APP_DIR}..."
if [ ! -d "${APP_DIR}/.git" ]; then
  git clone --branch "${BRANCH}" "${REPO_URL}" "${APP_DIR}"
else
  git -C "${APP_DIR}" fetch origin "${BRANCH}"
  git -C "${APP_DIR}" checkout "${BRANCH}"
  git -C "${APP_DIR}" pull --ff-only origin "${BRANCH}"
fi
chown -R "${RUN_USER}:${RUN_USER}" "${APP_DIR}"

echo "[3/6] Creating Python virtualenv and installing dependencies..."
if [ ! -d "${APP_DIR}/.venv" ]; then
  sudo -u "${RUN_USER}" python3 -m venv "${APP_DIR}/.venv"
fi
sudo -u "${RUN_USER}" "${APP_DIR}/.venv/bin/pip" install --upgrade pip
sudo -u "${RUN_USER}" "${APP_DIR}/.venv/bin/pip" install -r "${APP_DIR}/requirements.txt"

if [ ! -f "${APP_DIR}/.env" ] && [ -f "${APP_DIR}/.env.example" ]; then
  cp "${APP_DIR}/.env.example" "${APP_DIR}/.env"
  chown "${RUN_USER}:${RUN_USER}" "${APP_DIR}/.env"
  echo "NOTE: Created ${APP_DIR}/.env from .env.example — add your API keys there."
fi

echo "[4/6] Installing systemd service..."
sed "s/^User=.*/User=${RUN_USER}/; s/^Group=.*/Group=${RUN_USER}/" \
  "${APP_DIR}/deploy/aws-ec2/austint.service" > /etc/systemd/system/austint.service
systemctl daemon-reload
systemctl enable austint
systemctl restart austint

echo "[5/6] Configuring Nginx reverse proxy..."
sed "s/YOUR_DOMAIN_OR_EC2_HOST/${DOMAIN}/g" \
  "${APP_DIR}/deploy/aws-ec2/nginx-austint.conf" > /etc/nginx/sites-available/austint
ln -sf /etc/nginx/sites-available/austint /etc/nginx/sites-enabled/austint
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl enable nginx
systemctl reload nginx

echo "[6/6] TLS / Certbot..."
if [ "${DOMAIN}" != "_" ]; then
  echo "Run the following to issue & auto-renew your TLS certificate:"
  echo "  sudo certbot --nginx -d ${DOMAIN}"
else
  echo "No domain provided. Skipping certbot. Once DNS points to this EC2 instance, run:"
  echo "  sudo sed -i 's/server_name _;/server_name your.domain.com;/' /etc/nginx/sites-available/austint"
  echo "  sudo nginx -t && sudo systemctl reload nginx"
  echo "  sudo certbot --nginx -d your.domain.com"
fi

echo "Done! Check service status with: sudo systemctl status austint"
