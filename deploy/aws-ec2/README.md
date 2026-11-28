# AWS EC2 Deployment (`systemctl` + `nginx` + `certbot`)

## 1. EC2 Instance & Security Group
- **AMI**: Ubuntu 22.04 LTS or 24.04 LTS (`t3.small` or `t3.medium` recommended for PyTorch/SentenceTransformers)
- **Inbound Security Group rules**:
  - `22` (SSH) from your IP
  - `80` (HTTP) from `0.0.0.0/0` (needed for Certbot HTTP-01 challenge & redirect)
  - `443` (HTTPS) from `0.0.0.0/0`

## 2. One-Command Provisioning on EC2
SSH into your EC2 instance and run:

```bash
curl -fsSL https://raw.githubusercontent.com/alaotach/austint/ui-redesign/deploy/aws-ec2/setup_ec2.sh | sudo bash -s -- osint.yourdomain.com
```

Or if you cloned the repo manually:

```bash
sudo bash deploy/aws-ec2/setup_ec2.sh osint.yourdomain.com
```

## 3. Configure Environment Keys
Edit `/opt/austint/.env` with your API keys (e.g., `OPENAI_API_KEY`, `EXA_API_KEY`, etc.), then restart the service:

```bash
sudo nano /opt/austint/.env
sudo systemctl restart austint
```

## 4. Enable HTTPS with Certbot
Once your domain's `A` record points to the EC2 Elastic IP:

```bash
sudo certbot --nginx -d osint.yourdomain.com
```

Verify auto-renewal:
```bash
sudo certbot renew --dry-run
```

## 5. Useful Operations
- **Service status & logs**:
  ```bash
  sudo systemctl status austint
  sudo journalctl -u austint -f
  ```
- **Deploy updates**:
  ```bash
  cd /opt/austint && git pull && sudo systemctl restart austint
  ```
