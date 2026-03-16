# Deploying AI Trading Bot on Hostinger VPS

## 1. Buy the Right Hostinger Plan

Go to hostinger.com → **VPS Hosting**

| Plan | RAM | vCPU | Recommended? |
|------|-----|------|--------------|
| KVM 1 | 1 GB | 1 | ❌ Too small (pandas + numpy need >1GB) |
| **KVM 2** | **2 GB** | **2** | **✅ Minimum recommended** |
| KVM 4 | 4 GB | 4 | ✅ Best for stable live trading |

- OS: **Ubuntu 22.04 LTS** (select during setup)
- Location: **Singapore** or **India** (closer to NSE = faster API calls)

---

## 2. First Login via SSH

Hostinger gives you the server IP and root password in the VPS dashboard.

```bash
ssh root@YOUR_SERVER_IP
```

Change root password when prompted on first login.

---

## 3. Secure the Server

```bash
# Update everything
apt update && apt upgrade -y

# Create a dedicated app user (don't run as root)
adduser tradebot
usermod -aG sudo tradebot

# Copy your SSH key to the new user (optional but recommended)
rsync --archive --chown=tradebot:tradebot ~/.ssh /home/tradebot

# Switch to app user
su - tradebot
```

---

## 4. Install System Dependencies

```bash
sudo apt install -y \
  python3.11 python3.11-venv python3-pip \
  nginx git curl ufw certbot python3-certbot-nginx

# Confirm Python version
python3.11 --version
```

---

## 5. Upload Your Code

### Option A — GitHub (recommended)

First, on your Mac, push the code to a **private GitHub repo**:

```bash
# On your Mac
cd /Users/karanravi/Desktop/Trade-Bot
git init
git add .
git commit -m "initial deploy"
git remote add origin https://github.com/YOUR_USERNAME/trade-bot.git
git push -u origin main
```

Then on the server:

```bash
# On the VPS
cd /var/www
sudo mkdir tradebot
sudo chown tradebot:tradebot tradebot
git clone https://github.com/YOUR_USERNAME/trade-bot.git tradebot
cd tradebot
```

### Option B — SFTP / SCP (no GitHub needed)

```bash
# On your Mac — upload the whole project folder
scp -r /Users/karanravi/Desktop/Trade-Bot/* tradebot@YOUR_SERVER_IP:/var/www/tradebot/
```

---

## 6. Python Virtual Environment + Dependencies

```bash
cd /var/www/tradebot
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 7. Set Up Environment Variables

```bash
# Create the .env file with your real API keys
nano /var/www/tradebot/.env
```

Paste exactly this (with your real keys):

```
ANTHROPIC_API_KEY=sk-ant-api03-YOUR_REAL_KEY
GROWW_API_TOKEN=YOUR_REAL_GROWW_JWT
GROWW_API_SECRET=YOUR_REAL_GROWW_SECRET
```

Save: `Ctrl+X` → `Y` → `Enter`

Lock down the file permissions:

```bash
chmod 600 /var/www/tradebot/.env
```

---

## 8. Create Log Directory

```bash
sudo mkdir -p /var/log/tradebot
sudo chown tradebot:tradebot /var/log/tradebot
```

---

## 9. Install as a systemd Service (auto-start on reboot)

```bash
# Copy the service file
sudo cp /var/www/tradebot/tradebot.service /etc/systemd/system/

# Reload systemd and enable the service
sudo systemctl daemon-reload
sudo systemctl enable tradebot
sudo systemctl start tradebot

# Check it's running
sudo systemctl status tradebot

# Watch live logs
sudo journalctl -u tradebot -f
```

---

## 10. Configure Nginx (Reverse Proxy)

```bash
# Copy the nginx config
sudo cp /var/www/tradebot/nginx.conf /etc/nginx/sites-available/tradebot

# Edit it to put in your domain
sudo nano /etc/nginx/sites-available/tradebot
# Replace "YOUR_DOMAIN.COM" with your actual domain (e.g. tradebot.yourdomain.com)

# Enable the site
sudo ln -s /etc/nginx/sites-available/tradebot /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default   # remove default placeholder

# Test nginx config
sudo nginx -t

# Reload nginx
sudo systemctl restart nginx
```

---

## 11. Point Your Domain to the Server

In **Hostinger hPanel → Domains → DNS Zone**:

| Type | Name | Value | TTL |
|------|------|-------|-----|
| A | `@` or `tradebot` | `YOUR_VPS_IP` | 300 |
| A | `www` | `YOUR_VPS_IP` | 300 |

Wait 5–15 minutes for DNS to propagate.

---

## 12. Free SSL Certificate (HTTPS)

```bash
sudo certbot --nginx -d YOUR_DOMAIN.COM -d www.YOUR_DOMAIN.COM

# Auto-renewal is set up automatically. Test it:
sudo certbot renew --dry-run
```

After certbot runs, your site will be available at `https://YOUR_DOMAIN.COM`.

---

## 13. Firewall

```bash
sudo ufw allow OpenSSH
sudo ufw allow 'Nginx Full'
sudo ufw enable
sudo ufw status
```

---

## 14. Useful Commands After Deploy

```bash
# Restart the bot
sudo systemctl restart tradebot

# Stop the bot
sudo systemctl stop tradebot

# Watch live logs
sudo journalctl -u tradebot -f

# Watch app-specific logs
tail -f /var/log/tradebot/app.log

# Pull latest code update from GitHub and restart
cd /var/www/tradebot && git pull && sudo systemctl restart tradebot
```

---

## 15. Updating the App

Whenever you make changes on your Mac:

```bash
# On your Mac
git add . && git commit -m "update" && git push

# On the VPS
cd /var/www/tradebot && git pull && sudo systemctl restart tradebot
```

---

## Architecture Overview

```
Internet
    │
    ▼
Nginx (port 80/443)   ← handles SSL, static files, WebSocket upgrade
    │
    ▼
Gunicorn (port 5001)  ← 1 worker, 8 threads, gthread mode
    │
    ▼
Flask App (app.py)    ← Socket.IO, REST API, UI
    │
    ├── Scheduler (threading)   ← background trading loop
    ├── trader.py               ← scan + AI logic
    ├── portfolio.json          ← persisted to disk
    └── logs/                   ← trading logs
```

---

## Important Security Notes

- **Never** commit `.env` to GitHub — it contains your API keys
- `.env` is already in `.gitignore` ✅
- `portfolio.json` is also in `.gitignore` ✅
- After deploy, back up `/var/www/tradebot/.env` and `/var/www/tradebot/portfolio.json` separately
- Consider setting up daily portfolio.json backups: `crontab -e` → `0 2 * * * cp /var/www/tradebot/portfolio.json /var/backups/portfolio_$(date +\%F).json`
