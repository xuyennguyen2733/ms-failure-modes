#!/bin/bash

# 1. Update system and install Python 3.9 + tmux
echo "--- Installing Python 3.9 and tmux ---"
apt update -y
apt install -y python3.9 python3.9-venv tmux

# 2. Setup Virtual Environment
echo "--- Setting up Virtual Environment ---"
if [ ! -d "venv" ]; then
    python3.9 -m venv venv
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

# 3. Configure rclone automatically
echo "--- Configuring rclone ---"
RCLONE_CONF_PATH="/root/.config/rclone/rclone.conf"
mkdir -p $(dirname $RCLONE_CONF_PATH)

# Note: We use a heredoc to create the config file directly
cat <<EOF > $RCLONE_CONF_PATH
[gdrive]
type = drive
scope = drive
token = {"access_token":"ya29.a0ATkoCc6fLYRoHbVW--q4jR6YCoVUeoVnRtqovVXwH6__MwCexu53l8vY_4R8W_tXLG_6HOgewTl2ZD0eLv9AWRfBw922OxJmUXgMNJRPJ87cQWZJvx5P8bqE_U6gEHAPI-vNZTcaFDxPFIeZN1nylAe_24eWpFF6LR-ttqNQxpEvC51UgE-tLGfYzQ48_5a0HH5vvX8aCgYKAWcQSARQSfGHGX2MitmonNyQ3H5LGsAQR79BLOw0206","token_type":"Bearer","refresh_token":"1//06FYU-4f-xgb7CgYIARAAGAYSNwF-L9Ir0nJIv9iy6dSUhCI25YLg8BekuMEhi06lX6O2tdsM2Q-OXfY0--nohyGmywEdT8orB6w","expiry":"2026-03-03T13:59:39.9422592-07:00","expires_in":3599}
EOF