#!/bin/bash
#apt update -y && apt install -y python3.9 python3.9-venv
# Function to handle interruption (Ctrl+C)
handle_interrupt() {
    echo -e "\n\nScript interrupted!"
    
    # Kill any running background jobs
    if [ -n "$(jobs -p)" ]; then
        kill $(jobs -p) 2>/dev/null
    fi

    if [ -n "$RUNPOD_POD_ID" ]; then
        # Ask user with 10s timeout
        read -t 10 -p "Stop the pod? (Y/n) [Default: Yes, Auto-stop in 10s]: " response
        
        # If timeout (exit code != 0) or response is empty or starts with y/Y
        if [ $? -ne 0 ] || [[ -z "$response" ]] || [[ "$response" =~ ^[Yy] ]]; then
            echo -e "\nStopping pod $RUNPOD_POD_ID..."
            runpodctl stop pod "$RUNPOD_POD_ID"
        else
            echo -e "\nPod kept running."
        fi
    fi
    exit 1
}

# Set trap for SIGINT (Ctrl+C) and SIGTERM
trap 'handle_interrupt' SIGINT SIGTERM
# Ensure jobs are killed on exit (e.g. normal finish)
trap 'kill $(jobs -p) 2>/dev/null' EXIT

# Get epochs from first argument, default to 300
N_EPOCHS=${1:-300}
echo "Training for $N_EPOCHS epochs."

# 1. Install Dependencies
#echo "Installing dependencies..."
#pip install -r requirements.txt

# 2. Set Paths (Assumes you upload the 'data' folder to the project root)
TRAIN_DATA="data/train/flair"
TRAIN_GTS="data/train/gt"
VAL_DATA="data/dev_in/flair"
VAL_GTS="data/dev_in/gt"

# Create output directories
mkdir -p experiments_unet
mkdir -p experiments_swin

# 3. Training Logic
# Check for NVIDIA GPU count
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    GPU_COUNT=0
fi

echo "Detected $GPU_COUNT GPUs."

if [ "$GPU_COUNT" -ge 2 ]; then
    echo "Running UNet on GPU 0 and Swin UNETR on GPU 1 in parallel..."
    
    # Run UNet on GPU 0
    (
        export CUDA_VISIBLE_DEVICES=0
        for seed in 1 2 3; do
            echo "Training UNet Seed $seed..."
            mkdir -p "experiments_unet/seed${seed}"
            python src/train_unet.py \
                --seed $seed \
                --n_epochs $N_EPOCHS \
                --path_train_data "$TRAIN_DATA" \
                --path_train_gts "$TRAIN_GTS" \
                --path_val_data "$VAL_DATA" \
                --path_val_gts "$VAL_GTS" \
                --path_save "experiments_unet/seed${seed}" &
        done
        wait
    ) &
    
    # Run Swin on GPU 1
    (
        export CUDA_VISIBLE_DEVICES=1
        for seed in 1 2 3; do
            echo "Training Swin Seed $seed..."
            mkdir -p "experiments_swin/seed${seed}"
            python src/train_swin.py \
                --seed $seed \
                --n_epochs $N_EPOCHS \
                --path_train_data "$TRAIN_DATA" \
                --path_train_gts "$TRAIN_GTS" \
                --path_val_data "$VAL_DATA" \
                --path_val_gts "$VAL_GTS" \
                --path_save "experiments_swin/seed${seed}" &
        done
        wait
    ) &
    
    wait
    echo "Parallel training complete."

else
    echo "Running seeds in parallel (Single GPU detected)..."
    
    # UNet
    for seed in 1 2 3; do
        echo "Training UNet Seed $seed..."
        mkdir -p "experiments_unet/seed${seed}"
        python src/train_unet.py \
            --seed $seed \
            --n_epochs $N_EPOCHS \
            --path_train_data "$TRAIN_DATA" \
            --path_train_gts "$TRAIN_GTS" \
            --path_val_data "$VAL_DATA" \
            --path_val_gts "$VAL_GTS" \
            --path_save "experiments_unet/seed${seed}" &
    done
    wait

    # Swin
    for seed in 1 2 3; do
        echo "Training Swin Seed $seed..."
        mkdir -p "experiments_swin/seed${seed}"
        python src/train_swin.py \
            --seed $seed \
            --n_epochs $N_EPOCHS \
            --path_train_data "$TRAIN_DATA" \
            --path_train_gts "$TRAIN_GTS" \
            --path_val_data "$VAL_DATA" \
            --path_val_gts "$VAL_GTS" \
            --path_save "experiments_swin/seed${seed}" &
    done
    wait
    
    echo "Training complete."
fi

# Stop the pod to avoid extra charges (RunPod specific)
if [ -n "$RUNPOD_POD_ID" ]; then
    echo "RunPod environment detected. Stopping pod $RUNPOD_POD_ID..."
    runpodctl stop pod "$RUNPOD_POD_ID"
fi