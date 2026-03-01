import argparse
import os
import subprocess
import sys

def install_requirements():
    print("\n>>> [1/4] Installing dependencies from requirements.txt...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        sys.exit(1)

def run_training(epochs, seeds=[1, 2, 3]):
    print(f"\n>>> [2/4] Starting Training (Epochs: {epochs})...")
    
    # Define paths relative to current working directory
    train_data = os.path.join("data", "train", "flair")
    train_gts = os.path.join("data", "train", "gt")
    val_data = os.path.join("data", "dev_in", "flair")
    val_gts = os.path.join("data", "dev_in", "gt")
    
    # Ensure directories exist
    os.makedirs("experiments_unet", exist_ok=True)
    os.makedirs("experiments_swin", exist_ok=True)

    models = [
        ("UNet", "src/train_unet.py", "experiments_unet"),
        ("Swin UNETR", "src/train_swin.py", "experiments_swin")
    ]

    for model_name, script, save_base in models:
        for seed in seeds:
            print(f"--- Training {model_name} | Seed {seed} ---")
            save_path = os.path.join(save_base, f"seed{seed}")
            os.makedirs(save_path, exist_ok=True)
            
            cmd = [
                sys.executable, script,
                "--seed", str(seed),
                "--n_epochs", str(epochs),
                "--path_train_data", train_data,
                "--path_train_gts", train_gts,
                "--path_val_data", val_data,
                "--path_val_gts", val_gts,
                "--path_save", save_path
            ]
            
            try:
                subprocess.check_call(cmd)
            except subprocess.CalledProcessError as e:
                print(f"Training failed for {model_name} seed {seed}: {e}")
                sys.exit(1)

def run_evaluation(seeds=[1, 2, 3]):
    print("\n>>> [3/4] Starting Evaluation...")
    
    test_data = os.path.join("data", "dev_in", "flair")
    test_gts = os.path.join("data", "dev_in", "gt")
    test_bm = os.path.join("data", "dev_in", "fg_mask")
    
    evals = [
        ("UNet", "src/test_unet.py", "experiments_unet"),
        ("Swin UNETR", "src/test_swin.py", "experiments_swin")
    ]

    for model_name, script, model_dir in evals:
        print(f"--- Evaluating {model_name} ---")
        cmd = [
            sys.executable, script,
            "--path_model", model_dir,
            "--path_data", test_data,
            "--path_gts", test_gts,
            "--path_bm", test_bm,
            "--threshold", "0.35",
            "--seeds"
        ] + [str(s) for s in seeds]
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as e:
            print(f"Evaluation failed for {model_name}: {e}")
            sys.exit(1)

def run_audit():
    print("\n>>> [4/4] Starting Failure Mode Audit...")
    
    audit_data = os.path.join("data", "dev_out", "flair")
    audit_gts = os.path.join("data", "dev_out", "gt")
    audit_bm = os.path.join("data", "dev_out", "fg_mask")
    
    cmd = [
        sys.executable, "src/audit.py",
        "--path_unet", "experiments_unet",
        "--path_swin", "experiments_swin",
        "--path_data", audit_data,
        "--path_gts", audit_gts,
        "--path_bm", audit_bm
    ]
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print(f"Audit failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full MS Lesion Segmentation pipeline.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs (default: 100)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3], help="List of seeds to train (default: 1 2 3)")
    parser.add_argument("--skip_install", action="store_true", help="Skip dependency installation")
    parser.add_argument("--skip_train", action="store_true", help="Skip training phase")
    parser.add_argument("--skip_eval", action="store_true", help="Skip evaluation phase")
    parser.add_argument("--skip_audit", action="store_true", help="Skip audit phase")
    
    args = parser.parse_args()

    if not args.skip_install:
        install_requirements()
    
    if not args.skip_train:
        run_training(args.epochs, seeds=args.seeds)
        
    if not args.skip_eval:
        run_evaluation(seeds=args.seeds)
        
    if not args.skip_audit:
        run_audit()