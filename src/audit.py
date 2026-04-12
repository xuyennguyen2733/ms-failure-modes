"""
Audit script for failure-mode analysis.

Primary analysis (always runs, per backbone):
  - Uncertainty Calibration Audit: predictive entropy at FP and FN sites
    for whichever ensemble(s) are provided (UNet, Swin, or both).

Secondary analysis (only when BOTH backbones are provided AND --no_comparison
is not set):
  - Spatial Overlap Audit: IoU of false-positive locations between UNet and
    Swin, used to test whether their failure modes are spatially distinct.
"""

import argparse
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet, SwinUNETR
import numpy as np
from data_load import remove_connected_components, get_val_dataloader
from uncertainty import ensemble_uncertainties_classification

parser = argparse.ArgumentParser(description='Audit one or two backbones.')
# Model paths — at least one is required.
parser.add_argument('--path_unet', type=str, default=None,
                    help='Path to UNet experiments directory (optional)')
parser.add_argument('--path_swin', type=str, default=None,
                    help='Path to Swin UNETR experiments directory (optional)')
parser.add_argument('--no_comparison', action='store_true',
                    help='Skip the cross-model FP-overlap comparison even when '
                         'both backbones are provided. Useful when the project '
                         'has shifted to a single-backbone locality study and '
                         'the two-model comparison is only a secondary check.')
parser.add_argument('--num_models', type=int, default=3,
                    help='Number of models in each ensemble')
# Data
parser.add_argument('--path_data', type=str, required=True,
                    help='Path to FLAIR images')
parser.add_argument('--path_gts', type=str, required=True,
                    help='Path to ground truth masks')
parser.add_argument('--path_bm', type=str, required=True,
                    help='Path to brain masks')
parser.add_argument('--num_workers', type=int, default=1,
                    help='Number of workers')
# Hyperparameters
parser.add_argument('--threshold', type=float, default=0.35,
                    help='Probability threshold')
parser.add_argument('--patch_size', type=int, default=96,
                    help='Cubic sliding-window patch size. Must match the '
                         'patch_size used at training time for both models.')
parser.add_argument('--path_save', type=str, default='visualization',
                    help='Path to save audit plots')


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def load_ensemble(model_class, path, num_models, device, **kwargs):
    models = []
    for i in range(num_models):
        model = model_class(**kwargs).to(device)
        # Load weights
        weights_path = os.path.join(path, f"seed{i + 1}", "Best_model_finetuning.pth")
        if not os.path.exists(weights_path):
            print(f"Warning: Could not find model at {weights_path}")
            continue
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
        models.append(model)
    return models


def get_inference(models, inputs, roi_size, sw_batch_size, act):
    all_outputs = []
    for model in models:
        outputs = sliding_window_inference(inputs, roi_size, sw_batch_size, model, mode='gaussian')
        outputs = act(outputs).cpu().numpy()
        outputs = np.squeeze(outputs[0, 1])
        all_outputs.append(outputs)
    all_outputs = np.asarray(all_outputs)
    
    # Mean prediction
    mean_pred = np.mean(all_outputs, axis=0)
    
    # Uncertainty (Predictive Entropy)
    # Shape input for uncertainty: [Models, X, Y, Z, Classes]
    # We construct [Models, X, Y, Z, 2]
    unc_input = np.concatenate(
        (np.expand_dims(all_outputs, axis=-1),
         np.expand_dims(1. - all_outputs, axis=-1)),
        axis=-1)
    
    uncs = ensemble_uncertainties_classification(unc_input)
    entropy = uncs['entropy_of_expected'] # Predictive Entropy
    
    return mean_pred, entropy


def main(args):
    do_unet = args.path_unet is not None
    do_swin = args.path_swin is not None
    if not (do_unet or do_swin):
        raise ValueError("At least one of --path_unet or --path_swin must be provided.")
    do_comparison = do_unet and do_swin and not args.no_comparison

    device = get_default_device()

    # 1. Load Data
    val_loader = get_val_dataloader(flair_path=args.path_data,
                                    gts_path=args.path_gts,
                                    num_workers=args.num_workers,
                                    bm_path=args.path_bm)

    # 2. Load whichever ensembles were requested
    unet_models, swin_models = [], []
    if do_unet:
        print("Loading UNet Ensemble...")
        unet_models = load_ensemble(UNet, args.path_unet, args.num_models, device,
                                    spatial_dims=3, in_channels=1, out_channels=2,
                                    channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2),
                                    num_res_units=0)
    if do_swin:
        print("Loading Swin UNETR Ensemble...")
        swin_models = load_ensemble(SwinUNETR, args.path_swin, args.num_models, device,
                                    img_size=(args.patch_size, args.patch_size, args.patch_size),
                                    in_channels=1, out_channels=2,
                                    feature_size=48, use_checkpoint=False, spatial_dims=3)

    act = torch.nn.Softmax(dim=1)
    roi_size = (args.patch_size, args.patch_size, args.patch_size)
    sw_batch_size = 4
    th = args.threshold

    # Audit Accumulators (some stay empty when a backbone is absent)
    fp_ious = []
    unet_fp_entropy, swin_fp_entropy = [], []
    unet_fn_entropy, swin_fn_entropy = [], []

    print(f"Starting Audit on {len(val_loader)} subjects "
          f"(unet={do_unet}, swin={do_swin}, comparison={do_comparison})...")

    def process_seg(prob_map):
        seg = prob_map.copy()
        seg[seg >= th] = 1
        seg[seg < th] = 0
        return remove_connected_components(seg)

    with torch.no_grad():
        for count, batch_data in enumerate(val_loader):
            inputs = batch_data["image"].to(device)
            gt = np.squeeze(batch_data["label"].cpu().numpy())
            bm = np.squeeze(batch_data["brain_mask"].cpu().numpy())

            # --- Inference (per requested backbone) ---
            if do_unet:
                pred_unet_prob, ent_unet = get_inference(unet_models, inputs, roi_size, sw_batch_size, act)
                seg_unet = process_seg(pred_unet_prob)
                fp_unet_mask = (seg_unet == 1) & (gt == 0) & (bm == 1)
                fn_unet_mask = (seg_unet == 0) & (gt == 1) & (bm == 1)
                if fp_unet_mask.sum() > 0:
                    unet_fp_entropy.append(np.mean(ent_unet[fp_unet_mask]))
                if fn_unet_mask.sum() > 0:
                    unet_fn_entropy.append(np.mean(ent_unet[fn_unet_mask]))

            if do_swin:
                pred_swin_prob, ent_swin = get_inference(swin_models, inputs, roi_size, sw_batch_size, act)
                seg_swin = process_seg(pred_swin_prob)
                fp_swin_mask = (seg_swin == 1) & (gt == 0) & (bm == 1)
                fn_swin_mask = (seg_swin == 0) & (gt == 1) & (bm == 1)
                if fp_swin_mask.sum() > 0:
                    swin_fp_entropy.append(np.mean(ent_swin[fp_swin_mask]))
                if fn_swin_mask.sum() > 0:
                    swin_fn_entropy.append(np.mean(ent_swin[fn_swin_mask]))

            # --- Cross-model spatial overlap of FPs (secondary, both required) ---
            if do_comparison:
                intersection = np.logical_and(fp_unet_mask, fp_swin_mask).sum()
                union = np.logical_or(fp_unet_mask, fp_swin_mask).sum()
                fp_ious.append(intersection / union if union > 0 else np.nan)

            if (count + 1) % 5 == 0:
                print(f"Processed {count + 1}/{len(val_loader)}")

    # ============================== Reporting ==============================
    print("\n" + "=" * 40)
    print("AUDIT RESULTS")
    print("=" * 40)

    print("\nUncertainty Calibration Audit (Mean Predictive Entropy):")
    print("(Higher Entropy = Model is 'uncertain' about its error)")
    if do_unet:
        print(f"  UNet FP Entropy: {np.mean(unet_fp_entropy):.4f}  "
              f"FN Entropy: {np.mean(unet_fn_entropy):.4f}")
    if do_swin:
        print(f"  Swin FP Entropy: {np.mean(swin_fp_entropy):.4f}  "
              f"FN Entropy: {np.mean(swin_fn_entropy):.4f}")

    valid_ious = [x for x in fp_ious if not np.isnan(x)]
    if do_comparison:
        avg_fp_iou = np.mean(valid_ious) if valid_ious else 0.0
        print(f"\nSpatial Overlap Audit (FP IoU, UNet vs Swin): {avg_fp_iou:.4f}")
        print("(Lower IoU = Distinct Failure Modes)")
    else:
        print("\nSpatial Overlap Audit: SKIPPED "
              "(requires both backbones and --no_comparison not set).")

    # ============================== Plotting ==============================
    if args.path_save:
        os.makedirs(args.path_save, exist_ok=True)
        sns.set_theme(style="whitegrid")

        # Entropy boxplot — only includes whichever lists are non-empty.
        data = [unet_fp_entropy, swin_fp_entropy, unet_fn_entropy, swin_fn_entropy]
        labels = ['UNet FP', 'Swin FP', 'UNet FN', 'Swin FN']
        plot_data, plot_labels = [], []
        for d, lab in zip(data, labels):
            if len(d) > 0:
                plot_data.append(d)
                plot_labels.append(lab)

        if plot_data:
            plt.figure(figsize=(10, 6))
            plt.boxplot(plot_data, labels=plot_labels, patch_artist=True)
            plt.title('Uncertainty Calibration at Failure Sites')
            plt.ylabel('Predictive Entropy')
            save_path = os.path.join(args.path_save, 'uncertainty_audit.png')
            plt.savefig(save_path)
            plt.close()
            print(f"\nVisualization saved to: {save_path}")
        else:
            print("\nNo failure data available to plot.")

        if do_comparison and valid_ious:
            plt.figure(figsize=(10, 6))
            sns.histplot(valid_ious, kde=True, bins=10, color='purple')
            plt.title('Distribution of Spatial Overlap (IoU) of False Positives')
            plt.xlabel('Intersection over Union (IoU)')
            plt.ylabel('Frequency (Number of Subjects)')
            plt.xlim(0, 1.0)
            save_path_iou = os.path.join(args.path_save, 'spatial_overlap_audit.png')
            plt.savefig(save_path_iou)
            plt.close()
            print(f"Visualization saved to: {save_path_iou}")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)