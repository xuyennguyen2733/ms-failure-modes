"""
Audit script to compare 3D UNet and Swin UNETR models.
Performs:
1. Spatial Overlap Audit (Intersection of False Positives).
2. Uncertainty Calibration Audit (Predictive Entropy at failure sites).
"""

import argparse
import os
import torch
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet, SwinUNETR
import numpy as np
from data_load import remove_connected_components, get_val_dataloader
from uncertainty import ensemble_uncertainties_classification

parser = argparse.ArgumentParser(description='Audit and compare models.')
# Model paths
parser.add_argument('--path_unet', type=str, required=True,
                    help='Path to UNet experiments directory')
parser.add_argument('--path_swin', type=str, required=True,
                    help='Path to Swin UNETR experiments directory')
parser.add_argument('--num_models', type=int, default=3,
                    help='Number of models in each ensemble')
# Data
parser.add_argument('--path_data', type=str, required=True,
                    help='Path to FLAIR images')
parser.add_argument('--path_gts', type=str, required=True,
                    help='Path to ground truth masks')
parser.add_argument('--path_bm', type=str, required=True,
                    help='Path to brain masks')
parser.add_argument('--num_workers', type=int, default=4,
                    help='Number of workers')
# Hyperparameters
parser.add_argument('--threshold', type=float, default=0.35,
                    help='Probability threshold')


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
    device = get_default_device()
    
    # 1. Load Data
    val_loader = get_val_dataloader(flair_path=args.path_data,
                                    gts_path=args.path_gts,
                                    num_workers=args.num_workers,
                                    bm_path=args.path_bm)
    
    # 2. Load Models
    print("Loading UNet Ensemble...")
    unet_models = load_ensemble(UNet, args.path_unet, args.num_models, device,
                                spatial_dims=3, in_channels=1, out_channels=2,
                                channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2), num_res_units=0)
    
    print("Loading Swin UNETR Ensemble...")
    swin_models = load_ensemble(SwinUNETR, args.path_swin, args.num_models, device,
                                img_size=(96, 96, 96), in_channels=1, out_channels=2,
                                feature_size=48, use_checkpoint=True, spatial_dims=3)

    act = torch.nn.Softmax(dim=1)
    roi_size = (96, 96, 96)
    sw_batch_size = 4
    th = args.threshold

    # Audit Accumulators
    fp_ious = []
    unet_fp_entropy = []
    swin_fp_entropy = []
    unet_fn_entropy = []
    swin_fn_entropy = []
    
    print(f"Starting Audit on {len(val_loader)} subjects...")

    with torch.no_grad():
        for count, batch_data in enumerate(val_loader):
            inputs = batch_data["image"].to(device)
            gt = np.squeeze(batch_data["label"].cpu().numpy())
            bm = np.squeeze(batch_data["brain_mask"].cpu().numpy())
            
            # Inference
            pred_unet_prob, ent_unet = get_inference(unet_models, inputs, roi_size, sw_batch_size, act)
            pred_swin_prob, ent_swin = get_inference(swin_models, inputs, roi_size, sw_batch_size, act)
            
            # Threshold & Post-process
            def process_seg(prob_map):
                seg = prob_map.copy()
                seg[seg >= th] = 1
                seg[seg < th] = 0
                seg = remove_connected_components(seg)
                return seg

            seg_unet = process_seg(pred_unet_prob)
            seg_swin = process_seg(pred_swin_prob)
            
            # --- Audit 1: Spatial Overlap of False Positives ---
            # FP = Predicted 1 BUT Ground Truth 0 (masked by brain mask)
            fp_unet_mask = (seg_unet == 1) & (gt == 0) & (bm == 1)
            fp_swin_mask = (seg_swin == 1) & (gt == 0) & (bm == 1)
            
            intersection = np.logical_and(fp_unet_mask, fp_swin_mask).sum()
            union = np.logical_or(fp_unet_mask, fp_swin_mask).sum()
            
            if union > 0:
                fp_ious.append(intersection / union)
            else:
                fp_ious.append(np.nan) # No FPs for either model

            # --- Audit 2: Uncertainty Calibration at Failure Sites ---
            # FN = Predicted 0 BUT Ground Truth 1
            fn_unet_mask = (seg_unet == 0) & (gt == 1) & (bm == 1)
            fn_swin_mask = (seg_swin == 0) & (gt == 1) & (bm == 1)
            
            # Extract entropies at these locations
            if fp_unet_mask.sum() > 0: unet_fp_entropy.append(np.mean(ent_unet[fp_unet_mask]))
            if fp_swin_mask.sum() > 0: swin_fp_entropy.append(np.mean(ent_swin[fp_swin_mask]))
            
            if fn_unet_mask.sum() > 0: unet_fn_entropy.append(np.mean(ent_unet[fn_unet_mask]))
            if fn_swin_mask.sum() > 0: swin_fn_entropy.append(np.mean(ent_swin[fn_swin_mask]))

            if (count + 1) % 5 == 0:
                print(f"Processed {count + 1}/{len(val_loader)}")

    # --- Reporting ---
    print("\n" + "="*40)
    print("AUDIT RESULTS")
    print("="*40)
    
    # 1. Spatial Overlap
    valid_ious = [x for x in fp_ious if not np.isnan(x)]
    avg_fp_iou = np.mean(valid_ious) if valid_ious else 0.0
    print(f"1. Spatial Overlap Audit (FP IoU): {avg_fp_iou:.4f}")
    print(f"   (Lower IoU = Distinct Failure Modes)")

    # 2. Uncertainty Calibration
    print("\n2. Uncertainty Calibration Audit (Mean Predictive Entropy):")
    print(f"   UNet FP Entropy: {np.mean(unet_fp_entropy):.4f} vs Swin FP Entropy: {np.mean(swin_fp_entropy):.4f}")
    print(f"   UNet FN Entropy: {np.mean(unet_fn_entropy):.4f} vs Swin FN Entropy: {np.mean(swin_fn_entropy):.4f}")
    print("   (Higher Entropy = Model is 'uncertain' about its error)")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)