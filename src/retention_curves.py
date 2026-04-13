"""
Build nDSC retention curve plot for one backbone (UNet or SwinUNETR).

Persists both:
  - Raw per-subject retention curves (nDSC_rc_all.npy) so the plot can be
    regenerated or re-aggregated offline without re-running inference.
  - The mean curve (nDSC_rc.npy) and the rendered PNG.
"""

import argparse
import os
import torch
from joblib import Parallel
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet, SwinUNETR
import numpy as np
from data_load import remove_connected_components, get_val_dataloader
from metrics import ndsc_retention_curve
from uncertainty import ensemble_uncertainties_classification
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
from sklearn import metrics

parser = argparse.ArgumentParser(description='Get all command line arguments.')

parser.add_argument('--num_models', type=int, default=3,
                    help='Number of models in ensemble')
parser.add_argument('--path_model', type=str, default='',
                    help='Specify the dir to al the trained models')

parser.add_argument('--path_data', type=str, required=True,
                    help='Specify the path to the directory with FLAIR images')
parser.add_argument('--path_gts', type=str, required=True,
                    help='Specify the path to the directory with ground truth binary masks')
parser.add_argument('--path_bm', type=str, required=True,
                    help='Specify the path to the directory with brain masks')

parser.add_argument('--num_workers', type=int, default=1,
                    help='Number of workers to preprocess images')
parser.add_argument('--n_jobs', type=int, default=1,
                    help='Number of parallel workers for F1 score computation')

parser.add_argument('--threshold', type=float, default=0.35,
                    help='Probability threshold')
parser.add_argument('--patch_size', type=int, default=96,
                    help='Cubic sliding-window patch size. Must match the '
                         'patch_size used at training time.')
parser.add_argument('--sw_batch_size', type=int, default=4,
                    help='Sliding-window batch size (patches per forward pass).')
parser.add_argument('--model_name', type=str, default='UNet',
                    choices=['UNet', 'SwinUNETR'],
                    help='Backbone to load from --path_model.')
parser.add_argument('--curve_label', type=str, default=None,
                    help='Label used on the plot legend and in the saved file '
                         'names. Defaults to model_name_p{patch_size}.')

parser.add_argument('--path_save', type=str, required=True,
                    help='Specify the path to the directory where retention '
                         'curves will be saved')


def get_default_device():
    """ Set device """
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def main(args):
    os.makedirs(args.path_save, exist_ok=True)
    device = get_default_device()
    torch.multiprocessing.set_sharing_strategy('file_system')

    '''' Initialise dataloaders '''
    val_loader = get_val_dataloader(flair_path=args.path_data,
                                    gts_path=args.path_gts,
                                    num_workers=args.num_workers,
                                    bm_path=args.path_bm)

    ''' Load trained models  '''
    K = args.num_models
    models = []
    for i in range(K):
        if args.model_name == 'SwinUNETR':
            models.append(SwinUNETR(
                img_size=(args.patch_size, args.patch_size, args.patch_size),
                in_channels=1,
                out_channels=2,
                feature_size=48,
                use_checkpoint=False,
                spatial_dims=3
            ).to(device))
        else:
            models.append(UNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=2,
                channels=(32, 64, 128, 256, 512),
                strides=(2, 2, 2, 2),
                num_res_units=0).to(device))

    for i, model in enumerate(models):
        model.load_state_dict(torch.load(os.path.join(args.path_model,
                                                      f"seed{i + 1}",
                                                      "Best_model_finetuning.pth"),
                                         map_location=device))
        model.eval()

    act = torch.nn.Softmax(dim=1)
    th = args.threshold
    roi_size = (args.patch_size, args.patch_size, args.patch_size)
    sw_batch_size = args.sw_batch_size
    curve_label = args.curve_label or f"{args.model_name}_p{args.patch_size}"

    fracs_retained = np.log(np.arange(200 + 1)[1:])
    fracs_retained /= np.amax(fracs_retained)

    ndsc_rc = []

    ''' Evaluatioin loop '''
    with Parallel(n_jobs=args.n_jobs) as parallel_backend:
        with torch.no_grad():
            for count, batch_data in enumerate(val_loader):
                inputs, gt, brain_mask = (
                    batch_data["image"].to(device),
                    batch_data["label"].cpu().numpy(),
                    batch_data["brain_mask"].cpu().numpy()
                )
                
                all_outputs = []
                for model in models:
                    outputs = sliding_window_inference(inputs, roi_size,
                                                       sw_batch_size, model,
                                                       mode='gaussian')
                    outputs = act(outputs).cpu().numpy()
                    outputs = np.squeeze(outputs[0, 1])
                    all_outputs.append(outputs)
                all_outputs = np.asarray(all_outputs)

                seg = np.mean(all_outputs, axis=0)
                seg[seg >= th] = 1
                seg[seg < th] = 0
                seg = np.squeeze(seg)
                seg = remove_connected_components(seg)

                gt = np.squeeze(gt)
                brain_mask = np.squeeze(brain_mask)

                uncs_map = ensemble_uncertainties_classification(np.concatenate(
                    (np.expand_dims(all_outputs, axis=-1),
                     np.expand_dims(1. - all_outputs, axis=-1)),
                    axis=-1))['reverse_mutual_information']

                ndsc_rc += [ndsc_retention_curve(ground_truth=gt[brain_mask == 1].flatten(),
                                                 predictions=seg[brain_mask == 1].flatten(),
                                                 uncertainties=uncs_map[brain_mask == 1].flatten(),
                                                 fracs_retained=fracs_retained,
                                                 parallel_backend=parallel_backend)]

    ndsc_rc = np.asarray(ndsc_rc)
    y = np.mean(ndsc_rc, axis=0)

    # Persist raw data so downstream plots can be regenerated without re-running
    # GPU inference. The per-subject matrix enables uncertainty bands / stddev
    # overlays if we want them later.
    np.save(os.path.join(args.path_save, f'nDSC_rc_{curve_label}.npy'), y)
    np.save(os.path.join(args.path_save, f'nDSC_rc_all_{curve_label}.npy'), ndsc_rc)
    np.save(os.path.join(args.path_save, 'fracs_retained.npy'), fracs_retained)

    r_auc = 1.0 - metrics.auc(fracs_retained, y)

    plt.figure(figsize=(6, 5))
    plt.plot(fracs_retained, y, label=f"{curve_label} (R-AUC: {r_auc:.4f})")
    plt.xlabel("Retention Fraction")
    plt.ylabel("nDSC")
    plt.xlim([0.0, 1.01])
    plt.legend()
    plt.title("nDSC Retention Curve")
    plt.savefig(os.path.join(args.path_save, f'nDSC_RC_{curve_label}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[retention_curves] {curve_label}: R-AUC={r_auc:.4f}")
    print(f"[retention_curves] Saved to {args.path_save}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
