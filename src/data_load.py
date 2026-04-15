"""
Contains implementations of transforms and dataloaders needed for training, validation and inference.
"""
import numpy as np
import os
from glob import glob
import re
from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    AddChanneld, Compose, LoadImaged, RandCropByPosNegLabeld,
    Spacingd, ToTensord, NormalizeIntensityd, RandFlipd,
    RandRotate90d, RandShiftIntensityd, RandAffined, RandSpatialCropd,
    RandScaleIntensityd, RandBiasFieldd, RandAdjustContrastd,
    RandHistogramShiftd, RandGaussianNoised, RandGaussianSmoothd,
    RandGibbsNoised)
from scipy import ndimage

AUG_PROFILES = {
    "full":            {"acquisition": True,  "intensity": True,  "geometric": True},
    "no_acquisition":  {"acquisition": False, "intensity": True,  "geometric": True},
    "no_geometric":    {"acquisition": True,  "intensity": True,  "geometric": False},
    "no_intensity":    {"acquisition": True,  "intensity": False, "geometric": True},
    "minimal":         {"acquisition": False, "intensity": False, "geometric": False},
}


def get_train_transforms(patch_size=96, aug_profile="full"):
    """ Get transforms for training on FLAIR images and ground truth.

    Args:
      patch_size: int, size of the cubic training patch (P, P, P). This
                  controls how much spatial context each training example
                  exposes.
      aug_profile: str, one of the keys in AUG_PROFILES. Controls which
                   augmentation groups are active.
    """
    if aug_profile not in AUG_PROFILES:
        raise ValueError(f"Unknown aug_profile={aug_profile!r}. "
                         f"Valid choices: {list(AUG_PROFILES.keys())}")
    flags = AUG_PROFILES[aug_profile]
    P = patch_size
    outer = P + 32  # lesion-biased crop size — slightly larger so later
                    # RandSpatialCropd / RandAffined have room to jitter.

    t = [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
    ]

    # --- Group A (pre-normalize): acquisition-shift simulators ---
    if flags["acquisition"]:
        t += [
            RandBiasFieldd(keys="image", coeff_range=(0.0, 0.1), degree=3, prob=0.3),
            RandAdjustContrastd(keys="image", gamma=(0.7, 1.5), prob=0.3),
            RandHistogramShiftd(keys="image", num_control_points=10, prob=0.2),
        ]

    t.append(NormalizeIntensityd(keys=["image"], nonzero=True))

    # --- Group B (post-normalize): always-on intensity jitter ---
    if flags["intensity"]:
        t += [
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        ]

    # --- Group A (post-normalize): noise/smooth/gibbs — still "acquisition" ---
    if flags["acquisition"]:
        t += [
            RandGaussianNoised(keys="image", mean=0.0, std=0.05, prob=0.2),
            RandGaussianSmoothd(keys="image",
                                sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0),
                                prob=0.2),
            RandGibbsNoised(keys="image", alpha=(0.0, 0.5), prob=0.2),
        ]

    # --- Cropping (unconditional — infrastructure, not augmentation) ---
    t += [
        RandCropByPosNegLabeld(keys=["image", "label"],
                               label_key="label", image_key="image",
                               spatial_size=(outer, outer, outer), num_samples=32,
                               pos=4, neg=1),
        RandSpatialCropd(keys=["image", "label"],
                         roi_size=(P, P, P),
                         random_center=True, random_size=False),
    ]

    # --- Group C: geometric augmentation ---
    if flags["geometric"]:
        t += [
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=(0, 1, 2)),
            RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 1)),
            RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(1, 2)),
            RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 2)),
            RandAffined(keys=['image', 'label'], mode=('bilinear', 'nearest'),
                        prob=1.0, spatial_size=(P, P, P),
                        rotate_range=(np.pi / 12, np.pi / 12, np.pi / 12),
                        scale_range=(0.1, 0.1, 0.1), padding_mode='border'),
        ]
    t.append(ToTensord(keys=["image", "label"]))
    return Compose(t)


def get_val_transforms(keys=["image", "label"], image_keys=["image"]):
    """ Get transforms for testing on FLAIR images and ground truth:
    - Loads 3D images and masks from Nifti file
    - Adds channel dimention
    - Applies intensity normalisation to scans
    - Converts to torch.Tensor()
    """
    return Compose(
        [
            LoadImaged(keys=keys),
            AddChanneld(keys=keys),
            NormalizeIntensityd(keys=image_keys, nonzero=True),
            ToTensord(keys=keys),
        ]
    )


def get_train_dataloader(flair_path, gts_path, num_workers, cache_rate=0.1,
                         patch_size=96, aug_profile="full"):
    """
    Get dataloader for training 
    Args:
      flair_path: `str`, path to directory with FLAIR images from Train set.
      gts_path:  `str`, path to directory with ground truth lesion segmentation 
                    binary masks images from Train set.
      num_workers:  `int`,  number of worker threads to use for parallel processing
                    of images
      cache_rate:  `float` in (0.0, 1.0], percentage of cached data in total.
    Returns:
      monai.data.DataLoader() class object.
    """
    flair = sorted(glob(os.path.join(flair_path, "*FLAIR_isovox.nii.gz")),
                   key=lambda i: int(re.sub('\D', '', i)))  # Collect all flair images sorted
    segs = sorted(glob(os.path.join(gts_path, "*gt_isovox.nii.gz")),
                  key=lambda i: int(re.sub('\D', '', i)))  # Collect all corresponding ground truths

    files = [{"image": fl, "label": seg} for fl, seg in zip(flair, segs)]

    print("Number of training files:", len(files))

    ds = CacheDataset(
        data=files,
        transform=get_train_transforms(patch_size=patch_size, aug_profile=aug_profile),
        cache_rate=cache_rate, num_workers=num_workers,
    )
    return DataLoader(ds, batch_size=1, shuffle=True,
                      num_workers=num_workers)


def get_val_dataloader(flair_path, gts_path, num_workers, cache_rate=0.1, bm_path=None):
    """
    Get dataloader for validation and testing. Either with or without brain masks.

    Args:
      flair_path: `str`, path to directory with FLAIR images.
      gts_path:  `str`, path to directory with ground truth lesion segmentation 
                    binary masks images.
      num_workers:  `int`,  number of worker threads to use for parallel processing
                    of images
      cache_rate:  `float` in (0.0, 1.0], percentage of cached data in total.
      bm_path:   `None|str`. If `str`, then defines path to directory with
                 brain masks. If `None`, dataloader does not return brain masks. 
    Returns:
      monai.data.DataLoader() class object.
    """
    flair = sorted(glob(os.path.join(flair_path, "*FLAIR_isovox.nii.gz")),
                   key=lambda i: int(re.sub('\D', '', i)))  # Collect all flair images sorted
    segs = sorted(glob(os.path.join(gts_path, "*_isovox.nii.gz")),
                  key=lambda i: int(re.sub('\D', '', i)))  # Collect all corresponding ground truths

    if bm_path is not None:
        bms = sorted(glob(os.path.join(bm_path, "*isovox_fg_mask.nii.gz")),
                     key=lambda i: int(re.sub('\D', '', i)))  # Collect all corresponding brain masks

        assert len(flair) == len(segs) == len(bms), f"Some files must be missing: {[len(flair), len(segs), len(bms)]}"

        files = [
            {"image": fl, "label": seg, "brain_mask": bm} for fl, seg, bm
            in zip(flair, segs, bms)
        ]

        val_transforms = get_val_transforms(keys=["image", "label", "brain_mask"])
    else:
        assert len(flair) == len(segs), f"Some files must be missing: {[len(flair), len(segs)]}"

        files = [{"image": fl, "label": seg} for fl, seg in zip(flair, segs)]

        val_transforms = get_val_transforms()

    print("Number of validation files:", len(files))

    ds = CacheDataset(data=files, transform=val_transforms,
                      cache_rate=cache_rate, num_workers=num_workers)
    return DataLoader(ds, batch_size=1, shuffle=False,
                      num_workers=num_workers)


def get_flair_dataloader(flair_path, num_workers, cache_rate=0.1, bm_path=None):
    """
    Get dataloader with FLAIR images only for inference
    
    Args:
      flair_path: `str`, path to directory with FLAIR images from Train set.
      num_workers:  `int`,  number of worker threads to use for parallel processing
                    of images
      cache_rate:  `float` in (0.0, 1.0], percentage of cached data in total.
      bm_path:   `None|str`. If `str`, then defines path to directory with
                 brain masks. If `None`, dataloader does not return brain masks.
    Returns:
      monai.data.DataLoader() class object.
    """
    flair = sorted(glob(os.path.join(flair_path, "*FLAIR_isovox.nii.gz")),
                   key=lambda i: int(re.sub('\D', '', i)))  # Collect all flair images sorted

    if bm_path is not None:
        bms = sorted(glob(os.path.join(bm_path, "*isovox_fg_mask.nii.gz")),
                     key=lambda i: int(re.sub('\D', '', i)))  # Collect all corresponding brain masks

        assert len(flair) == len(bms), f"Some files must be missing: {[len(flair), len(bms)]}"

        files = [{"image": fl, "brain_mask": bm} for fl, bm in zip(flair, bms)]

        val_transforms = get_val_transforms(keys=["image", "brain_mask"])
    else:
        files = [{"image": fl} for fl in flair]

        val_transforms = get_val_transforms(keys=["image"])

    print("Number of FLAIR files:", len(files))

    ds = CacheDataset(data=files, transform=val_transforms,
                      cache_rate=cache_rate, num_workers=num_workers)
    return DataLoader(ds, batch_size=1, shuffle=False,
                      num_workers=num_workers)


def remove_connected_components(segmentation, l_min=9):
    """
    Remove all lesions with less or equal amount of voxels than `l_min` from a 
    binary segmentation mask `segmentation`.
    Args:
      segmentation: `numpy.ndarray` of shape [H, W, D], with a binary lesions segmentation mask.
      l_min:  `int`, minimal amount of voxels in a lesion.
    Returns:
      Binary lesion segmentation mask (`numpy.ndarray` of shape [H, W, D])
      only with connected components that have more than `l_min` voxels.
    """
    labeled_seg, num_labels = ndimage.label(segmentation)
    label_list = np.unique(labeled_seg)
    num_elements_by_lesion = ndimage.labeled_comprehension(segmentation, labeled_seg, label_list, np.sum, float, 0)

    seg2 = np.zeros_like(segmentation)
    for i_el, n_el in enumerate(num_elements_by_lesion):
        if n_el > l_min:
            current_voxels = np.stack(np.where(labeled_seg == i_el), axis=1)
            seg2[current_voxels[:, 0],
                 current_voxels[:, 1],
                 current_voxels[:, 2]] = 1
    return seg2
