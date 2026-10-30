"""
common/features.py
------------------
Turning a leaf image into a FEATURE VECTOR (a point in pattern space).

Two feature sets are produced:

1. Lesion features  (used by Solution 1)
   Computed from the k-means segmentation: how much of the leaf is diseased,
   how many spots there are, how big they are and what colour they are.
   ~24 numbers, each with a clear physical meaning.

2. Handcrafted global features  (used by Solutions 2 and 3)
   Describe the whole leaf without segmenting lesions:
     * colour moments  - mean / std / skewness of HSV and Lab channels  (18)
     * colour histogram - hue, saturation, value histograms            (32)
     * GLCM texture    - Haralick-style co-occurrence statistics        (12)
     * LBP texture     - local binary pattern histograms                (28)
   90 numbers in total.

Every function takes the image plus a boolean leaf mask so that the
background never contaminates the features.
"""

import time

import numpy as np
from scipy import ndimage as ndi
from scipy.stats import skew
from skimage import color
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from common.segmentation import segment_leaf, segment_lesions

# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------

def color_moments(image, mask):
    """Mean, standard deviation and skewness of each HSV and Lab channel (18 values).

    Colour moments are a compact summary of the colour distribution:
    a yellowing leaf shifts the hue mean, mottling raises the std, etc.
    """
    hsv = color.rgb2hsv(image)[mask]
    lab = color.rgb2lab(image)[mask]
    lab = lab / np.array([100.0, 128.0, 128.0])   # bring Lab to ~[-1, 1]
    feats = []
    for channels in (hsv, lab):
        feats.extend(channels.mean(axis=0))
        feats.extend(channels.std(axis=0))
        feats.extend(np.nan_to_num(skew(channels, axis=0)))
    return np.asarray(feats)


def color_histogram(image, mask, hue_bins=16, sv_bins=8):
    """Normalised histograms of hue (16 bins), saturation (8) and value (8)."""
    hsv = color.rgb2hsv(image)[mask]
    h = np.histogram(hsv[:, 0], bins=hue_bins, range=(0, 1))[0]
    s = np.histogram(hsv[:, 1], bins=sv_bins, range=(0, 1))[0]
    v = np.histogram(hsv[:, 2], bins=sv_bins, range=(0, 1))[0]
    hist = np.concatenate([h, s, v]).astype(float)
    return hist / max(mask.sum(), 1)


def glcm_features(gray, mask, levels=32, distances=(1, 3)):
    """Grey-Level Co-occurrence Matrix texture statistics (12 values).

    The GLCM counts how often grey level i sits next to grey level j.
    From it we compute contrast, dissimilarity, homogeneity, energy,
    correlation and ASM, averaged over 4 directions, for 2 distances.

    Background pixels are mapped to level 0 and that row/column of the
    matrix is dropped, so only leaf-to-leaf pixel pairs are counted.
    """
    q = (gray * (levels - 2)).astype(np.uint8) + 1      # leaf -> 1 .. levels-1
    q[~mask] = 0                                         # background -> 0
    glcm = graycomatrix(q, distances=list(distances),
                        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=levels, symmetric=True, normed=False)
    glcm = glcm[1:, 1:, :, :].astype(float)              # drop background level
    glcm /= np.maximum(glcm.sum(axis=(0, 1), keepdims=True), 1)
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    feats = []
    for p in props:
        vals = graycoprops(glcm, p)                      # (n_distances, n_angles)
        feats.extend(np.nan_to_num(vals.mean(axis=1)))   # average over angles
    return np.asarray(feats)


def lbp_histogram(gray, mask):
    """Uniform Local Binary Pattern histograms at two scales (10 + 18 values).

    LBP compares each pixel with its neighbours on a circle and encodes the
    result as a binary number; the histogram of codes describes micro-texture
    (smooth healthy tissue vs. rough, spotted lesions).
    """
    g = (gray * 255).astype(np.uint8)
    feats = []
    for P, R in ((8, 1), (16, 2)):
        lbp = local_binary_pattern(g, P, R, method="uniform")
        hist = np.histogram(lbp[mask], bins=P + 2, range=(0, P + 2))[0].astype(float)
        feats.extend(hist / max(mask.sum(), 1))
    return np.asarray(feats)


# --------------------------------------------------------------------------
# Feature set 1 : lesion features (Solution 1)
# --------------------------------------------------------------------------

LESION_FEATURE_NAMES = (
    ["lesion_area_ratio", "n_spots", "mean_spot_area", "largest_spot_ratio", "spot_area_std"]
    + [f"lesion_{s}_{c}" for s in ("mean", "std") for c in ("L", "a", "b")]
    + [f"healthy_mean_{c}" for c in ("L", "a", "b")]
    + [f"lesion_minus_healthy_{c}" for c in ("L", "a", "b")]
    + ["leaf_mean_hue", "leaf_std_hue", "leaf_mean_sat", "yellow_ratio", "brown_ratio", "dark_ratio"]
)


def lesion_features(image, return_masks=False):
    """Segment the leaf, run k-means, and describe the diseased regions.

    Returns a vector of len(LESION_FEATURE_NAMES) values (and optionally the
    masks, which are used for the segmentation figures).
    """
    leaf_mask = segment_leaf(image)
    lesion_mask, cluster_map, _ = segment_lesions(image, leaf_mask)

    leaf_px = max(leaf_mask.sum(), 1)
    lab = color.rgb2lab(image)
    hsv = color.rgb2hsv(image)

    # --- spot geometry (connected components of the lesion mask) -----------
    labelled, n_spots = ndi.label(lesion_mask)
    if n_spots > 0:
        areas = ndi.sum(lesion_mask, labelled, range(1, n_spots + 1)) / leaf_px
        # ignore 1-2 pixel specks when counting spots
        real = areas[areas > 5.0 / leaf_px]
        n_real = len(real)
        mean_area = real.mean() if n_real else 0.0
        largest = areas.max()
        area_std = real.std() if n_real > 1 else 0.0
    else:
        n_real, mean_area, largest, area_std = 0, 0.0, 0.0, 0.0
    geometry = [lesion_mask.sum() / leaf_px, np.log1p(n_real), mean_area, largest, area_std]

    # --- colour of lesions vs. healthy tissue ------------------------------
    healthy_mask = leaf_mask & ~lesion_mask
    scale = np.array([100.0, 128.0, 128.0])
    les = lab[lesion_mask] / scale if lesion_mask.any() else np.zeros((1, 3))
    hea = lab[healthy_mask] / scale if healthy_mask.any() else np.zeros((1, 3))
    les_mean, les_std, hea_mean = les.mean(0), les.std(0), hea.mean(0)
    colour = list(les_mean) + list(les_std) + list(hea_mean) + list(les_mean - hea_mean)

    # --- simple colour-category ratios over the whole leaf -----------------
    h, s, v = hsv[..., 0][leaf_mask], hsv[..., 1][leaf_mask], hsv[..., 2][leaf_mask]
    yellow = ((h > 0.11) & (h < 0.19) & (s > 0.3)).mean()     # yellow halos / chlorosis
    brown = ((h < 0.11) & (s > 0.25) & (v < 0.75)).mean()     # necrotic brown spots
    dark = (v < 0.25).mean()                                  # black mould / dead tissue
    global_c = [h.mean(), h.std(), s.mean(), yellow, brown, dark]

    feats = np.asarray(geometry + colour + global_c, dtype=float)
    if return_masks:
        return feats, leaf_mask, lesion_mask, cluster_map
    return feats


# --------------------------------------------------------------------------
# Feature set 2 : handcrafted global features (Solutions 2 and 3)
# --------------------------------------------------------------------------

HANDCRAFTED_FEATURE_NAMES = (
    [f"{sp}_{st}_{c}" for sp, chans in (("hsv", "HSV"), ("lab", "Lab"))
     for st in ("mean", "std", "skew") for c in chans]
    + [f"hue_hist_{i}" for i in range(16)]
    + [f"sat_hist_{i}" for i in range(8)]
    + [f"val_hist_{i}" for i in range(8)]
    + [f"glcm_{p}_d{d}" for p in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")
       for d in (1, 3)]
    + [f"lbp_p8_{i}" for i in range(10)]
    + [f"lbp_p16_{i}" for i in range(18)]
)


def handcrafted_features(image):
    """Colour + texture description of the whole leaf (90 values)."""
    mask = segment_leaf(image)
    gray = color.rgb2gray(image)
    return np.concatenate([
        color_moments(image, mask),
        color_histogram(image, mask),
        glcm_features(gray, mask),
        lbp_histogram(gray, mask),
    ])


# --------------------------------------------------------------------------
# Batch helper
# --------------------------------------------------------------------------

def extract_batch(images, feature_fn, desc="features", verbose=True):
    """Apply `feature_fn` to every image and stack the results into a matrix.

    Prints a simple progress line; returns (X, seconds_per_image).
    """
    start = time.time()
    rows = []
    n = len(images)
    for i, img in enumerate(images):
        rows.append(feature_fn(img))
        if verbose and (i + 1) % max(1, n // 10) == 0:
            print(f"  [{desc}] {i + 1}/{n} images")
    X = np.vstack(rows)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, (time.time() - start) / max(n, 1)
