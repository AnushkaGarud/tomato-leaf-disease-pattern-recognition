"""
common/segmentation.py
----------------------
Two-stage segmentation of a leaf image.

Stage 1 - leaf vs. background
    Leaves are strongly coloured, backgrounds (lab table, paper, sky) are
    mostly grey. So we threshold the HSV *saturation* channel with Otsu's
    method, clean the mask with morphology and keep the largest blob.

Stage 2 - healthy tissue vs. lesions (UNSUPERVISED, k-means)
    Inside the leaf mask every pixel is described by its colour in the
    CIE-Lab space (a* = green<->red, b* = blue<->yellow). K-means groups the
    pixels into K colour clusters. The cluster with the most negative a*
    (i.e. the greenest) is healthy tissue; everything else is treated as
    diseased tissue (brown spots, yellow halos, mould).

No labels are used anywhere in this file - it is pure clustering.
"""

import numpy as np
from scipy import ndimage as ndi
from skimage import color, filters, morphology
from sklearn.cluster import KMeans

import config


def segment_leaf(image):
    """Return a boolean mask that is True on leaf pixels.

    Parameters
    ----------
    image : np.ndarray (H, W, 3) uint8 RGB
    """
    hsv = color.rgb2hsv(image)
    saturation, value = hsv[..., 1], hsv[..., 2]

    # Otsu picks the threshold that best separates two intensity groups
    try:
        t = filters.threshold_otsu(saturation)
    except ValueError:            # completely flat image
        return np.ones(image.shape[:2], dtype=bool)
    mask = (saturation > t) & (value > 0.08)   # also drop near-black shadows

    # Morphological clean-up: remove specks, fill holes left by lesions
    mask = ndi.binary_opening(mask, structure=morphology.disk(2))
    mask = ndi.binary_fill_holes(mask)

    # Keep only the largest connected component (the leaf itself)
    labelled, n = ndi.label(mask)
    if n == 0:
        return np.ones(image.shape[:2], dtype=bool)   # fall back to whole image
    sizes = ndi.sum(mask, labelled, range(1, n + 1))
    mask = labelled == (np.argmax(sizes) + 1)

    # Sanity check: if the "leaf" is tiny the threshold failed, use everything
    if mask.mean() < 0.05:
        return np.ones(image.shape[:2], dtype=bool)
    return mask


def segment_lesions(image, leaf_mask=None, k=config.KMEANS_CLUSTERS,
                    max_pixels=config.KMEANS_MAX_PIXELS, seed=config.RANDOM_STATE):
    """Cluster leaf pixels by colour and mark the non-green clusters as lesions.

    Returns
    -------
    lesion_mask : bool array (H, W)   True on diseased pixels
    cluster_map : int array (H, W)    cluster index per pixel, -1 = background
    healthy_cluster : int             index of the cluster judged healthy
    """
    if leaf_mask is None:
        leaf_mask = segment_leaf(image)

    # Ignore a 2-pixel rim: edge pixels blend leaf and background colours and
    # would otherwise be mistaken for a thin lesion around the leaf outline.
    inner = ndi.binary_erosion(leaf_mask, iterations=2)
    if inner.sum() > 100:
        leaf_mask = inner

    lab = color.rgb2lab(image)
    pixels = lab[leaf_mask]                   # (n_leaf_pixels, 3) : L, a, b
    # Use a* and b* (colour) plus a down-weighted L (brightness) so that
    # dark necrotic spots and light healthy tissue still separate a little.
    feats = np.column_stack([pixels[:, 0] * 0.5, pixels[:, 1], pixels[:, 2]])

    cluster_map = np.full(image.shape[:2], -1, dtype=int)
    if len(feats) < k * 10:                   # leaf too small to cluster
        cluster_map[leaf_mask] = 0
        return np.zeros(image.shape[:2], dtype=bool), cluster_map, 0

    # Fit k-means on a random sample of pixels (fast), then label all pixels
    rng = np.random.default_rng(seed)
    sample = feats if len(feats) <= max_pixels else feats[rng.choice(len(feats), max_pixels, replace=False)]
    km = KMeans(n_clusters=k, n_init=3, random_state=seed).fit(sample)
    labels = km.predict(feats)
    cluster_map[leaf_mask] = labels

    # Healthy = greenest centre (most negative a*). Column 1 is a*.
    healthy_cluster = int(np.argmin(km.cluster_centers_[:, 1]))
    healthy_a = km.cluster_centers_[healthy_cluster, 1]

    # A cluster counts as lesion only if it is clearly less green than the
    # healthy one. This stops a healthy leaf (two shades of green) from being
    # labelled "diseased" just because k-means is forced to make K clusters.
    lesion_clusters = [c for c in range(k)
                       if c != healthy_cluster and km.cluster_centers_[c, 1] - healthy_a > 8.0]
    lesion_mask = np.isin(cluster_map, lesion_clusters) & leaf_mask
    return lesion_mask, cluster_map, healthy_cluster
