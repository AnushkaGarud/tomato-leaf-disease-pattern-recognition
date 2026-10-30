"""
common/robustness.py
--------------------
Simulating poor photo conditions for the robustness experiment.

PlantVillage photos are taken in a lab: one leaf, plain background, even
lighting. Real farm photos are noisy, blurry and badly lit. To measure how
each solution copes, the TEST images are degraded in four ways and the
already-trained model is evaluated again (it never sees degraded images
during training).

If a real field dataset is available (e.g. PlantDoc, with the same class
folder names) the run scripts can also evaluate on it via --field-dir.
"""

import numpy as np
from scipy import ndimage as ndi

import config


def add_gaussian_noise(images, std, seed=config.RANDOM_STATE):
    """Sensor noise: add N(0, std) to every pixel of [0, 1]-scaled images."""
    rng = np.random.default_rng(seed)
    x = images.astype(float) / 255.0
    x = x + rng.normal(0.0, std, size=x.shape)
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def gaussian_blur(images, sigma):
    """Out-of-focus / motion: blur each colour channel spatially."""
    return np.stack([
        ndi.gaussian_filter(img.astype(float), sigma=(sigma, sigma, 0)).clip(0, 255).astype(np.uint8)
        for img in images
    ])


def change_brightness(images, factor):
    """Under- (factor < 1) or over-exposure (factor > 1)."""
    return np.clip(images.astype(float) * factor, 0, 255).astype(np.uint8)


def perturb(images, kind, strength):
    """Dispatch to one of the perturbations by name."""
    if kind == "gaussian_noise":
        return add_gaussian_noise(images, strength)
    if kind == "blur":
        return gaussian_blur(images, strength)
    if kind in ("dark", "bright"):
        return change_brightness(images, strength)
    raise ValueError(f"Unknown perturbation: {kind}")


def all_perturbations(images):
    """Yield (name, degraded_images) for every perturbation in config."""
    for kind, strength in config.PERTURBATIONS.items():
        yield kind, perturb(images, kind, strength)
