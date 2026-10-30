"""
common/dataset.py
-----------------
Loading leaf images from disk and splitting them into train / test sets.

Expected folder layout (one sub-folder per class, folder name = class name):

    data/tomato/
        Tomato___healthy/        img001.jpg, img002.jpg, ...
        Tomato___Early_blight/   ...
        Tomato___Late_blight/    ...

This is exactly how the PlantVillage dataset is distributed, so the tomato
folders can be copied in without renaming.
"""

from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

import config


def list_class_folders(data_dir, class_prefix=None):
    """Return the sorted list of class sub-folders inside `data_dir`.

    Parameters
    ----------
    data_dir : str or Path
        Root folder that contains one folder per class.
    class_prefix : str, optional
        Keep only folders whose name starts with this text (e.g. "Tomato").
        Handy if the full PlantVillage folder (all crops) is used directly.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data folder not found: {data_dir}\n"
            "Download the dataset (see README) or create a test set with:\n"
            "    python scripts/make_synthetic_data.py"
        )
    folders = sorted(p for p in data_dir.iterdir() if p.is_dir())
    if class_prefix:
        folders = [p for p in folders if p.name.lower().startswith(class_prefix.lower())]
    if len(folders) < 2:
        raise ValueError(f"Need at least 2 class folders in {data_dir}, found {len(folders)}.")
    return folders


def pretty_class_name(folder_name):
    """Turn 'Tomato___Early_blight' into 'Early blight' for nicer plot labels."""
    name = folder_name.split("___")[-1]
    return name.replace("_", " ").strip()


def load_image(path, img_size=config.IMG_SIZE):
    """Read one image as an RGB uint8 array of shape (img_size, img_size, 3)."""
    with Image.open(path) as img:
        img = img.convert("RGB").resize((img_size, img_size), Image.BILINEAR)
        return np.asarray(img, dtype=np.uint8)


def load_dataset(data_dir=config.DATA_DIR,
                 img_size=config.IMG_SIZE,
                 max_per_class=config.MAX_IMAGES_PER_CLASS,
                 class_prefix=None,
                 class_names=None,
                 seed=config.RANDOM_STATE,
                 verbose=True):
    """Load every class folder into memory.

    Parameters
    ----------
    class_names : list of str, optional
        If given (e.g. the class list of the training set), the folders are
        matched to these names in this order. Used for the field dataset so
        that label i means the same disease in both datasets.

    Returns
    -------
    images : np.ndarray, shape (N, H, W, 3), uint8
    labels : np.ndarray, shape (N,), int
    class_names : list of str   (folder names, index = label)
    """
    rng = np.random.default_rng(seed)
    folders = list_class_folders(data_dir, class_prefix)

    if class_names is not None:
        by_name = {f.name: f for f in folders}
        missing = [c for c in class_names if c not in by_name]
        if missing and verbose:
            print(f"[dataset] warning: classes missing in {data_dir}: {missing}")
        folders = [by_name[c] for c in class_names if c in by_name]
        label_of = {c: i for i, c in enumerate(class_names)}
    else:
        class_names = [f.name for f in folders]
        label_of = {c: i for i, c in enumerate(class_names)}

    images, labels = [], []
    for folder in folders:
        files = sorted(p for p in folder.iterdir()
                       if p.suffix.lower() in config.IMAGE_EXTENSIONS)
        # Random (but reproducible) subset so big classes don't dominate
        if max_per_class and len(files) > max_per_class:
            idx = rng.choice(len(files), size=max_per_class, replace=False)
            files = [files[i] for i in sorted(idx)]
        for f in files:
            try:
                images.append(load_image(f, img_size))
                labels.append(label_of[folder.name])
            except OSError:
                if verbose:
                    print(f"[dataset] skipping unreadable file {f}")
        if verbose:
            print(f"[dataset] {folder.name:<45s} {len(files):5d} images")

    images = np.stack(images)
    labels = np.asarray(labels, dtype=int)
    if verbose:
        print(f"[dataset] total: {len(images)} images, {len(class_names)} classes")
    return images, labels, class_names


def split_dataset(images, labels, test_size=config.TEST_SIZE, seed=config.RANDOM_STATE):
    """Stratified train / test split (same class proportions in both parts).

    Returns the indices as well, so feature matrices computed once for the
    whole dataset can be split the same way.
    """
    idx = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, stratify=labels, random_state=seed)
    return train_idx, test_idx


def find_healthy_class(class_names):
    """Index of the 'healthy' class, or None if there isn't one."""
    for i, name in enumerate(class_names):
        if "healthy" in name.lower():
            return i
    return None
