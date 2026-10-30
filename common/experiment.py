"""
common/experiment.py
--------------------
Boilerplate shared by the three run scripts:

* the common command-line options (data folder, image size, ...)
* loading the dataset and making the SAME train/test split for everyone
* caching feature matrices on disk, so re-running a solution (or running
  Solution 3 after Solution 2, which uses the same features) is instant.
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np

import config
from common.dataset import load_dataset, split_dataset
from common.features import extract_batch


def base_arg_parser(description):
    """Argument parser with the options every solution understands."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--data-dir", default=str(config.DATA_DIR),
                   help="folder with one sub-folder per class (default: %(default)s)")
    p.add_argument("--class-prefix", default=None,
                   help="only use class folders starting with this text, e.g. Tomato")
    p.add_argument("--img-size", type=int, default=config.IMG_SIZE)
    p.add_argument("--max-per-class", type=int, default=config.MAX_IMAGES_PER_CLASS,
                   help="cap on images per class (0 = use all)")
    p.add_argument("--results-dir", default=str(config.RESULTS_DIR))
    p.add_argument("--robustness", action="store_true",
                   help="also evaluate on noisy / blurred / dark / bright test images")
    p.add_argument("--field-dir", default=None,
                   help="optional field-photo dataset with the same class folders")
    p.add_argument("--no-cache", action="store_true", help="recompute features even if cached")
    return p


def load_and_split(args):
    """Load images and return everything a solution needs."""
    images, labels, class_names = load_dataset(
        args.data_dir, args.img_size, args.max_per_class, args.class_prefix)
    train_idx, test_idx = split_dataset(images, labels)
    print(f"[split] train={len(train_idx)}  test={len(test_idx)}")
    return images, labels, class_names, train_idx, test_idx


def cached_features(images, feature_fn, tag, args):
    """Compute (or load from cache) the feature matrix for `images`.

    The cache file name includes a hash of the data folder and settings, so
    changing the dataset or image size never reuses stale features.
    """
    key = f"{Path(args.data_dir).resolve()}|{args.class_prefix}|{args.img_size}|{args.max_per_class}|{len(images)}"
    h = hashlib.md5(key.encode()).hexdigest()[:10]
    path = Path(args.results_dir) / "cache" / f"{tag}_{h}.npz"

    if path.exists() and not args.no_cache:
        data = np.load(path)
        print(f"[cache] loaded {tag} features from {path.name}")
        return data["X"], float(data["sec_per_img"])

    print(f"[features] extracting {tag} features for {len(images)} images ...")
    X, sec = extract_batch(images, feature_fn, desc=tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=X, sec_per_img=sec)
    print(f"[features] done: {X.shape[1]} features, {sec * 1000:.1f} ms / image")
    return X, sec


def load_field_set(args, class_names):
    """Load the optional field dataset with labels aligned to the training classes."""
    if not args.field_dir or not Path(args.field_dir).exists():
        return None
    images, labels, _ = load_dataset(args.field_dir, args.img_size, 0,
                                     class_names=class_names)
    return images, labels
