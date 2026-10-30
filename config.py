"""
config.py
---------
Central place for every tunable setting in the project.

All three solutions read their defaults from here, so the experiments stay
comparable (same image size, same train/test split, same random seed).
Most values can also be overridden from the command line of each run script.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

# Folder with ONE SUB-FOLDER PER CLASS, e.g.
#   data/tomato/Tomato___healthy/*.jpg
#   data/tomato/Tomato___Early_blight/*.jpg
DATA_DIR = PROJECT_ROOT / "data" / "tomato"

# Optional second dataset of real field photos (same class folder names),
# used only for the robustness experiment (e.g. PlantDoc).
FIELD_DIR = PROJECT_ROOT / "data" / "tomato_field"

RESULTS_DIR = PROJECT_ROOT / "results"
CACHE_DIR = RESULTS_DIR / "cache"

# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
IMG_SIZE = 128                # every image is resized to IMG_SIZE x IMG_SIZE
MAX_IMAGES_PER_CLASS = 400    # cap per class so the experiments run on a laptop
TEST_SIZE = 0.2               # 80 / 20 stratified train-test split
RANDOM_STATE = 42             # one seed for every random step
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

# --------------------------------------------------------------------------
# Solution 1 : K-means segmentation + Bayes classifier
# --------------------------------------------------------------------------
KMEANS_CLUSTERS = 3           # colour clusters inside the leaf (healthy / lesion / other)
KMEANS_MAX_PIXELS = 4000      # pixels sampled per image to fit k-means (speed)
BAYES_REG = 1e-3              # covariance regularisation (keeps matrices invertible)
MISS_COST = 5.0               # cost of calling a diseased leaf "healthy" (minimum-risk rule)

# --------------------------------------------------------------------------
# Solution 2 : Fisher score + PCA / LDA + kNN / SVM
# --------------------------------------------------------------------------
TOP_K_FEATURES = 60           # features kept after Fisher-score ranking
PCA_VARIANCE = 0.95           # keep enough components to explain 95 % variance
KNN_NEIGHBOURS = 5

# --------------------------------------------------------------------------
# Solution 3 : Neural networks
# --------------------------------------------------------------------------
MLP_HIDDEN = (128, 64)        # hidden layer sizes of the from-scratch MLP
MLP_LR = 0.01
MLP_EPOCHS = 150
MLP_BATCH = 32
MLP_L2 = 1e-4
MLP_MOMENTUM = 0.9
MLP_PATIENCE = 20             # early stopping patience (epochs)

CNN_EPOCHS = 30               # optional small CNN (needs Keras); early stopping usually ends it sooner
CNN_BATCH = 32
CNN_LR = 1e-3

# --------------------------------------------------------------------------
# Robustness experiment : perturbations applied to the test images
# --------------------------------------------------------------------------
PERTURBATIONS = {
    "gaussian_noise": 0.08,   # std of noise on [0, 1] images
    "blur": 2.0,              # gaussian blur sigma (pixels)
    "dark": 0.55,             # brightness multiplier (< 1 = darker)
    "bright": 1.5,            # brightness multiplier (> 1 = brighter)
}
