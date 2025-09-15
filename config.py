# Global defaults you can tweak in one place.

# Raw images folder (your original dataset)
RAW_DIR = "/Users/andreaprocopio/Desktop/models_experiments/segformer_polygons/Exp3b_Images"

# Output dataset folder (built as /train and /val, with images+GT masks)
DATA_DIR = "data"

# Model & training
MODEL_ID = "nvidia/segformer-b0-finetuned-ade-512-512"  # backbone; head is reset to 2 classes
NUM_LABELS = 2  # background=0, polygon=1
IMAGE_H, IMAGE_W = 400, 1000  # fixed, no rescaling

# Dataset split
VAL_FRACTION = 0.2
RNG_SEED = 42

# Thresholding (for GT)
THRESHOLDS = (12,)  # modified from (15, 10, 5)

# Training hyperparams (defaults; can be overridden via CLI)
LR = 6e-5
EPOCHS = 30
BATCH_TRAIN = 1
BATCH_EVAL = 1 
WEIGHT_DECAY = 0.01
LOG_STEPS = 50
EVAL_STEPS = 500
SAVE_STEPS = 2000

# Overlay export (evaluate.py)
OVERLAY_OUT = "overlays"
