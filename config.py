"""Central configuration for mouse-accuracy-bot."""
import torch
import numpy as np

# --- Screen Capture ---
# Set this to your game window region: (left, top, width, height) in LOGICAL pixels
# Run `python capture.py` to calibrate
# Chrome fullscreen on 1512x982 logical display
# Captures from top of browser (includes Chrome UI + game header — labeler masks these out)
GAME_REGION = (0, 80, 1512, 861)

# Retina scale factor: mss returns logical pixels on this Mac (1512x982),
# and PyAutoGUI also uses logical pixels, so scale factor is 1.0
SCALE_FACTOR = 1.0

# --- Model Input/Output ---
INPUT_SIZE = (320, 180)      # (width, height) model input resolution
HEATMAP_SIZE = (80, 45)      # (width, height) model output resolution (INPUT_SIZE / 4)

# --- HSV Color Thresholds for Red Target Detection ---
# Red wraps around H=0/180 in HSV, so we need two ranges
HSV_RED_LOWER_1 = np.array([0, 80, 80])
HSV_RED_UPPER_1 = np.array([12, 255, 255])
HSV_RED_LOWER_2 = np.array([165, 80, 80])
HSV_RED_UPPER_2 = np.array([180, 255, 255])

# Minimum contour area (in pixels at capture resolution) to count as a target
MIN_CONTOUR_AREA = 50

# Minimum circularity ratio (area / enclosing_circle_area) to filter non-circles
MIN_CIRCULARITY = 0.55

# --- Heatmap Generation ---
GAUSSIAN_SIGMA = 3.0  # Sigma for Gaussian blobs in heatmap

# --- Data Collection ---
CAPTURE_FPS = 10  # Frames per second during collection

# --- Game Window ---
# From end-of-game screen: Resolution 1512 x 861 logical pixels
GAME_RESOLUTION = (1512, 861)

# --- Gameplay ---
CLICK_DELAY = 0.05           # Seconds between clicks
CONFIDENCE_THRESHOLD = 0.5   # Heatmap peak threshold for detection

# --- Target Selection Strategy ---
# Options: "confidence" (highest model score), "nearest" (closest to cursor), "largest"
TARGET_STRATEGY = "confidence"

# --- Reward Function Weights ---
REWARD_SCORE_WEIGHT = 1.0       # Weight for raw score
REWARD_ACCURACY_WEIGHT = 50.0   # Weight for accuracy % (penalize misclicks)
REWARD_EFFICIENCY_WEIGHT = 50.0 # Weight for efficiency % (penalize missed targets)
REWARD_MISS_PENALTY = 2.0       # Per-miss penalty

# --- Training ---
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
TRAIN_SPLIT = 0.8

# --- Device ---
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# --- HUD Exclusion ---
# Fraction of the right side of the frame to mask out (score/timer area)
HUD_RIGHT_FRACTION = 0.12
