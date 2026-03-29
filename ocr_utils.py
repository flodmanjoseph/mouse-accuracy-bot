"""OCR utilities for reading game metrics from the end-of-game results screen.

Reads: Score, Accuracy %, Efficiency %, Hits, Misses, Targets count.
Uses template matching on the known layout rather than full OCR.
"""
import re
import cv2
import numpy as np
from capture import grab_screen
from config import GAME_REGION


def extract_text_region(frame_bgr, brightness_threshold=180):
    """Extract bright text from a dark background using thresholding.

    The results screen has light text on a dark card. We threshold to
    isolate the text, then use contour analysis to find text regions.

    Returns:
        Binary mask where white = text pixels.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, brightness_threshold, 255, cv2.THRESH_BINARY)
    return binary


def find_results_card(frame_bgr):
    """Locate the results card popup in the frame.

    The results card is a dark rectangle with rounded corners that appears
    centered on screen after the game ends. It contains bright text.

    Returns:
        Cropped BGR image of just the results card, or None if not found.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # The results card is a dark region (~30-60 brightness) against
    # an even darker background (~10-25). Look for the card boundary.
    # The text inside is bright (>180).
    # Strategy: threshold for text, find bounding box of text cluster.
    _, text_mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Find contours of text regions
    contours, _ = cv2.findContours(text_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Get bounding box that encompasses all text
    all_points = np.vstack(contours)
    x, y, w, h = cv2.boundingRect(all_points)

    # Add padding around the text region
    pad = 20
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(frame_bgr.shape[1] - x, w + 2 * pad)
    h = min(frame_bgr.shape[0] - y, h + 2 * pad)

    return frame_bgr[y:y+h, x:x+w]


def parse_metrics_from_screen(frame_bgr=None, region=None):
    """Capture the end-of-game screen and parse metrics.

    Expected layout (from the game):
        Score: <number>
        Accuracy: <number>%
        Efficiency: <number>%
        Hits/Misses: <h>/<m>
        Targets: <number>

    Args:
        frame_bgr: Pre-captured frame, or None to capture now.
        region: Screen region, or None for GAME_REGION.

    Returns:
        Dict with parsed metrics, or None if parsing failed.
        {
            "score": int,
            "accuracy": float,  # 0-100
            "efficiency": float,  # 0-100
            "hits": int,
            "misses": int,
            "targets": int,
        }
    """
    if frame_bgr is None:
        if region is None:
            region = GAME_REGION
        if region is None:
            print("ERROR: GAME_REGION not set")
            return None
        frame_bgr = grab_screen(region)

    # Save the raw capture for debugging
    cv2.imwrite("debug_endscreen.png", frame_bgr)

    # Try pytesseract if available, otherwise fall back to digit detection
    try:
        import pytesseract
        return _parse_with_tesseract(frame_bgr)
    except ImportError:
        print("pytesseract not installed. Trying manual digit detection...")
        return _parse_with_digit_detection(frame_bgr)


def _parse_with_tesseract(frame_bgr):
    """Parse metrics using Tesseract OCR."""
    import pytesseract

    # Preprocess: upscale, convert to grayscale, threshold
    scale = 2
    h, w = frame_bgr.shape[:2]
    upscaled = cv2.resize(frame_bgr, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Invert if needed (tesseract prefers dark text on light background)
    binary_inv = cv2.bitwise_not(binary)

    # Run OCR
    text = pytesseract.image_to_string(binary_inv, config="--psm 6")

    return _parse_text(text)


def _parse_text(text):
    """Parse metric values from OCR text output."""
    metrics = {
        "score": 0,
        "accuracy": 0.0,
        "efficiency": 0.0,
        "hits": 0,
        "misses": 0,
        "targets": 0,
    }

    lines = text.strip().split("\n")
    full_text = " ".join(lines)

    # Score: look for a standalone large number or "Score" label
    score_match = re.search(r"Score\s*[:\s]*(\d+)", full_text, re.IGNORECASE)
    if score_match:
        metrics["score"] = int(score_match.group(1))

    # Accuracy
    acc_match = re.search(r"Accuracy\s*[:\s]*(\d+(?:\.\d+)?)\s*%?", full_text, re.IGNORECASE)
    if acc_match:
        metrics["accuracy"] = float(acc_match.group(1))

    # Efficiency
    eff_match = re.search(r"Efficiency\s*[:\s]*(\d+(?:\.\d+)?)\s*%?", full_text, re.IGNORECASE)
    if eff_match:
        metrics["efficiency"] = float(eff_match.group(1))

    # Hits/Misses
    hm_match = re.search(r"Hits\s*/\s*Misses\s*[:\s]*(\d+)\s*/\s*(\d+)", full_text, re.IGNORECASE)
    if hm_match:
        metrics["hits"] = int(hm_match.group(1))
        metrics["misses"] = int(hm_match.group(2))

    # Targets
    tgt_match = re.search(r"Targets\s*[:\s]*(\d+)", full_text, re.IGNORECASE)
    if tgt_match:
        metrics["targets"] = int(tgt_match.group(1))

    return metrics


def _parse_with_digit_detection(frame_bgr):
    """Fallback: extract digits using contour-based digit segmentation.

    Less reliable than Tesseract but works without extra dependencies.
    For now, returns None and suggests installing pytesseract.
    """
    print("Manual digit detection not yet implemented.")
    print("Install pytesseract for best results:")
    print("  brew install tesseract")
    print("  pip install pytesseract")
    return None


def compute_reward(metrics):
    """Compute a reward score from game metrics.

    Higher reward = better performance. Balances speed and accuracy.

    reward = score * w_score
           + accuracy * w_accuracy
           + efficiency * w_efficiency
           - misses * miss_penalty

    Args:
        metrics: Dict from parse_metrics_from_screen().

    Returns:
        Float reward value.
    """
    from config import (
        REWARD_SCORE_WEIGHT,
        REWARD_ACCURACY_WEIGHT,
        REWARD_EFFICIENCY_WEIGHT,
        REWARD_MISS_PENALTY,
    )

    reward = (
        metrics["score"] * REWARD_SCORE_WEIGHT
        + metrics["accuracy"] * REWARD_ACCURACY_WEIGHT
        + metrics["efficiency"] * REWARD_EFFICIENCY_WEIGHT
        - metrics["misses"] * REWARD_MISS_PENALTY
    )

    return reward


if __name__ == "__main__":
    import sys
    import time

    if len(sys.argv) > 1:
        # Parse from an image file
        img = cv2.imread(sys.argv[1])
        if img is None:
            print(f"Could not load {sys.argv[1]}")
            sys.exit(1)
        metrics = parse_metrics_from_screen(frame_bgr=img)
    else:
        # Capture from screen
        print("Capturing end screen in 3 seconds...")
        print("Make sure the results screen is visible!")
        time.sleep(3)
        metrics = parse_metrics_from_screen()

    if metrics:
        print("\n=== Parsed Metrics ===")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        reward = compute_reward(metrics)
        print(f"\nReward: {reward:.1f}")
    else:
        print("Failed to parse metrics")
