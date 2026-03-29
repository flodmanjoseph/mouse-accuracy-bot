"""Inference loop: capture screen -> model prediction -> PyAutoGUI click.

KILL SWITCH: Press ESC at any time to stop.
FAILSAFE:    Move mouse to any screen corner to abort instantly.
BACKUP:      Ctrl+C in terminal also works.
"""
import sys
import time
import signal
import argparse
import threading
import numpy as np
import torch
import pyautogui
import cv2
from scipy import ndimage
from capture import grab_screen
from labeler import find_targets
from model import TargetDetector
from config import (
    GAME_REGION, SCALE_FACTOR, INPUT_SIZE, HEATMAP_SIZE,
    CONFIDENCE_THRESHOLD, CLICK_DELAY, DEVICE,
)

# PyAutoGUI settings
pyautogui.FAILSAFE = True   # Move mouse to corner to abort
pyautogui.PAUSE = 0.005

# ─── Kill Switch (multi-layered) ─────────────────────────────────
_kill = False
_kill_lock = threading.Lock()


def trigger_kill(source="unknown"):
    """Set the kill flag from any thread. Thread-safe."""
    global _kill
    with _kill_lock:
        if not _kill:
            _kill = True
            print(f"\n{'!'*50}")
            print(f"  KILL SWITCH ACTIVATED ({source})")
            print(f"{'!'*50}")


def _listen_esc_quartz():
    """Poll for ESC using macOS Quartz (no special permissions needed)."""
    global _kill
    try:
        import Quartz
        ESC_KEYCODE = 53
        while not _kill:
            pressed = Quartz.CGEventSourceKeyState(
                Quartz.kCGEventSourceStateHIDSystemState, ESC_KEYCODE
            )
            if pressed:
                trigger_kill("ESC key - Quartz")
                return
            time.sleep(0.05)
    except ImportError:
        pass


def _listen_esc_pynput():
    """Listen for ESC using pynput (needs Input Monitoring permission)."""
    global _kill
    try:
        from pynput import keyboard
        def on_press(key):
            if key == keyboard.Key.esc:
                trigger_kill("ESC key - pynput")
                return False
        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()
    except Exception:
        pass


def start_kill_listener():
    """Start ALL kill switch listeners."""
    global _kill
    _kill = False

    # Layer 1: Quartz ESC polling (most reliable on macOS)
    t1 = threading.Thread(target=_listen_esc_quartz, daemon=True)
    t1.start()

    # Layer 2: pynput ESC listener (backup)
    t2 = threading.Thread(target=_listen_esc_pynput, daemon=True)
    t2.start()

    # Layer 3: Ctrl+C signal handler
    def signal_handler(sig, frame):
        trigger_kill("Ctrl+C")
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("  Kill switches: ESC (Quartz + pynput), Ctrl+C, mouse-to-corner")


def is_in_game_region(x, y):
    """Check if screen coordinates are inside the game region."""
    left, top, gw, gh = GAME_REGION
    margin = 10
    return (left + margin <= x <= left + gw - margin and
            top + margin <= y <= top + gh - margin)


def focus_chrome():
    """Bring Chrome to the foreground using AppleScript."""
    import subprocess
    subprocess.run([
        "osascript", "-e",
        'tell application "Google Chrome" to activate'
    ], capture_output=True)
    time.sleep(0.5)  # Wait for window to come to front


def load_model(checkpoint_path="checkpoints/best.pth"):
    """Load trained model for inference."""
    model = TargetDetector()
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    return model


def preprocess(frame_bgr):
    """Convert captured BGR frame to model input tensor."""
    resized = cv2.resize(frame_bgr, INPUT_SIZE, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    return tensor.to(DEVICE)


def find_peaks(heatmap_np, threshold=None):
    """Find target peaks in the heatmap using connected components.

    Args:
        heatmap_np: 2D numpy array (H, W) with values [0, 1].
        threshold: Minimum confidence to consider a peak.

    Returns:
        List of (y, x, confidence) tuples sorted by confidence descending.
    """
    if threshold is None:
        threshold = CONFIDENCE_THRESHOLD

    binary = (heatmap_np > threshold).astype(np.int32)
    labeled, num_features = ndimage.label(binary)

    peaks = []
    for i in range(1, num_features + 1):
        region = (labeled == i)
        region_values = heatmap_np[region]
        max_idx = np.argmax(region_values)

        # Get coordinates of the max value within this region
        ys, xs = np.where(region)
        peak_y = ys[max_idx]
        peak_x = xs[max_idx]
        confidence = region_values[max_idx]

        peaks.append((peak_y, peak_x, float(confidence)))

    # Sort by confidence descending
    peaks.sort(key=lambda p: p[2], reverse=True)
    return peaks


def heatmap_to_screen(peak_y, peak_x, region):
    """Convert heatmap coordinates to screen coordinates (logical pixels).

    Args:
        peak_y, peak_x: Coordinates in heatmap space (HEATMAP_SIZE).
        region: Game region (left, top, width, height) in logical pixels.

    Returns:
        (screen_x, screen_y) in logical pixels for PyAutoGUI.
    """
    left, top, width, height = region
    hm_w, hm_h = HEATMAP_SIZE

    # Scale from heatmap to game region
    screen_x = left + (peak_x / hm_w) * width
    screen_y = top + (peak_y / hm_h) * height

    return int(screen_x), int(screen_y)


def play_with_model(model, region=None, delay=None, conf_threshold=None):
    """Main game loop using the trained CNN model."""
    if region is None:
        region = GAME_REGION
    if delay is None:
        delay = CLICK_DELAY
    if conf_threshold is None:
        conf_threshold = CONFIDENCE_THRESHOLD

    if region is None:
        print("ERROR: GAME_REGION not set in config.py")
        print("Run `python capture.py` first to calibrate.")
        return

    print("=== MODEL MODE ===")
    print(f"Region: {region}")
    print(f"Device: {DEVICE}")
    print(f"Confidence threshold: {conf_threshold}")
    print(f"Click delay: {delay}s")
    print("Move mouse to screen corner to abort (failsafe)")

    # Bring Chrome to foreground so we capture the game, not VS Code
    print("\nSwitching to Chrome...")
    focus_chrome()
    start_kill_listener()
    print("Press ESC at any time to stop the bot.")

    # Click the START button (centered horizontally, ~84% down the game area)
    left, top, gw, gh = region
    start_x = left + gw // 2
    start_y = top + int(gh * 0.84)
    print(f"Clicking START button at ({start_x}, {start_y})...")
    pyautogui.click(start_x, start_y)
    time.sleep(0.5)

    print("Waiting for game to start (looking for targets)...")

    clicks = 0
    frames = 0
    no_target_streak = 0
    game_started = False  # Don't trigger game-over until we've seen at least one target
    max_no_target = 50    # Stop after 50 consecutive frames with no targets (~6s at 8fps = game over)
    start_time = time.time()
    while not _kill:
        frame = grab_screen(region)
        tensor = preprocess(frame)
        with torch.no_grad():
            heatmap = model(tensor)
        heatmap_np = heatmap.squeeze().cpu().numpy()
        peaks = find_peaks(heatmap_np, threshold=conf_threshold)
        if peaks:
            print("Targets detected — game on!")
            break
        if time.time() - start_time > 15:
            print("Timed out waiting for game to start.")
            return
        time.sleep(0.05)

    if _kill:
        print("Aborted.")
        return

    start_time = time.time()  # Reset timer to actual game start

    try:
        while not _kill:
            frame = grab_screen(region)
            tensor = preprocess(frame)

            with torch.no_grad():
                heatmap = model(tensor)

            heatmap_np = heatmap.squeeze().cpu().numpy()
            peaks = find_peaks(heatmap_np, threshold=conf_threshold)

            if peaks:
                no_target_streak = 0
                game_started = True
                # Click the highest confidence target
                best_y, best_x, conf = peaks[0]
                screen_x, screen_y = heatmap_to_screen(best_y, best_x, region)
                # SAFETY: Only click inside the game region
                if is_in_game_region(screen_x, screen_y) and not _kill:
                    pyautogui.click(screen_x, screen_y)
                    clicks += 1
            else:
                no_target_streak += 1
                if game_started and no_target_streak >= max_no_target:
                    print(f"\nNo targets for {max_no_target} frames — game over detected.")
                    break

            frames += 1
            if frames % 20 == 0:
                elapsed = time.time() - start_time
                fps = frames / elapsed
                print(f"  Frames: {frames} | Clicks: {clicks} | FPS: {fps:.1f}")

            time.sleep(delay)

    except KeyboardInterrupt:
        pass
    except pyautogui.FailSafeException:
        print("\nFailsafe triggered (mouse moved to corner)")

    elapsed = time.time() - start_time
    print("\n=== Session Complete ===")
    print(f"Frames: {frames} | Clicks: {clicks} | Time: {elapsed:.1f}s")
    if elapsed > 0:
        print(f"FPS: {frames/elapsed:.1f}")


def play_with_cv(region=None, delay=None):
    """Fallback mode: use the color-thresholding labeler directly (no model).

    Good baseline to compare against the trained CNN.
    """
    if region is None:
        region = GAME_REGION
    if delay is None:
        delay = CLICK_DELAY

    if region is None:
        print("ERROR: GAME_REGION not set in config.py")
        return

    print("=== CV MODE (no model, color thresholding) ===")
    print(f"Region: {region}")
    print("\nSwitching to Chrome...")
    focus_chrome()
    start_kill_listener()
    print("Press ESC at any time to stop the bot.")

    clicks = 0
    frames = 0
    no_target_streak = 0
    max_no_target = 50
    game_started = False
    start_time = time.time()

    try:
        while not _kill:
            frame = grab_screen(region)
            targets = find_targets(frame)

            if targets:
                no_target_streak = 0
                game_started = True
                # Click the largest target
                best = max(targets, key=lambda t: t["radius"])

                # Convert from physical pixels to logical screen coordinates
                left, top, width, height = region
                phys_w = frame.shape[1]
                phys_h = frame.shape[0]

                screen_x = int(left + (best["x"] / phys_w) * width)
                screen_y = int(top + (best["y"] / phys_h) * height)

                # SAFETY: Only click inside game region
                if is_in_game_region(screen_x, screen_y) and not _kill:
                    pyautogui.click(screen_x, screen_y)
                    clicks += 1
            else:
                no_target_streak += 1
                if game_started and no_target_streak >= max_no_target:
                    print(f"\nNo targets for {max_no_target} frames — game over detected.")
                    break

            frames += 1
            if frames % 20 == 0:
                elapsed = time.time() - start_time
                fps = frames / elapsed
                print(f"  Frames: {frames} | Clicks: {clicks} | FPS: {fps:.1f}")

            time.sleep(delay)

    except KeyboardInterrupt:
        trigger_kill("Ctrl+C")
    except pyautogui.FailSafeException:
        trigger_kill("mouse to corner")

    elapsed = time.time() - start_time
    print("\n=== Session Complete ===")
    print(f"Frames: {frames} | Clicks: {clicks} | Time: {elapsed:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play Mouse Accuracy with AI")
    parser.add_argument("--mode", choices=["model", "cv"], default="model",
                        help="Detection mode: 'model' (CNN) or 'cv' (color threshold)")
    parser.add_argument("--checkpoint", default="checkpoints/best.pth",
                        help="Model checkpoint path")
    parser.add_argument("--delay", type=float, default=None,
                        help="Seconds between clicks")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Confidence threshold for model mode")
    args = parser.parse_args()

    if args.mode == "model":
        model = load_model(args.checkpoint)
        play_with_model(model, delay=args.delay, conf_threshold=args.threshold)
    else:
        play_with_cv(delay=args.delay)
