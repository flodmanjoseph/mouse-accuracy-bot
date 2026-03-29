"""Data collection: capture screenshots during gameplay and auto-label targets."""
import os
import json
import time
import argparse
import cv2
from capture import grab_screen
from labeler import find_targets, draw_targets
from config import GAME_REGION, CAPTURE_FPS


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
LABEL_DIR = os.path.join(DATA_DIR, "labels")


def collect(region=None, fps=None, duration=None):
    """Capture screenshots and auto-label them during gameplay.

    Args:
        region: Screen region (left, top, width, height) or None for GAME_REGION.
        fps: Capture rate (frames/sec).
        duration: Max seconds to collect, or None for unlimited (Ctrl+C to stop).
    """
    if region is None:
        region = GAME_REGION
    if fps is None:
        fps = CAPTURE_FPS

    if region is None:
        print("ERROR: GAME_REGION not set in config.py")
        print("Run `python capture.py` first to calibrate.")
        return

    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(LABEL_DIR, exist_ok=True)

    interval = 1.0 / fps
    frame_count = 0
    total_targets = 0
    start_time = time.time()

    print(f"Collecting at {fps} FPS from region {region}")
    print("Start the Mouse Accuracy game now!")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            loop_start = time.time()

            if duration and (loop_start - start_time) >= duration:
                break

            # Capture
            frame = grab_screen(region)
            h, w = frame.shape[:2]

            # Auto-label
            targets = find_targets(frame)

            # Save image
            timestamp = f"{loop_start:.3f}"
            img_path = os.path.join(IMAGE_DIR, f"{timestamp}.png")
            cv2.imwrite(img_path, frame)

            # Save labels
            label_data = {
                "width": w,
                "height": h,
                "targets": targets,
                "timestamp": float(timestamp),
            }
            label_path = os.path.join(LABEL_DIR, f"{timestamp}.json")
            with open(label_path, "w") as f:
                json.dump(label_data, f)

            frame_count += 1
            total_targets += len(targets)

            # Progress
            elapsed = time.time() - start_time
            if frame_count % 10 == 0:
                print(f"  Frames: {frame_count} | Targets found: {total_targets} | "
                      f"Elapsed: {elapsed:.1f}s | Avg targets/frame: {total_targets/frame_count:.1f}")

            # Maintain target FPS
            sleep_time = interval - (time.time() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        pass

    elapsed = time.time() - start_time
    print(f"\n=== Collection Complete ===")
    print(f"Frames:  {frame_count}")
    print(f"Targets: {total_targets}")
    print(f"Time:    {elapsed:.1f}s")
    print(f"Avg FPS: {frame_count/elapsed:.1f}")
    if frame_count > 0:
        print(f"Avg targets/frame: {total_targets/frame_count:.1f}")


def verify(num_samples=10):
    """Load saved data and draw labels on images for visual inspection."""
    if not os.path.exists(IMAGE_DIR) or not os.path.exists(LABEL_DIR):
        print("No data to verify. Run collection first.")
        return

    images = sorted(f for f in os.listdir(IMAGE_DIR) if f.endswith(".png"))
    if not images:
        print("No images found.")
        return

    # Sample evenly across the dataset
    step = max(1, len(images) // num_samples)
    samples = images[::step][:num_samples]

    os.makedirs("verify_output", exist_ok=True)

    for img_name in samples:
        key = os.path.splitext(img_name)[0]
        img_path = os.path.join(IMAGE_DIR, img_name)
        label_path = os.path.join(LABEL_DIR, f"{key}.json")

        img = cv2.imread(img_path)
        if img is None:
            continue

        if os.path.exists(label_path):
            with open(label_path) as f:
                label_data = json.load(f)
            targets = label_data["targets"]
            vis = draw_targets(img, targets)
            count = len(targets)
        else:
            vis = img
            count = 0

        out_path = os.path.join("verify_output", f"verify_{key}.png")
        cv2.imwrite(out_path, vis)
        print(f"  {img_name}: {count} targets → {out_path}")

    print(f"\nVerification images saved to verify_output/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect training data for mouse accuracy bot")
    parser.add_argument("--verify", action="store_true", help="Verify existing labels visually")
    parser.add_argument("--fps", type=int, default=None, help="Capture FPS")
    parser.add_argument("--duration", type=int, default=None, help="Max collection time in seconds")
    parser.add_argument("--samples", type=int, default=10, help="Number of verification samples")
    args = parser.parse_args()

    if args.verify:
        verify(num_samples=args.samples)
    else:
        collect(fps=args.fps, duration=args.duration)
