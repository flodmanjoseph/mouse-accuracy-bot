"""Screen capture utilities using mss for speed."""
import numpy as np
import mss
import cv2
from config import GAME_REGION, SCALE_FACTOR, INPUT_SIZE


_sct = mss.mss()


def grab_screen(region=None):
    """Capture a screen region, return as BGR numpy array at physical resolution.

    Args:
        region: (left, top, width, height) in logical pixels, or None for full screen.

    Returns:
        BGR numpy array at physical pixel resolution.
    """
    if region is None:
        monitor = _sct.monitors[1]  # Primary monitor
    else:
        left, top, width, height = region
        monitor = {
            "left": int(left * SCALE_FACTOR),
            "top": int(top * SCALE_FACTOR),
            "width": int(width * SCALE_FACTOR),
            "height": int(height * SCALE_FACTOR),
        }

    img = _sct.grab(monitor)
    frame = np.array(img)[:, :, :3]  # Drop alpha, BGRA -> BGR
    return frame


def grab_and_resize(region=None, target_size=None):
    """Capture screen and resize to model input dimensions.

    Returns:
        BGR numpy array at target_size resolution.
    """
    if target_size is None:
        target_size = INPUT_SIZE
    frame = grab_screen(region)
    resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
    return resized


def calibrate():
    """Interactive calibration: capture full screen and print dimensions.

    Helps the user determine GAME_REGION coordinates.
    """
    import time

    print("=== Screen Capture Calibration ===")
    print(f"Scale factor: {SCALE_FACTOR}")

    monitor = _sct.monitors[1]
    print(f"Primary monitor (physical): {monitor['width']}x{monitor['height']}")
    print(f"Primary monitor (logical):  {monitor['width']/SCALE_FACTOR:.0f}x{monitor['height']/SCALE_FACTOR:.0f}")

    print("\nCapturing full screen in 3 seconds...")
    print("Make sure the Mouse Accuracy game is visible!")
    time.sleep(3)

    frame = grab_screen()
    print(f"Captured frame shape: {frame.shape} (height, width, channels)")

    # Save for inspection
    out_path = "calibration_screenshot.png"
    cv2.imwrite(out_path, frame)
    print(f"Saved to {out_path}")
    print("\nOpen the screenshot and note the game window boundaries.")
    print("Then set GAME_REGION = (left, top, width, height) in config.py")
    print("Use LOGICAL pixel coordinates (divide physical by scale factor).")


if __name__ == "__main__":
    calibrate()
