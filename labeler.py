"""Auto-labeling: detect red circle targets using HSV thresholding + contour detection."""
import cv2
import numpy as np
from config import (
    HSV_RED_LOWER_1, HSV_RED_UPPER_1,
    HSV_RED_LOWER_2, HSV_RED_UPPER_2,
    MIN_CONTOUR_AREA, MIN_CIRCULARITY,
    HUD_RIGHT_FRACTION,
)


def find_targets(frame_bgr, exclude_hud=True):
    """Detect red circle targets in a BGR frame.

    Args:
        frame_bgr: BGR numpy array (any resolution).
        exclude_hud: If True, mask out the right-side HUD area.

    Returns:
        List of dicts: [{"x": int, "y": int, "radius": int}, ...]
        Coordinates are in the frame's pixel space.
    """
    h, w = frame_bgr.shape[:2]

    # Convert to HSV
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # Create red mask (two ranges because red wraps at H=0/180)
    mask1 = cv2.inRange(hsv, HSV_RED_LOWER_1, HSV_RED_UPPER_1)
    mask2 = cv2.inRange(hsv, HSV_RED_LOWER_2, HSV_RED_UPPER_2)
    mask = mask1 | mask2

    # Exclude HUD region (right side: timer, score, etc.)
    if exclude_hud:
        hud_left = int(w * (1 - HUD_RIGHT_FRACTION))
        mask[:, hud_left:] = 0
        # Also exclude top bar (header area, ~8% of height)
        header_bottom = int(h * 0.08)
        mask[:header_bottom, :] = 0

    # Morphological cleanup: remove noise, fill small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    targets = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_CONTOUR_AREA:
            continue

        # Fit minimum enclosing circle
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        circle_area = np.pi * radius * radius

        # Check circularity — use a lower threshold for bigger contours
        # because large circles (especially rings/outlines) have lower fill ratios
        fill_ratio = area / circle_area if circle_area > 0 else 0

        # For large contours (radius > 30px), accept lower fill ratios
        # This handles hollow/ring-style targets in the game
        if radius > 30:
            min_circ = 0.20  # Rings have low fill but are still valid
        else:
            min_circ = MIN_CIRCULARITY

        # Also check perimeter-based circularity (works better for rings)
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            perimeter_circularity = 4 * np.pi * area / (perimeter * perimeter)
        else:
            perimeter_circularity = 0

        # Accept if EITHER fill ratio OR perimeter circularity passes
        if fill_ratio >= min_circ or perimeter_circularity >= 0.4:
            targets.append({
                "x": int(cx),
                "y": int(cy),
                "radius": int(radius),
            })

    return targets


def draw_targets(frame_bgr, targets, color=(0, 255, 0), thickness=2):
    """Draw detected targets on a frame copy for visualization.

    Returns:
        BGR frame with green circles drawn at target locations.
    """
    vis = frame_bgr.copy()
    for t in targets:
        cv2.circle(vis, (t["x"], t["y"]), t["radius"], color, thickness)
        cv2.circle(vis, (t["x"], t["y"]), 3, (0, 0, 255), -1)  # Center dot
    return vis


def show_mask_debug(frame_bgr):
    """Show the HSV mask for debugging threshold values.

    Saves debug images to help tune HSV ranges.
    """
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, HSV_RED_LOWER_1, HSV_RED_UPPER_1)
    mask2 = cv2.inRange(hsv, HSV_RED_LOWER_2, HSV_RED_UPPER_2)
    mask = mask1 | mask2

    # Save debug images
    cv2.imwrite("debug_mask_raw.png", mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite("debug_mask_clean.png", mask_clean)

    # Overlay mask on original
    overlay = frame_bgr.copy()
    overlay[mask_clean > 0] = (0, 255, 0)
    blended = cv2.addWeighted(frame_bgr, 0.7, overlay, 0.3, 0)
    cv2.imwrite("debug_mask_overlay.png", blended)

    targets = find_targets(frame_bgr)
    vis = draw_targets(frame_bgr, targets)
    cv2.imwrite("debug_detections.png", vis)

    print(f"Found {len(targets)} targets")
    for t in targets:
        print(f"  ({t['x']}, {t['y']}) r={t['radius']}")
    print("Debug images saved: debug_mask_raw.png, debug_mask_clean.png, debug_mask_overlay.png, debug_detections.png")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # Load an image file for testing
        img = cv2.imread(sys.argv[1])
        if img is None:
            print(f"Could not load {sys.argv[1]}")
            sys.exit(1)
        show_mask_debug(img)
    else:
        # Capture live screen
        from capture import grab_screen
        import time
        print("Capturing screen in 3 seconds...")
        time.sleep(3)
        frame = grab_screen()
        show_mask_debug(frame)
