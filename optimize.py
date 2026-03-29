"""Reward-driven parameter optimization for the mouse accuracy bot.

Runs multiple game sessions with different parameter configurations,
reads the end-of-game metrics via OCR, computes rewards, and finds
the optimal settings for speed + accuracy.

Usage:
    python optimize.py --rounds 10 --mode model
    python optimize.py --rounds 5 --mode cv
"""
import os
import json
import time
import random
import argparse
import numpy as np
from datetime import datetime

from ocr_utils import parse_metrics_from_screen, compute_reward
from config import GAME_REGION, DEVICE


RESULTS_FILE = os.path.join(os.path.dirname(__file__), "optimization_results.json")


# Parameter search space
SEARCH_SPACE = {
    "confidence_threshold": [0.3, 0.4, 0.5, 0.6, 0.7],
    "click_delay": [0.02, 0.03, 0.05, 0.08, 0.10],
    "target_strategy": ["confidence", "nearest", "largest"],
}


def generate_configs(n_random=10):
    """Generate parameter configurations to test.

    Uses a mix of:
    1. Grid search over key parameters
    2. Random sampling for fine-tuning
    """
    configs = []

    # Grid search over the most impactful parameters
    for threshold in SEARCH_SPACE["confidence_threshold"]:
        for delay in SEARCH_SPACE["click_delay"]:
            configs.append({
                "confidence_threshold": threshold,
                "click_delay": delay,
                "target_strategy": "confidence",
            })

    # Random configs for exploration
    for _ in range(n_random):
        configs.append({
            "confidence_threshold": round(random.uniform(0.2, 0.8), 2),
            "click_delay": round(random.uniform(0.01, 0.12), 3),
            "target_strategy": random.choice(SEARCH_SPACE["target_strategy"]),
        })

    return configs


def run_single_session(config, mode="model", checkpoint="checkpoints/best.pth"):
    """Run one game session with the given parameters.

    This function:
    1. Sets up the bot with the config
    2. Waits for the user to start the game
    3. Plays the game
    4. Waits for the results screen
    5. OCRs the metrics
    6. Returns the metrics + reward

    Args:
        config: Dict of parameter values.
        mode: "model" or "cv".
        checkpoint: Model checkpoint path.

    Returns:
        Dict with config, metrics, and reward.
    """
    import pyautogui
    from capture import grab_screen

    if GAME_REGION is None:
        print("ERROR: GAME_REGION not set in config.py")
        return None

    conf_threshold = config["confidence_threshold"]
    click_delay = config["click_delay"]
    strategy = config["target_strategy"]

    print(f"\n--- Config: threshold={conf_threshold}, delay={click_delay}, strategy={strategy} ---")

    if mode == "model":
        from play import load_model, preprocess, find_peaks, heatmap_to_screen
        import torch
        model = load_model(checkpoint)

    from labeler import find_targets

    print("Starting in 3 seconds... make sure the game is running!")
    time.sleep(3)

    clicks = 0
    frames = 0
    start_time = time.time()
    game_duration = 35  # 30s game + 5s buffer for results screen

    try:
        while (time.time() - start_time) < game_duration:
            frame = grab_screen(GAME_REGION)

            if mode == "model":
                tensor = preprocess(frame)
                with torch.no_grad():
                    heatmap = model(tensor)
                heatmap_np = heatmap.squeeze().cpu().numpy()
                peaks = find_peaks(heatmap_np, threshold=conf_threshold)

                if peaks:
                    if strategy == "nearest":
                        # Pick closest to current mouse position
                        mx, my = pyautogui.position()
                        left, top, w, h = GAME_REGION
                        from config import HEATMAP_SIZE
                        hm_w, hm_h = HEATMAP_SIZE

                        def dist_to_cursor(p):
                            sx = left + (p[1] / hm_w) * w
                            sy = top + (p[0] / hm_h) * h
                            return (sx - mx)**2 + (sy - my)**2

                        best = min(peaks, key=dist_to_cursor)
                    else:
                        # "confidence" or "largest" — highest confidence
                        best = peaks[0]

                    best_y, best_x, conf = best
                    screen_x, screen_y = heatmap_to_screen(best_y, best_x, GAME_REGION)
                    pyautogui.click(screen_x, screen_y)
                    clicks += 1

            else:
                targets = find_targets(frame)
                if targets:
                    if strategy == "largest":
                        best = max(targets, key=lambda t: t["radius"])
                    elif strategy == "nearest":
                        mx, my = pyautogui.position()
                        left, top, w, h = GAME_REGION
                        phys_w, phys_h = frame.shape[1], frame.shape[0]

                        def dist_to_cursor(t):
                            sx = left + (t["x"] / phys_w) * w
                            sy = top + (t["y"] / phys_h) * h
                            return (sx - mx)**2 + (sy - my)**2

                        best = min(targets, key=dist_to_cursor)
                    else:
                        best = targets[0]

                    left, top, w, h = GAME_REGION
                    phys_w, phys_h = frame.shape[1], frame.shape[0]
                    screen_x = left + (best["x"] / phys_w) * w
                    screen_y = top + (best["y"] / phys_h) * h
                    pyautogui.click(int(screen_x), int(screen_y))
                    clicks += 1

            frames += 1
            time.sleep(click_delay)

    except KeyboardInterrupt:
        print("Session interrupted")
        return None
    except pyautogui.FailSafeException:
        print("Failsafe triggered")
        return None

    # Wait for results screen to appear
    print(f"Game phase done ({clicks} clicks, {frames} frames). Waiting for results...")
    time.sleep(3)

    # OCR the results screen
    metrics = parse_metrics_from_screen()

    if metrics is None:
        print("Could not parse metrics")
        return None

    reward = compute_reward(metrics)

    result = {
        "config": config,
        "metrics": metrics,
        "reward": reward,
        "clicks": clicks,
        "frames": frames,
        "timestamp": datetime.now().isoformat(),
    }

    print(f"  Score: {metrics['score']} | Acc: {metrics['accuracy']}% | "
          f"Eff: {metrics['efficiency']}% | Reward: {reward:.1f}")

    return result


def optimize(n_rounds=10, mode="model", checkpoint="checkpoints/best.pth"):
    """Run multiple game sessions and find optimal parameters.

    Args:
        n_rounds: Number of parameter configs to test.
        mode: "model" or "cv".
        checkpoint: Model checkpoint path.
    """
    configs = generate_configs()
    random.shuffle(configs)
    configs = configs[:n_rounds]

    print(f"=== Parameter Optimization ===")
    print(f"Mode: {mode}")
    print(f"Rounds: {n_rounds}")
    print(f"Configs to test: {len(configs)}")
    print()

    results = []

    # Load previous results if they exist
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
        print(f"Loaded {len(results)} previous results")

    for i, config in enumerate(configs):
        print(f"\n=== Round {i+1}/{n_rounds} ===")

        input("Press Enter when the game START button is visible, then quickly start the game...")

        result = run_single_session(config, mode=mode, checkpoint=checkpoint)
        if result:
            results.append(result)

            # Save after each round
            with open(RESULTS_FILE, "w") as f:
                json.dump(results, f, indent=2)

    # Find best config
    if results:
        best = max(results, key=lambda r: r["reward"])
        print(f"\n{'='*50}")
        print(f"=== BEST CONFIG (reward={best['reward']:.1f}) ===")
        print(f"{'='*50}")
        print(f"  confidence_threshold: {best['config']['confidence_threshold']}")
        print(f"  click_delay:          {best['config']['click_delay']}")
        print(f"  target_strategy:      {best['config']['target_strategy']}")
        print(f"  Score: {best['metrics']['score']}")
        print(f"  Accuracy: {best['metrics']['accuracy']}%")
        print(f"  Efficiency: {best['metrics']['efficiency']}%")
        print(f"\nAll results saved to {RESULTS_FILE}")
    else:
        print("No successful rounds completed")


def show_results():
    """Display previous optimization results sorted by reward."""
    if not os.path.exists(RESULTS_FILE):
        print("No results found. Run optimization first.")
        return

    with open(RESULTS_FILE) as f:
        results = json.load(f)

    results.sort(key=lambda r: r["reward"], reverse=True)

    print(f"{'Rank':<5} {'Reward':<10} {'Score':<8} {'Acc%':<8} {'Eff%':<8} "
          f"{'Thresh':<8} {'Delay':<8} {'Strategy':<12}")
    print("-" * 75)

    for i, r in enumerate(results):
        c = r["config"]
        m = r["metrics"]
        print(f"{i+1:<5} {r['reward']:<10.1f} {m['score']:<8} {m['accuracy']:<8.1f} "
              f"{m['efficiency']:<8.1f} {c['confidence_threshold']:<8} "
              f"{c['click_delay']:<8} {c['target_strategy']:<12}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize bot parameters using game rewards")
    parser.add_argument("--rounds", type=int, default=5, help="Number of game sessions")
    parser.add_argument("--mode", choices=["model", "cv"], default="model")
    parser.add_argument("--checkpoint", default="checkpoints/best.pth")
    parser.add_argument("--results", action="store_true", help="Show previous results")
    args = parser.parse_args()

    if args.results:
        show_results()
    else:
        optimize(n_rounds=args.rounds, mode=args.mode, checkpoint=args.checkpoint)
