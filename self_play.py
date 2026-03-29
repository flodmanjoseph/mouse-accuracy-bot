"""Self-playing RL loop: play games, read results, adjust parameters, improve.

Press ESC at any time to stop.

Usage:
    python self_play.py                  # Run continuous self-play
    python self_play.py --rounds 5       # Run 5 games
    python self_play.py --results        # Show history
"""
import os
import json
import time
import random
import argparse
import threading
import numpy as np
import pyautogui
import cv2
from datetime import datetime

from capture import grab_screen
from labeler import find_targets
from config import GAME_REGION, HEATMAP_SIZE, INPUT_SIZE, DEVICE

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.01

# Kill switch
_kill = False
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "self_play_results.json")


def _listen_for_esc():
    global _kill
    try:
        from pynput import keyboard
        def on_press(key):
            global _kill
            if key == keyboard.Key.esc:
                _kill = True
                print("\n*** ESC pressed — stopping ***")
                return False
        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()
    except ImportError:
        print("Install pynput for ESC support: pip install pynput")


def focus_chrome():
    import subprocess
    subprocess.run(['osascript', '-e', 'tell application "Google Chrome" to activate'], capture_output=True)
    time.sleep(0.5)


def click_start_or_restart():
    """Click START or RESTART button on the game/results page."""
    left, top, gw, gh = GAME_REGION

    # Capture to figure out which page we're on
    frame = grab_screen(GAME_REGION)

    # Check if we see "RESTART" (results page) by looking for the button area
    # RESTART button is centered, near bottom of page
    # START button is also centered, near bottom
    # Both are in roughly the same position — just click center-bottom
    btn_x = left + gw // 2
    btn_y = top + int(gh * 0.95)  # Very bottom for RESTART on results page

    pyautogui.click(btn_x, btn_y)
    time.sleep(0.3)

    # Also try the game page START position (slightly higher)
    btn_y2 = top + int(gh * 0.84)
    pyautogui.click(btn_x, btn_y2)
    time.sleep(0.5)


def wait_for_game_start(timeout=20):
    """Wait for the game countdown to finish and targets to appear."""
    start = time.time()
    while not _kill and (time.time() - start) < timeout:
        frame = grab_screen(GAME_REGION)
        targets = find_targets(frame)
        if targets:
            return True
        time.sleep(0.1)
    return False


def play_one_game(params):
    """Play a single game with the given parameters.

    Args:
        params: dict with click_delay, target_strategy

    Returns:
        dict with clicks, frames, duration
    """
    click_delay = params.get("click_delay", 0.03)
    strategy = params.get("target_strategy", "largest")

    clicks = 0
    frames = 0
    no_target_streak = 0
    game_started = False
    start_time = time.time()

    while not _kill:
        frame = grab_screen(GAME_REGION)
        targets = find_targets(frame)

        if targets:
            no_target_streak = 0
            game_started = True

            # Pick target based on strategy
            if strategy == "nearest":
                mx, my = pyautogui.position()
                left, top, gw, gh = GAME_REGION
                phys_w, phys_h = frame.shape[1], frame.shape[0]

                def screen_dist(t):
                    sx = left + (t["x"] / phys_w) * gw
                    sy = top + (t["y"] / phys_h) * gh
                    return (sx - mx)**2 + (sy - my)**2

                best = min(targets, key=screen_dist)
            elif strategy == "largest":
                best = max(targets, key=lambda t: t["radius"])
            else:
                best = targets[0]  # first found

            # Convert to screen coordinates and click
            left, top, gw, gh = GAME_REGION
            phys_w, phys_h = frame.shape[1], frame.shape[0]
            screen_x = left + (best["x"] / phys_w) * gw
            screen_y = top + (best["y"] / phys_h) * gh
            pyautogui.click(int(screen_x), int(screen_y))
            clicks += 1
        else:
            no_target_streak += 1
            # Only check for game over after game has started
            if game_started and no_target_streak >= 50:
                break

        frames += 1
        if frames % 50 == 0:
            elapsed = time.time() - start_time
            print(f"    Frames: {frames} | Clicks: {clicks} | FPS: {frames/elapsed:.1f}")

        time.sleep(click_delay)

    elapsed = time.time() - start_time
    return {"clicks": clicks, "frames": frames, "duration": elapsed}


def read_results_screen():
    """Wait for results screen and parse metrics via OCR or pattern matching."""
    time.sleep(3)  # Wait for results page to load

    frame = grab_screen(GAME_REGION)
    cv2.imwrite("last_results.png", frame)

    # Try pytesseract OCR
    try:
        from ocr_utils import parse_metrics_from_screen
        metrics = parse_metrics_from_screen(frame_bgr=frame)
        if metrics and metrics.get("targets", 0) > 0:
            return metrics
    except Exception as e:
        print(f"    OCR failed: {e}")

    # Fallback: return None, we'll use click-based metrics
    return None


def compute_reward(metrics, game_stats):
    """Compute reward from game metrics.

    Higher = better. Balances accuracy and efficiency.
    """
    if metrics:
        reward = (
            metrics["score"] * 1.0
            + metrics["accuracy"] * 0.5
            + metrics["efficiency"] * 0.5
            - metrics["misses"] * 2.0
        )
        return reward
    else:
        # Fallback: just use click count as rough proxy
        return game_stats["clicks"] * 0.5


def mutate_params(params, reward_history):
    """Adjust parameters based on recent performance.

    Simple evolutionary strategy:
    - Keep parameters that improved reward
    - Randomly mutate parameters that didn't
    """
    new_params = params.copy()

    if len(reward_history) >= 2:
        last_reward = reward_history[-1]
        prev_reward = reward_history[-2]

        if last_reward > prev_reward:
            # Getting better — small mutations
            mutation_scale = 0.1
            print("    Reward improved! Small mutations.")
        else:
            # Getting worse — larger mutations
            mutation_scale = 0.3
            print("    Reward decreased. Larger mutations.")
    else:
        mutation_scale = 0.2

    # Mutate click_delay
    delay = params["click_delay"]
    delay *= (1.0 + random.uniform(-mutation_scale, mutation_scale))
    new_params["click_delay"] = max(0.01, min(0.1, round(delay, 3)))

    # Occasionally switch strategy
    if random.random() < 0.2:
        strategies = ["largest", "nearest", "first"]
        new_params["target_strategy"] = random.choice(strategies)

    return new_params


def self_play(max_rounds=None):
    """Main self-play loop."""
    global _kill
    _kill = False

    # Start ESC listener
    esc_thread = threading.Thread(target=_listen_for_esc, daemon=True)
    esc_thread.start()

    print("=" * 50)
    print("  SELF-PLAY MODE")
    print("  Press ESC to stop at any time")
    print("=" * 50)

    # Load history
    history = []
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            history = json.load(f)
        print(f"Loaded {len(history)} previous games")

    # Starting parameters
    if history:
        # Use best params from history
        best_game = max(history, key=lambda g: g.get("reward", 0))
        params = best_game.get("params", {"click_delay": 0.03, "target_strategy": "largest"})
        print(f"Starting with best params: {params}")
    else:
        params = {"click_delay": 0.03, "target_strategy": "largest"}

    reward_history = [g.get("reward", 0) for g in history]
    round_num = len(history) + 1

    focus_chrome()
    time.sleep(1)

    while not _kill:
        if max_rounds and (round_num - len(history) - 1) >= max_rounds:
            break

        print(f"\n{'='*50}")
        print(f"  ROUND {round_num}")
        print(f"  Params: delay={params['click_delay']}, strategy={params['target_strategy']}")
        print(f"{'='*50}")

        # Click START/RESTART
        print("  Clicking START...")
        click_start_or_restart()

        if _kill:
            break

        # Wait for game
        print("  Waiting for countdown...")
        if not wait_for_game_start(timeout=15):
            print("  Game didn't start. Trying again...")
            click_start_or_restart()
            if not wait_for_game_start(timeout=10):
                print("  Failed to start game. Skipping round.")
                continue

        if _kill:
            break

        # Play!
        print("  Playing...")
        game_stats = play_one_game(params)
        print(f"  Game done: {game_stats['clicks']} clicks in {game_stats['duration']:.1f}s")

        if _kill:
            break

        # Read results
        print("  Reading results...")
        metrics = read_results_screen()
        reward = compute_reward(metrics, game_stats)

        if metrics:
            print(f"  Score: {metrics['score']} | Acc: {metrics['accuracy']}% | Eff: {metrics['efficiency']}%")
        print(f"  Reward: {reward:.1f}")

        # Save
        result = {
            "round": round_num,
            "params": params,
            "game_stats": game_stats,
            "metrics": metrics,
            "reward": reward,
            "timestamp": datetime.now().isoformat(),
        }
        history.append(result)
        reward_history.append(reward)

        with open(RESULTS_FILE, "w") as f:
            json.dump(history, f, indent=2)

        # Evolve parameters for next round
        params = mutate_params(params, reward_history)
        round_num += 1

    # Summary
    print(f"\n{'='*50}")
    print("  SESSION COMPLETE")
    print(f"  Games played: {len(history)}")
    if history:
        rewards = [g.get("reward", 0) for g in history]
        print(f"  Best reward: {max(rewards):.1f}")
        best = max(history, key=lambda g: g.get("reward", 0))
        print(f"  Best params: {best.get('params', {})}")
        if best.get("metrics"):
            m = best["metrics"]
            print(f"  Best score: {m['score']} | Acc: {m['accuracy']}% | Eff: {m['efficiency']}%")
    print(f"{'='*50}")


def show_results():
    """Display history sorted by reward."""
    if not os.path.exists(RESULTS_FILE):
        print("No results yet.")
        return

    with open(RESULTS_FILE) as f:
        history = json.load(f)

    history.sort(key=lambda g: g.get("reward", 0), reverse=True)

    print(f"{'#':<4} {'Reward':<9} {'Score':<7} {'Acc%':<7} {'Eff%':<7} "
          f"{'Delay':<7} {'Strategy':<10}")
    print("-" * 55)

    for g in history:
        m = g.get("metrics") or {}
        p = g.get("params", {})
        print(f"{g['round']:<4} {g.get('reward',0):<9.1f} {m.get('score','?'):<7} "
              f"{m.get('accuracy','?'):<7} {m.get('efficiency','?'):<7} "
              f"{p.get('click_delay','?'):<7} {p.get('target_strategy','?'):<10}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-playing RL loop")
    parser.add_argument("--rounds", type=int, default=None, help="Max games to play (default: infinite)")
    parser.add_argument("--results", action="store_true", help="Show game history")
    args = parser.parse_args()

    if args.results:
        show_results()
    else:
        self_play(max_rounds=args.rounds)
