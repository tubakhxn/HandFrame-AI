import argparse
import time
import subprocess
import sys


def _ensure_packages():
 
    required_core = ["opencv-python", "mediapipe", "numpy", "Pillow", "fal-client"]

    import importlib

    def _missing(pkgs):
        name_map = {
            "opencv-python": "cv2",
            "Pillow": "PIL",
            "fal-client": "fal_client",
        }
        missing = []
        for pkg in pkgs:
            mod_name = name_map.get(pkg, pkg)
            try:
                importlib.import_module(mod_name)
            except ImportError:
                missing.append(pkg)
        return missing

    missing_core = _missing(required_core)
    if missing_core:
        print(f"[setup] Installing required packages: {missing_core} ...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", *missing_core],
            check=True,
        )


_ensure_packages()

import cv2
import numpy as np

from hand_tracker import HandTracker
from perspective import FloatingPlane, QuadSmoother, quad_to_capture_rect
from inference import AsyncFluxEngine
from styles import get_style, style_count
import ui_overlay


def parse_args():
    p = argparse.ArgumentParser(description="Real-time AI hand-frame style transfer")
    p.add_argument("--camera", type=int, default=0, help="webcam index")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fal_key", type=str,
                    default="a1e22bab-8819-4c0c-b282-3e0883db30c8:8706cee467f8ccaf91043b7eec0977fb",
                    help="fal.ai API key for real FLUX.2 [klein] 4B image-to-"
                         "image generation on a held pinch. Get one free at "
                         "fal.ai/dashboard/keys. Can also be set via the "
                         "FAL_KEY environment variable. Omit to use the fast "
                         "OpenCV fallback stylizer for the AI-render path too.")
    p.add_argument("--strength", type=float, default=0.65)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--capture_size", type=int, default=512,
                    help="square resolution of the region sent to the AI model "
                         "on a held pinch")
    p.add_argument("--live_size", type=int, default=320,
                    help="square resolution of the region re-stylized every "
                         "frame for the live AR-filter effect")
    p.add_argument("--hold_threshold", type=float, default=0.6,
                    help="seconds a pinch must be held to trigger a full AI "
                         "generation job instead of an instant filter cycle")
    p.add_argument("--cycle_cooldown", type=float, default=0.35,
                    help="minimum seconds between quick-pinch style cycles")
    p.add_argument("--mirror", action="store_true", default=True,
                    help="mirror the webcam feed (selfie view)")
    return p.parse_args()


class FPSMeter:
    def __init__(self, smoothing=0.9):
        self._t_prev = time.time()
        self._fps = 0.0
        self._smoothing = smoothing

    def tick(self):
        t = time.time()
        dt = max(t - self._t_prev, 1e-6)
        inst_fps = 1.0 / dt
        self._fps = self._smoothing * self._fps + (1 - self._smoothing) * inst_fps
        self._t_prev = t
        return self._fps


class AppState:
    """
    Tracks what the floating plane currently shows.

    mode:
      IDLE       - no frame gesture active
      LIVE       - frame gesture active; quad is re-stylized every frame
                   with the current (fast) filter, like an AR lens
      GENERATING - a held-pinch fired an async AI job; spinner shown
                   over the live filter until the result lands
      AI_RESULT  - showing a one-off high-quality AI result (frozen,
                   but still pinned/warped to the live quad)
    """

    IDLE = "idle"
    LIVE = "live"
    GENERATING = "generating"
    AI_RESULT = "ai_result"

    def __init__(self):
        self.mode = self.IDLE
        self.style_idx = 0
        self.pending_job_id = None
        self.ai_result_image = None

        # Pinch edge-detection / hold-duration tracking
        self.pinch_active = False
        self.pinch_start_time = 0.0
        self.last_style_cycle_time = 0.0
        self.held_pinch_fired = False


def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check --camera index.")

    tracker = HandTracker(max_hands=2)
    plane = FloatingPlane()
    quad_smoother = QuadSmoother(min_cutoff=1.0, beta=0.35)

    engine = AsyncFluxEngine(
        fal_key=args.fal_key,
        strength=args.strength,
        num_inference_steps=args.steps,
    )
    engine.start()

    state = AppState()
    fps_meter = FPSMeter()

    window_name = "HandFrame AI Style Transfer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if args.mirror:
                frame = cv2.flip(frame, 1)

            now = time.time()
            tracker.process(frame, t=now)
            quad = tracker.get_frame_gesture()

            status_text = "Show both hands as an L-frame to begin"

            if quad is not None:
                smoothed_quad = quad_smoother.smooth(quad, t=now)

                if state.mode == AppState.IDLE:
                    state.mode = AppState.LIVE

                # ---------------- Pinch edge / hold detection ----------------
                pinched_now = tracker.get_pinch("Right")

                if pinched_now and not state.pinch_active:
                    # Pinch just started
                    state.pinch_active = True
                    state.pinch_start_time = now
                    state.held_pinch_fired = False

                elif pinched_now and state.pinch_active:
                    # Pinch is being held -- check if it crossed the hold
                    # threshold, in which case fire one AI generation job
                    # (only once per hold, not every frame).
                    held_for = now - state.pinch_start_time
                    if held_for >= args.hold_threshold and not state.held_pinch_fired \
                            and state.mode != AppState.GENERATING:
                        capture = quad_to_capture_rect(
                            frame, smoothed_quad,
                            out_size=(args.capture_size, args.capture_size),
                        )
                        style = get_style(state.style_idx)
                        job_id = engine.submit(capture, style)
                        if job_id is not None:
                            state.pending_job_id = job_id
                            state.mode = AppState.GENERATING
                        state.held_pinch_fired = True

                elif not pinched_now and state.pinch_active:
                    # Pinch released -- if it was a *quick* pinch (released
                    # before the hold threshold / before an AI job fired),
                    # instantly cycle to the next style like an AR lens swipe.
                    held_for = now - state.pinch_start_time
                    can_cycle = (now - state.last_style_cycle_time) > args.cycle_cooldown
                    if not state.held_pinch_fired and held_for < args.hold_threshold and can_cycle:
                        state.style_idx = (state.style_idx + 1) % style_count()
                        state.last_style_cycle_time = now
                        if state.mode == AppState.AI_RESULT:
                            state.mode = AppState.LIVE  # new style -> back to live filter
                    state.pinch_active = False

                # ---------------- Poll async AI result (non-blocking) --------
                result = engine.poll_result()
                if result is not None and result.job_id == state.pending_job_id:
                    if result.error is None:
                        state.ai_result_image = result.image_bgr
                        state.mode = AppState.AI_RESULT
                        status_text = f"AI styled in {result.elapsed:.2f}s"
                    else:
                        state.mode = AppState.LIVE
                        status_text = f"Generation failed: {result.error}"
                    state.pending_job_id = None

                # ---------------- Render plane content ------------------------
                if state.mode == AppState.LIVE:
                    # Re-stylize the live crop every frame with the current
                    # style's fast filter -> instant AR-lens feel, no lag.
                    live_crop = quad_to_capture_rect(
                        frame, smoothed_quad, out_size=(args.live_size, args.live_size)
                    )
                    style = get_style(state.style_idx)
                    stylized = AsyncFluxEngine._run_fallback_filter(live_crop, style)
                    plane.set_image(stylized)
                    plane.render(frame, smoothed_quad)
                    status_text = ("Quick pinch = next filter   |   "
                                   f"hold pinch {args.hold_threshold:.1f}s = AI render")

                elif state.mode == AppState.GENERATING:
                    live_crop = quad_to_capture_rect(
                        frame, smoothed_quad, out_size=(args.live_size, args.live_size)
                    )
                    style = get_style(state.style_idx)
                    stylized = AsyncFluxEngine._run_fallback_filter(live_crop, style)
                    plane.set_image(stylized)
                    plane.render(frame, smoothed_quad, opacity=0.6)
                    ui_overlay.draw_quad_loading_overlay(frame, smoothed_quad,
                                                          label="Generating AI style...")
                    status_text = "Running async AI inference..."

                elif state.mode == AppState.AI_RESULT:
                    plane.set_image(state.ai_result_image)
                    plane.render(frame, smoothed_quad)
                    status_text = "Quick pinch for next filter, hold pinch to re-render with AI"

            else:
                # Gesture lost: reset to idle so a fresh frame gesture
                # starts clean, and clear any in-progress pinch tracking.
                if state.mode != AppState.GENERATING:
                    state.mode = AppState.IDLE
                state.pinch_active = False

            fps = fps_meter.tick()
            style_name = get_style(state.style_idx)["name"]
            ui_overlay.draw_hud(frame, style_name, status_text, fps)
            ui_overlay.draw_instructions(frame)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(']'):
                state.style_idx = (state.style_idx + 1) % style_count()
                if state.mode == AppState.AI_RESULT:
                    state.mode = AppState.LIVE
            elif key == ord('['):
                state.style_idx = (state.style_idx - 1) % style_count()
                if state.mode == AppState.AI_RESULT:
                    state.mode = AppState.LIVE
            elif key == ord('r'):
                state.mode = AppState.LIVE if quad is not None else AppState.IDLE
                state.ai_result_image = None

    finally:
        engine.stop()
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()