import math
import time
import cv2
import numpy as np


def draw_loading_spinner(frame, center, radius=28, color=(255, 255, 255), thickness=3):
    """Draws a rotating arc spinner centered at `center` (x, y)."""
    t = time.time()
    start_angle = (t * 320) % 360
    sweep = 110
    cv2.ellipse(
        frame, center, (radius, radius), 0,
        start_angle, start_angle + sweep, color, thickness, cv2.LINE_AA
    )
    # faint full ring for context
    cv2.circle(frame, center, radius, (color[0], color[1], color[2]), 1, cv2.LINE_AA)
    return frame


def draw_quad_loading_overlay(frame, quad_pts, label="Generating..."):
    """Dims the quad region and draws a centered spinner + label."""
    quad_pts = quad_pts.astype(np.int32)
    overlay = frame.copy()
    cv2.fillConvexPoly(overlay, quad_pts, (20, 20, 20))
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, dst=frame)

    cx = int(np.mean(quad_pts[:, 0]))
    cy = int(np.mean(quad_pts[:, 1]))
    draw_loading_spinner(frame, (cx, cy))

    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    cv2.putText(
        frame, label, (cx - text_size[0] // 2, cy + 50),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
    )
    return frame


def draw_hud(frame, style_name, status_text, fps):
    h, w = frame.shape[:2]
    banner_h = 64
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)

    cv2.putText(frame, f"Style: {style_name}", (16, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, status_text, (16, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{fps:.1f} FPS", (w - 110, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 2, cv2.LINE_AA)
    return frame


def draw_instructions(frame):
    h, w = frame.shape[:2]
    lines = [
        "Form an L-frame with both hands to open the live filter panel",
        "Quick pinch (right hand) = next filter  |  hold pinch = AI render  |  [ ] cycle  |  Q quit",
    ]
    y = h - 46
    for line in lines:
        cv2.putText(frame, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        y += 22
    return frame