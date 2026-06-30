"""
perspective.py
---------------
Handles warping a source image (webcam capture or AI-generated style)
onto a quadrilateral defined by the user's hand-frame gesture, and
compositing it back onto the live video feed with alpha blending so it
reads as a floating "pane of glass" pinned between the hands -- exactly
like the reference clips, where the panel tilts/scales/rotates with the
hands in real time.
"""

import cv2
import numpy as np


class FloatingPlane:
    """
    Holds the current source image (the thing being displayed on the
    floating plane) and warps it into the live frame each tick.
    """

    def __init__(self):
        self.source_image = None  # BGR or BGRA numpy array
        self.alpha = 1.0          # overall opacity, used for fade in/out

    def set_image(self, image_bgr_or_bgra):
        self.source_image = image_bgr_or_bgra

    def render(self, frame, quad_pts, corner_smoother=None, t=None, opacity=None):
        """
        Warp self.source_image into quad_pts (TL, TR, BR, BL pixel coords)
        and alpha-composite onto `frame` in place. Returns the frame.
        """
        if self.source_image is None or quad_pts is None:
            return frame

        if corner_smoother is not None:
            quad_pts = corner_smoother.smooth(quad_pts, t=t)

        h, w = frame.shape[:2]
        src = self.source_image
        src_h, src_w = src.shape[:2]

        src_corners = np.array(
            [[0, 0], [src_w - 1, 0], [src_w - 1, src_h - 1], [0, src_h - 1]],
            dtype=np.float32,
        )
        dst_corners = quad_pts.astype(np.float32)

        M = cv2.getPerspectiveTransform(src_corners, dst_corners)

        # Ensure we have an alpha channel to warp along with the color,
        # so areas outside the quad don't get drawn (clean edges).
        if src.shape[2] == 3:
            src_rgba = cv2.cvtColor(src, cv2.COLOR_BGR2BGRA)
            src_rgba[:, :, 3] = 255
        else:
            src_rgba = src.copy()

        warped = cv2.warpPerspective(
            src_rgba, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

        alpha = (warped[:, :, 3].astype(np.float32) / 255.0)
        op = self.alpha if opacity is None else opacity
        alpha *= np.clip(op, 0.0, 1.0)
        alpha_3 = alpha[:, :, None]

        warped_bgr = warped[:, :, :3].astype(np.float32)
        frame_f = frame.astype(np.float32)

        composited = warped_bgr * alpha_3 + frame_f * (1 - alpha_3)
        frame[:] = composited.astype(np.uint8)

        # Subtle "glass" border so the panel reads as a distinct surface,
        # matching the bordered look of the floating panel in the clips.
        pts_i = dst_corners.astype(np.int32)
        cv2.polylines(frame, [pts_i], isClosed=True,
                       color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

        return frame


def quad_to_capture_rect(frame, quad_pts, out_size=(512, 512)):
    """
    Inverse operation: given the current frame and the hand-frame quad,
    extract (warp) that region of the webcam image into a clean
    axis-aligned square, suitable to send to the style-transfer model.
    """
    out_w, out_h = out_size
    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    src = quad_pts.astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    capture = cv2.warpPerspective(frame, M, (out_w, out_h), flags=cv2.INTER_CUBIC)
    return capture


class QuadSmoother:
    """Light-weight smoother specifically for the 4 quad corner points."""

    def __init__(self, min_cutoff=1.0, beta=0.3):
        from smoothing import LandmarkSmoother
        self._smoother = LandmarkSmoother(4, min_cutoff=min_cutoff, beta=beta)

    def smooth(self, quad_pts, t=None):
        return self._smoother.smooth(quad_pts, t=t)
