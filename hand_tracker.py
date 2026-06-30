import math
import numpy as np
import mediapipe as mp


WRIST = 0
THUMB_TIP = 4
THUMB_IP = 3
INDEX_MCP = 5
INDEX_TIP = 8
MIDDLE_TIP = 12
MIDDLE_MCP = 9
RING_TIP = 16
PINKY_TIP = 20


class HandState:
    """Per-hand smoothed landmark data and derived geometry."""

    def __init__(self, label):
        self.label = label  # "Left" or "Right" (as reported by MediaPipe, mirrored)
        self.landmarks_px = None  # (21, 2) smoothed pixel coords
        self.present = False


class HandTracker:
    def __init__(self, max_hands=2, det_conf=0.6, track_conf=0.6,
                 smooth_min_cutoff=1.2, smooth_beta=0.4):
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=det_conf,
            min_tracking_confidence=track_conf,
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

        # One smoother per tracked hand slot (left/right), persistent across frames
        from smoothing import LandmarkSmoother
        self._smoothers = {
            "Left": LandmarkSmoother(21, min_cutoff=smooth_min_cutoff, beta=smooth_beta),
            "Right": LandmarkSmoother(21, min_cutoff=smooth_min_cutoff, beta=smooth_beta),
        }

        self.hands = {"Left": HandState("Left"), "Right": HandState("Right")}

    def close(self):
        self._hands.close()

    def process(self, frame_bgr, t=None):
        """
        Run MediaPipe on a BGR frame. Updates self.hands in place.
        Returns the raw MediaPipe result (useful for debug drawing).
        """
        h, w = frame_bgr.shape[:2]
        frame_rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        frame_rgb.flags.writeable = False
        result = self._hands.process(frame_rgb)

        # Reset presence each frame; we'll mark present hands below.
        for hs in self.hands.values():
            hs.present = False

        if result.multi_hand_landmarks and result.multi_handedness:
            for lm_set, handedness in zip(result.multi_hand_landmarks,
                                           result.multi_handedness):
                label = handedness.classification[0].label  # "Left"/"Right"
                pts = np.array(
                    [[p.x * w, p.y * h] for p in lm_set.landmark],
                    dtype=np.float64,
                )
                smoothed = self._smoothers[label].smooth(pts, t=t)
                hs = self.hands[label]
                hs.landmarks_px = smoothed
                hs.present = True

        return result

    # ---------------------- Gesture detection ---------------------------

    def get_frame_gesture(self):
        """
        SIMPLE MODE: triggers as soon as both hands are visible in frame --
        no strict L-shape / angle checks. Builds the floating-plane quad
        from each hand's thumb-tip and index-tip positions, so just
        raising both hands up (in roughly an L or open shape) on either
        side of your face is enough to open the panel.
        """
        left, right = self.hands.get("Left"), self.hands.get("Right")
        if not (left and right and left.present and right.present):
            return None

        l_thumb = left.landmarks_px[THUMB_TIP]
        l_index = left.landmarks_px[INDEX_TIP]
        r_thumb = right.landmarks_px[THUMB_TIP]
        r_index = right.landmarks_px[INDEX_TIP]

        # Build 4 corners from the 4 fingertip anchors. We sort them
        # spatially so the quad stays consistently ordered even as hands
        # rotate, which keeps the perspective warp stable (no flipping).
        pts = np.array([l_thumb, l_index, r_thumb, r_index])
        return self._order_quad(pts)

    @staticmethod
    def _is_l_shape(lm, angle_thresh_deg=35):
        """
        Heuristic: index finger extended and roughly perpendicular to the
        thumb, with the other three fingers curled -- i.e. an 'L' shape.
        """
        if lm is None:
            return False

        wrist = lm[WRIST]
        thumb_tip, thumb_ip = lm[THUMB_TIP], lm[THUMB_IP]
        index_tip, index_mcp = lm[INDEX_TIP], lm[INDEX_MCP]
        middle_tip, middle_mcp = lm[MIDDLE_TIP], lm[MIDDLE_MCP]

        # Index must be clearly extended (tip far from its MCP relative to wrist scale)
        hand_scale = np.linalg.norm(index_mcp - wrist) + 1e-6
        index_extended = np.linalg.norm(index_tip - index_mcp) / hand_scale > 1.0

        # Middle/ring/pinky should be curled relative to wrist (closer to palm)
        middle_curled = np.linalg.norm(middle_tip - wrist) < np.linalg.norm(middle_mcp - wrist) * 1.6

        # Thumb extended outward from index (forming the L corner)
        thumb_vec = thumb_tip - thumb_ip
        index_vec = index_tip - index_mcp
        thumb_extended = np.linalg.norm(thumb_vec) > 1e-3

        if not (index_extended and thumb_extended):
            return False

        # Angle between thumb and index directions should be roughly 60-130 deg
        cos_ang = np.dot(thumb_vec, index_vec) / (
            np.linalg.norm(thumb_vec) * np.linalg.norm(index_vec) + 1e-6
        )
        cos_ang = np.clip(cos_ang, -1.0, 1.0)
        angle = math.degrees(math.acos(cos_ang))
        l_angle_ok = 40 <= angle <= 150

        return l_angle_ok and middle_curled

    @staticmethod
    def _order_quad(pts):
        """Order 4 points as TL, TR, BR, BL using sum/diff heuristic."""
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).ravel()
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(d)]
        bl = pts[np.argmax(d)]
        return np.array([tl, tr, br, bl], dtype=np.float64)

    def get_pinch(self, hand_label="Right", thresh_ratio=0.45):
        """
        Returns True if thumb tip and index tip of the given hand are
        touching (pinch gesture), normalized by hand size so it works at
        any distance from the camera.
        """
        hs = self.hands.get(hand_label)
        if not hs or not hs.present:
            return False
        lm = hs.landmarks_px
        thumb_tip, index_tip = lm[THUMB_TIP], lm[INDEX_TIP]
        wrist, index_mcp = lm[WRIST], lm[INDEX_MCP]
        hand_scale = np.linalg.norm(index_mcp - wrist) + 1e-6
        dist = np.linalg.norm(thumb_tip - index_tip)
        return (dist / hand_scale) < thresh_ratio

    def get_swipe_hand_center(self, hand_label="Left"):
        """Returns the palm-center pixel position for a hand, or None."""
        hs = self.hands.get(hand_label)
        if not hs or not hs.present:
            return None
        return hs.landmarks_px[MIDDLE_MCP]