"""
smoothing.py
------------
Temporal filtering utilities used to stabilize MediaPipe hand landmarks
frame-to-frame. We use a One-Euro Filter, which adapts its smoothing
strength to the speed of motion: it smooths heavily when the hand is
nearly still (killing jitter) and relaxes when the hand moves fast
(killing lag). This is the standard choice for low-latency,
low-jitter point tracking in AR-style interactions.
"""

import time
import math
import numpy as np


class _LowPassFilter:
    """Simple exponential low-pass filter, the building block of OneEuro."""

    def __init__(self):
        self._initialized = False
        self._x_prev = None

    def filter(self, x, alpha):
        if not self._initialized:
            self._x_prev = x
            self._initialized = True
            return x
        x_hat = alpha * x + (1.0 - alpha) * self._x_prev
        self._x_prev = x_hat
        return x_hat


class OneEuroFilter:
    """
    One-Euro Filter for a scalar or numpy-array signal.

    min_cutoff: lower -> more smoothing at low speed (less jitter)
    beta:       higher -> less lag at high speed (more responsive)
    d_cutoff:   cutoff for the derivative estimate
    """

    def __init__(self, min_cutoff=1.0, beta=0.02, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self._x_filter = _LowPassFilter()
        self._dx_filter = _LowPassFilter()
        self._t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def filter(self, x, t=None):
        x = np.asarray(x, dtype=np.float64)
        if t is None:
            t = time.time()

        if self._t_prev is None:
            dt = 1.0 / 30.0
        else:
            dt = max(t - self._t_prev, 1e-6)
        self._t_prev = t

        # Estimate derivative of the signal
        if self._x_filter._x_prev is None:
            dx = np.zeros_like(x)
        else:
            dx = (x - self._x_filter._x_prev) / dt

        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = self._dx_filter.filter(dx, a_d)

        speed = np.linalg.norm(dx_hat)
        cutoff = self.min_cutoff + self.beta * speed
        a = self._alpha(cutoff, dt)
        x_hat = self._x_filter.filter(x, a)
        return x_hat


class LandmarkSmoother:
    """
    Wraps one OneEuroFilter per landmark index for a full 21-point hand
    (or any fixed-size point set), so each point is smoothed independently
    but consistently.
    """

    def __init__(self, num_points, min_cutoff=1.0, beta=0.02, d_cutoff=1.0):
        self.filters = [
            OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
            for _ in range(num_points)
        ]

    def smooth(self, points, t=None):
        """points: (N, 2) or (N, 3) numpy array -> smoothed array, same shape."""
        points = np.asarray(points, dtype=np.float64)
        out = np.zeros_like(points)
        for i in range(points.shape[0]):
            out[i] = self.filters[i].filter(points[i], t=t)
        return out

    def reset(self):
        for f in self.filters:
            f._x_filter = _LowPassFilter()
            f._dx_filter = _LowPassFilter()
            f._t_prev = None
