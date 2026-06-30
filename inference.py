import os
import io
import time
import tempfile
import threading
import queue
import numpy as np
import cv2
from PIL import Image

_HAS_FAL = False
try:
    import fal_client  # type: ignore
    _HAS_FAL = True
except Exception:
    _HAS_FAL = False


class InferenceJob:
    """A single requested generation: input frame + target style."""

    __slots__ = ("job_id", "image_bgr", "style", "created_at")

    def __init__(self, job_id, image_bgr, style):
        self.job_id = job_id
        self.image_bgr = image_bgr
        self.style = style
        self.created_at = time.time()


class InferenceResult:
    __slots__ = ("job_id", "image_bgr", "elapsed", "error")

    def __init__(self, job_id, image_bgr=None, elapsed=0.0, error=None):
        self.job_id = job_id
        self.image_bgr = image_bgr
        self.elapsed = elapsed
        self.error = error


class AsyncFluxEngine:
    """
    Thread-backed async inference engine.

    Usage:
        engine = AsyncFluxEngine(fal_key="your-fal-key")
        engine.start()
        job_id = engine.submit(frame_bgr, style_dict)
        ... keep rendering ...
        result = engine.poll_result()  # non-blocking, returns None if not ready
    """

    FAL_EDIT_ENDPOINT = "fal-ai/flux-2/klein/4b/edit"

    def __init__(self, fal_key=None, model_path=None, device=None, strength=0.65,
                 guidance_scale=3.5, num_inference_steps=8, max_queue=1):
        # fal_key takes priority; FAL_KEY env var is also respected by
        # fal_client automatically if set.
        self.fal_key = fal_key or os.environ.get("FAL_KEY")
        if self.fal_key:
            os.environ["FAL_KEY"] = self.fal_key

        self.use_cloud = _HAS_FAL and bool(self.fal_key)

        self.strength = strength
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps

        self._in_q: "queue.Queue[InferenceJob]" = queue.Queue(maxsize=max_queue)
        self._out_q: "queue.Queue[InferenceResult]" = queue.Queue()
        self._thread = None
        self._stop_flag = threading.Event()
        self._busy = False
        self._next_job_id = 0
        self._lock = threading.Lock()

    # --------------------------- lifecycle --------------------------

    def start(self):
        if not self.use_cloud:
            if not _HAS_FAL:
                print("[inference] fal_client not installed -- held-pinch AI "
                      "render will use the fast filter instead. "
                      "Run: pip install fal-client")
            elif not self.fal_key:
                print("[inference] No FAL_KEY set -- held-pinch AI render will "
                      "use the fast filter instead. Pass --fal_key YOUR_KEY.")
        self._stop_flag.clear()
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_flag.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def is_busy(self):
        return self._busy

    # --------------------------- API ---------------------------------

    def submit(self, frame_bgr, style):
        """
        Non-blocking submit. Drops the request if the engine is already
        busy/queue is full, so we never build up backlog/latency -- the
        UI should treat "still busy" as "show loading animation".
        Returns job_id if accepted, else None.
        """
        with self._lock:
            job_id = self._next_job_id
            self._next_job_id += 1
        job = InferenceJob(job_id, frame_bgr.copy(), style)
        try:
            self._in_q.put_nowait(job)
            return job_id
        except queue.Full:
            return None

    def poll_result(self):
        """Non-blocking; returns an InferenceResult or None."""
        try:
            return self._out_q.get_nowait()
        except queue.Empty:
            return None

    # --------------------------- worker -------------------------------

    def _worker_loop(self):
        while not self._stop_flag.is_set():
            try:
                job = self._in_q.get(timeout=0.1)
            except queue.Empty:
                continue

            self._busy = True
            t0 = time.time()
            try:
                out_bgr = self._run_inference(job.image_bgr, job.style)
                elapsed = time.time() - t0
                self._out_q.put(InferenceResult(job.job_id, out_bgr, elapsed))
            except Exception as e:  # noqa: BLE001
                self._out_q.put(InferenceResult(job.job_id, error=str(e)))
            finally:
                self._busy = False

    def _run_inference(self, image_bgr, style):
        if self.use_cloud:
            try:
                return self._run_flux_cloud(image_bgr, style)
            except Exception as e:  # noqa: BLE001
                print(f"[inference] fal.ai request failed ({e}); "
                      f"falling back to fast filter for this frame.")
                return self._run_fallback_filter(image_bgr, style)
        return self._run_fallback_filter(image_bgr, style)

    # ---------------- Real AI via fal.ai hosted FLUX.2 [klein] --------

    def _run_flux_cloud(self, image_bgr, style):
        """
        Sends the captured frame to fal.ai's hosted FLUX.2 [klein] 4B
        edit endpoint and returns the styled result. Runs entirely on
        fal's GPUs -- no local GPU/model download required.
        """
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # fal_client needs a local file path (or URL) to upload.
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            pil_img.save(tmp.name, format="PNG")
            tmp_path = tmp.name

        try:
            image_url = fal_client.upload_file(tmp_path)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        result = fal_client.subscribe(
            self.FAL_EDIT_ENDPOINT,
            arguments={
                "prompt": style["prompt"],
                "image_urls": [image_url],
            },
        )

        out_url = result["images"][0]["url"]
        import urllib.request
        with urllib.request.urlopen(out_url, timeout=30) as resp:
            out_bytes = resp.read()

        out_pil = Image.open(io.BytesIO(out_bytes)).convert("RGB")
        out_rgb = np.array(out_pil)
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
        return out_bgr

    # ---------------- CPU fallback (instant live filter mode) ---------

    _face_cascade = None
    _bg_cache = {}  # (kind, w, h) -> cached procedural background (huge speedup)
    _grid_cache = {}  # (w, h) -> cached thermal grid overlay

    @classmethod
    def _get_face_mask(cls, img):
        """Soft elliptical mask around the detected face (or a sane
        center-frame guess if no face found), used to keep the real
        face visible while the background gets the heavy stylization.
        Detection runs on a small downscaled copy -- the Haar cascade is
        the slow part, and detecting at e.g. 160px wide is just as
        reliable for this purpose as full resolution but far cheaper."""
        if cls._face_cascade is None:
            cls._face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
        h, w = img.shape[:2]
        det_w = 160
        scale = det_w / w
        small_gray = cv2.cvtColor(
            cv2.resize(img, (det_w, int(h * scale)), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2GRAY,
        )
        sh, sw = small_gray.shape[:2]
        faces = cls._face_cascade.detectMultiScale(small_gray, 1.15, 5, minSize=(sw // 6, sh // 6))
        if len(faces) > 0:
            x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            x, y, fw, fh = x / scale, y / scale, fw / scale, fh / scale
            cx, cy = x + fw / 2, y + fh / 2
            rx, ry = fw * 0.85, fh * 1.15
        else:
            cx, cy = w / 2, h * 0.48
            rx, ry = w * 0.46, h * 0.56

        mask = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(mask, (int(cx), int(cy)), (int(rx), int(ry)), 0, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(w, h) * 0.03)
        return mask[:, :, None]

    @classmethod
    def _neon_city_bg(cls, w, h, t=0):
        """Procedural neon-skyline background: dark navy base, glowing
        vertical 'building' bands in magenta/cyan, faint horizon haze.
        Cached per resolution since the layout is static -- regenerating
        this from scratch every frame is what made the live filter slow."""
        key = ("neon", w, h)
        if key in cls._bg_cache:
            return cls._bg_cache[key]

        bg = np.zeros((h, w, 3), dtype=np.uint8)
        bg[:] = (40, 10, 5)  # deep blue-black (BGR)
        rng = np.random.RandomState(7)
        n_bldgs = 14
        xs = np.linspace(0, w, n_bldgs, endpoint=False).astype(int)
        palette = [(255, 60, 200), (255, 200, 40), (255, 120, 60), (200, 255, 255)]
        for i, x0 in enumerate(xs):
            bw = int(w / n_bldgs * rng.uniform(0.5, 0.95))
            bh = int(h * rng.uniform(0.35, 0.95))
            color = palette[i % len(palette)]
            cv2.rectangle(bg, (x0, h - bh), (x0 + bw, h), (color[0]//4, color[1]//4, color[2]//4), -1)
            # glowing window strip down the middle of each building
            cv2.line(bg, (x0 + bw // 2, h - bh), (x0 + bw // 2, h), color, 2, cv2.LINE_AA)
            for wy in range(h - bh + 6, h, 10):
                if rng.rand() > 0.4:
                    cv2.rectangle(bg, (x0 + 3, wy), (x0 + bw - 3, wy + 3), color, -1)
        glow = cv2.GaussianBlur(bg, (0, 0), sigmaX=9)
        bg = cv2.addWeighted(bg, 0.55, glow, 0.75, 0)
        # rain-streak vertical gradient haze
        haze = np.tile(np.linspace(0.15, 0.55, h, dtype=np.float32)[::-1, None, None], (1, w, 3))
        bg = (bg.astype(np.float32) * (0.6 + haze)).clip(0, 255).astype(np.uint8)

        cls._bg_cache[key] = bg
        return bg

    @classmethod
    def _psychedelic_bg(cls, w, h, seed=3):
        """Procedural rainbow swirl-noise background, like a trippy
        kaleidoscope poster backdrop. Cached per resolution -- this is
        the single biggest cost in the live filter loop otherwise."""
        key = ("psy", w, h, seed)
        if key in cls._bg_cache:
            return cls._bg_cache[key]

        rng = np.random.RandomState(seed)
        small = rng.rand(24, 24).astype(np.float32)
        noise = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = w / 2, h / 2
        angle = np.arctan2(yy - cy, xx - cx)
        radius = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        swirl = (angle / (2 * np.pi) + radius / max(w, h) * 2.0 + noise * 0.6)
        hue = (swirl * 179.0) % 179.0
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[..., 0] = hue.astype(np.uint8)
        hsv[..., 1] = 230
        hsv[..., 2] = 220
        bg = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        bg = cv2.GaussianBlur(bg, (0, 0), sigmaX=1.2)

        cls._bg_cache[key] = bg
        return bg

    @classmethod
    def _thermal_grid(cls, w, h):
        """Cached grid-line overlay for the thermal style (was being
        redrawn line-by-line every frame -- now built once per size)."""
        key = (w, h)
        if key in cls._grid_cache:
            return cls._grid_cache[key]
        grid = np.zeros((h, w, 3), dtype=np.uint8)
        step = max(8, w // 18)
        for x in range(0, w, step):
            cv2.line(grid, (x, 0), (x, h), (255, 60, 200), 1, cv2.LINE_AA)
        step_h = max(8, h // 14)
        for y in range(0, h, step_h):
            cv2.line(grid, (0, y), (w, y), (255, 60, 200), 1, cv2.LINE_AA)
        cls._grid_cache[key] = grid
        return grid

    @staticmethod
    def _process_at_scale(img, fn, max_dim=160):
        """
        Runs a slow per-pixel stylization (oil painting, pencil sketch,
        mean-shift segmentation, etc.) on a downscaled copy of the image,
        then upsamples the result back to full size. These filters look
        almost identical at reduced resolution but cost 5-10x less, which
        is what brings them under the live-filter frame budget.
        """
        h, w = img.shape[:2]
        scale = max_dim / max(h, w)
        if scale >= 1.0:
            return fn(img)
        small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        small_out = fn(small)
        return cv2.resize(small_out, (w, h), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _run_fallback_filter(image_bgr, style):
        """
        Deterministic OpenCV-only stylization used for the always-live
        quick-pinch filter mode (and as a safety fallback if the cloud
        AI call fails). Mimics the *category* of the requested style
        well enough for instant, every-frame re-styling.
        """
        name = style["name"].lower()
        img = image_bgr.copy()
        h, w = img.shape[:2]

        if "thermal" in name:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            out = cv2.applyColorMap(gray, cv2.COLORMAP_TWILIGHT)
            grid = AsyncFluxEngine._thermal_grid(w, h)
            out = cv2.addWeighted(out, 0.85, grid, 0.35, 0)
            hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[..., 1] = np.clip(hsv[..., 1] * 1.3, 0, 255)
            out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        elif "pencil" in name or "charcoal" in name:
            def _sketch(im):
                gray, _ = cv2.pencilSketch(im, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
                return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            out = AsyncFluxEngine._process_at_scale(img, _sketch, max_dim=180)
        elif "watercolor" in name or "ghibli" in name or "anime" in name:
            smooth = cv2.bilateralFilter(img, d=9, sigmaColor=200, sigmaSpace=200)
            smooth = cv2.bilateralFilter(smooth, d=9, sigmaColor=200, sigmaSpace=200)

            levels = 6
            quant = (smooth.astype(np.float32) / 255.0 * levels)
            quant = np.round(quant) / levels * 255.0
            quant = quant.astype(np.uint8)

            hsv = cv2.cvtColor(quant, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[..., 1] = np.clip(hsv[..., 1] * 1.25, 0, 255)
            hsv[..., 2] = np.clip(hsv[..., 2] * 1.08 + 6, 0, 255)
            quant = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_blur = cv2.medianBlur(gray, 5)
            edges = cv2.adaptiveThreshold(
                gray_blur, 255,
                cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 6
            )
            edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

            out = cv2.bitwise_and(quant, edges_3ch)
        elif "comic" in name or "graffiti" in name:
            edges = cv2.adaptiveThreshold(
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 255,
                cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2
            )
            color = cv2.bilateralFilter(img, 9, 250, 250)
            color = cv2.convertScaleAbs(color, alpha=1.4, beta=10)
            out = cv2.bitwise_and(color, color, mask=edges)
        elif "pop art" in name:
            smooth = cv2.bilateralFilter(img, d=11, sigmaColor=300, sigmaSpace=300)
            levels = 6
            quant = np.round(smooth.astype(np.float32) / 255.0 * levels) / levels * 255.0
            quant = quant.astype(np.uint8)
            hsv = cv2.cvtColor(quant, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[..., 1] = np.clip(hsv[..., 1] * 2.1, 0, 255)
            hsv[..., 2] = np.clip(hsv[..., 2] * 1.25 + 15, 0, 255)
            quant = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            edges = cv2.adaptiveThreshold(
                cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 7), 255,
                cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 7
            )
            ink = 255 - edges  # thin black linework only on strong edges
            out = quant.copy()
            out[ink > 0] = (out[ink > 0] * 0.15).astype(np.uint8)
        elif "psychedelic" in name or "swirl" in name:
            face_mask = AsyncFluxEngine._get_face_mask(img)
            bg = AsyncFluxEngine._psychedelic_bg(w, h)

            def _face_color(im):
                fc = cv2.detailEnhance(im, sigma_s=15, sigma_r=0.4)
                hsv = cv2.cvtColor(fc, cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[..., 1] = np.clip(hsv[..., 1] * 1.6, 0, 255)
                fc = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
                edges = cv2.Canny(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), 60, 140)
                fc[edges > 0] = (255, 255, 255)
                return fc

            face_color = AsyncFluxEngine._process_at_scale(img, _face_color, max_dim=200)
            out = (face_color.astype(np.float32) * face_mask +
                   bg.astype(np.float32) * (1 - face_mask)).astype(np.uint8)
        elif "neon" in name or "cyberpunk" in name or "vaporwave" in name:
            if "cyberpunk" in name or "neon" in name:
                face_mask = AsyncFluxEngine._get_face_mask(img)
                bg = AsyncFluxEngine._neon_city_bg(w, h)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
                lo = np.array([60, 10, 90], dtype=np.float32)
                hi = np.array([255, 120, 230], dtype=np.float32)
                t = (gray[..., None] / 255.0)
                face_duo = (lo * (1 - t) + hi * t).astype(np.uint8)
                edges = cv2.Canny(img, 70, 150)
                edges_color = cv2.applyColorMap(edges, cv2.COLORMAP_COOL)
                face_color = cv2.addWeighted(face_duo, 0.7, edges_color, 0.5, 0)
                out = (face_color.astype(np.float32) * face_mask +
                       bg.astype(np.float32) * (1 - face_mask)).astype(np.uint8)
                blur = cv2.GaussianBlur(out, (0, 0), sigmaX=4)
                out = cv2.addWeighted(out, 0.78, blur, 0.4, 0)
            else:
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[..., 1] = np.clip(hsv[..., 1] * 1.8, 0, 255)
                hsv[..., 2] = np.clip(hsv[..., 2] * 1.1, 0, 255)
                sat = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
                edges = cv2.Canny(img, 80, 160)
                edges_color = cv2.applyColorMap(edges, cv2.COLORMAP_COOL)
                out = cv2.addWeighted(sat, 0.75, edges_color, 0.5, 0)
        elif "oil" in name or "van gogh" in name:
            def _oil(im):
                if hasattr(cv2, "xphoto"):
                    return cv2.xphoto.oilPainting(im, 5, 1)
                return cv2.stylization(im, sigma_s=60, sigma_r=0.4)
            out = AsyncFluxEngine._process_at_scale(img, _oil, max_dim=180)
        elif "low poly" in name or "pixel" in name:
            small = cv2.resize(img, (32, 32), interpolation=cv2.INTER_LINEAR)
            out = cv2.resize(small, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        elif "stained glass" in name:
            def _glass(im):
                seg = cv2.pyrMeanShiftFiltering(im, 10, 25)
                edges = cv2.Canny(seg, 50, 120)
                seg[edges > 0] = (10, 10, 10)
                return seg
            out = AsyncFluxEngine._process_at_scale(img, _glass, max_dim=160)
        else:  # generic fallback
            out = cv2.detailEnhance(img, sigma_s=20, sigma_r=0.25)

        return out