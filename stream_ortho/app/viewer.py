"""Non-blocking live preview window for streaming orthophotos (`--show`)."""
from __future__ import annotations

import os
import queue
import threading
import numpy as np


class LiveOrthoViewer:
    """Renders incremental orthophoto tile snapshots in a dedicated OpenCV thread.

    Safe on headless systems: if OpenCV is unavailable or no GUI display is
    present, the viewer disables itself cleanly without affecting the pipeline.
    """

    def __init__(self, window_name: str = "Ortho-3DGS Live", max_w: int = 1400, max_h: int = 900):
        self.window_name = str(window_name)
        self.max_w = int(max_w)
        self.max_h = int(max_h)
        self._q: queue.Queue = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._disabled = False
        self._thread: threading.Thread | None = None
        if os.name != "nt" and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            self._disabled = True
            print("[viewer] no DISPLAY/WAYLAND_DISPLAY detected; --show preview disabled", flush=True)
            return
        self._thread = threading.Thread(target=self._run, name="ortho-live-viewer", daemon=True)
        self._thread.start()

    @property
    def active(self) -> bool:
        return (not self._disabled) and (not self._stop.is_set())

    def submit(self, tiles_snap, stats_text: str = ""):
        """Queue the latest tile snapshot, dropping any unrendered older snapshot."""
        if not self.active or not tiles_snap:
            return
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        try:
            self._q.put_nowait((tiles_snap, str(stats_text)))
        except queue.Full:
            pass

    def _render_canvas(self, tiles_snap, stats_text: str, cv2):
        from stream_ortho.app.mosaic_writer import stitch_arrays

        rgb, alpha = stitch_arrays(tiles_snap)
        if rgb is None:
            return None
        occ = np.nonzero(alpha > 0.02)
        if occ[0].size:
            r0, r1 = int(occ[0].min()), int(occ[0].max()) + 1
            c0, c1 = int(occ[1].min()), int(occ[1].max()) + 1
            rgb = rgb[r0:r1, c0:c1]
            alpha = alpha[r0:r1, c0:c1]

        bg = np.array([18.0 / 255.0, 22.0 / 255.0, 28.0 / 255.0], dtype=np.float32)
        mask = (alpha > 0.02)[..., None]
        rgb_vis = np.where(mask, np.clip(rgb, 0.0, 1.0), bg)
        u8_rgb = (rgb_vis * 255.0 + 0.5).astype(np.uint8)
        bgr = cv2.cvtColor(u8_rgb, cv2.COLOR_RGB2BGR)

        h, w = bgr.shape[:2]
        scale = min(self.max_w / max(w, 1), self.max_h / max(h, 1), 1.0)
        if scale < 1.0:
            nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
            bgr = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        elif max(h, w) < 400:
            up = min(400.0 / max(h, w), 3.0)
            bgr = cv2.resize(bgr, (int(round(w * up)), int(round(h * up))),
                             interpolation=cv2.INTER_NEAREST)

        if stats_text:
            cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 30), (16, 20, 26), thickness=-1)
            cv2.putText(
                bgr, stats_text, (10, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 240, 250), 1, cv2.LINE_AA,
            )
        return bgr

    def _run(self):
        try:
            import cv2
        except Exception as exc:
            self._disabled = True
            print(f"[viewer] OpenCV unavailable ({exc}); --show preview disabled", flush=True)
            return

        created = False
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            created = True
            while not self._stop.is_set():
                latest = None
                try:
                    latest = self._q.get(timeout=0.04)
                    while True:
                        try:
                            latest = self._q.get_nowait()
                        except queue.Empty:
                            break
                except queue.Empty:
                    pass

                if latest is not None:
                    tiles_snap, stats_text = latest
                    canvas = self._render_canvas(tiles_snap, stats_text, cv2)
                    if canvas is not None:
                        cv2.imshow(self.window_name, canvas)

                key = cv2.waitKey(15) & 0xFF
                if key in (ord("q"), 27):
                    self._stop.set()
                    break
                if created and cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    self._stop.set()
                    break
        except Exception as exc:
            self._disabled = True
            print(f"[viewer] GUI error ({exc}); --show preview disabled", flush=True)
        finally:
            if created:
                try:
                    cv2.destroyWindow(self.window_name)
                    cv2.waitKey(1)
                except Exception:
                    pass

    def close(self):
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
