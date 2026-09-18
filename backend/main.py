"""
CV Pick extension backend.

Real-time color and shape detection with a learning workflow.
A background thread captures webcam frames, runs detection for every learned
item, draws bounding-box overlays, and exposes the result as an MJPEG stream.
"""

import base64
import json
import os
import platform
import subprocess
import threading
import time
import uuid

import cv2
import numpy as np
from flask import Blueprint, Response, jsonify, request

blueprint = Blueprint("cv_pick", __name__)

# Distinct BGR colours assigned round-robin to learned items
_CANDIDATE_RESOLUTIONS = [
    (320, 240), (640, 480), (800, 600), (1024, 768),
    (1280, 720), (1280, 960), (1920, 1080), (2560, 1440),
    (3840, 2160),
]

_PALETTE = [
    (66, 133, 244),   # blue
    (52, 168, 83),    # green
    (234, 67, 53),    # red
    (251, 188, 4),    # yellow
    (0, 165, 255),    # orange
    (171, 71, 188),   # purple
    (0, 188, 212),    # cyan
    (255, 87, 34),    # deep-orange
]


# ---------------------------------------------------------------------------
#  Camera enumeration (friendly names when the OS provides them)
# ---------------------------------------------------------------------------

def _open_capture(index):
    """Open a capture with the platform backend that matches name order."""
    system = platform.system()
    backends = []
    if system == "Windows":
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF]
    elif system == "Darwin":
        backends = [cv2.CAP_AVFOUNDATION]
    for be in backends:
        cap = cv2.VideoCapture(index, be)
        if cap.isOpened():
            return cap
        cap.release()
    cap = cv2.VideoCapture(index)
    return cap


def _camera_names_windows():
    """DirectShow device names — index order matches OpenCV CAP_DSHOW."""
    try:
        from pygrabber.dshow_graph import FilterGraph
        names = FilterGraph().get_input_devices()
        if names:
            return list(names)
    except Exception:
        pass
    # Fallback: PnP camera/image devices (order may not match OpenCV)
    try:
        out = subprocess.check_output(
            [
                "powershell", "-NoProfile", "-Command",
                "Get-PnpDevice -Class Camera,Image -Status OK -ErrorAction SilentlyContinue "
                "| Select-Object -ExpandProperty FriendlyName",
            ],
            stderr=subprocess.DEVNULL,
            timeout=8,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        names = [ln.strip() for ln in out.splitlines() if ln.strip()]
        return names
    except Exception:
        return []


def _camera_names_macos():
    try:
        out = subprocess.check_output(
            ["system_profiler", "SPCameraDataType", "-json"],
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        data = json.loads(out.decode("utf-8", errors="replace"))
        names = []
        for cam in data.get("SPCameraDataType") or []:
            name = cam.get("_name")
            if name:
                names.append(str(name))
        return names
    except Exception:
        return []


def _camera_names_linux():
    names = []
    try:
        import glob
        paths = sorted(
            glob.glob("/sys/class/video4linux/video*"),
            key=lambda p: int(os.path.basename(p).replace("video", "") or 0),
        )
        for path in paths:
            # Skip metadata nodes when possible (index >= 64 often meta)
            base = os.path.basename(path)
            try:
                idx = int(base.replace("video", ""))
            except ValueError:
                idx = 0
            if idx >= 64:
                continue
            name_path = os.path.join(path, "name")
            try:
                with open(name_path, "r", encoding="utf-8", errors="replace") as f:
                    names.append(f.read().strip() or base)
            except OSError:
                names.append(base)
    except Exception:
        pass
    return names


def _platform_camera_names():
    system = platform.system()
    if system == "Windows":
        return _camera_names_windows()
    if system == "Darwin":
        return _camera_names_macos()
    if system == "Linux":
        return _camera_names_linux()
    return []


def enumerate_cameras(max_probe=10, skip_index=None):
    """
    Return [{index, name}, ...]. Prefer OS device names; fall back to probing
    indices 0..max_probe-1 as "Camera N".

    skip_index: if set, treat that index as available without re-opening
    (used when that camera is already held by the capture thread).
    """
    names = _platform_camera_names()
    if names:
        return [
            {"index": i, "name": (n.strip() if n and str(n).strip() else f"Camera {i}")}
            for i, n in enumerate(names)
        ]

    out = []
    for i in range(max_probe):
        if skip_index is not None and i == skip_index:
            out.append({"index": i, "name": f"Camera {i}"})
            continue
        cap = _open_capture(i)
        try:
            if cap is not None and cap.isOpened():
                out.append({"index": i, "name": f"Camera {i}"})
        finally:
            if cap is not None:
                cap.release()
    return out


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _put_label(img, text, x, y, color):
    """Draw *text* with a coloured background just above (x, y)."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x, y - th - 10), (x + tw + 6, y), color, -1)
    cv2.putText(img, text, (x + 3, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_preview(frame, display, mode, zx1, zy1, zx2, zy2):
    """Draw a live preview box for the candidate object in the learning zone."""
    zone = frame[zy1:zy2, zx1:zx2]
    zh, zw = zone.shape[:2]

    if mode == "color":
        my, mx = zh // 5, zw // 5
        sample = zone[my:zh - my, mx:zw - mx]
        if sample.size == 0:
            return
        hsv_sample = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
        med_h = float(np.median(hsv_sample[:, :, 0]))
        med_s = float(np.median(hsv_sample[:, :, 1]))
        med_v = float(np.median(hsv_sample[:, :, 2]))

        lower = np.array([max(0, med_h - 12), max(0, med_s - 55),
                          max(0, med_v - 55)], dtype=np.uint8)
        upper = np.array([min(179, med_h + 12), min(255, med_s + 55),
                          min(255, med_v + 55)], dtype=np.uint8)

        hsv_zone = cv2.cvtColor(zone, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_zone, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # derive the preview colour from the sampled median
        hsv_px = np.uint8([[[int(med_h), int(med_s), int(med_v)]]])
        bgr_px = cv2.cvtColor(hsv_px, cv2.COLOR_HSV2BGR)[0][0]
        pre_bgr = (int(bgr_px[0]), int(bgr_px[1]), int(bgr_px[2]))

        contours = [c for c in contours if cv2.contourArea(c) >= 200]
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            ox, oy = zx1 + x, zy1 + y
            cv2.rectangle(display, (ox, oy), (ox + w, oy + h),
                          (255, 255, 255), 1)
            _put_label(display, "candidate", ox, oy, pre_bgr)

    else:  # shape
        gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(blurred, 40, 120)
        edges = cv2.dilate(edges, None, iterations=2)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 100:
            return
        peri = cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, 0.04 * peri, True)
        nv = len(approx)
        _names = {3: "Triangle", 4: "Rectangle", 5: "Pentagon",
                  6: "Hexagon", 7: "Heptagon"}
        label = _names.get(
            nv, "Circle" if nv >= 8 else "{}-gon".format(nv))
        x, y, w, h = cv2.boundingRect(largest)
        ox, oy = zx1 + x, zy1 + y
        cv2.rectangle(display, (ox, oy), (ox + w, oy + h),
                      (255, 255, 255), 1)
        _put_label(display, "? " + label, ox, oy, (80, 80, 80))


# ---------------------------------------------------------------------------
#  NMS + detection routines
# ---------------------------------------------------------------------------

def _nms(candidates, iou_thresh=0.3):
    """Non-maximum suppression over (x, y, w, h, item, score) tuples.

    Sorted by *score* ascending (lower = better match).  For each box kept,
    any later box whose IoU exceeds *iou_thresh* is suppressed.
    """
    candidates.sort(key=lambda c: c[5])
    kept = []
    for cand in candidates:
        x, y, w, h = cand[:4]
        overlap = False
        for kx, ky, kw, kh, _, _ in kept:
            ix1, iy1 = max(x, kx), max(y, ky)
            ix2, iy2 = min(x + w, kx + kw), min(y + h, ky + kh)
            if ix1 < ix2 and iy1 < iy2:
                inter = (ix2 - ix1) * (iy2 - iy1)
                union = w * h + kw * kh - inter
                if union > 0 and inter / union > iou_thresh:
                    overlap = True
                    break
        if not overlap:
            kept.append(cand)
    return kept


def _draw_kept(display, kept, offset=(0, 0)):
    ox, oy = offset
    for x, y, w, h, item, _ in kept:
        bx, by = x + ox, y + oy
        bgr = tuple(item["display_color"])
        cv2.rectangle(display, (bx, by), (bx + w, by + h), bgr, 2)
        _put_label(display, item["name"], bx, by, bgr)


def _detect_colors(frame, display, colors, offset=(0, 0)):
    """Detect learned colours with winner-takes-all NMS.

    *frame* may be a ROI crop; *offset* shifts drawing coords back to
    full-frame space so bounding boxes appear in the right place.
    """
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fh, fw = frame.shape[:2]

    candidates = []
    for item in colors:
        lower = np.array(item["lower_hsv"], dtype=np.uint8)
        upper = np.array(item["upper_hsv"], dtype=np.uint8)
        mask = cv2.inRange(hsv_frame, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        med = item.get("median_hsv")
        if med is None:
            med = [(item["lower_hsv"][i] + item["upper_hsv"][i]) / 2.0
                   for i in range(3)]
        mh, ms, mv = float(med[0]), float(med[1]), float(med[2])

        for cnt in contours:
            if cv2.contourArea(cnt) < 500:
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = max(0, min(int(M["m10"] / M["m00"]), fw - 1))
            cy = max(0, min(int(M["m01"] / M["m00"]), fh - 1))
            ph = float(hsv_frame[cy, cx, 0])
            ps = float(hsv_frame[cy, cx, 1])
            pv = float(hsv_frame[cy, cx, 2])
            dist = abs(ph - mh) * 2.0 + abs(ps - ms) * 0.5 + abs(pv - mv) * 0.5
            x, y, w, h = cv2.boundingRect(cnt)
            candidates.append((x, y, w, h, item, dist))

    kept = _nms(candidates)
    _draw_kept(display, kept, offset)
    return len(kept)


def _detect_shapes(frame, display, shapes, offset=(0, 0)):
    """Detect learned shapes with winner-takes-all NMS."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 40, 120)
    edges = cv2.dilate(edges, None, iterations=2)
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 500:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        nv = len(approx)

        best_item, best_score = None, float("inf")
        for item in shapes:
            iv = item["vertex_count"]
            if iv < 8 and nv < 8 and abs(nv - iv) > 1:
                continue
            score = cv2.matchShapes(
                cnt, item["contour"], cv2.CONTOURS_MATCH_I1, 0)
            if score < 0.25 and score < best_score:
                best_score = score
                best_item = item

        if best_item is not None:
            x, y, w, h = cv2.boundingRect(cnt)
            candidates.append((x, y, w, h, best_item, best_score))

    kept = _nms(candidates)
    _draw_kept(display, kept, offset)
    return len(kept)


# ---------------------------------------------------------------------------
#  Shared pipeline state
# ---------------------------------------------------------------------------

class _CVState:
    def __init__(self):
        self.lock = threading.Lock()
        self.mode = "color"
        self.phase = "learning"      # "learning" or "inference"
        self.learned_colors = []
        self.learned_shapes = []
        self.roi = [0.25, 0.2, 0.75, 0.8]  # normalised [x1, y1, x2, y2]
        self.camera = None
        self._camera_index = 0
        self._supported_resolutions = []
        self.running = False
        self._thread = None
        self._raw_frame = None
        self._processed_frame = None
        self.calibration = None  # {a, b, tx, ty, z} similarity transform

    # -- camera lifecycle --------------------------------------------------

    def start(self, camera_index=0):
        if self.running:
            if self._camera_index == camera_index:
                return True
            self.stop()
        cap = _open_capture(int(camera_index))
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            return False
        self.camera = cap
        self._camera_index = int(camera_index)
        self._probe_resolutions(cap)
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def _probe_resolutions(self, cap):
        """Probe which resolutions the camera actually accepts."""
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        seen = {(orig_w, orig_h)}
        for w, h in _CANDIDATE_RESOLUTIONS:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if aw == w and ah == h:
                seen.add((w, h))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, orig_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, orig_h)
        self._supported_resolutions = sorted(seen)

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None
        if self.camera:
            self.camera.release()
            self.camera = None
        with self.lock:
            self._raw_frame = None
            self._processed_frame = None

    # -- thread-safe frame access ------------------------------------------

    def processed_frame(self):
        with self.lock:
            f = self._processed_frame
            return f.copy() if f is not None else None

    def raw_frame(self):
        with self.lock:
            f = self._raw_frame
            return f.copy() if f is not None else None

    # -- capture + process loop --------------------------------------------

    def _loop(self):
        target_dt = 1.0 / 30
        while self.running:
            t0 = time.monotonic()
            ret, frame = self.camera.read()
            if not ret:
                time.sleep(0.01)
                continue

            with self.lock:
                self._raw_frame = frame
                colors = list(self.learned_colors)
                shapes = list(self.learned_shapes)
                mode = self.mode
                phase = self.phase
                roi = list(self.roi)

            processed = self._process(frame, mode, phase, colors, shapes, roi)

            with self.lock:
                self._processed_frame = processed

            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, target_dt - elapsed))

    # -- per-frame processing ----------------------------------------------

    @staticmethod
    def _process(frame, mode, phase, colors, shapes, roi):
        display = frame.copy()
        fh, fw = frame.shape[:2]

        # ROI pixel coordinates (clamped)
        rx1 = max(0, int(roi[0] * fw))
        ry1 = max(0, int(roi[1] * fh))
        rx2 = min(fw, int(roi[2] * fw))
        ry2 = min(fh, int(roi[3] * fh))

        count = 0
        if phase == "learning":
            _draw_preview(frame, display, mode, rx1, ry1, rx2, ry2)
        else:
            roi_frame = frame[ry1:ry2, rx1:rx2]
            if roi_frame.size > 0:
                if mode == "color" and colors:
                    count = _detect_colors(roi_frame, display, colors,
                                           offset=(rx1, ry1))
                elif mode == "shape" and shapes:
                    count = _detect_shapes(roi_frame, display, shapes,
                                           offset=(rx1, ry1))

        # HUD
        mode_tag = "COLOR" if mode == "color" else "SHAPE"
        tag = ("LEARN " + mode_tag) if phase == "learning" else mode_tag
        cv2.putText(display, tag, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(display, tag, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        if phase != "learning" and count:
            txt = "{} found".format(count)
            cv2.putText(display, txt, (10, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(display, txt, (10, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1, cv2.LINE_AA)

        return display

    # -- learning ----------------------------------------------------------

    def _zone_crop(self, frame):
        fh, fw = frame.shape[:2]
        with self.lock:
            roi = list(self.roi)
        x1 = max(0, int(roi[0] * fw))
        y1 = max(0, int(roi[1] * fh))
        x2 = min(fw, int(roi[2] * fw))
        y2 = min(fh, int(roi[3] * fh))
        return frame[y1:y2, x1:x2]

    def learn_color(self, name=None):
        raw = self.raw_frame()
        if raw is None:
            return None

        zone = self._zone_crop(raw)
        # sample the inner 60 % to reduce edge noise
        zh, zw = zone.shape[:2]
        margin_y, margin_x = zh // 5, zw // 5
        sample = zone[margin_y:zh - margin_y, margin_x:zw - margin_x]

        hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
        med_h = float(np.median(hsv[:, :, 0]))
        med_s = float(np.median(hsv[:, :, 1]))
        med_v = float(np.median(hsv[:, :, 2]))

        lower = [max(0, med_h - 12), max(0, med_s - 55), max(0, med_v - 55)]
        upper = [min(179, med_h + 12), min(255, med_s + 55), min(255, med_v + 55)]

        # display colour from the learned median
        hsv_px = np.uint8([[[int(med_h), int(med_s), int(med_v)]]])
        bgr_px = cv2.cvtColor(hsv_px, cv2.COLOR_HSV2BGR)[0][0]
        disp = [int(bgr_px[0]), int(bgr_px[1]), int(bgr_px[2])]

        iid = uuid.uuid4().hex[:8]
        if not name:
            name = "Color {}".format(len(self.learned_colors) + 1)

        item = {
            "id": iid, "name": name, "mode": "color",
            "lower_hsv": [int(v) for v in lower],
            "upper_hsv": [int(v) for v in upper],
            "median_hsv": [int(med_h), int(med_s), int(med_v)],
            "display_color": disp,
        }
        with self.lock:
            self.learned_colors.append(item)
        return item

    def learn_shape(self, name=None):
        raw = self.raw_frame()
        if raw is None:
            return None

        zone = self._zone_crop(raw)
        gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(blurred, 40, 120)
        edges = cv2.dilate(edges, None, iterations=2)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 100:
            return None

        peri = cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, 0.04 * peri, True)
        n_verts = len(approx)

        iid = uuid.uuid4().hex[:8]
        if not name:
            _names = {3: "Triangle", 4: "Rectangle", 5: "Pentagon",
                      6: "Hexagon", 7: "Heptagon"}
            base = _names.get(n_verts,
                              "Circle" if n_verts >= 8 else "{}-gon".format(n_verts))
            name = "{} {}".format(base, len(self.learned_shapes) + 1)

        idx = len(self.learned_shapes) % len(_PALETTE)
        disp = list(_PALETTE[idx])

        item = {
            "id": iid, "name": name, "mode": "shape",
            "contour": largest,          # numpy array (not serialised)
            "vertex_count": n_verts,
            "display_color": disp,
        }
        with self.lock:
            self.learned_shapes.append(item)

        # return JSON-safe copy (no numpy)
        return {"id": iid, "name": name, "mode": "shape",
                "vertex_count": n_verts, "display_color": disp}

    # -- queries -----------------------------------------------------------

    def learned_list(self):
        with self.lock:
            items = []
            for c in self.learned_colors:
                items.append({k: v for k, v in c.items()})
            for s in self.learned_shapes:
                items.append({
                    "id": s["id"], "name": s["name"], "mode": "shape",
                    "vertex_count": s.get("vertex_count"),
                    "display_color": s["display_color"],
                })
            return items

    def remove_item(self, item_id):
        with self.lock:
            self.learned_colors = [
                c for c in self.learned_colors if c["id"] != item_id]
            self.learned_shapes = [
                s for s in self.learned_shapes if s["id"] != item_id]

    def clear_all(self):
        with self.lock:
            self.learned_colors.clear()
            self.learned_shapes.clear()

    # -- detection with positions ------------------------------------------

    def detect_objects(self):
        """Run detection on the current frame and return centroids.

        Returns a list of dicts sorted by area (largest first):
        [{"name", "mode", "center_px": [cx, cy], "bbox": [x,y,w,h],
          "area", "display_color"}]
        """
        raw = self.raw_frame()
        if raw is None:
            return []

        with self.lock:
            colors = list(self.learned_colors)
            shapes = list(self.learned_shapes)
            mode = self.mode
            roi = list(self.roi)

        fh, fw = raw.shape[:2]
        rx1 = max(0, int(roi[0] * fw))
        ry1 = max(0, int(roi[1] * fh))
        rx2 = min(fw, int(roi[2] * fw))
        ry2 = min(fh, int(roi[3] * fh))
        roi_frame = raw[ry1:ry2, rx1:rx2]
        if roi_frame.size == 0:
            return []

        results = []
        if mode == "color" and colors:
            results = self._detect_color_objects(roi_frame, colors, rx1, ry1)
        elif mode == "shape" and shapes:
            results = self._detect_shape_objects(roi_frame, shapes, rx1, ry1)

        results.sort(key=lambda d: d["area"], reverse=True)
        return results

    @staticmethod
    def _detect_color_objects(frame, colors, ox, oy):
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fh, fw = frame.shape[:2]
        out = []
        for item in colors:
            lower = np.array(item["lower_hsv"], dtype=np.uint8)
            upper = np.array(item["upper_hsv"], dtype=np.uint8)
            mask = cv2.inRange(hsv_frame, lower, upper)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 500:
                    continue
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"]) + ox
                cy = int(M["m01"] / M["m00"]) + oy
                x, y, w, h = cv2.boundingRect(cnt)
                out.append({
                    "name": item["name"], "mode": "color",
                    "center_px": [cx, cy],
                    "bbox": [x + ox, y + oy, w, h],
                    "area": int(area),
                    "display_color": item["display_color"],
                })
        return out

    @staticmethod
    def _detect_shape_objects(frame, shapes, ox, oy):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(blurred, 40, 120)
        edges = cv2.dilate(edges, None, iterations=2)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            nv = len(approx)
            best_item, best_score = None, float("inf")
            for item in shapes:
                iv = item["vertex_count"]
                if iv < 8 and nv < 8 and abs(nv - iv) > 1:
                    continue
                score = cv2.matchShapes(
                    cnt, item["contour"], cv2.CONTOURS_MATCH_I1, 0)
                if score < 0.25 and score < best_score:
                    best_score = score
                    best_item = item
            if best_item is not None:
                M = cv2.moments(cnt)
                cx = int(M["m10"] / M["m00"]) + ox if M["m00"] else ox
                cy = int(M["m01"] / M["m00"]) + oy if M["m00"] else oy
                x, y, w, h = cv2.boundingRect(cnt)
                out.append({
                    "name": best_item["name"], "mode": "shape",
                    "center_px": [cx, cy],
                    "bbox": [x + ox, y + oy, w, h],
                    "area": int(area),
                    "display_color": best_item["display_color"],
                })
        return out

    # -- calibration -------------------------------------------------------

    def set_calibration(self, pixel_pts, robot_pts, z_height):
        """Compute and store a 2D affine transform from 3 point pairs.

        pixel_pts: [[px1,py1], [px2,py2], [px3,py3]]
        robot_pts: [[rx1,ry1], [rx2,ry2], [rx3,ry3]]
        z_height:  float (robot Z for picking)

        Affine:  robot_x = a*px + b*py + tx
                 robot_y = c*px + d*py + ty
        """
        P = [[float(p[0]), float(p[1])] for p in pixel_pts]
        R = [[float(r[0]), float(r[1])] for r in robot_pts]

        # 6 equations, 6 unknowns: [a, b, tx, c, d, ty]
        A = np.array([
            [P[0][0], P[0][1], 1, 0, 0, 0],
            [0, 0, 0, P[0][0], P[0][1], 1],
            [P[1][0], P[1][1], 1, 0, 0, 0],
            [0, 0, 0, P[1][0], P[1][1], 1],
            [P[2][0], P[2][1], 1, 0, 0, 0],
            [0, 0, 0, P[2][0], P[2][1], 1],
        ], dtype=np.float64)
        B = np.array([
            R[0][0], R[0][1],
            R[1][0], R[1][1],
            R[2][0], R[2][1],
        ], dtype=np.float64)

        params = np.linalg.solve(A, B)
        a, b, tx, c, d, ty = params.tolist()

        self.calibration = {
            "a": a, "b": b, "tx": tx,
            "c": c, "d": d, "ty": ty,
            "z": float(z_height),
        }
        return self.calibration

    def pixel_to_robot(self, px, py):
        """Convert pixel coords to robot coords using stored calibration."""
        if not self.calibration:
            return None
        cal = self.calibration
        rx = cal["a"] * px + cal["b"] * py + cal["tx"]
        ry = cal["c"] * px + cal["d"] * py + cal["ty"]
        return {"x": round(rx, 2), "y": round(ry, 2), "z": round(cal["z"], 2)}


_state = _CVState()


# ---------------------------------------------------------------------------
#  Flask routes
# ---------------------------------------------------------------------------

@blueprint.route("/stream")
def stream_feed():
    """MJPEG stream of the processed camera frames."""
    if not _state.running:
        _state.start()

    def generate():
        enc = [cv2.IMWRITE_JPEG_QUALITY, 80]
        # wait for the first frame (camera warm-up)
        for _ in range(100):
            if _state.processed_frame() is not None:
                break
            time.sleep(0.03)

        while _state.running:
            frame = _state.processed_frame()
            if frame is not None:
                ok, jpg = cv2.imencode(".jpg", frame, enc)
                if ok:
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n"
                           + jpg.tobytes() + b"\r\n")
            time.sleep(1.0 / 30)

    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@blueprint.route("/frame")
def get_frame():
    """Return the latest processed frame as a base64 JPEG."""
    frame = _state.processed_frame()
    if frame is None:
        return jsonify({"success": False, "error": "No frame available"})
    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return jsonify({"success": False, "error": "Encode failed"})
    b64 = base64.b64encode(jpg.tobytes()).decode("ascii")
    return jsonify({"success": True, "image": b64})


@blueprint.route("/start", methods=["POST"])
def start_camera():
    data = request.get_json() or {}
    cam = int(data.get("camera", 0))
    if _state.start(cam):
        return jsonify({"success": True})
    return jsonify({"success": False, "error": "Cannot open camera"})


@blueprint.route("/stop", methods=["POST"])
def stop_camera():
    _state.stop()
    return jsonify({"success": True})


@blueprint.route("/cameras")
def list_cameras():
    """List cameras as [{index, name}, ...] with friendly OS names when possible."""
    skip = _state._camera_index if _state.running else None
    cameras = enumerate_cameras(max_probe=10, skip_index=skip)
    return jsonify({
        "success": True,
        "cameras": cameras,
        "current": _state._camera_index if _state.running else None,
    })


@blueprint.route("/resolution", methods=["GET", "POST"])
def resolution():
    if _state.camera is None or not _state.camera.isOpened():
        return jsonify({"success": False, "error": "Camera not running"})
    if request.method == "GET":
        w = int(_state.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(_state.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return jsonify({"success": True, "width": w, "height": h})
    data = request.get_json() or {}
    w = data.get("width")
    h = data.get("height")
    if not w or not h:
        return jsonify({"success": False, "error": "Need width and height"})
    _state.camera.set(cv2.CAP_PROP_FRAME_WIDTH, int(w))
    _state.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, int(h))
    aw = int(_state.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(_state.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return jsonify({"success": True, "width": aw, "height": ah})


@blueprint.route("/resolutions")
def list_resolutions():
    """Return the camera's supported resolutions (probed on start)."""
    w = h = 0
    if _state.camera and _state.camera.isOpened():
        w = int(_state.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(_state.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return jsonify({
        "success": True,
        "resolutions": [{"width": rw, "height": rh}
                        for rw, rh in _state._supported_resolutions],
        "current": {"width": w, "height": h},
    })


@blueprint.route("/phase", methods=["GET", "POST"])
def phase():
    if request.method == "GET":
        return jsonify({"success": True, "phase": _state.phase})
    data = request.get_json() or {}
    p = data.get("phase")
    if p in ("learning", "inference"):
        _state.phase = p
        return jsonify({"success": True, "phase": p})
    return jsonify({"success": False, "error": "Invalid phase"})


@blueprint.route("/mode", methods=["GET", "POST"])
def mode():
    if request.method == "GET":
        return jsonify({"success": True, "mode": _state.mode})
    data = request.get_json() or {}
    m = data.get("mode")
    if m in ("color", "shape"):
        _state.mode = m
        return jsonify({"success": True, "mode": m})
    return jsonify({"success": False, "error": "Invalid mode"})


@blueprint.route("/learn", methods=["POST"])
def learn():
    data = request.get_json() or {}
    name = (data.get("name") or "").strip() or None

    if _state.mode == "color":
        item = _state.learn_color(name)
    else:
        item = _state.learn_shape(name)

    if item:
        return jsonify({"success": True, "item": item})
    return jsonify({
        "success": False,
        "error": "Nothing detected in the zone. "
                 "Place an object inside the ROI and try again."
    })


@blueprint.route("/learned", methods=["GET"])
def learned():
    return jsonify({"success": True, "items": _state.learned_list()})


@blueprint.route("/remove", methods=["POST"])
def remove():
    data = request.get_json() or {}
    iid = data.get("id")
    if not iid:
        return jsonify({"success": False, "error": "Missing id"})
    _state.remove_item(iid)
    return jsonify({"success": True})


@blueprint.route("/roi", methods=["GET", "POST"])
def roi():
    if request.method == "GET":
        with _state.lock:
            r = list(_state.roi)
        return jsonify({"success": True, "roi": r})
    data = request.get_json() or {}
    coords = data.get("roi")
    if not coords or len(coords) != 4:
        return jsonify({"success": False, "error": "Need roi=[x1,y1,x2,y2]"})
    try:
        coords = [float(c) for c in coords]
    except (TypeError, ValueError):
        return jsonify({"success": False, "error": "Invalid coordinates"})
    coords = [max(0.0, min(1.0, c)) for c in coords]
    if coords[2] - coords[0] < 0.05 or coords[3] - coords[1] < 0.05:
        return jsonify({"success": False, "error": "ROI too small"})
    with _state.lock:
        _state.roi = coords
    return jsonify({"success": True, "roi": coords})


@blueprint.route("/clear", methods=["POST"])
def clear():
    _state.clear_all()
    return jsonify({"success": True})


# ---------------------------------------------------------------------------
#  Detection & Calibration routes
# ---------------------------------------------------------------------------

@blueprint.route("/detections")
def detections():
    """Return detected objects with pixel centroids (sorted by area desc)."""
    objs = _state.detect_objects()
    return jsonify({"success": True, "detections": objs})


@blueprint.route("/calibration", methods=["GET", "POST", "DELETE"])
def calibration():
    if request.method == "GET":
        return jsonify({"success": True, "calibration": _state.calibration})

    if request.method == "DELETE":
        _state.calibration = None
        return jsonify({"success": True})

    # POST — compute calibration from 2 point pairs
    data = request.get_json() or {}
    pixel_pts = data.get("pixel_points")
    robot_pts = data.get("robot_points")
    z = data.get("z")

    if (not pixel_pts or not robot_pts
            or len(pixel_pts) != 3 or len(robot_pts) != 3 or z is None):
        return jsonify({"success": False,
                        "error": "Need pixel_points (3), robot_points (3), z"})
    try:
        cal = _state.set_calibration(pixel_pts, robot_pts, float(z))
        return jsonify({"success": True, "calibration": cal})
    except np.linalg.LinAlgError:
        return jsonify({"success": False,
                        "error": "Points are collinear — place markers in an L-shape"})


@blueprint.route("/pick-position", methods=["POST"])
def pick_position():
    """Convert pixel coordinates to robot coordinates."""
    data = request.get_json() or {}
    px = data.get("px")
    py = data.get("py")
    if px is None or py is None:
        return jsonify({"success": False, "error": "Need px and py"})
    pos = _state.pixel_to_robot(float(px), float(py))
    if pos is None:
        return jsonify({"success": False, "error": "Not calibrated"})
    return jsonify({"success": True, "position": pos})