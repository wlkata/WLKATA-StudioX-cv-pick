"""
CV Pick extension backend.

Load a local YOLO26n checkpoint and run real-time detection in the ROI.
A background thread captures webcam frames, overlays boxes, and exposes
the result as JPEG frames.
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
# Listing only needs to prove the device works. Sweeping 4K on every
# tab-open overruns the 30s proxy and surfaces as "Could not list cameras".
_LIST_RESOLUTIONS = [
    (640, 480), (1280, 720), (800, 600), (1920, 1080),
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

_YOLO26N_MARKERS = ("yolo26n", "yolov26n")
_YOLO26_OTHER = (
    "yolo26s", "yolo26m", "yolo26l", "yolo26x",
    "yolov26s", "yolov26m", "yolov26l", "yolov26x",
)
_CKPT_EXTS = (".pt", ".ckpt")


def _warmup_yolo_import():
    """Import torch/ultralytics in the background so Load is not the first hit."""
    try:
        import torch  # noqa: F401
        import ultralytics  # noqa: F401
    except Exception:
        pass


threading.Thread(target=_warmup_yolo_import, daemon=True).start()


def _ckpt_hints(path, model=None):
    hints = [os.path.basename(path or "").lower()]
    ckpt = getattr(model, "ckpt", None) if model is not None else None
    if model is not None:
        ov = getattr(model, "overrides", None) or {}
        if ov.get("model"):
            hints.append(str(ov["model"]).lower())
        yaml = getattr(getattr(model, "model", None), "yaml", None) or {}
        if isinstance(yaml, dict):
            for key in ("yaml_file", "scale"):
                if yaml.get(key) is not None:
                    hints.append(str(yaml[key]).lower())
    if isinstance(ckpt, dict):
        args = ckpt.get("train_args") or {}
        for key in ("model", "name"):
            if args.get(key):
                hints.append(str(args[key]).lower())
    return " ".join(hints)


def _is_yolo26n(path, model=None):
    blob = _ckpt_hints(path, model)
    if any(tag in blob for tag in _YOLO26_OTHER):
        return False
    if any(tag in blob for tag in _YOLO26N_MARKERS):
        return True
    yaml = getattr(getattr(model, "model", None), "yaml", None) or {}
    if not isinstance(yaml, dict):
        return False
    scale = str(yaml.get("scale") or "").lower()
    yaml_file = str(yaml.get("yaml_file") or "").lower()
    return scale == "n" and ("yolo26" in yaml_file or "yolov26" in yaml_file)


def _allowed_ckpt_filename(name):
    lower = (name or "").lower()
    return any(lower.endswith(ext) for ext in _CKPT_EXTS)


# ---------------------------------------------------------------------------
#  Camera enumeration (friendly names when the OS provides them)
# ---------------------------------------------------------------------------

def _is_windows():
    return platform.system() == "Windows"


def _configure_size(cap, width, height):
    """Apply size (and MJPG on Windows so UVC can leave 640x480)."""
    if cap is None or not width or not height or width <= 0 or height <= 0:
        return
    if _is_windows():
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))


def _is_blank_frame(frame):
    """True for empty or USB-warmup all-black frames (not a dim room)."""
    if frame is None or getattr(frame, "size", 0) == 0:
        return True
    try:
        return int(frame.max()) < 12
    except Exception:
        return False


def _read_frame_size(cap, tries=8, require_content=False, pause=0.02):
    """Return (w, h) from a decoded frame, or None if the device is silent."""
    last = None
    for _ in range(tries):
        ret, frame = cap.read()
        if ret and frame is not None and getattr(frame, "size", 0):
            h, w = int(frame.shape[0]), int(frame.shape[1])
            if w > 0 and h > 0:
                last = (w, h)
                if not require_content or not _is_blank_frame(frame):
                    return last
        if pause:
            time.sleep(pause)
    return None if require_content else last


def _warmup_open(cap):
    """isOpened() is not enough — require at least one decoded frame."""
    if cap is None or not cap.isOpened():
        return False
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return _read_frame_size(cap, tries=12, require_content=False, pause=0.04) is not None


def _release_cap(cap):
    if cap is None:
        return
    try:
        cap.release()
    except Exception:
        pass


def _open_capture(index, width=None, height=None):
    """
    Open a capture with the platform backend that matches name order.

    isOpened() can be true on a dead DirectShow handle; try the next backend
    if the first one cannot actually deliver a frame.
    """
    system = platform.system()
    backends = []
    if system == "Windows":
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF]
    elif system == "Darwin":
        backends = [cv2.CAP_AVFOUNDATION]
    for be in backends:
        cap = cv2.VideoCapture(int(index), be)
        if not cap.isOpened():
            _release_cap(cap)
            continue
        if width and height:
            _configure_size(cap, width, height)
        if _warmup_open(cap):
            return cap
        _release_cap(cap)
    cap = cv2.VideoCapture(int(index))
    if cap is not None and cap.isOpened():
        if width and height:
            _configure_size(cap, width, height)
        if _warmup_open(cap):
            return cap
    _release_cap(cap)
    return None


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


_names_cache = (None, 0.0)  # (names, monotonic ts)


def _platform_camera_names(force=False):
    global _names_cache
    names, ts = _names_cache
    if (not force and names is not None
            and (time.monotonic() - ts) < 20):
        return list(names)
    system = platform.system()
    if system == "Windows":
        result = _camera_names_windows()
    elif system == "Darwin":
        result = _camera_names_macos()
    elif system == "Linux":
        result = _camera_names_linux()
    else:
        result = []
    _names_cache = (list(result), time.monotonic())
    return result


def _resolution_matches(requested, actual, tol=0.02):
    rw, rh = requested
    aw, ah = actual
    if aw <= 0 or ah <= 0 or rw <= 0 or rh <= 0:
        return False
    return (
        abs(aw - rw) <= max(2, int(rw * tol))
        and abs(ah - rh) <= max(2, int(rh * tol))
    )


def _try_resolution(cap, w, h):
    """
    Ask the driver for (w, h) and verify a frame actually arrives at that size.
    Returns the measured (aw, ah) on success, else None.
    """
    _configure_size(cap, w, h)
    size = _read_frame_size(cap, tries=8, pause=0.02)
    if not size:
        return None
    if _resolution_matches((w, h), size):
        return size
    return None


def probe_camera_resolutions(cap, candidates=None):
    """
    Test candidate resolutions by capturing frames.
    Returns ([(w,h), ...], default_(w,h)) or ([], None) if unusable.
    """
    found = {}

    baseline = _read_frame_size(cap, tries=16, pause=0.04)
    if baseline:
        found[baseline] = True

    modes = _LIST_RESOLUTIONS if candidates is None else candidates
    for w, h in modes:
        if baseline and _resolution_matches((w, h), baseline):
            continue
        got = _try_resolution(cap, w, h)
        if got:
            found[got] = True

    if not found:
        return [], None

    res_list = sorted(found.keys(), key=lambda wh: (wh[0] * wh[1], wh[0]))
    default = res_list[0]
    for preferred in ((640, 480), (1280, 720), (800, 600), (1280, 960)):
        for cand in res_list:
            if _resolution_matches(preferred, cand):
                default = cand
                break
        else:
            continue
        break

    # Leave the device parked on a known-good mode.
    _configure_size(cap, default[0], default[1])
    _read_frame_size(cap, tries=6, pause=0.02)
    return res_list, default


def test_camera_device(index, name=None):
    """
    Open *index*, verify it can deliver frames, probe a small resolution set.
    Returns a catalog entry or None if the camera is unusable.
    """
    cap = _open_capture(int(index))
    if cap is None or not cap.isOpened():
        _release_cap(cap)
        return None
    try:
        res_list, default = probe_camera_resolutions(cap)
        if not res_list or not default:
            return None
        label = (name or "").strip() or f"Camera {index}"
        return {
            "index": int(index),
            "name": label,
            "resolutions": [{"width": w, "height": h} for w, h in res_list],
            "default": {"width": default[0], "height": default[1]},
        }
    except Exception:
        return None
    finally:
        _release_cap(cap)


# Last full catalog from /cameras (index -> entry). Used by start() and
# subsequent list calls so we do not re-open devices we already tested.
_device_catalog = {}
_device_catalog_lock = threading.Lock()
_enumerate_lock = threading.Lock()
# Indices that produced a usable catalog entry. Failed opens are retried
# when we still have no working list (cold plug / first-open warmup).
_probed_indices = set()
_failed_indices = set()


def _catalog_snapshot():
    with _device_catalog_lock:
        return [dict(entry) for _, entry in sorted(_device_catalog.items())]


def _catalog_copy():
    with _device_catalog_lock:
        return {int(k): dict(v) for k, v in _device_catalog.items()}


def _norm_cam_name(name):
    return " ".join(str(name or "").strip().lower().split())


def _same_camera(cached_name, os_name):
    """True only when both sides have a name and they match."""
    a = _norm_cam_name(cached_name)
    b = _norm_cam_name(os_name)
    return bool(a) and bool(b) and a == b


def _reuse_entry(index, name, prev):
    """
    Reuse a catalog entry without opening the device.

    Index alone is not identity: plugging in a USB camera on Windows often
    takes index 0 and shifts OBS Virtual Camera to 1. Only reuse when the
    OS name at this index still matches the cached device.
    """
    entry = prev.get(int(index))
    if not entry or not entry.get("resolutions"):
        return None
    if name and entry.get("name") and not _same_camera(entry.get("name"), name):
        return None
    out = dict(entry)
    out["index"] = int(index)
    if name:
        out["name"] = name
    return out


def _reuse_by_name(name, prev, index):
    """Reuse a cached device that moved to a new OpenCV index."""
    if not name or not prev:
        return None
    for entry in prev.values():
        if not entry or not entry.get("resolutions"):
            continue
        if not _same_camera(entry.get("name"), name):
            continue
        out = dict(entry)
        out["index"] = int(index)
        out["name"] = name
        return out
    return None


def _parse_index_set(raw):
    out = set()
    for part in str(raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.add(int(part))
        except ValueError:
            continue
    return out


def enumerate_cameras(max_probe=10, skip_index=None, skip_entry=None,
                      reuse_cached=True, never_probe=None):
    """
    Return usable cameras only: each must produce frames at ≥1 resolution.

    skip_index / skip_entry: camera already held by the live session — do not
    re-open it; reuse the running session's probed list instead.

    reuse_cached: never re-open an index already in the catalog *if the OS
    name still matches*. A new name at a cached index is probed. A cached
    name that moved to a new index is reused without opening. Failed opens
    are retried only when the catalog is still empty.

    never_probe: extra indices the client already has cached — ignored when
    the name at that index changed.
    """
    with _enumerate_lock:
        return _enumerate_cameras_locked(
            max_probe=max_probe,
            skip_index=skip_index,
            skip_entry=skip_entry,
            reuse_cached=reuse_cached,
            never_probe=set(never_probe or ()),
        )


def _enumerate_cameras_locked(max_probe, skip_index, skip_entry, reuse_cached,
                              never_probe):
    # Always refresh OS names on enumerate so a just-plugged USB camera is
    # visible. The 20s cache would hide hotplug during discover.
    names = _platform_camera_names(force=True)
    prev = _catalog_copy()

    if names:
        candidates = [
            (i, (n.strip() if n and str(n).strip() else f"Camera {i}"))
            for i, n in enumerate(names)
        ]
    elif reuse_cached and prev:
        # No OS names: do not rescan empty 0..max_probe slots.
        candidates = [
            (i, (prev[i].get("name") or f"Camera {i}"))
            for i in sorted(prev)
        ]
    else:
        candidates = [(i, f"Camera {i}") for i in range(max_probe)]

    out = []
    for i, name in candidates:
        i = int(i)
        if skip_index is not None and i == int(skip_index):
            if skip_entry and skip_entry.get("resolutions"):
                entry = dict(skip_entry)
                entry["index"] = i
                live_name = skip_entry.get("name") or ""
                # Keep the held device's name; do not relabel it as a newly
                # plugged camera that stole this index in the OS list.
                entry["name"] = live_name or name
                out.append(entry)
                _probed_indices.add(i)
                continue
            reused = _reuse_entry(i, name, prev) if reuse_cached else None
            if reused:
                out.append(reused)
                _probed_indices.add(i)
            continue

        if reuse_cached:
            reused = _reuse_entry(i, name, prev)
            if not reused:
                reused = _reuse_by_name(name, prev, i)
            if reused:
                out.append(reused)
                _probed_indices.add(i)
                _failed_indices.discard(i)
                continue
            # Known-dead only skipped once we already have a working list.
            if i in _failed_indices and prev:
                continue

        info = test_camera_device(i, name)
        if info:
            out.append(info)
            _probed_indices.add(i)
            _failed_indices.discard(i)
        else:
            _failed_indices.add(i)

    with _device_catalog_lock:
        _device_catalog.clear()
        for entry in out:
            _device_catalog[int(entry["index"])] = entry
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


def _draw_yolo_dets(display, dets):
    for d in dets:
        x, y, w, h = d["bbox"]
        bgr = tuple(int(c) for c in d["display_color"])
        cv2.rectangle(display, (x, y), (x + w, y + h), bgr, 2)
        label = d["name"]
        if d.get("conf") is not None:
            label = "{} {:.2f}".format(label, d["conf"])
        _put_label(display, label, x, y, bgr)
    return len(dets)


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
        # Applied only on the capture thread (never call cap.set from HTTP).
        self._pending_resolution = None  # (w, h) or None
        self._resolution_event = threading.Event()
        self._last_resolution = (0, 0)
        self._last_resolution_ok = True  # False if last change was reverted
        self._camera_name = ""
        self._lifecycle = threading.Lock()
        self._started_at = 0.0
        self.yolo_model = None
        self.yolo_classes = []
        self.yolo_path = None
        self.yolo_hidden = set()
        self.yolo_conf = 0.25
        self._last_yolo_dets = []
        self._yolo_infer_lock = threading.Lock()

    # -- camera lifecycle --------------------------------------------------

    def start(self, camera_index=0):
        with self._lifecycle:
            return self._start_body(camera_index)

    def _start_body(self, camera_index=0):
        if self.running:
            if self._camera_index == camera_index:
                return True
            self._stop_body()

        idx = int(camera_index)
        with _device_catalog_lock:
            cached = dict(_device_catalog.get(idx) or {})

        # Prefer catalog probe from /cameras; re-test if missing.
        res_list = []
        default = None
        if cached.get("resolutions"):
            res_list = [
                (int(r["width"]), int(r["height"]))
                for r in cached["resolutions"]
                if int(r.get("width") or 0) > 0 and int(r.get("height") or 0) > 0
            ]
            d = cached.get("default") or {}
            if int(d.get("width") or 0) > 0 and int(d.get("height") or 0) > 0:
                default = (int(d["width"]), int(d["height"]))

        # Open already parked on the known-good size, with MJPG on Windows.
        if default:
            cap = _open_capture(idx, default[0], default[1])
        else:
            cap = _open_capture(idx)
        if cap is None or not cap.isOpened():
            _release_cap(cap)
            return False

        if not res_list or not default:
            res_list, default = probe_camera_resolutions(
                cap, candidates=_CANDIDATE_RESOLUTIONS)
        if not res_list or not default:
            _release_cap(cap)
            return False

        # Park on default (MJPG + size) and wait out black warmup frames.
        _configure_size(cap, default[0], default[1])
        size = _read_frame_size(
            cap, tries=24, require_content=True, pause=0.04)
        if not size:
            size = _read_frame_size(cap, tries=8, require_content=False, pause=0.03)
        if not size:
            _release_cap(cap)
            return False

        self.camera = cap
        self._camera_index = idx
        self._camera_name = cached.get("name") or f"Camera {idx}"
        self._supported_resolutions = sorted(set(res_list) | {size})
        self._last_resolution = size
        self._last_resolution_ok = True
        self._pending_resolution = None
        self._resolution_event.set()
        self._started_at = time.monotonic()
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def catalog_entry(self):
        """Snapshot of the live camera for /cameras while it is held open."""
        with self.lock:
            res = [
                {"width": w, "height": h}
                for w, h in self._supported_resolutions
                if w > 0 and h > 0
            ]
            lw, lh = self._last_resolution
        if not res:
            return None
        default = {"width": lw, "height": lh} if lw > 0 and lh > 0 else res[0]
        return {
            "index": self._camera_index,
            "name": self._camera_name or f"Camera {self._camera_index}",
            "resolutions": res,
            "default": default,
        }

    def request_resolution(self, width, height):
        """
        Queue a resolution change for the capture thread.
        Returns dict {width, height, reverted} or None.
        """
        if self.camera is None or not self.running:
            return None
        req = (int(width), int(height))
        # Reject modes we already know are unsupported.
        with self.lock:
            supported = list(self._supported_resolutions)
        if supported and not any(_resolution_matches(req, s) for s in supported):
            with self.lock:
                cw, ch = self._last_resolution
            return {"width": cw, "height": ch, "reverted": True}

        self._resolution_event.clear()
        with self.lock:
            self._pending_resolution = req
            self._last_resolution_ok = True
        # Windows reopen can take several seconds.
        wait_s = 12.0 if _is_windows() else 6.0
        self._resolution_event.wait(timeout=wait_s)
        with self.lock:
            cw, ch = self._last_resolution
            ok = self._last_resolution_ok
        return {
            "width": cw,
            "height": ch,
            "reverted": (not ok) or (not _resolution_matches(req, (cw, ch))),
        }

    def stop(self):
        with self._lifecycle:
            self._stop_body()

    def _stop_body(self):
        self.running = False
        self._resolution_event.set()
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.join(timeout=2.5)
        cap = self.camera
        self.camera = None
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        # DirectShow / AVFoundation keep an exclusive lock briefly after
        # release(); a short pause avoids the next open seeing "busy".
        if platform.system() in ("Windows", "Darwin"):
            time.sleep(0.2)
        with self.lock:
            self._raw_frame = None
            self._processed_frame = None
            self._pending_resolution = None

    # -- thread-safe frame access ------------------------------------------

    def processed_frame(self):
        with self.lock:
            f = self._processed_frame
            return f.copy() if f is not None else None

    def raw_frame(self):
        with self.lock:
            f = self._raw_frame
            return f.copy() if f is not None else None

    def _swap_camera(self, cap):
        """Replace the live capture; caller owns the new *cap*."""
        old = self.camera
        self.camera = cap
        if old is not None:
            try:
                old.release()
            except Exception:
                pass

    def _reopen_at(self, width, height):
        """
        Release and reopen this camera at (width, height).
        Returns measured (w, h) or None.
        """
        idx = self._camera_index
        old = self.camera
        self.camera = None
        if old is not None:
            try:
                old.release()
            except Exception:
                pass
        cap = _open_capture(idx, width, height)
        if cap is None or not cap.isOpened():
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            return None
        size = _read_frame_size(cap, tries=15)
        if not size or size[0] <= 0 or size[1] <= 0:
            try:
                cap.release()
            except Exception:
                pass
            return None
        self.camera = cap
        return size

    def _restore_resolution(self, prev):
        """Restore a known-good mode. Windows always reopens."""
        if not prev or prev[0] <= 0 or prev[1] <= 0:
            return (0, 0)
        if _is_windows():
            return self._reopen_at(prev[0], prev[1]) or (0, 0)
        if self.camera is None:
            return self._reopen_at(prev[0], prev[1]) or (0, 0)
        _configure_size(self.camera, prev[0], prev[1])
        size = _read_frame_size(self.camera, tries=12)
        if size and size[0] > 0:
            return size
        # In-place set left the handle dead — reopen.
        return self._reopen_at(prev[0], prev[1]) or (0, 0)

    def _apply_pending_resolution(self):
        """Run only on the capture thread."""
        with self.lock:
            pending = self._pending_resolution
            self._pending_resolution = None
            prev = tuple(self._last_resolution)
        if not pending:
            return
        w, h = pending
        reverted = False
        try:
            if _is_windows():
                # DirectShow/MSMF: set() on a live handle often kills read().
                size = self._reopen_at(w, h)
            else:
                if self.camera is None:
                    size = self._reopen_at(w, h)
                else:
                    _configure_size(self.camera, w, h)
                    size = _read_frame_size(self.camera, tries=12)
                    if not size or size[0] <= 0:
                        size = self._reopen_at(w, h)

            if size and _resolution_matches((w, h), size):
                aw, ah = size
            elif size and size[0] > 0 and size[1] > 0:
                # Driver opened at a different real size — keep it if known.
                if any(_resolution_matches(size, s) for s in self._supported_resolutions):
                    aw, ah = size
                else:
                    restored = self._restore_resolution(prev)
                    aw, ah = restored if restored and restored[0] > 0 else prev
                    reverted = True
            else:
                restored = self._restore_resolution(prev)
                aw, ah = restored if restored and restored[0] > 0 else prev
                reverted = True
            if aw <= 0 or ah <= 0:
                aw, ah = prev if prev[0] > 0 else (w, h)
                reverted = True
            with self.lock:
                self._last_resolution = (int(aw), int(ah))
                self._last_resolution_ok = not reverted
        except Exception:
            restored = self._restore_resolution(prev)
            aw, ah = restored if restored and restored[0] > 0 else prev
            with self.lock:
                if aw > 0 and ah > 0:
                    self._last_resolution = (int(aw), int(ah))
                self._last_resolution_ok = False
        finally:
            self._resolution_event.set()

    # -- capture + process loop --------------------------------------------

    def _loop(self):
        target_dt = 1.0 / 30
        while self.running:
            t0 = time.monotonic()
            self._apply_pending_resolution()
            cap = self.camera
            if cap is None:
                time.sleep(0.02)
                continue
            ret, frame = cap.read()
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
            processed = self._process(
                frame, mode, phase, colors, shapes, roi)

            # Hold the JPEG feed until auto-exposure produces a real picture.
            if (_is_blank_frame(processed)
                    and (time.monotonic() - self._started_at) < 2.5):
                elapsed = time.monotonic() - t0
                time.sleep(max(0.0, target_dt - elapsed))
                continue

            with self.lock:
                self._processed_frame = processed

            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, target_dt - elapsed))

    # -- per-frame processing ----------------------------------------------

    def _process(self, frame, mode, phase, colors, shapes, roi):
        display = frame.copy()
        fh, fw = frame.shape[:2]

        # ROI pixel coordinates (clamped)
        rx1 = max(0, int(roi[0] * fw))
        ry1 = max(0, int(roi[1] * fh))
        rx2 = min(fw, int(roi[2] * fw))
        ry2 = min(fh, int(roi[3] * fh))

        count = 0
        if mode == "yolo":
            roi_frame = frame[ry1:ry2, rx1:rx2]
            dets = self._run_yolo(roi_frame, rx1, ry1) if roi_frame.size else []
            with self.lock:
                self._last_yolo_dets = dets
            count = _draw_yolo_dets(display, dets)
            tag = "YOLO26n" if self.yolo_model is not None else "NO YOLO"
        elif phase == "learning":
            _draw_preview(frame, display, mode, rx1, ry1, rx2, ry2)
            mode_tag = "COLOR" if mode == "color" else "SHAPE"
            tag = "LEARN " + mode_tag
        else:
            roi_frame = frame[ry1:ry2, rx1:rx2]
            if roi_frame.size > 0:
                if mode == "color" and colors:
                    count = _detect_colors(roi_frame, display, colors,
                                           offset=(rx1, ry1))
                elif mode == "shape" and shapes:
                    count = _detect_shapes(roi_frame, display, shapes,
                                           offset=(rx1, ry1))
            mode_tag = "COLOR" if mode == "color" else "SHAPE"
            tag = mode_tag

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

    def load_yolo(self, path):
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            raise ValueError("Checkpoint not found")
        if not _allowed_ckpt_filename(path):
            raise ValueError("Need a .pt or .ckpt file")
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ValueError(
                "ultralytics is not installed. Add it to this extension env.")

        model = YOLO(path)
        task = getattr(model, "task", "detect") or "detect"
        if task != "detect":
            raise ValueError(
                "Need a YOLO26n detection checkpoint, not task '{}'".format(task))
        if not _is_yolo26n(path, model):
            raise ValueError(
                "Only YOLO26n checkpoints are supported. "
                "Load a yolo26n.pt (or a YOLO26n fine-tune).")

        names = getattr(model, "names", None) or {}
        if isinstance(names, dict):
            pairs = sorted(names.items(), key=lambda kv: int(kv[0]))
        else:
            pairs = list(enumerate(names))
        classes = []
        for idx, name in pairs:
            classes.append({
                "id": "yolo-{}".format(int(idx)),
                "name": str(name),
                "mode": "yolo",
                "class_id": int(idx),
                "display_color": list(_PALETTE[int(idx) % len(_PALETTE)]),
            })
        with self._yolo_infer_lock:
            with self.lock:
                self.yolo_model = model
                self.yolo_classes = classes
                self.yolo_path = path
                self.yolo_hidden = set()
                self._last_yolo_dets = []
        return {
            "path": path,
            "filename": os.path.basename(path),
            "classes": len(classes),
            "items": classes,
        }

    def unload_yolo(self):
        with self._yolo_infer_lock:
            with self.lock:
                self.yolo_model = None
                self.yolo_classes = []
                self.yolo_path = None
                self.yolo_hidden = set()
                self._last_yolo_dets = []

    def set_yolo_conf(self, conf):
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            raise ValueError("Threshold must be a number")
        conf = max(0.05, min(0.95, conf))
        with self.lock:
            self.yolo_conf = conf
        return conf

    def yolo_status(self):
        with self.lock:
            status = {"loaded": False, "conf": self.yolo_conf}
            if self.yolo_model is None:
                return status
            status.update({
                "loaded": True,
                "path": self.yolo_path,
                "filename": os.path.basename(self.yolo_path or ""),
                "classes": len(self.yolo_classes),
            })
            return status

    def _run_yolo(self, roi_bgr, ox, oy):
        with self.lock:
            model = self.yolo_model
            classes = list(self.yolo_classes)
            hidden = set(self.yolo_hidden)
            conf = self.yolo_conf
        if model is None or roi_bgr is None or getattr(roi_bgr, "size", 0) == 0:
            return []
        class_by_id = {c["class_id"]: c for c in classes}
        try:
            with self._yolo_infer_lock:
                results = model.predict(
                    roi_bgr, verbose=False, imgsz=640, conf=conf)
        except Exception:
            return []
        if not results:
            return []
        boxes = getattr(results[0], "boxes", None)
        if boxes is None:
            return []
        try:
            xyxy = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
        except Exception:
            return []
        dets = []
        for i in range(len(xyxy)):
            item = class_by_id.get(int(cls[i]))
            if item is None or item["id"] in hidden:
                continue
            x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
            x, y = int(x1), int(y1)
            w, h = max(0, int(x2 - x1)), max(0, int(y2 - y1))
            if w < 2 or h < 2:
                continue
            dets.append({
                "name": item["name"],
                "mode": "yolo",
                "center_px": [int(ox + x + w / 2), int(oy + y + h / 2)],
                "bbox": [ox + x, oy + y, w, h],
                "area": w * h,
                "display_color": item["display_color"],
                "conf": float(conf[i]),
            })
        dets.sort(key=lambda d: d["area"], reverse=True)
        return dets

    def learned_list(self):
        with self.lock:
            items = []
            if self.yolo_model is not None:
                hidden = set(self.yolo_hidden)
                items.extend(
                    dict(c) for c in self.yolo_classes if c["id"] not in hidden)
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
            if str(item_id).startswith("yolo-"):
                self.yolo_hidden.add(item_id)
                return
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

        with self.lock:
            cached = list(self._last_yolo_dets) if mode == "yolo" else []

        results = []
        if mode == "yolo":
            results = cached if cached else self._run_yolo(roi_frame, rx1, ry1)
        elif mode == "color" and colors:
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
    """
    List only cameras that can deliver frames, each with tested resolutions.
    Entry shape: {index, name, resolutions:[{width,height}], default:{width,height}}

    Default: return the last catalog immediately (no device I/O) when one
    exists. Pass ?discover=1 to look for newly plugged cameras only — cached
    devices are not re-opened or re-tested.
    """
    discover = str(request.args.get("discover") or "").lower() in (
        "1", "true", "yes",
    )
    cached = _catalog_snapshot()
    if cached and not discover:
        return jsonify({
            "success": True,
            "cameras": cached,
            "current": _state._camera_index if _state.running else None,
            "cached": True,
        })

    skip = _state._camera_index if _state.running else None
    skip_entry = _state.catalog_entry() if _state.running else None
    cameras = enumerate_cameras(
        max_probe=10,
        skip_index=skip,
        skip_entry=skip_entry,
        reuse_cached=True,
        never_probe=_parse_index_set(request.args.get("skip")),
    )
    return jsonify({
        "success": True,
        "cameras": cameras,
        "current": _state._camera_index if _state.running else None,
        "cached": False,
    })


@blueprint.route("/resolution", methods=["GET", "POST"])
def resolution():
    if _state.camera is None or not _state.running:
        return jsonify({"success": False, "error": "Camera not running"})
    if request.method == "GET":
        with _state.lock:
            w, h = _state._last_resolution
        return jsonify({"success": True, "width": w, "height": h})
    data = request.get_json() or {}
    w = data.get("width")
    h = data.get("height")
    if not w or not h:
        return jsonify({"success": False, "error": "Need width and height"})
    # Apply on the capture thread so we never race camera.read().
    applied = _state.request_resolution(int(w), int(h))
    if not applied:
        return jsonify({"success": False, "error": "Camera not running"})
    return jsonify({
        "success": True,
        "width": applied["width"],
        "height": applied["height"],
        "reverted": bool(applied.get("reverted")),
    })


@blueprint.route("/resolutions")
def list_resolutions():
    """Return the camera's frame-tested resolutions."""
    with _state.lock:
        w, h = _state._last_resolution
        res_list = [
            (rw, rh) for rw, rh in _state._supported_resolutions
            if rw > 0 and rh > 0
        ]
    return jsonify({
        "success": True,
        "resolutions": [{"width": rw, "height": rh} for rw, rh in res_list],
        "current": {"width": w, "height": h} if w > 0 and h > 0 else None,
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
    if m in ("color", "shape", "yolo"):
        _state.mode = m
        return jsonify({"success": True, "mode": m})
    return jsonify({"success": False, "error": "Invalid mode"})


@blueprint.route("/model", methods=["GET"])
def model_status():
    return jsonify({"success": True, **_state.yolo_status()})


@blueprint.route("/load-model", methods=["POST"])
def load_model():
    path = None
    upload = request.files.get("file")
    if upload and upload.filename:
        filename = os.path.basename(upload.filename)
        if not _allowed_ckpt_filename(filename):
            return jsonify({"success": False, "error": "Need a .pt or .ckpt file"})
        dest_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_models")
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, filename)
        upload.save(dest)
        path = dest
    else:
        data = request.get_json(silent=True) or {}
        path = (data.get("path") or "").strip() or None

    if not path:
        return jsonify({"success": False, "error": "Choose a local YOLO26n checkpoint"})
    try:
        info = _state.load_yolo(path)
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)})
    return jsonify({"success": True, **info})


@blueprint.route("/unload-model", methods=["POST"])
def unload_model():
    _state.unload_yolo()
    return jsonify({"success": True})


@blueprint.route("/yolo-conf", methods=["GET", "POST"])
def yolo_conf():
    if request.method == "GET":
        return jsonify({"success": True, "conf": _state.yolo_status()["conf"]})
    data = request.get_json(silent=True) or {}
    try:
        conf = _state.set_yolo_conf(data.get("conf"))
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)})
    return jsonify({"success": True, "conf": conf})


@blueprint.route("/learn", methods=["POST"])
def learn():
    data = request.get_json() or {}
    name = (data.get("name") or "").strip() or None

    if _state.mode == "yolo":
        return jsonify({"success": False,
                        "error": "Switch to Color or Shape to learn items"})
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