(function () {
  var feed        = document.getElementById('cvpick-feed');
  var placeholder = document.getElementById('cvpick-placeholder');
  var feedWrap    = document.getElementById('cvpick-feed-wrap');
  var roiOverlay  = document.getElementById('cvpick-roi-overlay');
  var roiRect     = document.getElementById('cvpick-roi-rect');
  var nameInput   = document.getElementById('cvpick-name');
  var learnBtn    = document.getElementById('cvpick-learn-btn');
  var clearBtn    = document.getElementById('cvpick-clear-btn');
  var modelFileInput = document.getElementById('cvpick-model-file');
  var modelBrowseBtn = document.getElementById('cvpick-model-browse');
  var modelFilename  = document.getElementById('cvpick-model-filename');
  var loadBtn     = document.getElementById('cvpick-load-btn');
  var unloadBtn   = document.getElementById('cvpick-unload-btn');
  var yoloConfInput = document.getElementById('cvpick-yolo-conf');
  var yoloSettings = document.getElementById('cvpick-yolo-settings');
  var learnedList = document.getElementById('cvpick-learned-list');
  var colorBtn    = document.getElementById('cvpick-mode-color');
  var shapeBtn    = document.getElementById('cvpick-mode-shape');
  var learnPhBtn  = document.getElementById('cvpick-phase-learn');
  var detectPhBtn = document.getElementById('cvpick-phase-detect');
  var modelPhBtn  = document.getElementById('cvpick-phase-model');
  var learnControls = document.getElementById('cvpick-learn-controls');
  var cameraSelect  = document.getElementById('cvpick-camera-select');
  var cameraStartBtn = document.getElementById('cvpick-camera-start');
  var resSelect     = document.getElementById('cvpick-resolution-select');

  // Calibration DOM refs
  var calibDetectBtn = document.getElementById('cvpick-calib-detect');
  var calibPxA       = document.getElementById('cvpick-calib-px-a');
  var calibPxB       = document.getElementById('cvpick-calib-px-b');
  var calibRobotA    = document.getElementById('cvpick-calib-robot-a');
  var calibRobotB    = document.getElementById('cvpick-calib-robot-b');
  var calibRecordA   = document.getElementById('cvpick-calib-record-a');
  var calibRecordB   = document.getElementById('cvpick-calib-record-b');
  var calibGotoA     = document.getElementById('cvpick-calib-goto-a');
  var calibGotoB     = document.getElementById('cvpick-calib-goto-b');
  var calibPxC       = document.getElementById('cvpick-calib-px-c');
  var calibRobotC    = document.getElementById('cvpick-calib-robot-c');
  var calibRecordC   = document.getElementById('cvpick-calib-record-c');
  var calibGotoC     = document.getElementById('cvpick-calib-goto-c');
  var liftInput      = document.getElementById('cvpick-lift-height');
  var scanOnceBtn    = document.getElementById('cvpick-scan-once');
  var scanEachBtn    = document.getElementById('cvpick-scan-each');
  var calibSaveBtn   = document.getElementById('cvpick-calib-save');
  var calibStatus    = document.getElementById('cvpick-calib-status');
  var markerA        = document.getElementById('cvpick-marker-a');
  var markerB        = document.getElementById('cvpick-marker-b');
  var markerC        = document.getElementById('cvpick-marker-c');

  var polling = false;
  var currentMode = 'color';
  var currentPhase = 'learning';
  var LIFT_MIN = 1;
  var LIFT_MAX = 100;
  var CONF_MIN = 0.05;
  var CONF_MAX = 0.95;
  var liftHeight = 10;
  var scanEach = false;
  var yoloConf = 0.25;
  var pickRunning = false;

  function clampLift(v) {
    if (!isFinite(v)) return null;
    if (v < LIFT_MIN) return LIFT_MIN;
    if (v > LIFT_MAX) return LIFT_MAX;
    return v;
  }

  function savePickSettings() {
    ExtensionAPI.setData('cv-pick', 'pickSettings', {
      liftHeight: liftHeight,
      scanEach: scanEach,
      yoloConf: yoloConf
    });
  }

  function applyLiftHeight(raw, writeInput) {
    var clamped = clampLift(raw);
    if (clamped == null) return false;
    liftHeight = clamped;
    if (writeInput && liftInput) liftInput.value = String(clamped);
    savePickSettings();
    return true;
  }

  function setScanEach(on) {
    scanEach = !!on;
    if (scanOnceBtn) scanOnceBtn.classList.toggle('active', !scanEach);
    if (scanEachBtn) scanEachBtn.classList.toggle('active', scanEach);
    savePickSettings();
  }

  (function loadPickSettings() {
    var s = ExtensionAPI.getData('cv-pick', 'pickSettings');
    if (s) {
      if (s.liftHeight != null) {
        var lh = clampLift(parseFloat(s.liftHeight));
        if (lh != null) liftHeight = lh;
      }
      if (s.scanEach) scanEach = true;
      if (s.yoloConf != null) {
        var yc = parseFloat(s.yoloConf);
        if (isFinite(yc)) {
          if (yc < CONF_MIN) yc = CONF_MIN;
          if (yc > CONF_MAX) yc = CONF_MAX;
          yoloConf = yc;
        }
      }
    }
    if (liftInput) liftInput.value = String(liftHeight);
    if (yoloConfInput) yoloConfInput.value = String(yoloConf);
    if (scanOnceBtn) scanOnceBtn.classList.toggle('active', !scanEach);
    if (scanEachBtn) scanEachBtn.classList.toggle('active', scanEach);
    ExtensionAPI.fetch('cv-pick', '/yolo-conf', {
      method: 'POST',
      body: JSON.stringify({ conf: yoloConf })
    }).catch(function () {});
  })();

  if (liftInput) {
    liftInput.addEventListener('input', function () {
      var raw = parseFloat(this.value);
      if (!isFinite(raw)) return;
      if (raw > LIFT_MAX) {
        this.value = String(LIFT_MAX);
        applyLiftHeight(LIFT_MAX, false);
      } else if (raw >= LIFT_MIN) {
        applyLiftHeight(raw, false);
      }
    });
    liftInput.addEventListener('change', function () {
      var clamped = clampLift(parseFloat(this.value));
      if (clamped == null) clamped = liftHeight;
      this.value = String(clamped);
      applyLiftHeight(clamped, false);
    });
  }
  if (scanOnceBtn) {
    scanOnceBtn.addEventListener('click', function () { setScanEach(false); });
  }
  if (scanEachBtn) {
    scanEachBtn.addEventListener('click', function () { setScanEach(true); });
  }

  function applyYoloConf(raw, writeInput) {
    if (!isFinite(raw)) return false;
    if (raw < CONF_MIN) raw = CONF_MIN;
    if (raw > CONF_MAX) raw = CONF_MAX;
    yoloConf = raw;
    if (writeInput && yoloConfInput) yoloConfInput.value = String(raw);
    savePickSettings();
    ExtensionAPI.fetch('cv-pick', '/yolo-conf', {
      method: 'POST',
      body: JSON.stringify({ conf: yoloConf })
    }).catch(function () {});
    return true;
  }

  if (yoloConfInput) {
    yoloConfInput.addEventListener('input', function () {
      var raw = parseFloat(this.value);
      if (!isFinite(raw)) return;
      if (raw > CONF_MAX) {
        this.value = String(CONF_MAX);
        applyYoloConf(CONF_MAX, false);
      } else if (raw >= CONF_MIN) {
        applyYoloConf(raw, false);
      }
    });
    yoloConfInput.addEventListener('change', function () {
      var raw = parseFloat(this.value);
      if (!applyYoloConf(raw, true)) {
        this.value = String(yoloConf);
      }
    });
  }

  // ---- robot jog controls -----------------------------------------------

  var jogStep = 5;
  var jogBusy = false;
  var pumpOn  = false;
  var selectedPort = ExtensionAPI.getData('cv-pick', 'selectedPort') || null;
  var robotSelect  = document.getElementById('cvpick-robot-select');
  var robotRefresh = document.getElementById('cvpick-robot-refresh');

  function withPort(body) {
    if (selectedPort) body.port = selectedPort;
    return body;
  }

  function requirePort() {
    if (selectedPort) return selectedPort;
    ExtensionAPI.showNotification('Select a robotic arm first', 'error');
    return null;
  }

  function loadRobots() {
    ExtensionAPI.getDevices().then(function (data) {
      if (!data.success || !data.ports) return;
      var prev = selectedPort || robotSelect.value;
      robotSelect.innerHTML = '<option value="">-- select a robot --</option>';
      data.ports.forEach(function (d) {
        if (!d.connected) return;
        var opt = document.createElement('option');
        opt.value = d.port;
        opt.textContent = d.model + ' (' + d.port + ')';
        robotSelect.appendChild(opt);
      });
      if (prev) robotSelect.value = prev;
      if (!robotSelect.value) {
        var def = ExtensionAPI.pickDefaultPort(data.ports);
        if (def) robotSelect.value = def;
      }
      selectedPort = robotSelect.value || null;
      if (selectedPort) ExtensionAPI.setData('cv-pick', 'selectedPort', selectedPort);
    }).catch(function () {});
  }

  robotSelect.addEventListener('change', function () {
    selectedPort = robotSelect.value || null;
    ExtensionAPI.setData('cv-pick', 'selectedPort', selectedPort);
  });
  robotRefresh.addEventListener('click', loadRobots);
  loadRobots();
  ExtensionAPI.onActivate('cv-pick', loadRobots);

  // Step selector (presets + custom 1–50)
  var STEP_MIN = 1;
  var STEP_MAX = 50;
  var stepBtns = document.querySelectorAll('.cvpick-jog-step');
  var stepInput = document.getElementById('cvpick-jog-step-custom');

  function clampJogStep(step) {
    if (!isFinite(step)) return null;
    if (step < STEP_MIN) return STEP_MIN;
    if (step > STEP_MAX) return STEP_MAX;
    return step;
  }

  function applyJogStep(step, fromPreset) {
    var clamped = clampJogStep(step);
    if (clamped == null) return false;
    jogStep = clamped;
    var matched = false;
    for (var k = 0; k < stepBtns.length; k++) {
      var btnStep = parseFloat(stepBtns[k].getAttribute('data-step'));
      var on = btnStep === clamped;
      stepBtns[k].classList.toggle('active', on);
      if (on) matched = true;
    }
    if (stepInput) {
      if (fromPreset) stepInput.value = String(clamped);
      stepInput.classList.toggle('custom-active', !matched);
    }
    return true;
  }

  for (var si = 0; si < stepBtns.length; si++) {
    stepBtns[si].addEventListener('click', function () {
      applyJogStep(parseFloat(this.getAttribute('data-step')), true);
    });
  }
  if (stepInput) {
    stepInput.addEventListener('input', function () {
      var raw = parseFloat(this.value);
      if (!isFinite(raw)) return;
      if (raw > STEP_MAX) {
        this.value = String(STEP_MAX);
        applyJogStep(STEP_MAX, false);
      } else if (raw >= STEP_MIN) {
        applyJogStep(raw, false);
      }
    });
    stepInput.addEventListener('change', function () {
      var clamped = clampJogStep(parseFloat(this.value));
      if (clamped == null) clamped = jogStep;
      this.value = String(clamped);
      applyJogStep(clamped, false);
    });
  }

  // Axis jog buttons — use /cmd/jog for single-axis incremental moves
  var jogBtns = document.querySelectorAll('.cvpick-jog-btn');
  for (var ji = 0; ji < jogBtns.length; ji++) {
    jogBtns[ji].addEventListener('click', function () {
      if (jogBusy) return;
      if (!requirePort()) return;
      var axis = this.getAttribute('data-axis').toUpperCase();
      var dir  = parseInt(this.getAttribute('data-dir'), 10);
      jogBusy = true;

      fetch(ExtensionAPI.getServerUrl() + '/cmd/jog', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(withPort({ mode: 'coord', axis: axis, step: dir * jogStep }))
      }).then(function () { jogBusy = false; })
        .catch(function () { jogBusy = false; });
    });
  }

  // Suction buttons — use /cmd/pump
  var pumpOnBtn  = document.getElementById('cvpick-pump-on');
  var pumpOffBtn = document.getElementById('cvpick-pump-off');
  pumpOnBtn.addEventListener('click', function () {
    if (!requirePort()) return;
    cmdPump(1);
    pumpOn = true;
    pumpOnBtn.classList.add('pump-active');
    pumpOffBtn.classList.remove('pump-active');
  });
  pumpOffBtn.addEventListener('click', function () {
    if (!requirePort()) return;
    cmdPump(0);
    pumpOn = false;
    pumpOffBtn.classList.add('pump-active');
    pumpOnBtn.classList.remove('pump-active');
    setTimeout(function () { pumpOffBtn.classList.remove('pump-active'); }, 300);
  });

  // ---- ROI state & interaction ------------------------------------------

  var roi = { x1: 0.25, y1: 0.2, x2: 0.75, y2: 0.8 };
  var drag = null; // null | { type, startMx, startMy, startRoi }

  /** Rendered image bounds inside the feed element. */
  function feedImageRect() {
    var r = feed.getBoundingClientRect();
    return { left: 0, top: 0, width: r.width, height: r.height };
  }

  /** Hide ROI + calibration markers (used while loading / idle). */
  function hideOverlays() {
    if (roiRect) roiRect.style.display = 'none';
    if (markerA) markerA.style.display = 'none';
    if (markerB) markerB.style.display = 'none';
    if (markerC) markerC.style.display = 'none';
  }

  /** Position the ROI overlay div from normalised roi coords. */
  function updateRoiVisual() {
    // Never show the detection square while loading or without a live feed.
    if (cameraBusy || !cameraRunning || feed.style.display === 'none') {
      if (roiRect) roiRect.style.display = 'none';
      return;
    }
    roiRect.style.display = '';
    var b = feedImageRect();
    roiRect.style.left   = (b.left + roi.x1 * b.width) + 'px';
    roiRect.style.top    = (b.top  + roi.y1 * b.height) + 'px';
    roiRect.style.width  = ((roi.x2 - roi.x1) * b.width) + 'px';
    roiRect.style.height = ((roi.y2 - roi.y1) * b.height) + 'px';
  }

  /** Convert a mouse event to normalised (0-1) image coordinates. */
  function mouseToNorm(e) {
    var wr = feedWrap.getBoundingClientRect();
    var b  = feedImageRect();
    return {
      x: (e.clientX - wr.left - b.left) / b.width,
      y: (e.clientY - wr.top  - b.top)  / b.height
    };
  }

  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

  function roiMouseDown(e) {
    if (cameraBusy || !cameraRunning || feed.style.display === 'none') return;
    // Determine what was grabbed
    var target = e.target;
    var type = null;
    if (target.dataset.corner) type = target.dataset.corner;
    else if (target.dataset.edge) type = target.dataset.edge;
    else if (target === roiRect) type = 'move';
    if (!type) return;

    var m = mouseToNorm(e);
    drag = {
      type: type,
      startMx: m.x,
      startMy: m.y,
      startRoi: { x1: roi.x1, y1: roi.y1, x2: roi.x2, y2: roi.y2 }
    };
    e.preventDefault();
  }

  function roiMouseMove(e) {
    if (!drag) return;
    var m  = mouseToNorm(e);
    var dx = m.x - drag.startMx;
    var dy = m.y - drag.startMy;
    var s  = drag.startRoi;
    var MIN = 0.05;

    switch (drag.type) {
      case 'move':
        var w = s.x2 - s.x1, h = s.y2 - s.y1;
        roi.x1 = clamp(s.x1 + dx, 0, 1 - w);
        roi.y1 = clamp(s.y1 + dy, 0, 1 - h);
        roi.x2 = roi.x1 + w;
        roi.y2 = roi.y1 + h;
        break;
      // corners
      case 'tl':
        roi.x1 = clamp(s.x1 + dx, 0, s.x2 - MIN);
        roi.y1 = clamp(s.y1 + dy, 0, s.y2 - MIN);
        break;
      case 'tr':
        roi.x2 = clamp(s.x2 + dx, s.x1 + MIN, 1);
        roi.y1 = clamp(s.y1 + dy, 0, s.y2 - MIN);
        break;
      case 'bl':
        roi.x1 = clamp(s.x1 + dx, 0, s.x2 - MIN);
        roi.y2 = clamp(s.y2 + dy, s.y1 + MIN, 1);
        break;
      case 'br':
        roi.x2 = clamp(s.x2 + dx, s.x1 + MIN, 1);
        roi.y2 = clamp(s.y2 + dy, s.y1 + MIN, 1);
        break;
      // edges
      case 't':
        roi.y1 = clamp(s.y1 + dy, 0, s.y2 - MIN);
        break;
      case 'b':
        roi.y2 = clamp(s.y2 + dy, s.y1 + MIN, 1);
        break;
      case 'l':
        roi.x1 = clamp(s.x1 + dx, 0, s.x2 - MIN);
        break;
      case 'r':
        roi.x2 = clamp(s.x2 + dx, s.x1 + MIN, 1);
        break;
    }
    updateRoiVisual();
  }

  function roiMouseUp() {
    if (!drag) return;
    drag = null;
    sendRoi();
  }

  function sendRoi() {
    ExtensionAPI.fetch('cv-pick', '/roi', {
      method: 'POST',
      body: JSON.stringify({ roi: [roi.x1, roi.y1, roi.x2, roi.y2] })
    });
  }

  // Attach ROI listeners
  roiRect.addEventListener('mousedown', roiMouseDown);
  document.addEventListener('mousemove', roiMouseMove);
  document.addEventListener('mouseup', roiMouseUp);
  feed.addEventListener('load', function () { updateRoiVisual(); updateMarkers(); });
  window.addEventListener('resize', function () { updateRoiVisual(); updateMarkers(); });

  // ---- camera & resolution dropdowns ------------------------------------

  var cameraBusy = false;
  var cameraRunning = false;
  /** index -> { resolutions:[{width,height}], default:{width,height} } */
  var cameraMeta = {};
  /** Last confirmed live resolution key, e.g. "1280x720". */
  var lastGoodResKey = '';
  /** Last successful /cameras payload — shown instantly on tab re-enter. */
  var lastCameraPayload = null;
  /** In-flight /stop; re-enter waits so the device is actually released. */
  var stopPromise = Promise.resolve();
  /** Bumped on each activate/deactivate so stale list requests skip UI. */
  var listRequestId = 0;

  /**
   * Show the full-area cover over the feed (hides ROI / blank frame).
   * @param {string} msg
   * @param {boolean} [loading] show spinner when true
   */
  function setPlaceholder(msg, loading) {
    feed.style.display = 'none';
    placeholder.style.display = '';
    placeholder.classList.toggle('is-loading', !!loading);
    if (feedWrap) feedWrap.classList.toggle('is-loading', !!loading);
    placeholder.querySelector('p').textContent =
      msg || 'Select a camera and click Start';
    hideOverlays();
    updateRoiVisual();
  }

  function setIdlePlaceholder(msg) {
    setPlaceholder(msg || 'Select a camera and click Start', false);
  }

  function showLiveFeed() {
    placeholder.classList.remove('is-loading');
    if (feedWrap) feedWrap.classList.remove('is-loading');
    placeholder.style.display = 'none';
    feed.style.display = 'block';
    updateRoiVisual();
  }

  function updateStartBtn() {
    if (!cameraStartBtn) return;
    cameraStartBtn.textContent = cameraRunning ? 'Stop' : 'Start';
    cameraStartBtn.classList.toggle('is-stop', cameraRunning);
    cameraStartBtn.title = cameraRunning
      ? 'Stop the camera'
      : 'Start the selected camera';
  }

  function setCameraBusy(busy) {
    cameraBusy = !!busy;
    if (cameraSelect) cameraSelect.disabled = cameraBusy;
    if (cameraStartBtn) cameraStartBtn.disabled = cameraBusy;
    if (resSelect) resSelect.disabled = cameraBusy || !cameraRunning;
    if (feedWrap) feedWrap.classList.toggle('is-loading', cameraBusy);
    if (cameraBusy) hideOverlays();
    updateRoiVisual();
  }

  function selectedCameraName() {
    if (!cameraSelect || cameraSelect.selectedIndex < 0) return '';
    var opt = cameraSelect.options[cameraSelect.selectedIndex];
    return (opt && opt.textContent) || '';
  }

  function cameraNameSet(payload) {
    var set = {};
    var cams = (payload && payload.cameras) || [];
    for (var i = 0; i < cams.length; i++) {
      var n = (cams[i] && typeof cams[i] === 'object') ? cams[i].name : '';
      if (n) set[n] = true;
    }
    return set;
  }

  function populateCameras(data) {
    if (!data || !data.success || !cameraSelect) return;
    cameraMeta = {};
    cameraSelect.innerHTML = '';
    (data.cameras || []).forEach(function (cam) {
      var idx = (cam && typeof cam === 'object') ? cam.index : cam;
      var name = (cam && typeof cam === 'object' && cam.name)
        ? cam.name
        : ('Camera ' + idx);
      // Backend already drops cameras with no working resolution.
      if (cam && typeof cam === 'object') {
        cameraMeta[String(idx)] = {
          resolutions: Array.isArray(cam.resolutions) ? cam.resolutions : [],
          default: cam.default || null
        };
      }
      var opt = document.createElement('option');
      opt.value = String(idx);
      opt.textContent = name;
      cameraSelect.appendChild(opt);
    });
  }

  function restoreCameraSelection(data, keepName, oldNames) {
    if (!cameraSelect) return;
    if (data && data.current !== null && data.current !== undefined) {
      cameraSelect.value = String(data.current);
      return;
    }
    var cams = (data && data.cameras) || [];
    if (oldNames) {
      for (var i = 0; i < cams.length; i++) {
        var n = (cams[i] && cams[i].name) || '';
        if (n && !oldNames[n] && cams[i].index !== undefined
            && cams[i].index !== null && cams[i].index !== '') {
          cameraSelect.value = String(cams[i].index);
          return;
        }
      }
    }
    if (keepName) {
      for (var j = 0; j < cameraSelect.options.length; j++) {
        if (cameraSelect.options[j].textContent === keepName) {
          cameraSelect.selectedIndex = j;
          return;
        }
      }
    }
  }

  function applyCameraList(data, opts) {
    opts = opts || {};
    if (!data || !data.success) return false;
    lastCameraPayload = data;
    populateCameras(data);
    restoreCameraSelection(data, opts.keepName, opts.oldNames);
    return !!(data.cameras && data.cameras.length);
  }

  function cachedCameraIndices() {
    var cams = (lastCameraPayload && lastCameraPayload.cameras) || [];
    var ids = [];
    for (var i = 0; i < cams.length; i++) {
      var cam = cams[i];
      var idx = (cam && typeof cam === 'object') ? cam.index : cam;
      if (idx === undefined || idx === null || idx === '') continue;
      ids.push(String(idx));
    }
    return ids;
  }

  function mergeCameraPayload(prev, fresh) {
    if (!fresh || !fresh.success) return prev;
    if (!prev || !prev.cameras || !prev.cameras.length) return fresh;
    // Fresh enumerate is the occupancy source (OS index + name).
    // Do not keep stale prev-at-index-0 (OBS) after USB takes that slot.
    var prevByName = {};
    (prev.cameras || []).forEach(function (cam) {
      if (cam && cam.name) prevByName[cam.name] = cam;
    });
    var cameras = (fresh.cameras || []).map(function (cam) {
      if (!cam || typeof cam !== 'object') return cam;
      if (cam.resolutions && cam.resolutions.length) return cam;
      return prevByName[cam.name] || cam;
    });
    return {
      success: true,
      cameras: cameras,
      current: (fresh.current != null && fresh.current !== undefined)
        ? fresh.current
        : prev.current
    };
  }

  function loadCameras(query) {
    var path = '/cameras' + (query || '');
    return ExtensionAPI.fetch('cv-pick', path).then(function (data) {
      return data;
    }).catch(function () { return null; });
  }

  function delayMs(ms) {
    return new Promise(function (resolve) { setTimeout(resolve, ms); });
  }

  /** Retry a cold first list: backend spawn + camera warmup often miss once. */
  function loadCamerasRetry(query, attempts) {
    attempts = attempts || 1;
    function once(left) {
      return loadCameras(query).then(function (data) {
        var ok = data && data.success && data.cameras && data.cameras.length;
        if (ok || left <= 1) return data;
        return delayMs(800).then(function () { return once(left - 1); });
      });
    }
    return once(attempts);
  }

  /** Ensure resSelect has option w×h and select it (avoids blank dropdown). */
  function selectResolution(w, h) {
    w = parseInt(w, 10);
    h = parseInt(h, 10);
    // OpenCV can report -1 when CAP_PROP is unavailable — never show that.
    if (!resSelect || !w || !h || w < 0 || h < 0) return;
    var key = w + 'x' + h;
    if (!resSelect.querySelector('option[value="' + key + '"]')) {
      var opt = document.createElement('option');
      opt.value = key;
      opt.textContent = key;
      resSelect.appendChild(opt);
    }
    resSelect.value = key;
    lastGoodResKey = key;
  }

  function fillResolutionSelect(resolutions, current) {
    if (!resSelect) return;
    resSelect.innerHTML = '';
    (resolutions || []).forEach(function (r) {
      var w = parseInt(r.width, 10);
      var h = parseInt(r.height, 10);
      if (!w || !h || w < 0 || h < 0) return;
      var opt = document.createElement('option');
      opt.value = w + 'x' + h;
      opt.textContent = w + 'x' + h;
      resSelect.appendChild(opt);
    });
    if (current && current.width > 0 && current.height > 0) {
      selectResolution(current.width, current.height);
    } else if (resSelect.options.length) {
      resSelect.selectedIndex = 0;
    }
  }

  /** Prefer frame-tested list from /cameras catalog; fall back to /resolutions. */
  function loadResolutions(camIdx) {
    var meta = cameraMeta[String(camIdx)];
    if (meta && meta.resolutions && meta.resolutions.length) {
      fillResolutionSelect(meta.resolutions, meta.default);
      // Sync current from backend if live.
      ExtensionAPI.fetch('cv-pick', '/resolutions').then(function (data) {
        if (data && data.success && data.current && data.current.width > 0) {
          selectResolution(data.current.width, data.current.height);
        }
      }).catch(function () {});
      return;
    }
    ExtensionAPI.fetch('cv-pick', '/resolutions').then(function (data) {
      if (!data.success) return;
      fillResolutionSelect(data.resolutions, data.current);
    }).catch(function () {});
  }

  /** Poll /frame until a JPEG arrives or timeout (~12s, covers AE warmup). */
  function waitForFirstFrame() {
    var attempts = 0;
    var maxAttempts = 60;
    return new Promise(function (resolve) {
      function tick() {
        ExtensionAPI.fetch('cv-pick', '/frame').then(function (data) {
          if (data && data.success && data.image) {
            feed.src = 'data:image/jpeg;base64,' + data.image;
            showLiveFeed();
            resolve(true);
            return;
          }
          attempts += 1;
          if (attempts >= maxAttempts) {
            resolve(false);
            return;
          }
          setTimeout(tick, 200);
        }).catch(function () {
          attempts += 1;
          if (attempts >= maxAttempts) {
            resolve(false);
            return;
          }
          setTimeout(tick, 250);
        });
      }
      tick();
    });
  }

  /**
   * Open a camera index. Disables controls until success (first frame)
   * or failure (unavailable / timeout / backend error).
   */
  function openCamera(idx) {
    setCameraBusy(true);
    polling = false;
    setPlaceholder('Loading camera\u2026', true);

    return ExtensionAPI.fetch('cv-pick', '/start', {
      method: 'POST',
      body: JSON.stringify({ camera: idx })
    }).then(function (data) {
      if (!data || !data.success) {
        cameraRunning = false;
        updateStartBtn();
        setPlaceholder((data && data.error) || 'Camera unavailable', false);
        ExtensionAPI.showNotification(
          (data && data.error) || 'Camera failed', 'error');
        setCameraBusy(false);
        return false;
      }
      setPlaceholder('Loading camera\u2026', true);
      return waitForFirstFrame().then(function (ok) {
        if (ok) {
          cameraRunning = true;
          loadResolutions(idx);
          updateStartBtn();
          // Clear busy before starting the poll loop — pollFrame() no-ops
          // while cameraBusy is true, which froze the feed on frame 1.
          setCameraBusy(false);
          polling = true;
          pollFrame();
        } else {
          cameraRunning = false;
          setPlaceholder('Camera unavailable', false);
          updateStartBtn();
          setCameraBusy(false);
        }
        return ok;
      });
    }).catch(function () {
      cameraRunning = false;
      updateStartBtn();
      setPlaceholder('Backend not reachable', false);
      setCameraBusy(false);
      return false;
    });
  }

  function releaseCameraBackend() {
    stopPromise = ExtensionAPI.fetch('cv-pick', '/stop', { method: 'POST' })
      .catch(function () { return null; });
    return stopPromise;
  }

  function stopCameraStream() {
    polling = false;
    cameraRunning = false;
    updateStartBtn();
    releaseCameraBackend();
    if (resSelect) resSelect.innerHTML = '';
    setIdlePlaceholder('Select a camera and click Start');
    setCameraBusy(false);
  }

  // Changing camera only switches while a session is already running.
  cameraSelect.addEventListener('change', function () {
    if (cameraBusy || !cameraRunning) return;
    var idx = parseInt(cameraSelect.value, 10);
    if (isNaN(idx)) return;
    openCamera(idx);
  });

  function discoverThenStart() {
    var keepName = selectedCameraName();
    var oldNames = cameraNameSet(lastCameraPayload);
    var skip = cachedCameraIndices();
    var q = '?discover=1';
    if (skip.length) q += '&skip=' + encodeURIComponent(skip.join(','));
    setCameraBusy(true);
    setPlaceholder('Loading camera\u2026', true);
    return loadCameras(q).then(function (data) {
      if (data && data.success) {
        data = lastCameraPayload
          ? mergeCameraPayload(lastCameraPayload, data)
          : data;
        applyCameraList(data, { keepName: keepName, oldNames: oldNames });
      }
      var idx = parseInt(cameraSelect.value, 10);
      if (isNaN(idx)) {
        setCameraBusy(false);
        ExtensionAPI.showNotification('Select a camera first', 'error');
        return false;
      }
      return openCamera(idx);
    }).catch(function () {
      var idx = parseInt(cameraSelect.value, 10);
      if (isNaN(idx)) {
        setCameraBusy(false);
        setPlaceholder('Backend not reachable', false);
        return false;
      }
      return openCamera(idx);
    });
  }

  if (cameraStartBtn) {
    cameraStartBtn.addEventListener('click', function () {
      if (cameraBusy) return;
      if (cameraRunning) {
        stopCameraStream();
        return;
      }
      discoverThenStart();
    });
  }

  resSelect.addEventListener('change', function () {
    if (!cameraRunning || cameraBusy) return;
    var parts = resSelect.value.split('x');
    var reqW = parseInt(parts[0], 10);
    var reqH = parseInt(parts[1], 10);
    if (!reqW || !reqH) return;
    var fallbackKey = lastGoodResKey;

    setCameraBusy(true);
    setPlaceholder('Loading camera\u2026', true);

    ExtensionAPI.fetch('cv-pick', '/resolution', {
      method: 'POST',
      body: JSON.stringify({ width: reqW, height: reqH })
    }).then(function (data) {
      if (data && data.success && data.width > 0 && data.height > 0) {
        selectResolution(data.width, data.height);
        if (data.reverted) {
          ExtensionAPI.showNotification(
            'That resolution is not usable; kept ' + data.width + 'x' + data.height,
            'error');
        }
      } else if (fallbackKey && resSelect.querySelector('option[value="' + fallbackKey + '"]')) {
        resSelect.value = fallbackKey;
        lastGoodResKey = fallbackKey;
      }
      setCameraBusy(false);
      if (cameraRunning && polling) {
        showLiveFeed();
      }
    }).catch(function () {
      if (fallbackKey && resSelect.querySelector('option[value="' + fallbackKey + '"]')) {
        resSelect.value = fallbackKey;
        lastGoodResKey = fallbackKey;
      }
      setCameraBusy(false);
      if (cameraRunning && polling) {
        showLiveFeed();
      }
    });
  });

  // ---- helpers ----------------------------------------------------------

  function escHtml(s) {
    var el = document.createElement('span');
    el.textContent = s;
    return el.innerHTML;
  }

  // ---- frame polling (replaces MJPEG — works with file:// origin) -------

  function pollFrame() {
    if (!polling) return;
    // Skip work while opening/switching, but keep the loop alive so frames
    // resume automatically when busy clears.
    if (cameraBusy) {
      setTimeout(pollFrame, 100);
      return;
    }
    ExtensionAPI.fetch('cv-pick', '/frame').then(function (data) {
      if (!polling) return;
      if (cameraBusy) {
        setTimeout(pollFrame, 100);
        return;
      }
      if (data.success) {
        feed.src = 'data:image/jpeg;base64,' + data.image;
        showLiveFeed();
      }
      if (polling) setTimeout(pollFrame, 33); // ~30 fps cap
    }).catch(function () {
      if (polling) setTimeout(pollFrame, 500); // back off on error
    });
  }

  var refreshTimer = null;

  function showListResult(data, hadCache, keepName, oldNames) {
    setCameraBusy(false);
    if (data && data.success) {
      applyCameraList(data, { keepName: keepName, oldNames: oldNames });
      if (!data.cameras || !data.cameras.length) {
        setPlaceholder('No cameras found', false);
      } else {
        setIdlePlaceholder('Select a camera and click Start');
      }
    } else if (!hadCache) {
      setPlaceholder('Could not list cameras', false);
    }
    refreshLearned();
  }

  /**
   * Tab open / return from another tab.
   * If we already listed cameras, paint that list immediately and look for
   * newly plugged devices in the background (cached devices are not re-tested).
   */
  function prepareCameraTab() {
    var req = ++listRequestId;
    var hadCache = !!(lastCameraPayload && lastCameraPayload.cameras &&
      lastCameraPayload.cameras.length);
    var keepName = selectedCameraName();
    var oldNames = cameraNameSet(lastCameraPayload);

    ExtensionAPI.fetch('cv-pick', '/roi').then(function (data) {
      if (data.success && data.roi && data.roi.length === 4) {
        roi.x1 = data.roi[0]; roi.y1 = data.roi[1];
        roi.x2 = data.roi[2]; roi.y2 = data.roi[3];
      }
    }).catch(function () {});

    cameraRunning = false;
    polling = false;
    updateStartBtn();
    if (resSelect) resSelect.innerHTML = '';

    if (hadCache) {
      applyCameraList(lastCameraPayload, { keepName: keepName });
      setCameraBusy(false);
      setIdlePlaceholder('Select a camera and click Start');
    } else {
      setCameraBusy(true);
      setPlaceholder('Loading camera list\u2026', true);
    }

    function afterStop() {
      if (req !== listRequestId) return;
      var load;
      if (hadCache) {
        var skip = cachedCameraIndices();
        var q = '?discover=1';
        if (skip.length) q += '&skip=' + encodeURIComponent(skip.join(','));
        load = loadCameras(q);
      } else {
        load = loadCamerasRetry('', 3);
      }
      return load.then(function (data) {
        if (hadCache && data && data.success) {
          data = mergeCameraPayload(lastCameraPayload, data);
        }
        if (data && data.success) lastCameraPayload = data;
        if (req !== listRequestId) return;
        showListResult(data, hadCache, keepName, oldNames);
      }).catch(function () {
        if (req !== listRequestId) return;
        setCameraBusy(false);
        if (!hadCache) setPlaceholder('Backend not reachable', false);
      });
    }

    (stopPromise || Promise.resolve()).then(afterStop).catch(afterStop);

    if (!refreshTimer) {
      refreshTimer = setInterval(function () {
        refreshLearned();
        refreshPickTargets();
      }, 3000);
    }
  }

  function stopCamera() {
    polling = false;
    cameraRunning = false;
    updateStartBtn();
    setCameraBusy(false);
    hideOverlays();
    listRequestId += 1;
    if (refreshTimer) {
      clearInterval(refreshTimer);
      refreshTimer = null;
    }
    // Release the device before another tab (or a later re-enter) opens it.
    releaseCameraBackend();
  }

  // Lifecycle: list cameras on enter; release device when leaving
  ExtensionAPI.onActivate('cv-pick', prepareCameraTab);
  ExtensionAPI.onDeactivate('cv-pick', stopCamera);

  // First open (IIFE runs on first tab click via lazy loading)
  prepareCameraTab();
  updateStartBtn();

  // ---- mode toggle ------------------------------------------------------

  function listMode() {
    return currentPhase === 'yolo' ? 'yolo' : currentMode;
  }

  function syncBackendMode() {
    var mode = listMode();
    var phase = currentPhase === 'yolo' ? 'inference' : currentPhase;
    ExtensionAPI.fetch('cv-pick', '/mode', {
      method: 'POST',
      body: JSON.stringify({ mode: mode })
    });
    ExtensionAPI.fetch('cv-pick', '/phase', {
      method: 'POST',
      body: JSON.stringify({ phase: phase })
    });
  }

  function setMode(mode) {
    if (currentPhase === 'yolo') return;
    currentMode = mode;
    colorBtn.classList.toggle('active', mode === 'color');
    shapeBtn.classList.toggle('active', mode === 'shape');
    syncBackendMode();
    refreshLearned();
    refreshPickTargets();
  }

  colorBtn.addEventListener('click', function () { setMode('color'); });
  shapeBtn.addEventListener('click', function () { setMode('shape'); });

  // ---- phase toggle (Learn / Detect / Model) ----------------------------

  function setPhase(phase) {
    currentPhase = phase;
    learnPhBtn.classList.toggle('active', phase === 'learning');
    detectPhBtn.classList.toggle('active', phase === 'inference');
    if (modelPhBtn) modelPhBtn.classList.toggle('active', phase === 'yolo');
    colorBtn.disabled = phase === 'yolo';
    shapeBtn.disabled = phase === 'yolo';
    if (yoloSettings) yoloSettings.hidden = phase !== 'yolo';
    if (learnControls) {
      learnControls.style.display = phase === 'learning' ? '' : 'none';
    }
    syncBackendMode();
    if (phase === 'learning') {
      calibState.pixelA = null;
      calibState.pixelB = null;
      calibState.pixelC = null;
      updateCalibUI();
    } else {
      updateCalibDetectBtn();
    }
    refreshLearned();
    refreshPickTargets();
  }

  learnPhBtn.addEventListener('click', function () { setPhase('learning'); });
  detectPhBtn.addEventListener('click', function () { setPhase('inference'); });
  if (modelPhBtn) {
    modelPhBtn.addEventListener('click', function () { setPhase('yolo'); });
  }

  // ---- learn ------------------------------------------------------------

  learnBtn.addEventListener('click', function () {
    var name = nameInput.value.trim();
    learnBtn.disabled = true;
    learnBtn.textContent = 'Learning\u2026';

    ExtensionAPI.fetch('cv-pick', '/learn', {
      method: 'POST',
      body: JSON.stringify({ name: name || undefined })
    }).then(function (data) {
      learnBtn.disabled = false;
      learnBtn.textContent = 'Learn';
      if (data.success) {
        nameInput.value = '';
        refreshLearned();
        ExtensionAPI.showNotification('Learned: ' + data.item.name, 'info');
      } else {
        ExtensionAPI.showNotification(data.error || 'Learning failed', 'error');
      }
    }).catch(function () {
      learnBtn.disabled = false;
      learnBtn.textContent = 'Learn';
    });
  });

  // ---- load YOLO26n checkpoint ------------------------------------------

  var selectedModelFile = null;

  function setModelFilename(text) {
    var label = text || 'No file';
    if (modelFilename) {
      modelFilename.textContent = label;
      modelFilename.title = label;
    }
  }

  function selectedCkptPath() {
    if (!selectedModelFile) return '';
    return selectedModelFile.path || '';
  }

  if (modelBrowseBtn && modelFileInput) {
    modelBrowseBtn.addEventListener('click', function () {
      modelFileInput.click();
    });
    modelFileInput.addEventListener('change', function () {
      selectedModelFile = this.files && this.files[0] ? this.files[0] : null;
      setModelFilename(selectedModelFile ? selectedModelFile.name : '');
      if (loadBtn) loadBtn.disabled = !selectedModelFile;
    });
  }

  function loadModel() {
    if (!selectedModelFile) {
      ExtensionAPI.showNotification('Choose a YOLO26n .pt checkpoint', 'error');
      return;
    }
    loadBtn.disabled = true;
    loadBtn.textContent = 'Loading\u2026';

    var path = selectedCkptPath();
    var req;
    if (path) {
      req = ExtensionAPI.fetch('cv-pick', '/load-model', {
        method: 'POST',
        body: JSON.stringify({ path: path })
      });
    } else {
      var form = new FormData();
      form.append('file', selectedModelFile);
      req = fetch(ExtensionAPI.getServerUrl() + '/ext/cv-pick/load-model', {
        method: 'POST',
        body: form
      }).then(function (resp) { return resp.json(); });
    }

    req.then(function (data) {
      loadBtn.textContent = 'Load';
      loadBtn.disabled = !selectedModelFile;
      if (data.success) {
        setModelFilename(data.filename || selectedModelFile.name);
        refreshLearned();
        refreshPickTargets();
        ExtensionAPI.showNotification(
          'Loaded YOLO26n (' + (data.classes || 0) + ' classes)',
          'info');
      } else {
        ExtensionAPI.showNotification(data.error || 'Failed to load checkpoint', 'error');
      }
    }).catch(function () {
      loadBtn.textContent = 'Load';
      loadBtn.disabled = !selectedModelFile;
      ExtensionAPI.showNotification('Failed to load checkpoint', 'error');
    });
  }

  if (loadBtn) loadBtn.addEventListener('click', loadModel);

  function refreshModelStatus() {
    ExtensionAPI.fetch('cv-pick', '/model').then(function (data) {
      if (data.success && data.loaded) {
        setModelFilename(data.filename || 'YOLO26n loaded');
      }
    }).catch(function () {});
  }

  // ---- clear all --------------------------------------------------------

  clearBtn.addEventListener('click', function () {
    ExtensionAPI.fetch('cv-pick', '/clear', { method: 'POST' }).then(function () {
      refreshLearned();
      refreshPickTargets();
    });
  });

  if (unloadBtn) {
    unloadBtn.addEventListener('click', function () {
      ExtensionAPI.fetch('cv-pick', '/unload-model', { method: 'POST' }).then(function (data) {
        if (!data.success) {
          ExtensionAPI.showNotification(data.error || 'Unload failed', 'error');
          return;
        }
        if (!selectedModelFile) setModelFilename('');
        refreshLearned();
        refreshPickTargets();
        ExtensionAPI.showNotification('YOLO model unloaded', 'info');
      });
    });
  }

  // ---- learned-items list -----------------------------------------------

  function refreshLearned() {
    var titleEl = document.getElementById('cvpick-learned-title');
    var emptyMsg;
    if (listMode() === 'yolo') {
      if (titleEl) titleEl.textContent = 'YOLO Classes';
      emptyMsg = 'Load a YOLO26n checkpoint in Settings to detect objects.';
    } else if (currentMode === 'shape') {
      if (titleEl) titleEl.textContent = 'Learned Shapes';
      emptyMsg = 'No shapes learned yet. Place an object in the zone and click Learn.';
    } else {
      if (titleEl) titleEl.textContent = 'Learned Colors';
      emptyMsg = 'No colors learned yet. Place an object in the zone and click Learn.';
    }

    ExtensionAPI.fetch('cv-pick', '/learned').then(function (data) {
      var items = (data.success && data.items) ? data.items : [];
      var filtered = items.filter(function (item) {
        return item.mode === listMode();
      });

      if (filtered.length === 0) {
        learnedList.innerHTML = '<p class="cvpick-empty">' + emptyMsg + '</p>';
        return;
      }

      var html = '';
      filtered.forEach(function (item) {
        // display_color is BGR from OpenCV — swap to RGB for CSS
        var r = item.display_color[2];
        var g = item.display_color[1];
        var b = item.display_color[0];
        var rgb = 'rgb(' + r + ',' + g + ',' + b + ')';

        html += '<div class="cvpick-item">'
          + '<span class="cvpick-dot" style="background:' + rgb + '"></span>'
          + '<span class="cvpick-item-name">' + escHtml(item.name) + '</span>'
          + '<button class="cvpick-item-rm" data-id="' + item.id + '">'
          + '&times;</button>'
          + '</div>';
      });
      learnedList.innerHTML = html;

      var rmBtns = learnedList.querySelectorAll('.cvpick-item-rm');
      for (var i = 0; i < rmBtns.length; i++) {
        rmBtns[i].addEventListener('click', function () {
          var id = this.getAttribute('data-id');
          ExtensionAPI.fetch('cv-pick', '/remove', {
            method: 'POST',
            body: JSON.stringify({ id: id })
          }).then(refreshLearned);
        });
      }
    });
  }

  // Periodic refresh is now managed by startCamera() / stopCamera()

  // ---- calibration -------------------------------------------------------

  var calibState = {
    pixelA: null, pixelB: null, pixelC: null,   // [px, py]
    robotA: null, robotB: null, robotC: null,   // {x, y, z}
  };
  var isCalibrated = false;

  function fmtPx(pt) {
    return '(' + pt[0] + ', ' + pt[1] + ') px';
  }
  function fmtRobot(r) {
    return 'X' + r.x + ' Y' + r.y + ' Z' + r.z;
  }

  function positionMarker(el, pixelPt) {
    if (!pixelPt || !feed.naturalWidth) {
      el.style.display = 'none';
      return;
    }
    var scaleX = feed.clientWidth / feed.naturalWidth;
    var scaleY = feed.clientHeight / feed.naturalHeight;
    el.style.display = '';
    el.style.left = (pixelPt[0] * scaleX) + 'px';
    el.style.top  = (pixelPt[1] * scaleY) + 'px';
  }

  function updateMarkers() {
    positionMarker(markerA, calibState.pixelA);
    positionMarker(markerB, calibState.pixelB);
    positionMarker(markerC, calibState.pixelC);
  }

  function updateCalibUI() {
    calibPxA.textContent = calibState.pixelA ? fmtPx(calibState.pixelA) : '\u2014';
    calibPxB.textContent = calibState.pixelB ? fmtPx(calibState.pixelB) : '\u2014';
    calibPxC.textContent = calibState.pixelC ? fmtPx(calibState.pixelC) : '\u2014';
    calibRobotA.textContent = calibState.robotA ? fmtRobot(calibState.robotA) : '\u2014';
    calibRobotB.textContent = calibState.robotB ? fmtRobot(calibState.robotB) : '\u2014';
    calibRobotC.textContent = calibState.robotC ? fmtRobot(calibState.robotC) : '\u2014';

    calibRecordA.disabled = !calibState.pixelA;
    calibRecordB.disabled = !calibState.pixelB;
    calibRecordC.disabled = !calibState.pixelC;
    if (calibGotoA) {
      calibGotoA.hidden = !calibState.robotA;
      calibGotoA.disabled = !calibState.robotA || pickRunning;
    }
    if (calibGotoB) {
      calibGotoB.hidden = !calibState.robotB;
      calibGotoB.disabled = !calibState.robotB || pickRunning;
    }
    if (calibGotoC) {
      calibGotoC.hidden = !calibState.robotC;
      calibGotoC.disabled = !calibState.robotC || pickRunning;
    }
    calibSaveBtn.disabled = !(calibState.pixelA && calibState.pixelB && calibState.pixelC
                              && calibState.robotA && calibState.robotB && calibState.robotC);
    updateMarkers();
    updateCalibDetectBtn();
  }

  function updateCalibDetectBtn() {
    var inDetect = currentPhase === 'inference' || currentPhase === 'yolo';
    calibDetectBtn.disabled = !inDetect || pickRunning;
    calibDetectBtn.title = inDetect
      ? 'Detect 3 objects in the zone as calibration markers'
      : 'Switch to Detect or Model to detect markers';
  }

  function saveCalibPoses() {
    ExtensionAPI.setData('cv-pick', 'calibPoses', {
      robotA: calibState.robotA,
      robotB: calibState.robotB,
      robotC: calibState.robotC
    });
  }

  // Load existing calibration on startup
  function loadCalibration() {
    var savedPoses = ExtensionAPI.getData('cv-pick', 'calibPoses');
    if (savedPoses) {
      if (savedPoses.robotA) calibState.robotA = savedPoses.robotA;
      if (savedPoses.robotB) calibState.robotB = savedPoses.robotB;
      if (savedPoses.robotC) calibState.robotC = savedPoses.robotC;
      updateCalibUI();
    }
    ExtensionAPI.fetch('cv-pick', '/calibration').then(function (data) {
      if (data.success && data.calibration) {
        isCalibrated = true;
        calibStatus.textContent = 'Calibrated';
        calibStatus.classList.add('calibrated');
        updatePickUI();
      }
    }).catch(function () {});
  }

  // Detect markers — pick 3 largest detections (Detect mode only)
  calibDetectBtn.addEventListener('click', function () {
    if (currentPhase !== 'inference' && currentPhase !== 'yolo') {
      ExtensionAPI.showNotification('Switch to Detect or Model to detect markers', 'error');
      return;
    }
    calibDetectBtn.disabled = true;
    calibDetectBtn.textContent = 'Detecting\u2026';
    ExtensionAPI.fetch('cv-pick', '/detections').then(function (data) {
      calibDetectBtn.textContent = 'Detect Markers';
      updateCalibDetectBtn();
      if (currentPhase !== 'inference' && currentPhase !== 'yolo') return;
      if (!data.success || !data.detections || data.detections.length < 3) {
        ExtensionAPI.showNotification(
          'Need at least 3 detected objects. Learn an item, switch to Detect, and place 3 markers in an L-shape.',
          'error');
        return;
      }
      var d = data.detections;
      calibState.pixelA = d[0].center_px;
      calibState.pixelB = d[1].center_px;
      calibState.pixelC = d[2].center_px;
      calibState.robotA = null;
      calibState.robotB = null;
      calibState.robotC = null;
      saveCalibPoses();
      updateCalibUI();
      ExtensionAPI.showNotification(
        'Markers detected: A (red), B (blue), C (green)',
        'info');
    }).catch(function () {
      calibDetectBtn.textContent = 'Detect Markers';
      updateCalibDetectBtn();
    });
  });

  // Record robot position for point A
  calibRecordA.addEventListener('click', function () {
    if (!requirePort()) return;
    ExtensionAPI.getRobotStatus(selectedPort).then(function (status) {
      if (!status.success) {
        ExtensionAPI.showNotification(status.error || 'Cannot read robot position', 'error');
        return;
      }
      calibState.robotA = {
        x: status.coordinates.X,
        y: status.coordinates.Y,
        z: status.coordinates.Z
      };
      saveCalibPoses();
      updateCalibUI();
    });
  });

  // Record robot position for point B
  calibRecordB.addEventListener('click', function () {
    if (!requirePort()) return;
    ExtensionAPI.getRobotStatus(selectedPort).then(function (status) {
      if (!status.success) {
        ExtensionAPI.showNotification(status.error || 'Cannot read robot position', 'error');
        return;
      }
      calibState.robotB = {
        x: status.coordinates.X,
        y: status.coordinates.Y,
        z: status.coordinates.Z
      };
      saveCalibPoses();
      updateCalibUI();
    });
  });

  // Record robot position for point C
  calibRecordC.addEventListener('click', function () {
    if (!requirePort()) return;
    ExtensionAPI.getRobotStatus(selectedPort).then(function (status) {
      if (!status.success) {
        ExtensionAPI.showNotification(status.error || 'Cannot read robot position', 'error');
        return;
      }
      calibState.robotC = {
        x: status.coordinates.X,
        y: status.coordinates.Y,
        z: status.coordinates.Z
      };
      saveCalibPoses();
      updateCalibUI();
    });
  });

  function gotoCalibPose(pos) {
    if (!requirePort()) return;
    if (!pos) return;
    cmdMove(pos.x, pos.y, pos.z);
  }

  if (calibGotoA) {
    calibGotoA.addEventListener('click', function () { gotoCalibPose(calibState.robotA); });
  }
  if (calibGotoB) {
    calibGotoB.addEventListener('click', function () { gotoCalibPose(calibState.robotB); });
  }
  if (calibGotoC) {
    calibGotoC.addEventListener('click', function () { gotoCalibPose(calibState.robotC); });
  }

  // Save calibration
  calibSaveBtn.addEventListener('click', function () {
    var avgZ = (calibState.robotA.z + calibState.robotB.z + calibState.robotC.z) / 3;
    ExtensionAPI.fetch('cv-pick', '/calibration', {
      method: 'POST',
      body: JSON.stringify({
        pixel_points: [calibState.pixelA, calibState.pixelB, calibState.pixelC],
        robot_points: [
          [calibState.robotA.x, calibState.robotA.y],
          [calibState.robotB.x, calibState.robotB.y],
          [calibState.robotC.x, calibState.robotC.y]
        ],
        z: avgZ
      })
    }).then(function (data) {
      if (data.success) {
        isCalibrated = true;
        calibStatus.textContent = 'Calibrated';
        calibStatus.classList.add('calibrated');
        calibState.pixelA = null;
        calibState.pixelB = null;
        calibState.pixelC = null;
        saveCalibPoses();
        updateCalibUI();
        updatePickUI();
        ExtensionAPI.showNotification('Calibration saved! Z=' + avgZ.toFixed(1), 'info');
      } else {
        ExtensionAPI.showNotification(data.error || 'Calibration failed', 'error');
      }
    });
  });

  loadCalibration();

  // ---- pick & place -------------------------------------------------------

  var dropSetBtn   = document.getElementById('cvpick-drop-set');
  var dropPosEl    = document.getElementById('cvpick-drop-pos');
  var pickAllBtn   = document.getElementById('cvpick-pick-all');
  var pickOneBtn   = document.getElementById('cvpick-pick-one');
  var pickSelect   = document.getElementById('cvpick-pick-select');
  var pickStopBtn  = document.getElementById('cvpick-pick-stop');
  var pickProgress = document.getElementById('cvpick-pick-progress');

  var dropPosition = null;   // {x, y, z}
  var pickAborted  = false;

  function updatePickUI() {
    dropPosEl.textContent = dropPosition ? fmtRobot(dropPosition) : '\u2014';
    var canPick = !!dropPosition && isCalibrated && !pickRunning;
    pickAllBtn.disabled = !canPick;
    if (pickOneBtn) {
      pickOneBtn.disabled = !canPick || !pickSelect.value;
    }
    if (pickSelect) pickSelect.disabled = pickRunning || !isCalibrated;
    if (liftInput) liftInput.disabled = pickRunning;
    if (yoloConfInput) yoloConfInput.disabled = pickRunning;
    if (scanOnceBtn) scanOnceBtn.disabled = pickRunning;
    if (scanEachBtn) scanEachBtn.disabled = pickRunning;
    updateCalibUI();
  }

  function refreshPickTargets() {
    if (pickRunning || !pickSelect) return;
    Promise.all([
      ExtensionAPI.fetch('cv-pick', '/learned'),
      ExtensionAPI.fetch('cv-pick', '/detections')
    ]).then(function (results) {
      if (pickRunning) return;
      var learned = (results[0] && results[0].success && results[0].items) ? results[0].items : [];
      var dets = (results[1] && results[1].success && results[1].detections) ? results[1].detections : [];
      var prev = pickSelect.value;

      var names = [];
      learned.forEach(function (item) {
        if (item.mode !== listMode()) return;
        if (item.name && names.indexOf(item.name) === -1) names.push(item.name);
      });

      var counts = {};
      dets.forEach(function (d) {
        var n = d.name;
        if (!n) return;
        counts[n] = (counts[n] || 0) + 1;
      });

      names.sort(function (a, b) {
        var ca = counts[a] || 0;
        var cb = counts[b] || 0;
        if (cb !== ca) return cb - ca;
        return a < b ? -1 : a > b ? 1 : 0;
      });

      pickSelect.innerHTML = '<option value="">-- select group --</option>';
      names.forEach(function (name) {
        var opt = document.createElement('option');
        opt.value = name;
        opt.textContent = counts[name] ? (name + ' (' + counts[name] + ')') : name;
        pickSelect.appendChild(opt);
      });
      if (prev) pickSelect.value = prev;
      updatePickUI();
    }).catch(function () {});
  }

  // Restore saved drop position
  var savedDrop = ExtensionAPI.getData('cv-pick', 'dropPosition');
  if (savedDrop) {
    dropPosition = savedDrop;
    updatePickUI();
  }

  // Set drop position from current robot position
  dropSetBtn.addEventListener('click', function () {
    if (!requirePort()) return;
    ExtensionAPI.getRobotStatus(selectedPort).then(function (status) {
      if (!status.success) {
        ExtensionAPI.showNotification(status.error || 'Cannot read robot position', 'error');
        return;
      }
      dropPosition = {
        x: status.coordinates.X,
        y: status.coordinates.Y,
        z: status.coordinates.Z
      };
      ExtensionAPI.setData('cv-pick', 'dropPosition', dropPosition);
      updatePickUI();
      ExtensionAPI.showNotification('Drop position set: ' + fmtRobot(dropPosition), 'info');
    });
  });

  // Helpers
  function sleep(ms) {
    return new Promise(function (resolve) { setTimeout(resolve, ms); });
  }

  var _serverUrl = ExtensionAPI.getServerUrl();

  function cmdMove(x, y, z) {
    return fetch(_serverUrl + '/cmd/jog', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(withPort({ mode: 'coord', motion: 1, values: { x: x, y: y, z: z }, isAbsolute: true }))
    });
  }

  function cmdPump(mode) {
    return fetch(_serverUrl + '/cmd/pump', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(withPort({ mode: mode }))
    });
  }

  async function waitIdle(timeoutMs) {
    var deadline = Date.now() + (timeoutMs || 15000);
    while (Date.now() < deadline) {
      if (pickAborted) return;
      var st = await ExtensionAPI.getRobotStatus(selectedPort);
      if (st.success && st.state === 'Idle') return;
      await sleep(200);
    }
  }

  function setProgress(msg) {
    pickProgress.textContent = msg;
    pickProgress.classList.toggle('running', !!msg);
  }

  async function detectionsToTargets(detections) {
    var targets = [];
    for (var i = 0; i < detections.length; i++) {
      var px = detections[i].center_px;
      var pos = await ExtensionAPI.fetch('cv-pick', '/pick-position', {
        method: 'POST',
        body: JSON.stringify({ px: px[0], py: px[1] })
      });
      if (pos.success) {
        targets.push({ name: detections[i].name, pos: pos.position });
      }
    }
    return targets;
  }

  async function fetchPickDetections(groupName) {
    var det = await ExtensionAPI.fetch('cv-pick', '/detections');
    var list = (det && det.success && det.detections) ? det.detections : [];
    if (groupName) {
      list = list.filter(function (d) { return d.name === groupName; });
    }
    return list;
  }

  async function pickOneObject(target, index, total) {
    var dz = liftHeight;
    var d = dropPosition;
    var t = target.pos;
    var label = target.name || ('Object ' + index);
    var suffix = total ? (' (' + index + '/' + total + ')') : (' (' + index + ')');
    setProgress('Picking ' + label + suffix + '...');

    await cmdMove(t.x, t.y, t.z + dz);
    await waitIdle();
    if (pickAborted) return;

    await cmdPump(1);
    await sleep(300);

    await cmdMove(t.x, t.y, t.z);
    await waitIdle();
    if (pickAborted) return;
    await sleep(200);

    await cmdMove(t.x, t.y, t.z + dz);
    await waitIdle();
    if (pickAborted) return;

    setProgress('Dropping ' + label + suffix + '...');
    await cmdMove(d.x, d.y, d.z + dz);
    await waitIdle();
    if (pickAborted) return;

    await cmdMove(d.x, d.y, d.z);
    await waitIdle();
    if (pickAborted) return;

    await cmdPump(0);
    await sleep(300);

    await cmdMove(d.x, d.y, d.z + dz);
    await waitIdle();
  }

  function finishPick(picked, total, groupName, aborted) {
    pickRunning = false;
    pickStopBtn.disabled = true;
    updatePickUI();

    if (aborted) {
      setProgress('Stopped (' + picked + (total ? '/' + total : '') + ' picked)');
      ExtensionAPI.showNotification('Pick sequence stopped', 'info');
      return;
    }
    var label = groupName || 'objects';
    var doneMsg = groupName
      ? ('Done! ' + picked + ' ' + groupName + ' object' + (picked === 1 ? '' : 's') + ' picked')
      : ('Done! ' + picked + ' objects picked');
    setProgress(doneMsg);
    ExtensionAPI.showNotification(
      groupName ? ('Pick complete: ' + picked + ' ' + label) : ('Pick complete: ' + picked + ' objects'),
      'info');
  }

  async function runPickSequence(targets) {
    pickRunning = true;
    pickAborted = false;
    updatePickUI();
    pickStopBtn.disabled = false;

    var picked = 0;
    for (var j = 0; j < targets.length; j++) {
      if (pickAborted) break;
      await pickOneObject(targets[j], j + 1, targets.length);
      if (pickAborted) break;
      picked += 1;
    }

    await cmdPump(0);
    var groupName = targets[0] && targets.every(function (t) { return t.name === targets[0].name; })
      ? targets[0].name : null;
    finishPick(picked, targets.length, groupName, pickAborted);
  }

  async function runPickRetake(groupName) {
    pickRunning = true;
    pickAborted = false;
    updatePickUI();
    pickStopBtn.disabled = false;

    var picked = 0;
    while (!pickAborted) {
      setProgress(picked === 0 ? 'Detecting objects...' : 'Scanning for next object...');
      if (picked > 0) await sleep(600);
      var dets = await fetchPickDetections(groupName);
      if (!dets.length) {
        if (picked === 0) {
          pickRunning = false;
          pickStopBtn.disabled = true;
          updatePickUI();
          setProgress('');
          ExtensionAPI.showNotification(
            groupName
              ? ('No "' + groupName + '" objects detected. Switch to Detect mode and ensure they are visible.')
              : 'No objects detected. Switch to Detect mode and ensure items are visible.',
            'error');
          return;
        }
        break;
      }
      var targets = await detectionsToTargets([dets[0]]);
      if (!targets.length) {
        if (picked === 0) {
          pickRunning = false;
          pickStopBtn.disabled = true;
          updatePickUI();
          setProgress('');
          ExtensionAPI.showNotification('Could not convert positions — check calibration', 'error');
          return;
        }
        break;
      }
      picked += 1;
      await pickOneObject(targets[0], picked, null);
      if (pickAborted) {
        picked -= 1;
        break;
      }
    }

    await cmdPump(0);
    finishPick(picked, 0, groupName || null, pickAborted);
  }

  async function ensureCalibrated() {
    var calData = await ExtensionAPI.fetch('cv-pick', '/calibration');
    if (!calData.success || !calData.calibration) {
      ExtensionAPI.showNotification('Calibrate first before picking', 'error');
      return false;
    }
    return true;
  }

  async function startPicking(groupName) {
    if (!requirePort()) return;
    if (!(await ensureCalibrated())) return;

    if (scanEach) {
      await runPickRetake(groupName);
      return;
    }

    setProgress(groupName ? ('Detecting ' + groupName + '...') : 'Detecting objects...');
    var dets = await fetchPickDetections(groupName);
    if (!dets.length) {
      setProgress('');
      ExtensionAPI.showNotification(
        groupName
          ? ('No "' + groupName + '" objects detected. Switch to Detect mode and ensure they are visible.')
          : 'No objects detected. Switch to Detect mode and ensure items are visible.',
        'error');
      return;
    }

    var targets = await detectionsToTargets(dets);
    if (targets.length === 0) {
      setProgress('');
      ExtensionAPI.showNotification('Could not convert positions — check calibration', 'error');
      return;
    }

    await runPickSequence(targets);
  }

  pickAllBtn.addEventListener('click', async function () {
    await startPicking(null);
  });

  pickSelect.addEventListener('change', updatePickUI);

  pickOneBtn.addEventListener('click', async function () {
    var name = pickSelect.value;
    if (!name) {
      ExtensionAPI.showNotification('Select a group to pick', 'error');
      return;
    }
    await startPicking(name);
  });

  pickStopBtn.addEventListener('click', function () {
    pickAborted = true;
    pickStopBtn.disabled = true;
    setProgress('Stopping...');
    cmdPump(0);
  });

  updatePickUI();
  refreshPickTargets();
  refreshModelStatus();
})();