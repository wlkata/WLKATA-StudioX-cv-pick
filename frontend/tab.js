(function () {
  var feed        = document.getElementById('cvpick-feed');
  var placeholder = document.getElementById('cvpick-placeholder');
  var feedWrap    = document.getElementById('cvpick-feed-wrap');
  var roiOverlay  = document.getElementById('cvpick-roi-overlay');
  var roiRect     = document.getElementById('cvpick-roi-rect');
  var nameInput   = document.getElementById('cvpick-name');
  var learnBtn    = document.getElementById('cvpick-learn-btn');
  var clearBtn    = document.getElementById('cvpick-clear-btn');
  var learnedList = document.getElementById('cvpick-learned-list');
  var colorBtn    = document.getElementById('cvpick-mode-color');
  var shapeBtn    = document.getElementById('cvpick-mode-shape');
  var learnPhBtn  = document.getElementById('cvpick-phase-learn');
  var detectPhBtn = document.getElementById('cvpick-phase-detect');
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
  var calibPxC       = document.getElementById('cvpick-calib-px-c');
  var calibRobotC    = document.getElementById('cvpick-calib-robot-c');
  var calibRecordC   = document.getElementById('cvpick-calib-record-c');
  var calibSaveBtn   = document.getElementById('cvpick-calib-save');
  var calibStatus    = document.getElementById('cvpick-calib-status');
  var markerA        = document.getElementById('cvpick-marker-a');
  var markerB        = document.getElementById('cvpick-marker-b');
  var markerC        = document.getElementById('cvpick-marker-c');

  var polling = false;

  // ---- robot jog controls -----------------------------------------------

  var jogStep = 5;
  var jogBusy = false;
  var pumpOn  = false;

  // Step selector
  var stepBtns = document.querySelectorAll('.cvpick-jog-step');
  for (var si = 0; si < stepBtns.length; si++) {
    stepBtns[si].addEventListener('click', function () {
      for (var k = 0; k < stepBtns.length; k++) stepBtns[k].classList.remove('active');
      this.classList.add('active');
      jogStep = parseInt(this.getAttribute('data-step'), 10);
    });
  }

  // Axis jog buttons — use /cmd/jog for single-axis incremental moves
  var jogBtns = document.querySelectorAll('.cvpick-jog-btn');
  for (var ji = 0; ji < jogBtns.length; ji++) {
    jogBtns[ji].addEventListener('click', function () {
      if (jogBusy) return;
      var axis = this.getAttribute('data-axis').toUpperCase();
      var dir  = parseInt(this.getAttribute('data-dir'), 10);
      jogBusy = true;

      fetch(ExtensionAPI.getServerUrl() + '/cmd/jog', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: 'coord', axis: axis, step: dir * jogStep })
      }).then(function () { jogBusy = false; })
        .catch(function () { jogBusy = false; });
    });
  }

  // Suction buttons — use /cmd/pump
  var pumpOnBtn  = document.getElementById('cvpick-pump-on');
  var pumpOffBtn = document.getElementById('cvpick-pump-off');
  pumpOnBtn.addEventListener('click', function () {
    cmdPump(1);
    pumpOn = true;
    pumpOnBtn.classList.add('pump-active');
    pumpOffBtn.classList.remove('pump-active');
  });
  pumpOffBtn.addEventListener('click', function () {
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

  function populateCameras(data) {
    if (!data || !data.success || !cameraSelect) return;
    var prev = cameraSelect.value;
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
    if (data.current !== null && data.current !== undefined) {
      cameraSelect.value = String(data.current);
    } else if (prev !== '' && cameraSelect.querySelector('option[value="' + prev + '"]')) {
      cameraSelect.value = prev;
    }
  }

  function loadCameras() {
    return ExtensionAPI.fetch('cv-pick', '/cameras').then(function (data) {
      populateCameras(data);
      return data;
    }).catch(function () { return null; });
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

  /** Poll /frame until a JPEG arrives or timeout (~8s). */
  function waitForFirstFrame() {
    var attempts = 0;
    var maxAttempts = 40;
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

  function stopCameraStream() {
    polling = false;
    cameraRunning = false;
    updateStartBtn();
    ExtensionAPI.fetch('cv-pick', '/stop', { method: 'POST' }).catch(function () {});
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

  if (cameraStartBtn) {
    cameraStartBtn.addEventListener('click', function () {
      if (cameraBusy) return;
      if (cameraRunning) {
        stopCameraStream();
        return;
      }
      var idx = parseInt(cameraSelect.value, 10);
      if (isNaN(idx)) {
        ExtensionAPI.showNotification('Select a camera first', 'error');
        return;
      }
      openCamera(idx);
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

  /**
   * Tab open / return from another tab:
   * refresh camera list (plug/unplug) and stay idle until Start.
   */
  function prepareCameraTab() {
    ExtensionAPI.fetch('cv-pick', '/roi').then(function (data) {
      if (data.success && data.roi && data.roi.length === 4) {
        roi.x1 = data.roi[0]; roi.y1 = data.roi[1];
        roi.x2 = data.roi[2]; roi.y2 = data.roi[3];
      }
    }).catch(function () {});

    // Leaving the tab stops the stream; always re-enter idle and refresh list.
    cameraRunning = false;
    polling = false;
    updateStartBtn();
    if (resSelect) resSelect.innerHTML = '';

    setCameraBusy(true);
    setPlaceholder('Loading camera list\u2026', true);

    loadCameras().then(function (data) {
      setCameraBusy(false);
      if (!data || !data.success) {
        setPlaceholder('Could not list cameras', false);
      } else if (!data.cameras || !data.cameras.length) {
        setPlaceholder('No cameras found', false);
      } else {
        setIdlePlaceholder('Select a camera and click Start');
      }
      refreshLearned();
    }).catch(function () {
      setCameraBusy(false);
      setPlaceholder('Backend not reachable', false);
    });

    if (!refreshTimer) {
      refreshTimer = setInterval(refreshLearned, 3000);
    }
  }

  function stopCamera() {
    polling = false;
    cameraRunning = false;
    updateStartBtn();
    setCameraBusy(false);
    hideOverlays();
    ExtensionAPI.fetch('cv-pick', '/stop', { method: 'POST' }).catch(function () {});
    if (refreshTimer) {
      clearInterval(refreshTimer);
      refreshTimer = null;
    }
  }

  // Lifecycle: list cameras on enter; release device when leaving
  ExtensionAPI.onActivate('cv-pick', prepareCameraTab);
  ExtensionAPI.onDeactivate('cv-pick', stopCamera);

  // First open (IIFE runs on first tab click via lazy loading)
  prepareCameraTab();
  updateStartBtn();

  // ---- mode toggle ------------------------------------------------------

  function setMode(mode) {
    colorBtn.classList.toggle('active', mode === 'color');
    shapeBtn.classList.toggle('active', mode === 'shape');
    ExtensionAPI.fetch('cv-pick', '/mode', {
      method: 'POST',
      body: JSON.stringify({ mode: mode })
    });
  }

  colorBtn.addEventListener('click', function () { setMode('color'); });
  shapeBtn.addEventListener('click', function () { setMode('shape'); });

  // ---- phase toggle (learning / inference) ------------------------------

  function setPhase(phase) {
    learnPhBtn.classList.toggle('active', phase === 'learning');
    detectPhBtn.classList.toggle('active', phase === 'inference');
    learnControls.style.display = phase === 'learning' ? '' : 'none';
    ExtensionAPI.fetch('cv-pick', '/phase', {
      method: 'POST',
      body: JSON.stringify({ phase: phase })
    });
  }

  learnPhBtn.addEventListener('click', function () { setPhase('learning'); });
  detectPhBtn.addEventListener('click', function () { setPhase('inference'); });

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

  // ---- clear all --------------------------------------------------------

  clearBtn.addEventListener('click', function () {
    ExtensionAPI.fetch('cv-pick', '/clear', { method: 'POST' })
      .then(refreshLearned);
  });

  // ---- learned-items list -----------------------------------------------

  function refreshLearned() {
    ExtensionAPI.fetch('cv-pick', '/learned').then(function (data) {
      if (!data.success || !data.items || data.items.length === 0) {
        learnedList.innerHTML =
          '<p class="cvpick-empty">No items learned yet. '
          + 'Place an object in the zone and click Learn.</p>';
        return;
      }

      var html = '';
      data.items.forEach(function (item) {
        // display_color is BGR from OpenCV — swap to RGB for CSS
        var r = item.display_color[2];
        var g = item.display_color[1];
        var b = item.display_color[0];
        var rgb = 'rgb(' + r + ',' + g + ',' + b + ')';

        html += '<div class="cvpick-item">'
          + '<span class="cvpick-dot" style="background:' + rgb + '"></span>'
          + '<span class="cvpick-item-name">' + escHtml(item.name) + '</span>'
          + '<span class="cvpick-item-mode">' + item.mode + '</span>'
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
    calibSaveBtn.disabled = !(calibState.pixelA && calibState.pixelB && calibState.pixelC
                              && calibState.robotA && calibState.robotB && calibState.robotC);
    updateMarkers();
  }

  // Load existing calibration on startup
  function loadCalibration() {
    ExtensionAPI.fetch('cv-pick', '/calibration').then(function (data) {
      if (data.success && data.calibration) {
        calibStatus.textContent = 'Calibrated';
        calibStatus.classList.add('calibrated');
      }
    }).catch(function () {});
  }

  // Detect markers — pick 2 largest detections
  calibDetectBtn.addEventListener('click', function () {
    calibDetectBtn.disabled = true;
    calibDetectBtn.textContent = 'Detecting\u2026';
    ExtensionAPI.fetch('cv-pick', '/detections').then(function (data) {
      calibDetectBtn.disabled = false;
      calibDetectBtn.textContent = 'Detect Markers';
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
      updateCalibUI();
      ExtensionAPI.showNotification(
        'Markers detected: A (red), B (blue), C (green)',
        'info');
    }).catch(function () {
      calibDetectBtn.disabled = false;
      calibDetectBtn.textContent = 'Detect Markers';
    });
  });

  // Record robot position for point A
  calibRecordA.addEventListener('click', function () {
    ExtensionAPI.getRobotStatus().then(function (status) {
      if (!status.success) {
        ExtensionAPI.showNotification(status.error || 'Cannot read robot position', 'error');
        return;
      }
      calibState.robotA = {
        x: status.coordinates.X,
        y: status.coordinates.Y,
        z: status.coordinates.Z
      };
      updateCalibUI();
    });
  });

  // Record robot position for point B
  calibRecordB.addEventListener('click', function () {
    ExtensionAPI.getRobotStatus().then(function (status) {
      if (!status.success) {
        ExtensionAPI.showNotification(status.error || 'Cannot read robot position', 'error');
        return;
      }
      calibState.robotB = {
        x: status.coordinates.X,
        y: status.coordinates.Y,
        z: status.coordinates.Z
      };
      updateCalibUI();
    });
  });

  // Record robot position for point C
  calibRecordC.addEventListener('click', function () {
    ExtensionAPI.getRobotStatus().then(function (status) {
      if (!status.success) {
        ExtensionAPI.showNotification(status.error || 'Cannot read robot position', 'error');
        return;
      }
      calibState.robotC = {
        x: status.coordinates.X,
        y: status.coordinates.Y,
        z: status.coordinates.Z
      };
      updateCalibUI();
    });
  });

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
        calibStatus.textContent = 'Calibrated';
        calibStatus.classList.add('calibrated');
        calibState.pixelA = null;
        calibState.pixelB = null;
        calibState.pixelC = null;
        updateMarkers();
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
  var pickStopBtn  = document.getElementById('cvpick-pick-stop');
  var pickProgress = document.getElementById('cvpick-pick-progress');

  var dropPosition = null;   // {x, y, z}
  var pickRunning  = false;
  var pickAborted  = false;

  function updatePickUI() {
    dropPosEl.textContent = dropPosition ? fmtRobot(dropPosition) : '\u2014';
    pickAllBtn.disabled = !dropPosition || pickRunning;
  }

  // Restore saved drop position
  var savedDrop = ExtensionAPI.getData('cv-pick', 'dropPosition');
  if (savedDrop) {
    dropPosition = savedDrop;
    updatePickUI();
  }

  // Set drop position from current robot position
  dropSetBtn.addEventListener('click', function () {
    ExtensionAPI.getRobotStatus().then(function (status) {
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
      body: JSON.stringify({ mode: 'coord', motion: 1, values: { x: x, y: y, z: z }, isAbsolute: true })
    });
  }

  function cmdPump(mode) {
    return fetch(_serverUrl + '/cmd/pump', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode: mode })
    });
  }

  async function waitIdle(timeoutMs) {
    var deadline = Date.now() + (timeoutMs || 15000);
    while (Date.now() < deadline) {
      if (pickAborted) return;
      var st = await ExtensionAPI.getRobotStatus();
      if (st.success && st.state === 'Idle') return;
      await sleep(200);
    }
  }

  function setProgress(msg) {
    pickProgress.textContent = msg;
    pickProgress.classList.toggle('running', !!msg);
  }

  // Pick all sequence
  pickAllBtn.addEventListener('click', async function () {
    // Verify calibration exists
    var calData = await ExtensionAPI.fetch('cv-pick', '/calibration');
    if (!calData.success || !calData.calibration) {
      ExtensionAPI.showNotification('Calibrate first before picking', 'error');
      return;
    }

    // Detect all objects
    setProgress('Detecting objects...');
    var det = await ExtensionAPI.fetch('cv-pick', '/detections');
    if (!det.success || !det.detections || det.detections.length === 0) {
      setProgress('');
      ExtensionAPI.showNotification('No objects detected. Switch to Detect mode and ensure items are visible.', 'error');
      return;
    }

    // Convert all pixel positions to robot coords
    var targets = [];
    for (var i = 0; i < det.detections.length; i++) {
      var px = det.detections[i].center_px;
      var pos = await ExtensionAPI.fetch('cv-pick', '/pick-position', {
        method: 'POST',
        body: JSON.stringify({ px: px[0], py: px[1] })
      });
      if (pos.success) {
        targets.push({ name: det.detections[i].name, pos: pos.position });
      }
    }

    if (targets.length === 0) {
      setProgress('');
      ExtensionAPI.showNotification('Could not convert positions — check calibration', 'error');
      return;
    }

    // Start pick loop
    pickRunning = true;
    pickAborted = false;
    pickAllBtn.disabled = true;
    pickStopBtn.disabled = false;

    var dz = 10;
    var d = dropPosition;

    for (var j = 0; j < targets.length; j++) {
      if (pickAborted) break;

      var t = targets[j].pos;
      var label = targets[j].name || ('Object ' + (j + 1));
      setProgress('Picking ' + label + ' (' + (j + 1) + '/' + targets.length + ')...');

      // Move above target
      await cmdMove(t.x, t.y, t.z + dz);
      await waitIdle();
      if (pickAborted) break;

      // Pump on
      await cmdPump(1);
      await sleep(300);

      // Lower to pick
      await cmdMove(t.x, t.y, t.z);
      await waitIdle();
      if (pickAborted) break;
      await sleep(200);

      // Lift up
      await cmdMove(t.x, t.y, t.z + dz);
      await waitIdle();
      if (pickAborted) break;

      // Move to drop position (above)
      setProgress('Dropping ' + label + ' (' + (j + 1) + '/' + targets.length + ')...');
      await cmdMove(d.x, d.y, d.z + dz);
      await waitIdle();
      if (pickAborted) break;

      // Lower to drop
      await cmdMove(d.x, d.y, d.z);
      await waitIdle();
      if (pickAborted) break;

      // Pump off
      await cmdPump(0);
      await sleep(300);

      // Lift up from drop
      await cmdMove(d.x, d.y, d.z + dz);
      await waitIdle();
      if (pickAborted) break;
    }

    // Done
    await cmdPump(0);  // ensure pump off
    pickRunning = false;
    pickStopBtn.disabled = true;
    updatePickUI();

    if (pickAborted) {
      setProgress('Stopped (' + j + '/' + targets.length + ' picked)');
      ExtensionAPI.showNotification('Pick sequence stopped', 'info');
    } else {
      setProgress('Done! ' + targets.length + ' objects picked');
      ExtensionAPI.showNotification('Pick complete: ' + targets.length + ' objects', 'info');
    }
  });

  // Stop button
  pickStopBtn.addEventListener('click', function () {
    pickAborted = true;
    pickStopBtn.disabled = true;
    setProgress('Stopping...');
    cmdPump(0);  // turn off pump immediately
  });

  updatePickUI();
})();