"""Frames from the camera extension's shared manager.

CV Pick does not open the device. It registers as the client ``cv-pick``,
then reads the JPEG the manager's reader already stored.
"""

import json
import os
import urllib.error
import urllib.request

CLIENT = 'cv-pick'


def _base():
    port = os.environ.get('STUDIOX_PORT', '5080')
    return 'http://127.0.0.1:%s/ext/camera' % port


def _request(method, path, body=None, timeout=5):
    data = None
    headers = {'User-Agent': 'cv-pick'}
    if body is not None:
        data = json.dumps(body).encode('utf-8')
        headers['Content-Type'] = 'application/json'
    req = urllib.request.Request(
        _base() + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read(), resp.headers.get('Content-Type', '')
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), exc.headers.get('Content-Type', '')
    except Exception as exc:
        return 0, str(exc).encode('utf-8'), 'text/plain'


def _json(method, path, body=None, timeout=8):
    status, raw, ctype = _request(method, path, body=body, timeout=timeout)
    if 'json' in ctype or (raw[:1] in (b'{', b'[')):
        try:
            data = json.loads(raw.decode('utf-8'))
        except Exception:
            data = None
        if isinstance(data, dict):
            if status == 0 and not data.get('error'):
                data['success'] = False
                data['error'] = 'Camera extension is not available'
            return data
    if status == 0:
        text = raw.decode('utf-8', 'replace').strip() or 'Camera extension is not available'
        return {'success': False, 'error': text}
    return {'success': False, 'error': 'Camera extension returned HTTP %s' % status}


def list_cameras():
    """OS camera list from the manager, shaped for the CV Pick dropdown."""
    data = _json('GET', '/cameras', timeout=15)
    if not data.get('success') and 'cameras' not in data:
        raise RuntimeError(data.get('error') or 'Could not list cameras')
    cameras = []
    for item in data.get('cameras') or []:
        if not isinstance(item, dict):
            continue
        index = item.get('id', item.get('index'))
        if index is None:
            continue
        cameras.append({
            'index': int(index),
            'name': item.get('name') or ('Camera %s' % index),
            'resolutions': [],
            'default': None,
        })
    return cameras


def connect(index):
    return _json('POST', '/connect', {'id': int(index), 'client': CLIENT}, timeout=8)


def disconnect(index):
    return _json('POST', '/disconnect', {'id': int(index), 'client': CLIENT}, timeout=8)


def latest_jpeg(index):
    """Latest JPEG for this camera, or None when the manager has no frame yet."""
    status, raw, ctype = _request('GET', '/frame?id=%d' % int(index), timeout=2)
    if status != 200 or not raw or 'image' not in ctype:
        return None
    return raw
