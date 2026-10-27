"""Run a user-supplied Ultralytics YOLO training script in an isolated folder.

Invoked as:  python -u train_runner.py <run_dir> [source_cwd]

Accepts:
  - Python scripts from the Ultralytics Train docs
  - Shell snippets from Ultralytics Platform "Train local"
    (export ULTRALYTICS_API_KEY=... plus a multi-line `yolo train \\` command)

The user script is read from <run_dir>/script.py. Checkpoints go under
<run_dir>/runs/train/. A Platform `project=user/name` slug is kept so metrics
can stream to the platform; local files still land in the dedicated folder.
"""

from __future__ import print_function

import json
import os
import re
import sys
import traceback


_YOLO26N = ("yolo26n", "yolov26n")
_YOLO26_OTHER = (
    "yolo26s", "yolo26m", "yolo26l", "yolo26x",
    "yolov26s", "yolov26m", "yolov26l", "yolov26x",
)
_EXPORT_RE = re.compile(
    r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$"
)
_POWERSHELL_RE = re.compile(
    r"^\$env:([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$", re.I
)
_SECRET_RE = re.compile(r"ul_[0-9a-f]{16,}", re.I)
_ENV_SECRET_RE = re.compile(
    r"(ULTRALYTICS_API_KEY\s*=\s*)(['\"]?)([^'\"\s]+)\2", re.I
)


def _die(msg, code=1):
    print("ERROR: " + msg, flush=True)
    sys.exit(code)


def redact_secrets(text):
    text = _ENV_SECRET_RE.sub(r"\1\2***\2", text or "")
    return _SECRET_RE.sub("ul_***", text)


def _write_json(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def _num(value):
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _pick_device():
    try:
        import torch
        if torch.cuda.is_available():
            return 0
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _looks_like_yolo26n(text):
    blob = (text or "").lower()
    if any(tag in blob for tag in _YOLO26_OTHER):
        return False
    return any(tag in blob for tag in _YOLO26N)


def _assert_yolo26n(model):
    name = str(model)
    if name.lower() in ("best.pt", "last.pt"):
        return
    if _looks_like_yolo26n(name):
        return
    lower = name.lower()
    if lower.endswith(".pt") or lower.endswith(".ckpt"):
        return
    if not any(tag in lower for tag in _YOLO26_OTHER):
        if any(tag in lower for tag in _YOLO26N) or lower in ("", "none"):
            return
    _die(
        "Only YOLO26n training is supported. "
        "Use yolo26n / ul://…/yolo26n, not '{}'.".format(name)
    )


def _unquote(val):
    val = (val or "").strip()
    if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
        return val[1:-1]
    return val


def _join_continuations(script):
    lines = (script or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    joined = []
    buf = ""
    for line in lines:
        raw = line.rstrip()
        if buf:
            raw = buf + raw.lstrip()
            buf = ""
        if raw.endswith("\\"):
            buf = raw[:-1].rstrip() + " "
            continue
        joined.append(raw)
    if buf:
        joined.append(buf.rstrip())
    return joined


def _is_python_line(s):
    low = s.lstrip()
    return (
        low.startswith("from ")
        or low.startswith("import ")
        or low.startswith("def ")
        or low.startswith("class ")
        or "YOLO(" in s
        or s.lstrip().startswith("os.environ")
    )


def parse_train_script(script):
    """Return {'kind': 'cli'|'python', 'cmd': str|None, 'env': dict}."""
    env = {}
    yolo_cmd = None
    pythonish = False
    for raw in _join_continuations(script):
        s = raw.strip()
        if not s or s.startswith("#") or s.startswith("#!") or s.startswith("set "):
            continue
        if _is_python_line(s):
            pythonish = True
            continue
        m = _POWERSHELL_RE.match(s)
        if m:
            env[m.group(1)] = _unquote(m.group(2))
            continue
        low = s.lower()
        if low.startswith("yolo ") or low == "yolo":
            yolo_cmd = s
            continue
        m = _EXPORT_RE.match(s)
        if m and (s.startswith("export ") or m.group(1).isupper()):
            env[m.group(1)] = _unquote(m.group(2))
            continue
        if "=" in s and s.split("=", 1)[0].strip().isidentifier():
            key, val = s.split("=", 1)
            env[key.strip()] = _unquote(val)
            continue
    if yolo_cmd and not pythonish:
        return {"kind": "cli", "cmd": yolo_cmd, "env": env}
    return {"kind": "python", "cmd": None, "env": env}


def _python_source(script):
    """Drop shell export / yolo lines so mixed pastes still exec as Python."""
    keep = []
    for raw in (script or "").splitlines(True):
        s = raw.strip()
        if not s:
            keep.append(raw)
            continue
        if s.startswith("export ") or _POWERSHELL_RE.match(s):
            continue
        if s.lower().startswith("yolo ") or s.lower() == "yolo":
            continue
        keep.append(raw)
    return "".join(keep)


def _keep_platform_project(kwargs):
    proj = str(kwargs.get("project") or "").strip().strip("\"'")
    if not proj or os.path.isabs(proj) or "\\" in proj:
        return False
    if proj.count("/") != 1:
        return False
    blob = " ".join(str(kwargs.get(k) or "") for k in ("model", "data", "project"))
    return "ul://" in blob or bool(os.environ.get("ULTRALYTICS_API_KEY"))


def _apply_env(env):
    for key, value in (env or {}).items():
        os.environ[str(key)] = str(value)
        secret = (
            "KEY" in key.upper()
            or "TOKEN" in key.upper()
            or "SECRET" in key.upper()
        )
        shown = "***" if secret else value
        print("env: {}={}".format(key, shown), flush=True)


def main():
    if len(sys.argv) < 2:
        _die("usage: train_runner.py <run_dir> [source_cwd]")

    run_dir = os.path.abspath(sys.argv[1])
    source_cwd = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else run_dir
    script_path = os.path.join(run_dir, "script.py")
    progress_path = os.path.join(run_dir, "progress.json")
    project = os.path.join(run_dir, "runs")
    name = "train"
    local_save = os.path.join(project, name)

    if not os.path.isfile(script_path):
        _die("Missing script.py in " + run_dir)

    with open(script_path, "r", encoding="utf-8") as f:
        script = f.read()

    parsed = parse_train_script(script)
    _apply_env(parsed.get("env"))

    os.makedirs(local_save, exist_ok=True)
    os.chdir(source_cwd)

    try:
        from ultralytics import YOLO
    except ImportError:
        _die("ultralytics is not installed in this environment")

    orig_init = YOLO.__init__
    orig_train = YOLO.train

    def write_progress(payload):
        current = {}
        if os.path.isfile(progress_path):
            try:
                with open(progress_path, "r", encoding="utf-8") as f:
                    current = json.load(f) or {}
            except Exception:
                current = {}
        current.update(payload)
        _write_json(progress_path, current)

    def on_train_start(trainer):
        args = getattr(trainer, "args", None)
        write_progress({
            "phase": "training",
            "epoch": 0,
            "epochs": int(getattr(args, "epochs", 0) or getattr(trainer, "epochs", 0) or 0),
            "device": str(getattr(args, "device", "") or ""),
            "imgsz": getattr(args, "imgsz", None),
            "batch": getattr(args, "batch", None),
            "data": str(getattr(args, "data", "") or ""),
            "model": str(getattr(args, "model", "") or ""),
            "save_dir": str(trainer.save_dir),
            "best": str(trainer.best),
            "last": str(trainer.last),
            "best_exists": os.path.isfile(str(trainer.best)),
        })
        print("save_dir: {}".format(trainer.save_dir), flush=True)
        print("weights:  {}".format(trainer.best), flush=True)

    def on_fit_epoch_end(trainer):
        metrics = {}
        raw = getattr(trainer, "metrics", None) or {}
        try:
            items = dict(raw).items()
        except Exception:
            items = []
        for key, value in items:
            number = _num(value)
            if number is not None:
                metrics[str(key)] = number
        epochs = getattr(trainer, "epochs", None)
        if epochs is None:
            epochs = getattr(getattr(trainer, "args", None), "epochs", 0)
        write_progress({
            "phase": "epoch",
            "epoch": int(getattr(trainer, "epoch", 0) or 0) + 1,
            "epochs": int(epochs or 0),
            "metrics": metrics,
            "fitness": _num(getattr(trainer, "fitness", None)),
            "best_fitness": _num(getattr(trainer, "best_fitness", None)),
            "save_dir": str(trainer.save_dir),
            "best": str(trainer.best),
            "last": str(trainer.last),
            "best_exists": os.path.isfile(str(trainer.best)),
        })

    def on_train_end(trainer):
        write_progress({
            "phase": "done",
            "save_dir": str(trainer.save_dir),
            "best": str(trainer.best),
            "last": str(trainer.last),
            "best_exists": os.path.isfile(str(trainer.best)),
            "best_fitness": _num(getattr(trainer, "best_fitness", None)),
        })
        print("TRAIN_DONE best={} last={} save_dir={}".format(
            trainer.best, trainer.last, trainer.save_dir), flush=True)

    def patched_init(self, model="yolo26n.pt", *args, **kwargs):
        _assert_yolo26n(model)
        orig_init(self, model, *args, **kwargs)
        blob = str(model).lower()
        yaml = getattr(getattr(self, "model", None), "yaml", None) or {}
        if isinstance(yaml, dict):
            blob += " " + str(yaml.get("yaml_file") or "") + " " + str(yaml.get("scale") or "")
        ov = getattr(self, "overrides", None) or {}
        blob += " " + str(ov.get("model") or "")
        if any(tag in blob.lower() for tag in _YOLO26_OTHER):
            _die("Only YOLO26n checkpoints are supported")
        if not _looks_like_yolo26n(blob):
            scale = str(yaml.get("scale") or "").lower() if isinstance(yaml, dict) else ""
            yaml_file = str(yaml.get("yaml_file") or "").lower() if isinstance(yaml, dict) else ""
            if not (scale == "n" and ("yolo26" in yaml_file or "yolov26" in yaml_file)):
                ckpt = getattr(self, "ckpt", None) or {}
                args = ckpt.get("train_args") if isinstance(ckpt, dict) else None
                hint = " ".join(str(args.get(k) or "") for k in ("model", "name")) if isinstance(args, dict) else ""
                if not _looks_like_yolo26n(hint + " " + yaml_file + " " + scale):
                    _die("Only YOLO26n models can be trained here")

    def patched_train(self, *args, **kwargs):
        os.makedirs(local_save, exist_ok=True)
        if _keep_platform_project(kwargs):
            kwargs["save_dir"] = local_save
            kwargs.setdefault("name", name)
        else:
            kwargs.pop("save_dir", None)
            kwargs["project"] = project
            kwargs["name"] = name
        kwargs["exist_ok"] = True
        kwargs["verbose"] = True
        kwargs["save"] = True
        if kwargs.get("device") is None:
            kwargs["device"] = _pick_device()
        if sys.platform == "darwin" and "workers" not in kwargs:
            kwargs["workers"] = 0
        self.add_callback("on_train_start", on_train_start)
        self.add_callback("on_fit_epoch_end", on_fit_epoch_end)
        self.add_callback("on_train_end", on_train_end)
        print("Training device: {}".format(kwargs.get("device")), flush=True)
        print("Run folder:      {}".format(run_dir), flush=True)
        print("Checkpoints:     {}".format(os.path.join(local_save, "weights")), flush=True)
        if _keep_platform_project(kwargs):
            print("Platform project: {}".format(kwargs.get("project")), flush=True)
        return orig_train(self, *args, **kwargs)

    YOLO.__init__ = patched_init
    YOLO.train = patched_train

    write_progress({
        "phase": "starting",
        "run_dir": run_dir,
        "project": project,
        "name": name,
        "source_cwd": source_cwd,
        "kind": parsed.get("kind"),
    })

    print("CV Pick YOLO26n training", flush=True)
    print("script: {}".format(script_path), flush=True)
    print("cwd:    {}".format(source_cwd), flush=True)

    try:
        if parsed.get("kind") == "cli":
            from ultralytics.cfg import entrypoint
            cmd = (parsed.get("cmd") or "").strip()
            if cmd.lower().startswith("yolo"):
                cmd = cmd[4:].strip()
            print("CLI: yolo {}".format(redact_secrets(cmd)), flush=True)
            entrypoint(cmd)
        else:
            source = _python_source(script)
            ns = {"__name__": "__main__", "__file__": script_path}
            exec(compile(source, script_path, "exec"), ns, ns)
    except SystemExit as exc:
        code = exc.code
        if code not in (0, None):
            raise
    except Exception:
        traceback.print_exc()
        write_progress({"phase": "error"})
        sys.exit(1)

    best = os.path.join(local_save, "weights", "best.pt")
    last = os.path.join(local_save, "weights", "last.pt")
    write_progress({
        "phase": "done",
        "save_dir": local_save,
        "best": best,
        "last": last,
        "best_exists": os.path.isfile(best),
    })
    if os.path.isfile(best):
        print("Best checkpoint: {}".format(best), flush=True)
    elif os.path.isfile(last):
        print("Last checkpoint: {}".format(last), flush=True)
    else:
        print("WARNING: No weights written. The script must call model.train(...).",
              flush=True)


if __name__ == "__main__":
    main()
