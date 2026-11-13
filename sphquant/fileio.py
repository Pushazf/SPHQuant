"""Atomic file writes, hashing and the CLI receipt/log helpers shared by the sphquant CLIs."""
import hashlib
import json
import os
import sys


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def atomic_write(path, write):
    """Call write(tmp_path), fsync, then os.replace onto path."""
    tmp = f"{path}.tmp{os.getpid()}"
    try:
        write(tmp)
        with open(tmp, "rb+") as f:
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def write_json(path, obj):
    def write(tmp):
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(obj, f, indent=1, sort_keys=True)
            f.write("\n")
    atomic_write(path, write)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def config_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def log(*parts):
    print(*parts, file=sys.stderr, flush=True)


def receipt(obj):
    """Final one-line JSON receipt on stdout."""
    print(json.dumps(obj, sort_keys=True), flush=True)
