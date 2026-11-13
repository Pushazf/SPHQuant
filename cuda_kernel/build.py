"""Build sph_gemv.so (the W2 kernels and the decode attention kernel) with nvcc and check its dynamic section.

python -m cuda_kernel.build --out DIR [--arch sm_XY] [--nvcc PATH] [--ccbin PATH]

--arch defaults to the compute capability of GPU 0; --ccbin defaults to $CUDAHOSTCXX, else nvcc's own choice of
host compiler.
"""
import argparse
import hashlib
import json
import os
import posixpath
import re
import shutil
import subprocess
import sys
from pathlib import Path

SOURCES = [Path(__file__).resolve().with_name(n) for n in ("sph_gemv.cu", "decode_attention.cu")]
CUDART = "libcudart.so.13"


def nvcc_command(nvcc, ccbin, arch, out):
    """nvcc invocation building the shared library (nvcc takes the host -fPIC through -Xcompiler)."""
    sm = arch.removeprefix("sm_")
    host = ["-ccbin", ccbin] if ccbin else []
    return [nvcc, "-std=c++17", "-O3", "-Xcompiler", "-fPIC", "-shared", "--cudart=shared", "-fmad=false",
            *host, f"--generate-code=arch=compute_{sm},code=sm_{sm}", "-o", str(out), *map(str, SOURCES)]


def gpu_arch():
    """sm_XY of GPU 0, or None when torch sees no GPU."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability(0)
    return f"sm_{major}{minor}"


def parse_dynamic(text):
    """(NEEDED libraries, RPATH/RUNPATH entries) from `readelf -d` output."""
    needed = re.findall(r"\(NEEDED\)\s+Shared library: \[(.*?)\]", text)
    entries = re.findall(r"\((?:RPATH|RUNPATH)\)\s+Library r(?:un)?path: \[(.*?)\]", text)
    return needed, [p for entry in entries for p in entry.split(":") if p]


def dynamic_problems(needed, rpaths, toolkit_roots):
    """Reasons to reject the library: no libcudart.so.13 in NEEDED, or a search path inside the toolkit."""
    problems = [] if CUDART in needed else [f"NEEDED lacks {CUDART}: {needed}"]
    roots = [posixpath.normpath(r) for r in toolkit_roots]
    for p in rpaths:
        path = posixpath.normpath(p)
        if any(path == r or path.startswith(r + "/") for r in roots):
            problems.append(f"RPATH/RUNPATH points into the toolkit: {p}")
    return problems


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build cuda_kernel/sph_gemv.so")
    ap.add_argument("--out", required=True, help="output directory for sph_gemv.so")
    ap.add_argument("--arch", help="target, e.g. sm_103 (default: GPU 0's compute capability)")
    ap.add_argument("--nvcc", default="nvcc")
    ap.add_argument("--ccbin", default=os.environ.get("CUDAHOSTCXX"), help="host compiler (default: $CUDAHOSTCXX)")
    args = ap.parse_args(argv)

    nvcc = shutil.which(args.nvcc)
    if nvcc is None:
        sys.exit(f"nvcc not found: {args.nvcc}")
    arch = args.arch or gpu_arch()
    if arch is None:
        sys.exit("no GPU visible to torch; pass --arch, e.g. --arch sm_103")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp, final = out_dir / ".sph_gemv.tmp.so", out_dir / "sph_gemv.so"
    cmd = nvcc_command(nvcc, args.ccbin, arch, tmp)
    try:
        subprocess.run(cmd, check=True, stdout=sys.stderr)
        readelf = subprocess.run(["readelf", "-d", str(tmp)], check=True, capture_output=True, text=True).stdout
        needed, rpaths = parse_dynamic(readelf)
        roots = [os.path.dirname(os.path.dirname(p)) for p in (os.path.abspath(nvcc), os.path.realpath(nvcc))]
        problems = dynamic_problems(needed, rpaths, roots)
        if problems:
            sys.exit("; ".join(problems))
        os.replace(tmp, final)
    finally:
        tmp.unlink(missing_ok=True)
    version = subprocess.run([nvcc, "--version"], check=True, capture_output=True, text=True).stdout
    print(json.dumps({"ok": True, "so": str(final.resolve()), "sha256": sha256(final), "command": cmd,
                      "nvcc_version": version.strip().splitlines()[-1], "needed": needed, "rpath": rpaths}))


if __name__ == "__main__":
    main()
