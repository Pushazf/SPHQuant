"""Download one benchmark dataset at the revision pinned in its module and check every file's sha256.

python -m benchmark.fetch --dataset textvqa --out data/textvqa

Files already present with the right hash are not downloaded again. The revision is written to <out>.revision.
"""
import argparse
import json
from pathlib import Path

from benchmark import client, common


def bad_files(ds, out):
    """Pinned files of DS under OUT that are missing or have the wrong sha256."""
    return [name for name in sorted(ds.FILES)
            if not (Path(out) / name).is_file() or common.sha256(Path(out) / name) != ds.FILES[name]]


def fetch(name, out):
    ds = client.DATASETS[name]
    if bad_files(ds, out):
        from huggingface_hub import snapshot_download

        snapshot_download(ds.DATASET, repo_type="dataset", revision=ds.REVISION, local_dir=str(out),
                          allow_patterns=sorted(ds.FILES))
    bad = bad_files(ds, out)
    if bad:
        raise SystemExit(f"{name}: sha256 mismatch or missing after download: {bad}")
    Path(f"{out}.revision").write_text(ds.REVISION + "\n")
    return {"dataset": ds.DATASET_ID, "revision": ds.REVISION, "dir": str(out), "files": len(ds.FILES)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=sorted(client.DATASETS))
    p.add_argument("--out", required=True, help="snapshot directory of the dataset")
    args = p.parse_args(argv)
    print(json.dumps(fetch(args.dataset, args.out)))


if __name__ == "__main__":
    main()
