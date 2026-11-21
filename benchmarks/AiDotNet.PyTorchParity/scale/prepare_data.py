"""Build the datasets for the large-data benchmark in the shared binary format both harnesses read.

Every dataset is a directory holding raw uint8 arrays and a meta.json:

    train-x.u8  train-y.u8  test-x.u8  test-y.u8  meta.json

train-x.u8 is N samples of prod(sample_shape) bytes each, row-major in sample_shape order (CHW for images, time-major
for sequences). train-y.u8 is N class indices. meta.json records the sample shape, class count, the per-channel mean and
standard deviation of the training pixels (on the 0-1 scale), each file's size and SHA-256, and the SHA-256 of every
downloaded archive. Both frameworks normalise a byte b of channel c to (b / 255 - mean[c]) / std[c].

Datasets:
  emnist-digits    NIST EMNIST Digits, 240,000 training and 40,000 test handwritten digits, 1x28x28 (mlp, cnn)
  emnist-seq32     The same digits padded to 32x32 and read one row per time step, 32 steps of 32 (lstm, gru, transformer)
  cinic10          CINIC-10, 180,000 training (its train and valid splits) and 90,000 test RGB images, 3x32x32
                   (resnet18, resnet50, vgg11, mobilenetv2, densenet121, vit)
  synth-mnist      1,000,000 training and 10,000 test synthetic samples in each of the three shapes. Each class has a
  synth-seq32      fixed random template; a sample is 25% its class template and 75% fresh uniform noise, so the models
  synth-rgb32      can learn it and the loss is comparable, while no sample repeats.

Usage:
  python prepare_data.py --root DATA_DIR [--downloads DIR] [--datasets emnist-digits,cinic10,...]

The downloads are verified against the SHA-256 values below before anything is built.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

SOURCES = {
    "emnist": dict(
        url="https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip",
        file="emnist-gzip.zip",
    ),
    "cinic10": dict(
        url="https://datashare.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz",
        file="CINIC-10.tar.gz",
    ),
}
# SHA-256 of the archives this benchmark was built from. A different download is refused, so every run trains on the
# same bytes.
EXPECTED_SHA256 = {
    "emnist": "fb9bb67e33772a9cc0b895e4ecf36d2cf35be8b709693c3564cea2a019fcda8e",
    "cinic10": "31b095acf6d75e25a9e028bae82a07a0f94ff6b00671be2802d34ac4efa81a9e",
}

SYNTH_TRAIN, SYNTH_TEST, SYNTH_SEED = 1_000_000, 10_000, 2026
CINIC_CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
CHUNK = 20_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(name: str, downloads: Path) -> Path:
    source = SOURCES[name]
    path = downloads / source["file"]
    if not path.exists():
        downloads.mkdir(parents=True, exist_ok=True)
        print(f"downloading {source['url']}", flush=True)
        partial = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(source["url"]) as response, partial.open("wb") as out:
            while block := response.read(1 << 22):
                out.write(block)
        partial.replace(path)
    actual = sha256_file(path)
    if actual != EXPECTED_SHA256[name]:
        raise SystemExit(f"{path} has SHA-256 {actual}, expected {EXPECTED_SHA256[name]}. Delete it and download again.")
    return path


def read_idx(data: bytes) -> np.ndarray:
    """An IDX file (the MNIST/EMNIST container): magic 0x0000 0x08 ndim, ndim big-endian sizes, then uint8 data."""
    if data[:2] != b"\0\0" or data[2] != 0x08:
        raise ValueError("not an unsigned-byte IDX file")
    ndim = data[3]
    dims = [int.from_bytes(data[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


def channel_stats(x: np.ndarray, channels: int) -> tuple[list[float], list[float]]:
    """Per-channel mean and standard deviation of the pixels on the 0-1 scale, accumulated in float64 by chunk."""
    per_sample = x.reshape(x.shape[0], channels, -1)
    total = np.zeros(channels)
    total_sq = np.zeros(channels)
    for start in range(0, x.shape[0], CHUNK):
        chunk = per_sample[start:start + CHUNK].astype(np.float64) / 255.0
        total += chunk.sum(axis=(0, 2))
        total_sq += (chunk * chunk).sum(axis=(0, 2))
    count = per_sample.shape[0] * per_sample.shape[2]
    mean = total / count
    std = np.sqrt(total_sq / count - mean * mean)
    return [round(float(v), 6) for v in mean], [round(float(v), 6) for v in std]


LIMIT: int | None = None  # --limit: a small copy of every dataset for a quick end-to-end check


def write_dataset(root: Path, name: str, train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray,
                  test_y: np.ndarray, sample_shape: list[int], channels: int, description: str,
                  sources: dict[str, str], models: list[str], stats_x: np.ndarray | None = None) -> None:
    out = root / name
    out.mkdir(parents=True, exist_ok=True)
    if LIMIT:
        # A multiple of the 250-sample test batch, and whole training batches.
        train_x, train_y = train_x[:LIMIT], train_y[:LIMIT]
        test_x, test_y = test_x[:1000], test_y[:1000]
    mean, std = channel_stats(train_x if stats_x is None else stats_x, channels)
    files = {}
    for stem, array in (("train-x", train_x), ("train-y", train_y), ("test-x", test_x), ("test-y", test_y)):
        path = out / f"{stem}.u8"
        np.ascontiguousarray(array, dtype=np.uint8).tofile(path)
        files[stem] = dict(bytes=path.stat().st_size, sha256=sha256_file(path))
    meta = dict(name=name, description=description, sample_shape=sample_shape, classes=10,
                train_samples=int(train_x.shape[0]), test_samples=int(test_x.shape[0]), mean=mean, std=std,
                models=models, files=files, sources=sources, format=1)
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"{name}: {train_x.shape[0]:,} train, {test_x.shape[0]:,} test, shape {sample_shape}, mean {mean}, std {std}",
          flush=True)


def build_emnist(root: Path, downloads: Path, wanted: set[str]) -> None:
    archive = fetch("emnist", downloads)
    arrays = {}
    with zipfile.ZipFile(archive) as zf:
        for split in ("train", "test"):
            for kind in ("images-idx3", "labels-idx1"):
                with zf.open(f"gzip/emnist-digits-{split}-{kind}-ubyte.gz") as member:
                    arrays[split, kind] = read_idx(gzip.decompress(member.read()))
    # EMNIST stores each image transposed relative to MNIST; transpose so the digits are upright.
    images = {split: np.ascontiguousarray(arrays[split, "images-idx3"].transpose(0, 2, 1)) for split in ("train", "test")}
    labels = {split: arrays[split, "labels-idx1"] for split in ("train", "test")}
    sources = {"emnist-gzip.zip": EXPECTED_SHA256["emnist"]}
    if "emnist-digits" in wanted:
        write_dataset(root, "emnist-digits", images["train"], labels["train"], images["test"], labels["test"],
                      [1, 28, 28], 1, "NIST EMNIST Digits (Cohen et al. 2017), upright, 1x28x28", sources,
                      ["mlp", "cnn"])
    if "emnist-seq32" in wanted:
        padded = {split: np.pad(images[split], ((0, 0), (2, 2), (2, 2))) for split in ("train", "test")}
        write_dataset(root, "emnist-seq32", padded["train"], labels["train"], padded["test"], labels["test"],
                      [32, 32], 1, "EMNIST Digits zero-padded to 32x32, one image row per time step", sources,
                      ["lstm", "gru", "transformer"])


def build_cinic(root: Path, downloads: Path) -> None:
    archive = fetch("cinic10", downloads)
    from PIL import Image

    images: dict[str, list[tuple[str, int, bytes]]] = {"train": [], "valid": [], "test": []}
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf:
            parts = Path(member.name).parts
            if not member.isfile() or not member.name.endswith(".png"):
                continue
            split, cls = parts[-3], parts[-2]
            images[split].append((member.name, CINIC_CLASSES.index(cls), tf.extractfile(member).read()))

    def decode(entries: list[tuple[str, int, bytes]]) -> tuple[np.ndarray, np.ndarray]:
        # Sorted by path so the sample order does not depend on the archive's member order.
        entries.sort(key=lambda entry: entry[0])
        x = np.empty((len(entries), 3, 32, 32), dtype=np.uint8)
        for i, (_, _, png) in enumerate(entries):
            # A few CINIC-10 images are greyscale; RGB conversion repeats the channel.
            x[i] = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype=np.uint8).transpose(2, 0, 1)
        return x, np.array([entry[1] for entry in entries], dtype=np.uint8)

    train_x, train_y = decode(images["train"] + images["valid"])
    test_x, test_y = decode(images["test"])
    write_dataset(root, "cinic10", train_x, train_y, test_x, test_y, [3, 32, 32], 3,
                  "CINIC-10 (Darlow et al. 2018): train+valid for training, test for testing, 3x32x32",
                  {"CINIC-10.tar.gz": EXPECTED_SHA256["cinic10"]},
                  ["resnet18", "resnet50", "vgg11", "mobilenetv2", "densenet121", "vit"])


def build_synthetic(root: Path, name: str, sample_shape: list[int], channels: int, models: list[str]) -> None:
    rng = np.random.Generator(np.random.PCG64(SYNTH_SEED))
    size = int(np.prod(sample_shape))
    templates = rng.integers(0, 256, size=(10, size), dtype=np.int64)

    def make(count: int, generator: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        x = np.empty((count, size), dtype=np.uint8)
        y = generator.integers(0, 10, size=count, dtype=np.int64).astype(np.uint8)
        for start in range(0, count, CHUNK):
            stop = min(count, start + CHUNK)
            noise = generator.integers(0, 256, size=(stop - start, size), dtype=np.int64)
            # 25% class template, 75% fresh noise: learnable but not trivially, and no two samples are the same.
            x[start:stop] = ((25 * templates[y[start:stop]] + 75 * noise + 50) // 100).astype(np.uint8)
        return x, y

    train_x, train_y = make(SYNTH_TRAIN, np.random.Generator(np.random.PCG64(SYNTH_SEED + 1)))
    test_x, test_y = make(SYNTH_TEST, np.random.Generator(np.random.PCG64(SYNTH_SEED + 2)))
    write_dataset(root, name, train_x, train_y, test_x, test_y, sample_shape, channels,
                  f"{SYNTH_TRAIN:,} synthetic samples: a fixed random template per class mixed 25/75 with fresh noise "
                  f"(numpy PCG64, seed {SYNTH_SEED})", {}, models,
                  stats_x=train_x[:100_000])


ALL = ("emnist-digits", "emnist-seq32", "cinic10", "synth-mnist", "synth-seq32", "synth-rgb32")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="Directory the datasets are written to")
    parser.add_argument("--downloads", type=Path, help="Directory for the downloaded archives (default ROOT/downloads)")
    parser.add_argument("--datasets", default=",".join(ALL), help=f"Comma-separated subset of {', '.join(ALL)}")
    parser.add_argument("--limit", type=int, help="Keep only this many training samples (and 1,000 test samples) per "
                                                    "dataset, for a quick check of the pipeline")
    args = parser.parse_args()
    global LIMIT, SYNTH_TRAIN, SYNTH_TEST
    LIMIT = args.limit
    if LIMIT:
        SYNTH_TRAIN, SYNTH_TEST = LIMIT, 1000
    wanted = {name.strip() for name in args.datasets.split(",") if name.strip()}
    unknown = wanted - set(ALL)
    if unknown:
        raise SystemExit(f"unknown datasets: {', '.join(sorted(unknown))}")
    downloads = args.downloads or args.root / "downloads"
    if wanted & {"emnist-digits", "emnist-seq32"}:
        build_emnist(args.root, downloads, wanted)
    if "cinic10" in wanted:
        build_cinic(args.root, downloads)
    if "synth-mnist" in wanted:
        build_synthetic(args.root, "synth-mnist", [1, 28, 28], 1, ["mlp", "cnn"])
    if "synth-seq32" in wanted:
        build_synthetic(args.root, "synth-seq32", [32, 32], 1, ["lstm", "gru", "transformer"])
    if "synth-rgb32" in wanted:
        build_synthetic(args.root, "synth-rgb32", [3, 32, 32], 3,
                        ["resnet18", "resnet50", "vgg11", "mobilenetv2", "densenet121", "vit"])


if __name__ == "__main__":
    main()
