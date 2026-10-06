"""Shared helpers for the reference-data generators: load a fixture, compare recomputed values, rewrite it."""
import argparse
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def fixture_path(relative):
    return os.path.join(REPO, *relative.split("/"))


def parse_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--write", action="store_true",
                        help="rewrite the fixture's outputs from its committed weights and inputs (default: verify only)")
    return parser.parse_args()


def load(relative):
    with open(fixture_path(relative), encoding="utf-8") as handle:
        return json.load(handle)


def save(relative, data, script):
    data["generator"] = "tools/reference-data/" + os.path.basename(script)
    data["torch"] = torch.__version__
    with open(fixture_path(relative), "w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle)


def flat(tensor):
    return [float(v) for v in tensor.detach().to(torch.float32).reshape(-1).tolist()]


class Comparison:
    """Collects exact comparisons between committed values and recomputed ones."""

    def __init__(self, name):
        self.name = name
        self.failures = []

    def floats(self, label, committed, recomputed, relative_tolerance=0.0, dtype=np.float32):
        """Exact by default. A tolerance is relative to max(1, |committed|): compiled numerical libraries (pyworld,
        pycwt's FFT) differ from one build to another in the last few ulps."""
        want = np.asarray(committed, dtype=dtype)
        got = np.asarray(recomputed, dtype=dtype)
        if want.shape != got.shape:
            self.failures.append(f"{label}: {want.shape} committed, {got.shape} recomputed")
            return
        if not want.size:
            return
        excess = np.abs(want - got) - relative_tolerance * np.maximum(1.0, np.abs(want))
        if float(excess.max()) > 0.0:
            self.failures.append(f"{label}: max |difference| {float(np.abs(want - got).max()):.3g}"
                                 + (f" (tolerance {relative_tolerance:g} relative)" if relative_tolerance else ""))

    def exact(self, label, committed, recomputed):
        if committed != recomputed:
            self.failures.append(f"{label}: differs")

    def report(self, exact=True):
        if self.failures:
            print(f"{self.name}: MISMATCH")
            for failure in self.failures:
                print("  " + failure)
            sys.exit(1)
        print(f"{self.name}: matches the reference implementation" + (" exactly" if exact else " within tolerance"))
