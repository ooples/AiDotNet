"""Reference fixture for GPT-SoVITS's reference-audio resampler (inference_webui.resample with match_librosa=True:
resampy's kaiser_best table, float32 arithmetic).

Runs the release's own function (RVC-Boss/GPT-SoVITS at b4570b8) in the gptsovits-ref image (resampy 0.4.3 added) on a
short noisy chord at 44.1 kHz, 32 kHz and 8 kHz, each resampled to 16 kHz.

    docker run --rm -v <refsrc>:/ref -v <repo>:/repo -w /ref/GPT-SoVITS/GPT_SoVITS gptsovits-ref \
        python /repo/tools/reference-data/gptsovits_resample_reference.py

Writes tests/AiDotNet.Tests/TextToSpeech/ReferenceData/gptsovits_resample_reference.json.
"""
import ast
import json
import math
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
resampy_filter_cache = {}


def load_functions():
    # inference_webui.py starts the UI on import; take only the two resampling functions from its source.
    source = open("inference_webui.py", encoding="utf-8").read()
    tree = ast.parse(source)
    keep = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {"_resampy_gpu_filter", "resample"}]
    namespace = {"torch": torch, "np": np, "resampy_filter_cache": resampy_filter_cache, "resample_transform_dict": {}}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "inference_webui.py", "exec"), namespace)
    return namespace["resample"]


def main():
    resample = load_functions()
    torch.manual_seed(2)
    out = {"cases": []}
    for rate in [44100, 32000, 8000]:
        n = int(rate * 0.25)
        t = torch.arange(n, dtype=torch.float32) / rate
        wav = 0.3 * torch.sin(2 * math.pi * 220 * t) + 0.2 * torch.sin(2 * math.pi * 3100 * t) + 0.02 * torch.randn(n)
        y = resample(wav.unsqueeze(0), rate, 16000, "cpu", match_librosa=True)[0]
        out["cases"].append({"rate": rate, "wav": wav.double().tolist(), "out": y.double().tolist()})
    path = os.path.join(HERE, "..", "..", "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", "gptsovits_resample_reference.json")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(out, handle)
        handle.write("\n")
    print("wrote", os.path.getsize(path), "bytes")


if __name__ == "__main__":
    main()
