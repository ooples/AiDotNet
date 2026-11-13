"""Run the unmodified Transformers 4.48.3 Llama-3 RoPE functions used by OuteTTS 1.0.

Requires torch; no Transformers installation or model weights. AST selection removes only unrelated
module definitions/imports, leaving both author function bodies unchanged. The release config names
Transformers 4.48.3; the downloaded source is checked against its Git blob before execution.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from urllib.request import urlopen

import torch

SOURCE = "https://raw.githubusercontent.com/huggingface/transformers/v4.48.3/src/transformers/modeling_rope_utils.py"
BLOB = "b2d343e0237fa0492208e1a00ec899f548d7d5e4"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with urlopen(SOURCE, timeout=60) as response:
        source = response.read()
    blob = hashlib.sha1(b"blob " + str(len(source)).encode() + b"\0" + source).hexdigest()
    if blob != BLOB:
        raise RuntimeError(f"Author source changed: {blob} != {BLOB}")
    tree = ast.parse(source, filename=SOURCE)
    names = {"_compute_default_rope_parameters", "_compute_llama3_parameters"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(functions) != 2:
        raise RuntimeError("Both unmodified author functions are required")
    tree.body = ast.parse("from __future__ import annotations").body + functions
    namespace = {"torch": torch, "math": math}
    exec(compile(ast.fix_missing_locations(tree), SOURCE, "exec"), namespace)
    cases = []
    configurations = [(head_dim, low, high, original) for head_dim in (8, 64)
                      for low, high, original in [(1, 4, 8192), (2, 4, 8192), (1, 2, 8192), (1, 4, 4096)]]
    for head_dim, low, high, original in configurations:
        scaling = {"factor": 32.0, "low_freq_factor": low, "high_freq_factor": high,
                   "original_max_position_embeddings": original, "rope_type": "llama3"}
        config = SimpleNamespace(hidden_size=16, num_attention_heads=2, head_dim=head_dim,
                                 rope_theta=500000.0, rope_scaling=scaling)
        frequencies, attention_factor = namespace["_compute_llama3_parameters"](config, torch.device("cpu"))
        cases.append({"head_dim": head_dim, "low": low, "high": high, "original": original,
                      "frequencies": frequencies.tolist(), "attention_factor": attention_factor})
    data = {"source": SOURCE, "git_blob": BLOB, "torch": torch.__version__,
            "dtype": "float32", "cases": cases}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(cases)} author cases to {args.output}")


if __name__ == "__main__":
    main()
