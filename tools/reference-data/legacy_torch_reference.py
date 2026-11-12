"""A small checkpoint in torch.save's legacy (pre-1.6, non-ZIP) format, for WeightLoader's legacy reader.

A nested dictionary (as a training checkpoint nests its state) with a float32 matrix, a strided view of it (a transpose
and a narrow slice sharing its storage), a float64 vector and a float16 tensor. Writes
tests/AiDotNet.Tests/ComputerVision/ReferenceData/legacy_torch.pt and legacy_torch.json (the values as torch reads them).
Needs torch.
"""
import json

import torch

torch.manual_seed(211)
base = torch.randn(4, 6)
state = {"epoch": 3, "state_dict": {"model": {"w": base, "w_t": base.t(), "w_slice": base[1:3, 2:5], "v64": torch.randn(5, dtype=torch.float64),
                                              "h16": torch.randn(3).half()}}}
torch.save(state, "tests/AiDotNet.Tests/ComputerVision/ReferenceData/legacy_torch.pt", _use_new_zipfile_serialization=False)
loaded = torch.load("tests/AiDotNet.Tests/ComputerVision/ReferenceData/legacy_torch.pt", weights_only=False)
flat = {f"state_dict.model.{k}": {"shape": list(v.shape), "values": v.float().reshape(-1).tolist()} for k, v in loaded["state_dict"]["model"].items()}
json.dump(flat, open("tests/AiDotNet.Tests/ComputerVision/ReferenceData/legacy_torch.json", "w"), indent=1)
print("wrote", list(flat))
