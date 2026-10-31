"""Real-weights reference for Dia: the release's checkpoint (nari-labs/Dia-1.6B-0626, dia-v1.pth) in nari-labs/dia's
Dia (876125e) in float64, generating the README's script greedily (temperature 0) with CFG scale 3 and the CFG filter's
top-k 45, for 40 steps. The float32 arithmetic is replaced as in dia_reference.py.

Records the script's byte ids, the encoder's output, the first 4 steps' unconditional and conditional logits, and the codes
Dia.generate returns.

Usage: python dia_realweights_reference.py DIA_SNAPSHOT_DIR (with ~/source/repos/.venv-dia and the dia clone in
refsrc/dia). Writes tests/AiDotNet.Tests/TextToSpeech/ReferenceData/dia_realweights_reference.json.
"""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dia_reference  # noqa: E402  (float64 arithmetic; puts refsrc/dia on the path)
import dia.layers as dia_layers  # noqa: E402
import dia.model as dia_model  # noqa: E402
from dia.config import DiaConfig  # noqa: E402

SCRIPT = "[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices."
STEPS, RECORDED = 40, 4


def main():
    snapshot = sys.argv[1]
    config = DiaConfig.load(os.path.join(snapshot, "config.json"))
    dia = dia_model.Dia(config, device=torch.device("cpu"), load_dac=False)
    dia.compute_dtype = torch.float64
    state = torch.load(os.path.join(snapshot, "dia-v1.pth"), map_location="cpu", weights_only=True, mmap=True)
    dia.model.load_state_dict(state)
    del state
    dia.model = dia.model.double().eval()
    for module in dia.model.modules():
        if isinstance(module, dia_layers.RotaryEmbedding):
            dia_reference.timescale64(module)
        if hasattr(module, "compute_dtype"):
            module.compute_dtype = torch.float64
        if isinstance(module, dia_layers.MlpBlock):
            module.dtype = torch.float64

    text_ids = dia._encode_text(SCRIPT)
    with torch.no_grad():
        padded = dia._pad_text_input([text_ids])
        enc_state = dia_model.EncoderInferenceState.new(config, padded)
        encoded = dia.model.encoder(torch.cat([torch.zeros_like(padded[:, 0]), padded[:, 0]]), enc_state)[1, : len(text_ids)]

    steps = []
    decode_step = dia.model.decoder.decode_step

    def recording_step(tokens, state, current_idx):
        logits = decode_step(tokens, state, current_idx)
        if len(steps) < RECORDED:
            steps.append({"uncond": logits[0, -1].reshape(-1).tolist(), "cond": logits[1, -1].reshape(-1).tolist()})
        return logits

    dia.model.decoder.decode_step = recording_step
    with torch.no_grad():
        codes = dia.generate(SCRIPT, max_tokens=STEPS, cfg_scale=3.0, temperature=0.0, top_p=0.95, cfg_filter_top_k=45)

    result = {"script": SCRIPT, "text_ids": text_ids.tolist(), "encoded": encoded.reshape(-1).tolist(), "steps": steps,
              "max_tokens": STEPS, "generated": codes.tolist()}
    path = os.path.join(HERE, "..", "..", "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", "dia_realweights_reference.json")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle)
        handle.write("\n")
    print(len(text_ids), codes.shape, codes[:3].tolist())


if __name__ == "__main__":
    main()
