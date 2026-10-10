"""pyannote x-vector reference (Pheme's speaker encoder): pyannote.audio's XVectorSincNet.

Builds pyannote.audio's XVectorSincNet (the architecture of the pyannote/embedding checkpoint) with a seeded random
initialization, gives every batch norm non-trivial running statistics, stores the weights in float32, and runs it in
float64 in evaluation mode on an 8000-sample, float32-representable waveform. Writes weights, waveform and embedding to
tests/AiDotNet.Tests/TextToSpeech/ReferenceData/pyannote_xvector_reference.json. Default: verify; --write: regenerate.
"""
import base64

import torch
from pyannote.audio.models.embedding import XVectorSincNet
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

import _fixture

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/pyannote_xvector_reference.json"
SAMPLES = 8000


def build(state=None):
    torch.manual_seed(2294)
    model = XVectorSincNet(sample_rate=16000)
    if state is None:
        generator = torch.Generator().manual_seed(11)
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.BatchNorm1d):
                module.running_mean.copy_(torch.randn(module.num_features, generator=generator) * 0.1)
                module.running_var.copy_(torch.rand(module.num_features, generator=generator) + 0.5)
    else:
        model.load_state_dict(state, strict=True)
    return model.eval()


def main():
    args = _fixture.parse_args(__doc__)
    if args.write:
        model = build()
        state = {k: v.clone().contiguous() for k, v in model.state_dict().items() if not k.endswith("num_batches_tracked")}
        torch.manual_seed(5)
        waveform = (torch.randn(1, 1, SAMPLES) * 0.1)
        with torch.no_grad():
            embedding = model.double()(waveform.double())
        data = {"samples": SAMPLES, "safetensors_base64": base64.b64encode(save_safetensors(state)).decode("ascii"),
                "waveform": [float(v) for v in waveform.reshape(-1).tolist()],
                "embedding": [float(v) for v in embedding.reshape(-1).tolist()]}
        _fixture.save(FIXTURE, data, __file__)
        print("wrote " + FIXTURE)
        return
    data = _fixture.load(FIXTURE)
    state = load_safetensors(base64.b64decode(data["safetensors_base64"]))
    model = build({**state, **{k: torch.tensor(0) for k in build().state_dict() if k.endswith("num_batches_tracked")}})
    waveform = torch.tensor(data["waveform"], dtype=torch.float64).reshape(1, 1, -1)
    with torch.no_grad():
        embedding = model.double()(waveform)
    check = _fixture.Comparison("pyannote x-vector")
    check.floats("embedding", data["embedding"], embedding.reshape(-1).tolist(), 1e-12, float)
    check.report(exact=False)


if __name__ == "__main__":
    main()
