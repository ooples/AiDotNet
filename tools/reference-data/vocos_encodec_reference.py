"""Vocos EnCodec-token decoder reference: the vocos package (gemelo-ai/vocos 0.1.0) as charactr/vocos-encodec-24khz
configures it, at a tiny size.

Builds VocosBackbone (adaptive layer norms over 4 bandwidth classes) and ISTFTHead (padding "same") with the
package's code and a random codebook table (EncodecFeatures stores EnCodec's codebooks as
feature_extractor.codebook_weights; the parameter names do not depend on the sizes, so the fixture also checks the
loader's names), stores the weights in float32 and decodes codes of 8 codebooks (6 kbps, bandwidth class 2) in float64
the way Vocos.decode_codes does (codes_to_features, then decode), recording
tests/AiDotNet.Tests/TextToSpeech/ReferenceData/vocos_encodec_reference.json. The ISTFT's Hann window is rebuilt in
float64 for the float64 run (the package builds it in float32). Default: verify; --write: regenerate.
"""
import argparse
import base64

import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

import _fixture

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/vocos_encodec_reference.json"
CONFIG = dict(latent=8, dim=16, intermediate=32, layers=2, n_fft=64, hop=16, bins=32, max_codebooks=16)
CODEBOOKS, FRAMES = 8, 11


def build(state=None):
    from vocos.heads import ISTFTHead
    from vocos.models import VocosBackbone
    torch.manual_seed(2294)
    backbone = VocosBackbone(input_channels=CONFIG["latent"], dim=CONFIG["dim"], intermediate_dim=CONFIG["intermediate"],
                             num_layers=CONFIG["layers"], adanorm_num_embeddings=4)
    head = ISTFTHead(dim=CONFIG["dim"], n_fft=CONFIG["n_fft"], hop_length=CONFIG["hop"], padding="same")
    codebooks = torch.nn.Parameter(torch.randn(CONFIG["max_codebooks"] * CONFIG["bins"], CONFIG["latent"]))
    modules = torch.nn.ModuleDict({"backbone": backbone, "head": head})
    if state is None:
        with torch.no_grad():
            # Non-trivial adaptive norms (they start at one and zero) and layer scales.
            for name, p in backbone.named_parameters():
                if ".norm.scale" in name or name == "norm.scale.weight":
                    p.add_(0.3 * torch.randn_like(p))
                if ".norm.shift" in name or name == "norm.shift.weight":
                    p.add_(0.1 * torch.randn_like(p))
                if name.endswith("gamma"):
                    p.copy_(0.5 + torch.rand_like(p))
    else:
        modules.load_state_dict({k: v for k, v in state.items() if k != "feature_extractor.codebook_weights"}, strict=False)
        codebooks.data.copy_(state["feature_extractor.codebook_weights"])
    return modules.eval(), codebooks


def decode(modules, codebooks, codes):
    offsets = torch.arange(0, CONFIG["bins"] * codes.shape[0], CONFIG["bins"])
    features = torch.nn.functional.embedding(codes + offsets.view(-1, 1), codebooks).sum(dim=0)   # [frames, latent]
    features = features.transpose(0, 1).unsqueeze(0)
    head = modules["head"]
    head.istft.window = torch.hann_window(CONFIG["n_fft"], dtype=torch.float64)
    with torch.no_grad():
        x = modules["backbone"](features, bandwidth_id=torch.tensor([2]))
        return head(x)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="regenerate the fixture (default: verify it)")
    args = parser.parse_args()
    g = torch.Generator().manual_seed(7)
    codes = torch.randint(0, CONFIG["bins"], (CODEBOOKS, FRAMES), generator=g)
    if args.write:
        modules, codebooks = build()
        state = {k: v.float().clone().contiguous() for k, v in modules.state_dict().items() if not k.endswith("istft.window")}
        state["feature_extractor.codebook_weights"] = codebooks.detach().float().clone()
        modules, codebooks = build(state)
        audio = decode(modules.double(), codebooks.double(), codes)
        data = {"config": CONFIG, "codes": codes.tolist(),
                "safetensors_base64": base64.b64encode(save_safetensors(state)).decode("ascii"),
                "audio": [float(v) for v in audio.tolist()]}
        _fixture.save(FIXTURE, data, __file__)
        print("wrote " + FIXTURE)
        return
    data = _fixture.load(FIXTURE)
    modules, codebooks = build(load_safetensors(base64.b64decode(data["safetensors_base64"])))
    audio = decode(modules.double(), codebooks.double(), codes)
    check = _fixture.Comparison("Vocos EnCodec decoder")
    check.floats("audio", data["audio"], [float(v) for v in audio.tolist()], 1e-12, float)
    check.report(exact=False)


if __name__ == "__main__":
    main()
