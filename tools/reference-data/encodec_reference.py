"""EnCodec (Défossez et al. 2022) reference: Hugging Face transformers' EncodecModel, the layout of facebook/encodec_*.

Recomputes, for each named fixture in
tests/AiDotNet.Tests/Audio/Codecs/ReferenceData/encodec_official_layout_reference.json, the encoder latent, the
residual-VQ codes at the fixture's bandwidth and the decoded audio, from the fixture's own weights (safetensors) and
input audio. Default: verify; --write: rewrite those outputs.
"""
import base64

import torch
from safetensors.torch import load as load_safetensors
from transformers import EncodecConfig, EncodecModel

import _fixture

FIXTURE = "tests/AiDotNet.Tests/Audio/Codecs/ReferenceData/encodec_official_layout_reference.json"


def run(fixture):
    model = EncodecModel(EncodecConfig(**fixture["config"])).eval()
    model.load_state_dict(load_safetensors(base64.b64decode(fixture["safetensors_base64"])), strict=True)
    audio = torch.tensor(fixture["audio"], dtype=torch.float32).reshape(fixture["audio_shape"])
    with torch.no_grad():
        latent = model.encoder(audio)
        encoded = model.encode(audio, bandwidth=fixture["bandwidth"])
        decoded = model.decode(encoded.audio_codes, encoded.audio_scales).audio_values
    codes = encoded.audio_codes.reshape(encoded.audio_codes.shape[-2], -1).tolist()
    return latent, codes, decoded


def main():
    args = _fixture.parse_args(__doc__)
    data = _fixture.load(FIXTURE)
    failed = False
    for fixture in data["fixtures"]:
        latent, codes, decoded = run(fixture)
        if args.write:
            fixture.update(latent=_fixture.flat(latent), latent_shape=list(latent.shape), codes=codes,
                           decoded=_fixture.flat(decoded), decoded_shape=list(decoded.shape))
            continue
        check = _fixture.Comparison("EnCodec " + fixture["name"])
        check.floats("latent", fixture["latent"], _fixture.flat(latent))
        check.exact("latent_shape", fixture["latent_shape"], list(latent.shape))
        check.exact("codes", fixture["codes"], codes)
        check.floats("decoded", fixture["decoded"], _fixture.flat(decoded))
        check.exact("decoded_shape", fixture["decoded_shape"], list(decoded.shape))
        try:
            check.report()
        except SystemExit:
            failed = True
    if args.write:
        _fixture.save(FIXTURE, data, __file__)
        print("wrote " + FIXTURE)
    elif failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
