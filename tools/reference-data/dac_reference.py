"""DAC (Kumar et al. 2023) reference: Hugging Face transformers' DacModel, the layout of descript/dac_* on the Hub.

Recomputes the encoder latent, the residual-VQ codes and the decoded audio of
tests/AiDotNet.Tests/Audio/Codecs/ReferenceData/dac_official_layout_reference.json from the fixture's own weights
(safetensors) and input audio. Default: verify; --write: rewrite those three outputs.
"""
import base64

import torch
from safetensors.torch import load as load_safetensors
from transformers import DacConfig, DacModel

import _fixture

FIXTURE = "tests/AiDotNet.Tests/Audio/Codecs/ReferenceData/dac_official_layout_reference.json"


def main():
    args = _fixture.parse_args(__doc__)
    data = _fixture.load(FIXTURE)
    model = DacModel(DacConfig(**data["config"])).eval()
    model.load_state_dict(load_safetensors(base64.b64decode(data["safetensors_base64"])), strict=True)
    audio = torch.tensor(data["audio"], dtype=torch.float32).reshape(1, 1, -1)
    with torch.no_grad():
        latent = model.encoder(audio)
        encoded = model.encode(audio)
        decoded = model.decode(encoded.quantized_representation).audio_values.reshape(1, 1, -1)
    codes = encoded.audio_codes.reshape(encoded.audio_codes.shape[1], -1).tolist()
    if args.write:
        data.update(latent=_fixture.flat(latent), latent_shape=list(latent.shape), codes=codes,
                    decoded=_fixture.flat(decoded), decoded_shape=list(decoded.shape))
        _fixture.save(FIXTURE, data, __file__)
        print("wrote " + FIXTURE)
        return
    check = _fixture.Comparison("DAC")
    check.floats("latent", data["latent"], _fixture.flat(latent))
    check.exact("latent_shape", data["latent_shape"], list(latent.shape))
    check.exact("codes", data["codes"], codes)
    check.floats("decoded", data["decoded"], _fixture.flat(decoded))
    check.exact("decoded_shape", data["decoded_shape"], list(decoded.shape))
    check.report()


if __name__ == "__main__":
    main()
