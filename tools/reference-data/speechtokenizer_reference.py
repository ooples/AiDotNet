"""SpeechTokenizer (Zhang et al. 2024) reference: the authors' speechtokenizer package (ZhangXInFD/SpeechTokenizer).

Recomputes the encoder latent, the residual-VQ codes and the decoded audio of
tests/AiDotNet.Tests/Audio/Codecs/ReferenceData/speechtokenizer_official_layout_reference.json from the fixture's own
torch.save checkpoint and input audio. Default: verify; --write: rewrite those outputs.
"""
import base64
import io

import torch
from speechtokenizer import SpeechTokenizer

import _fixture

FIXTURE = "tests/AiDotNet.Tests/Audio/Codecs/ReferenceData/speechtokenizer_official_layout_reference.json"


def main():
    args = _fixture.parse_args(__doc__)
    data = _fixture.load(FIXTURE)
    model = SpeechTokenizer(data["config"]).eval()
    state = torch.load(io.BytesIO(base64.b64decode(data["checkpoint_base64"])), map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    audio = torch.tensor(data["audio"], dtype=torch.float32).reshape(1, 1, -1)
    with torch.no_grad():
        latent = model.encoder(audio)
        codes = model.encode(audio, n_q=data["config"]["n_q"])          # [n_q, batch, frames]
        decoded = model.decode(codes)
    code_rows = codes.reshape(codes.shape[0], -1).tolist()
    keys = sorted(state.keys())                                       # the fixture lists them sorted
    if args.write:
        data.update(latent=_fixture.flat(latent), latent_shape=list(latent.shape), codes=code_rows,
                    decoded=_fixture.flat(decoded), decoded_shape=list(decoded.shape), keys=keys)
        _fixture.save(FIXTURE, data, __file__)
        print("wrote " + FIXTURE)
        return
    check = _fixture.Comparison("SpeechTokenizer")
    check.floats("latent", data["latent"], _fixture.flat(latent))
    check.exact("latent_shape", data["latent_shape"], list(latent.shape))
    check.exact("codes", data["codes"], code_rows)
    check.floats("decoded", data["decoded"], _fixture.flat(decoded))
    check.exact("decoded_shape", data["decoded_shape"], list(decoded.shape))
    check.exact("keys", data["keys"], keys)
    check.report()


if __name__ == "__main__":
    main()
