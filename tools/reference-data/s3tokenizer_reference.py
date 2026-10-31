"""S3 speech tokenizer v1 reference: xingchensong/S3Tokenizer (commit 9bf5d845b5e043ffaf4657f4942939091c7697a2,
s3tokenizer/model.py and utils.py), the PyTorch form of CosyVoice 1's speech_tokenizer_v1.onnx. Pass the checkout with
--s3tokenizer.

Builds a tiny randomly initialized tokenizer (seeded, float64, a random codebook) and records in
tests/AiDotNet.Tests/TextToSpeech/ReferenceData/s3tokenizer_reference.json:
- the 128-band log-mel of a short random waveform (log_mel_spectrogram);
- the encoder states and tokens of a padded batch of two log-mels (lengths 13 and 10);
- the tokens of a 3100-frame log-mel, which takes the 30 s windowing path (_quantize_mixed_batch).

float32 steps removed (the other fixtures do the same):
- the model is built under a float64 default dtype, so the positional table is float64;
- mask_to_bias accepts float64;
- .float() leaves float64 tensors alone (LayerNorm, the attention softmax and VectorQuantization.encode call it);
- the mel filters are librosa's in float64. The npz Whisper ships holds librosa.filters.mel(sr=16000, n_fft=400,
  n_mels=128) in float32, and this script checks they round to it.

Default: verify; --write: regenerate.
"""
import argparse
import base64
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

import _fixture

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/s3tokenizer_reference.json"
CONFIG = dict(n_mels=4, n_audio_ctx=1500, n_audio_state=8, n_audio_head=2, n_audio_layer=2, n_codebook_size=16)
BATCH_LENGTHS = [13, 10]
LONG_FRAMES = 3100
AUDIO_SAMPLES = 1600


def flat(tensor):
    return [float(v) for v in tensor.detach().reshape(-1).tolist()]


def load_package(root):
    # utils imports onnx for onnx2torch, which the reference never calls here.
    sys.modules.setdefault("onnx", types.ModuleType("onnx"))
    sys.path.insert(0, root)
    import librosa
    import s3tokenizer.model as model_module
    import s3tokenizer.utils as utils

    def mask_to_bias(mask, dtype):
        assert mask.dtype == torch.bool
        return (1.0 - mask.to(dtype)) * -1.0e+10

    def mel_filters(device, n_mels):
        return torch.from_numpy(librosa.filters.mel(sr=16000, n_fft=400, n_mels=n_mels, dtype=np.float64)).to(device)

    shipped = np.load(f"{root}/s3tokenizer/assets/mel_filters.npz")["mel_128"]
    # The shipped float32 filters are these float64 ones to within 2 float32 ulps (measured: at most 1.31, in 4 of
    # 25,728 entries above 1). Whisper computed them in float32 with an older librosa.
    ulp = np.spacing(np.abs(shipped)).astype(np.float64)
    assert (np.abs(mel_filters("cpu", 128).numpy() - shipped) <= 2 * ulp).all(), "librosa no longer makes Whisper's filters"
    model_module.mask_to_bias = mask_to_bias
    # LayerNorm, the attention softmax and VectorQuantization.encode compute in float32 via .float() (an upcast for
    # fp16 that is a downcast here); float64 tensors are left as they are.
    to_float = torch.Tensor.float
    torch.Tensor.float = lambda self, *args, **kwargs: self if self.dtype == torch.float64 else to_float(self, *args, **kwargs)
    utils._mel_filters = mel_filters
    return model_module, utils


def build(model_module, state=None):
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(2294)
    model = model_module.S3Tokenizer("speech_tokenizer_v1", model_module.ModelConfig(**CONFIG)).eval()
    with torch.no_grad():
        # Row norms in [0.5, 1.5]. With N(0, 1) rows (norms near √d) the |e|² term picks the same entry for every state.
        # With unit rows, normalizing the state cannot change the argmax, so the tokens could not detect a missing
        # normalization.
        embed = model.quantizer._codebook.embed
        embed.copy_(F.normalize(torch.randn_like(embed), dim=-1) * (0.5 + torch.rand(embed.shape[0], 1)))
        # sinusoids() still yields float32 under a float64 default (numpy's float64 scalar times an int tensor gives the
        # default float of a scalar operand, float32); this is its formula evaluated in float64.
        channels = CONFIG["n_audio_state"]
        increment = np.log(10000) / (channels // 2 - 1)
        scaled = (torch.arange(CONFIG["n_audio_ctx"], dtype=torch.float64)[:, None]
                  * torch.exp(-increment * torch.arange(channels // 2, dtype=torch.float64))[None, :])
        # Assigned, not copied: copy_ would round into the float32 buffer.
        model.encoder.positional_embedding = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=1)
    if state is not None:
        model.load_state_dict(state, strict=True)
    return model


def run(model, utils, audio, batch, long):
    log_mel = utils.log_mel_spectrogram(audio, n_mels=128)
    lengths = torch.tensor(BATCH_LENGTHS)
    with torch.no_grad():
        hidden, hidden_lengths = model.encoder(batch, lengths)
    codes, code_lengths = model.quantize(batch, lengths)
    long_codes, long_length = model.quantize(long, torch.tensor([LONG_FRAMES]))
    batch_hidden = [hidden[i, :hidden_lengths[i]] for i in range(len(BATCH_LENGTHS))]
    batch_codes = [codes[i, :code_lengths[i]].tolist() for i in range(len(BATCH_LENGTHS))]
    return log_mel, batch_hidden, batch_codes, long_codes[0, :long_length[0]].tolist()


def inputs(seed):
    generator = torch.Generator().manual_seed(seed)
    audio = 0.3 * torch.randn(AUDIO_SAMPLES, generator=generator)
    batch = torch.randn(len(BATCH_LENGTHS), CONFIG["n_mels"], max(BATCH_LENGTHS), generator=generator)
    for i, length in enumerate(BATCH_LENGTHS):
        batch[i, :, length:] = 0
    long = torch.randn(1, CONFIG["n_mels"], LONG_FRAMES, generator=generator)
    return audio, batch, long


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s3tokenizer", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    model_module, utils = load_package(args.s3tokenizer)
    if args.write:
        model = build(model_module)
        state = {k: v.detach().clone().contiguous() for k, v in model.state_dict().items()}
        audio, batch, long = inputs(7)
        log_mel, hidden, codes, long_codes = run(model, utils, audio, batch, long)
        data = {"config": CONFIG, "batch_lengths": BATCH_LENGTHS,
                "safetensors_base64": base64.b64encode(save_safetensors(state)).decode("ascii"),
                "audio": flat(audio), "log_mel": flat(log_mel),
                "batch": flat(batch), "hidden": [flat(h) for h in hidden], "codes": codes,
                "long": flat(long), "long_codes": long_codes}
        _fixture.save(FIXTURE, data, __file__)
        print("wrote " + FIXTURE)
        return
    data = _fixture.load(FIXTURE)
    model = build(model_module, load_safetensors(base64.b64decode(data["safetensors_base64"])))
    audio = torch.tensor(data["audio"])
    batch = torch.tensor(data["batch"]).reshape(len(BATCH_LENGTHS), CONFIG["n_mels"], max(BATCH_LENGTHS))
    long = torch.tensor(data["long"]).reshape(1, CONFIG["n_mels"], LONG_FRAMES)
    log_mel, hidden, codes, long_codes = run(model, utils, audio, batch, long)
    check = _fixture.Comparison("S3 tokenizer")
    check.floats("log_mel", data["log_mel"], flat(log_mel), 1e-12, float)
    for i, h in enumerate(hidden):
        check.floats(f"hidden[{i}]", data["hidden"][i], flat(h), 1e-12, float)
    check.exact("codes", data["codes"], codes)
    check.exact("long_codes", data["long_codes"], long_codes)
    check.report(exact=False)


if __name__ == "__main__":
    main()
