"""Step-Audio's linguistic tokenizer reference (StepAudioTokenizer.get_vq02_code: the streaming Paraformer encoder through
funasr_detach's ParaformerStreaming.infer_encoder, then k-means over linguistic_tokenizer.npy).

The release's WavFrontendOnline (Kaldi fbank on 16-bit-scaled samples, LFR 7/6 with its splice cache, CMVN) and
SANMEncoderChunkOpt (StreamSinusoidalPositionEncoder, 5 overlap frames, SAN-M layers with look-back key/value caches)
are imported from Step-Audio's funasr_detach; infer_encoder's chunk loop is copied (it lives on the model, which needs
the full package). Tiny: 8 mel bands, width 16, 2 heads, feed-forward 32, 3 blocks, FSMN kernel 11, seeded random
weights and CMVN, 12 centroids near the encoded frames, float64 with dither 0, chunk [0, 4, 5] (240 ms strides) and encoder look-back 4 as the
tokenizer sets them, on 1.3 s of a chirp fed whole with is_final. The release dithers at 1.0 (Kaldi's default, as its
config leaves it); the comparison needs 0.
Usage: python tools/reference-data/stepaudio_linguistic_reference.py --stepaudio <Step-Audio checkout> [--write]
"""
import argparse
import math
import os
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fixture  # noqa: E402

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/stepaudio_linguistic_reference.json"
MELS, LFR_M, LFR_N, WIDTH = 8, 7, 6, 16
CHUNK, LOOK_BACK = [0, 4, 5], 4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stepaudio")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    root = args.stepaudio
    for name in ("funasr_detach", "funasr_detach.models", "funasr_detach.frontends", "funasr_detach.models.scama",
                 "funasr_detach.models.sanm", "funasr_detach.models.transformer"):
        module = types.ModuleType(name)
        module.__path__ = [os.path.join(root, *name.split("."))]
        sys.modules[name] = module
    register = types.ModuleType("funasr_detach.register")
    register.tables = types.SimpleNamespace(register=lambda *a, **k: (lambda c: c))
    sys.modules["funasr_detach.register"] = register
    sys.path.insert(0, root)
    import funasr_detach.frontends.wav_frontend as wav_frontend
    from funasr_detach.models.scama.encoder import SANMEncoderChunkOpt

    # The frontend's waveform moves to CUDA and its outputs to float32; keep the CPU and float64.
    torch.Tensor.cuda = lambda self, *a, **k: self
    torch.manual_seed(0)
    shift = (-8 + 0.5 * torch.randn(MELS * LFR_M))
    scale = (0.3 + 0.05 * torch.rand(MELS * LFR_M))
    frontend = wav_frontend.WavFrontendOnline(fs=16000, window="hamming", n_mels=MELS, frame_length=25, frame_shift=10, lfr_m=LFR_M,
                                              lfr_n=LFR_N, dither=0.0)
    frontend.cmvn = torch.stack([shift, scale])
    frontend.cmvn_file = "set"

    def apply_cmvn(inputs, cmvn):
        return (inputs + cmvn[0:1, :inputs.shape[1]]) * cmvn[1:2, :inputs.shape[1]]

    frontend.apply_cmvn = apply_cmvn
    # apply_lfr returns float32; re-run its source without the cast to keep float64.
    import inspect
    import textwrap
    lines = textwrap.dedent(inspect.getsource(wav_frontend.WavFrontendOnline.apply_lfr)).splitlines()
    source = chr(10).join(line for line in lines if not line.startswith("@staticmethod"))
    namespace = dict(wav_frontend.__dict__)
    exec(source.replace("LFR_outputs.type(torch.float32)", "LFR_outputs"), namespace)
    frontend.apply_lfr = namespace["apply_lfr"]
    encoder = SANMEncoderChunkOpt(input_size=MELS * LFR_M, output_size=WIDTH, attention_heads=2, linear_units=32, num_blocks=3,
                                  dropout_rate=0.0, attention_dropout_rate=0.0, input_layer="pe_online", kernel_size=11,
                                  chunk_size=[12, 15], stride=[8, 10], pad_left=[0, 0], encoder_att_look_back_factor=[4, 4],
                                  decoder_att_look_back_factor=[1, 1]).double().eval()
    with torch.no_grad():
        for name, p in encoder.named_parameters():
            p.copy_(1 + 0.2 * torch.randn_like(p) if "norm" in name and name.endswith("weight") else 0.3 * torch.randn_like(p))
    noise = torch.randn(12, WIDTH)

    n_samples = 20800
    i = torch.arange(n_samples, dtype=torch.float64)
    audio = 0.4 * torch.sin(2 * math.pi * 150 * i / 16000 * (1 + i / 30000)) + 0.1 * torch.sin(0.7 * i)

    # ParaformerStreaming.init_cache and infer_encoder's loop, as the release runs them for a whole recording.
    cache = {"encoder": {"start_idx": 0, "chunk_size": CHUNK, "encoder_chunk_look_back": LOOK_BACK, "last_chunk": False, "opt": None,
                         "feats": torch.zeros((1, CHUNK[0] + CHUNK[2], MELS * LFR_M)), "tail_chunk": False},
             "frontend": {}, "prev_samples": torch.empty(0)}
    stride = int(CHUNK[1] * 960)
    sample = torch.cat((cache["prev_samples"], audio))
    n = int(len(sample) // stride + 1)
    outs, features = [], []
    with torch.no_grad():
        for k in range(n):
            final = k == n - 1
            piece = sample[k * stride:(k + 1) * stride]
            if final and len(piece) == 0:
                break
            speech, lengths = frontend(piece[None, :], [piece.shape[0]], cache=cache["frontend"], is_final=final)
            if len(speech) == 0 and final:
                break
            features.append(speech[0].reshape(-1).tolist())
            out, _, _ = encoder.forward_chunk(speech.clone(), lengths, cache=cache["encoder"], is_final=final)
            outs.append(out[:, -int(lengths[0]):])
            if final:
                break
    encoded = torch.cat(outs, dim=1)[0]
    # Centroids near every other encoded frame, so the codes vary.
    means = encoded[torch.arange(0, 24, 2).clamp(max=encoded.shape[0] - 1)] + 0.05 * noise
    codes = torch.cdist(encoded, means).argmin(dim=1).tolist()
    data = {"encoded": encoded.reshape(-1).tolist(), "codes": codes, "chunk_features": features}
    if args.write:
        state = {k: v.reshape(-1).tolist() for k, v in encoder.state_dict().items()}
        _fixture.save(FIXTURE, {"audio": audio.tolist(), "shift": shift.tolist(), "scale": scale.tolist(), "means": means.reshape(-1).tolist(),
                                "state": state, **data}, __file__)
        print("wrote", FIXTURE, encoded.shape, codes)
        return
    stored = _fixture.load(FIXTURE)
    check = _fixture.Comparison("Step-Audio linguistic tokenizer")
    check.floats("encoded", stored["encoded"], data["encoded"], 1e-12, float)
    if stored["codes"] != codes:
        check.failures.append("codes differ")
    check.report(exact=False)


if __name__ == "__main__":
    main()
