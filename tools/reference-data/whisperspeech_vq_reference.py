"""WhisperSpeech's semantic tokenizer reference (collabora/WhisperSpeech vq_stoks.py RQBottleneckTransformer with
vector-quantize-pytorch 1.6.32's ResidualVQ, whose per-layer projections the released checkpoints carry, and openai-whisper 20240930): a tiny Whisper (80 mels, 16 wide, 2 heads,
one encoder and one decoder layer) as the frozen teacher and student decoder, and a tiny bottleneck with the released
whisper-vq-stoks-medium-en+pl tunables (cosine-similarity codebook through a 4-dim projection, 8 codes plus the padding
code, mean downsampling by 2, padding masked to the padding code's embedding, one rotary output block), seeded random
weights in float64 (LayerNorm's, the rotary table's, the codebook's and the decoder logits' float32 casts removed). Dead-code replacement
draws random samples, so the comparison runs with threshold_ema_dead_code 0 (the release's checkpoints use 2).

Records quantize() of a mel, dequantize() of codes, and one training step's loss (cross-entropy + KL to the teacher +
commitment) with the codebook's moving averages after it.
Usage: PYTHONPATH=<vector_quantize_pytorch 1.6.32 dir>;<dir holding whisper/> python tools/reference-data/whisperspeech_vq_reference.py --whisperspeech <checkout> [--write]
"""
import argparse
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fixture  # noqa: E402
from whisperspeech_t2s_reference import load as load_t2s  # noqa: E402
from whisperspeech_s2a_reference import patch_rotary  # noqa: E402

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/whisperspeech_vq_reference.json"
WHISPER = dict(n_mels=80, n_audio_ctx=1500, n_audio_state=16, n_audio_head=2, n_audio_layer=1, n_vocab=51865, n_text_ctx=448,
               n_text_state=16, n_text_head=2, n_text_layer=1)
VQ = dict(vq_codes=8, q_depth=1, depth=1, n_head=2, head_width=8, ffn_mult=4, codebook_dim=4, threshold_ema_dead_code=0,
          use_cosine_sim=True, downsample=2)
TUNABLES = dict(init_std=1.5, embeddings_std=0.045, embeddings_lr_scale=1, output_mult=1, query_mult=2, rope=True, mask_embs=True,
                downsample_conv=False, downsample_mean=True, codebook_dim=32, codebook_decay=0.9)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--whisperspeech")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    load_t2s(args.whisperspeech)
    import whisperspeech.modules as modules
    patch_rotary(modules)
    import whisper
    import whisper.model as wm
    import vector_quantize_pytorch.vector_quantize_pytorch as vqp
    # float64: the codebook casts its input to float32; whisper's LayerNorm and Linear cast to the input's dtype already.
    original = vqp.CosineSimCodebook.forward
    def forward(self, x, *a, **k):
        x = x.double()
        real_float = torch.Tensor.float
        torch.Tensor.float = lambda t: t
        try:
            return original(self, x, *a, **k)
        finally:
            torch.Tensor.float = real_float
    vqp.CosineSimCodebook.forward = forward
    decode = wm.TextDecoder.forward
    def decoder_forward(self, *a, **k):
        real_float = torch.Tensor.float
        torch.Tensor.float = lambda t: t     # the logits' float32 cast
        try:
            return decode(self, *a, **k)
        finally:
            torch.Tensor.float = real_float
    wm.TextDecoder.forward = decoder_forward
    wm.LayerNorm.forward = lambda self, x: torch.nn.LayerNorm.forward(self, x)
    # utils.py patches torio's media-metadata parser for dataset loading; torchaudio 2.9+ no longer ships torio.
    import types
    for name in ("torio", "torio.io", "torio.io._streaming_media_decoder"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["torio"].io = sys.modules["torio.io"]
    sys.modules["torio.io"]._streaming_media_decoder = sys.modules["torio.io._streaming_media_decoder"]
    from whisperspeech import vq_stoks

    torch.manual_seed(0)
    teacher = wm.Whisper(wm.ModelDimensions(**WHISPER)).double()
    with torch.no_grad():
        for name, p in teacher.named_parameters():
            # Four decimals keep the 51865-row vocabulary's fixture small.
            p.copy_(torch.round(1 + 0.1 * torch.randn_like(p) if "ln" in name and name.endswith("weight") else 0.2 * torch.randn_like(p), decimals=4))
    model = vq_stoks.RQBottleneckTransformer(**VQ, whisper_model_name="tiny", tunables=vq_stoks.Tunables(**TUNABLES)).double()
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(1 + 0.2 * torch.randn_like(p) if "ln" in name and name.endswith("weight") else 0.3 * torch.randn_like(p))
        cb = model.rq.layers[0]._codebook
        cb.embed.copy_(torch.nn.functional.normalize(torch.randn_like(cb.embed), dim=-1))
        cb.embed_avg.copy_(cb.embed.clone())
    model.whmodel = [teacher]

    n_samples = 16000
    i = torch.arange(n_samples, dtype=torch.float64)
    audio = (0.4 * torch.sin(2 * math.pi * 180 * i / 16000 * (1 + i / 20000)) + 0.1 * torch.sin(0.9 * i))[None]
    torch.set_default_dtype(torch.float32)   # the mel filters are stored in float32; the mel is only an input here
    mel = whisper.log_mel_spectrogram(audio.float(), 80).double()
    torch.set_default_dtype(torch.float64)
    padded = torch.nn.functional.pad(mel, (0, 3000 - mel.shape[-1]), value=-1.5)
    mask = torch.zeros(1, 1500, dtype=torch.bool)
    mask[0, :mel.shape[-1] // 2] = True
    # Random code vectors all sit on one side of the projected frames (the projection's bias dominates), so seed the
    # used codes from the frames, as k-means initialisation would, to give the codes and the EMA update spread.
    with torch.no_grad():
        frames = model.downsample_embeddings(teacher.encoder(padded))
        frames = torch.nn.functional.normalize(model.rq.layers[0].project_in(frames + model.mlp(model.mlp_ln(frames))), dim=-1)[0]
        cb.embed[0, :VQ["vq_codes"]] = frames[torch.linspace(0, mel.shape[-1] // 4 - 1, VQ["vq_codes"]).long()]
        cb.embed_avg.copy_(cb.embed.clone())
    input_toks = torch.tensor([[50258, 50259, 50359, 50363, 440, 1243, 286, 264]])
    output_toks = torch.tensor([[-100, -100, -100, 440, 1243, 286, 264, 50257]])

    state = {k: v.reshape(-1).tolist() for k, v in model.state_dict().items() if torch.is_tensor(v) and v.dtype.is_floating_point}
    teacher_state = {k: v.reshape(-1).tolist() for k, v in teacher.state_dict().items() if v.dtype.is_floating_point}
    model.eval()
    with torch.no_grad():
        embs = teacher.encoder(padded)
        stoks = model.quantize(embs)[0]
        dequantized = model.dequantize(stoks[: mel.shape[-1] // 2 // 2])
    model.train()
    # extract_teacher computes the mel itself; give it the padded one.
    model.log_mel_spectrogram = lambda samples: padded
    _, _, loss = model(audio, mask, input_toks, output_toks)
    after = {"embed": cb.embed.reshape(-1).tolist(), "embed_avg": cb.embed_avg.reshape(-1).tolist(), "cluster_size": cb.cluster_size.reshape(-1).tolist()}
    data = {"stoks": stoks.tolist(), "dequantized": dequantized.reshape(-1).tolist(), "loss": float(loss), "after": after}
    if args.write:
        _fixture.save(FIXTURE, {"whisper": WHISPER, "vq": VQ, "mel": padded.reshape(-1).tolist(), "mel_frames": mel.shape[-1],
                                "mask_frames": int(mask.sum()), "input_toks": input_toks[0].tolist(), "output_toks": output_toks[0].tolist(),
                                "state": state, "teacher": teacher_state, **data}, __file__)
        print("wrote", FIXTURE, float(loss), torch.bincount(stoks, minlength=9).tolist(), stoks.tolist()[:60])
        return
    stored = _fixture.load(FIXTURE)
    check = _fixture.Comparison("WhisperSpeech VQ")
    check.floats("loss", [stored["loss"]], [data["loss"]], 1e-12, float)
    check.floats("dequantized", stored["dequantized"], data["dequantized"], 1e-12, float)
    if stored["stoks"] != data["stoks"]:
        check.failures.append("codes differ")
    check.report(exact=False)


if __name__ == "__main__":
    main()
