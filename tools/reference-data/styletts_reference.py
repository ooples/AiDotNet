"""StyleTTS reference: yl4579/StyleTTS (models.py, Utils/ASR, Utils/JDC) and resemble-ai/monotonic_align, the code the
paper's (Li et al. 2022) gaps follow.

Builds a tiny randomly initialized StyleTTS with the reference's build_model (pass a checkout of yl4579/StyleTTS with
--styletts; it is not vendored here), stores its weights in float32, runs it in float64 in evaluation mode and records:
- the text encoder's phoneme features, the style encoder's style of a reference spectrogram;
- the duration encoder's output, the predicted durations, and inference as the reference's notebooks run it: the
  rounded durations, the predicted pitch and energy, and the decoder's mel spectrogram;
- the text aligner's token logits and attention energies (its random token masking turned off), and the pitch
  extractor's F0 (the JDC network at tiny widths: its own ResBlock and forward, the hard-coded 512 replaced by its width
  and its cast to float32 dropped);
- the discriminator's logit and, for every feature map, its first 64 values, sum and sum of squares;
- monotonic alignment search on a score matrix (monotonic_align.core.maximum_path_c).
in tests/AiDotNet.Tests/TextToSpeech/ReferenceData/styletts_reference.json. Default: verify; --write: regenerate.

Needs torch, torchaudio (the aligner's DCT), munch, pyyaml and monotonic_align (built from resemble-ai/monotonic_align).
"""
import argparse
import base64
import inspect
import sys
import textwrap

import numpy as np
import torch
from munch import Munch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors
from torch import nn

import _fixture

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/styletts_reference.json"
MODEL = dict(hidden_dim=16, n_token=178, style_dim=8, n_layer=2, dim_in=4, max_conv_dim=16, n_mels=80, dropout=0.2)
ASR = dict(input_dim=80, hidden_dim=16, n_token=178, n_layers=2, token_embedding_dim=8)
JDC_WIDTHS = [4, 8, 12, 16]
JDC_LSTM = 16
TOKENS = [0, 51, 157, 102, 16, 83, 56, 135, 47, 102, 4, 0]
FRAMES = 48
REFERENCE_FRAMES = 96


def load(root):
    sys.path.insert(0, root)
    import models
    from Utils.ASR.models import ASRCNN
    from Utils.JDC import model as jdc

    class TinyJDC(jdc.JDCNet):
        """The reference JDCNet at other widths: its layers as JDCNet builds them, its forward with the 512-wide
        classifier input replaced by this network's width."""

        def __init__(self, widths, lstm, num_class=1, leaky_relu_slope=0.01):
            nn.Module.__init__(self)
            w0, w1, w2, w3 = widths
            self.num_class = num_class
            self.pooled_width = 2 * w3
            self.conv_block = nn.Sequential(
                nn.Conv2d(1, w0, 3, padding=1, bias=False), nn.BatchNorm2d(w0),
                nn.LeakyReLU(leaky_relu_slope, inplace=True), nn.Conv2d(w0, w0, 3, padding=1, bias=False))
            self.res_block1 = jdc.ResBlock(w0, w1)
            self.res_block2 = jdc.ResBlock(w1, w2)
            self.res_block3 = jdc.ResBlock(w2, w3)
            self.pool_block = nn.Sequential(nn.BatchNorm2d(w3), nn.LeakyReLU(leaky_relu_slope, inplace=True),
                                            nn.MaxPool2d(kernel_size=(1, 4)), nn.Dropout(p=0.2))
            self.bilstm_classifier = nn.LSTM(input_size=2 * w3, hidden_size=lstm, batch_first=True, bidirectional=True)
            self.classifier = nn.Linear(2 * lstm, num_class)
            self.apply(self.init_weights)

    # Also drop the forward's x.float(), a cast to single precision, so the network runs in float64 like the rest.
    source = textwrap.dedent(inspect.getsource(jdc.JDCNet.forward)).replace("512", "self.pooled_width").replace("x.float()", "x")
    namespace = {}
    exec(source, jdc.__dict__, namespace)
    TinyJDC.forward = namespace["forward"]

    # The aligner's decoder masks a tenth of its input tokens at random in every forward; turn that off.
    original = models.ASRCNN.__mro__[0]
    s2s = sys.modules["Utils.ASR.models"].ASRS2S
    initialize = s2s.initialize_decoder_states

    def no_masking(self, memory, mask):
        initialize(self, memory, mask)
        self.random_mask = 0.0

    s2s.initialize_decoder_states = no_masking
    return models, ASRCNN, TinyJDC


def build(models, ASRCNN, TinyJDC, state=None):
    torch.manual_seed(2294)
    nets = models.build_model(Munch(MODEL), ASRCNN(**ASR), TinyJDC(JDC_WIDTHS, JDC_LSTM))
    module = nn.ModuleDict(dict(nets))
    if state is None:
        with torch.no_grad():
            # Batch norm statistics and spectral-norm vectors away from their defaults, so loading them is tested.
            g = torch.Generator().manual_seed(5)
            for m in module.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.running_mean.copy_(torch.randn(m.running_mean.shape, generator=g) * 0.1)
                    m.running_var.copy_(torch.rand(m.running_var.shape, generator=g) + 0.5)
                    m.weight.copy_(torch.rand(m.weight.shape, generator=g) + 0.5)
                    m.bias.copy_(torch.randn(m.bias.shape, generator=g) * 0.1)
                if isinstance(m, (nn.InstanceNorm1d,)) and m.affine:
                    m.weight.copy_(torch.rand(m.weight.shape, generator=g) + 0.5)
                    m.bias.copy_(torch.randn(m.bias.shape, generator=g) * 0.1)
            # Durations of a few frames each, so inference's rounding and alignment are exercised.
            projection = module["predictor"].duration_proj.linear_layer
            projection.weight.mul_(2.0)
            projection.bias.fill_(2.2)
            # A few power iterations, so u and v are the vectors a trained model stores.
            module.train()
            for _ in range(3):
                module["style_encoder"](torch.randn(1, 1, 80, REFERENCE_FRAMES, generator=g))
                module["discriminator"](torch.randn(1, 1, 80, REFERENCE_FRAMES, generator=g))
    else:
        module.load_state_dict(state, strict=True)
    return module.eval()


def inputs():
    g = torch.Generator().manual_seed(7)
    # float32 whatever the default dtype, so the stored inputs are the ones the outputs were computed from.
    mel = torch.randn(1, 80, FRAMES, generator=g, dtype=torch.float32) * 0.6 - 1.5
    reference = torch.randn(1, 80, REFERENCE_FRAMES, generator=g, dtype=torch.float32) * 0.6 - 1.5
    scores = torch.rand(1, 10, 30, generator=g, dtype=torch.float32)
    return mel, reference, scores


def flat(t):
    return [float(v) for v in t.detach().reshape(-1).tolist()]


def run(module):
    # The reference allocates some buffers with torch.zeros (the text encoder's padding), which follow the default dtype.
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        return run_double(module)
    finally:
        torch.set_default_dtype(previous)


def run_double(module):
    from monotonic_align.core import maximum_path_c
    mel, reference, scores = inputs()
    tokens = torch.tensor([TOKENS])
    lengths = torch.tensor([len(TOKENS)])
    mask = torch.zeros(1, len(TOKENS), dtype=torch.bool)
    out = {}
    with torch.no_grad():
        t_en = module["text_encoder"](tokens, lengths, mask)
        out["text"] = flat(t_en)
        s = module["style_encoder"](reference.unsqueeze(1).double())
        out["style"] = flat(s)
        d = module["predictor"].text_encoder(t_en, s, lengths, mask)
        out["duration_encoder"] = flat(d)
        x, _ = module["predictor"].lstm(d)
        duration = module["predictor"].duration_proj(x)
        out["durations"] = flat(duration)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)
        out["rounded_durations"] = [int(v) for v in pred_dur.tolist()]
        aln = torch.zeros(len(TOKENS), int(pred_dur.sum().item()), dtype=torch.float64)
        frame = 0
        for i in range(aln.size(0)):
            aln[i, frame:frame + int(pred_dur[i].item())] = 1
            frame += int(pred_dur[i].item())
        en = d.transpose(-1, -2) @ aln.unsqueeze(0)
        f0, n = module["predictor"].F0Ntrain(en, s)
        out["f0"] = flat(f0)
        out["energy"] = flat(n)
        synthesized = module["decoder"](t_en @ aln.unsqueeze(0), f0, n, s)
        out["mel"] = flat(synthesized)

        ctc, s2s_logits, s2s_attn = module["text_aligner"](mel.double(), torch.zeros(1, FRAMES // 2, dtype=torch.bool), tokens)
        out["aligner_logits"] = flat(s2s_logits)
        out["aligner_energies"] = flat(s2s_attn)
        out["aligner_ctc"] = flat(ctc)
        f0_real, _, _ = module["pitch_extractor"](mel.unsqueeze(1).double())
        out["pitch"] = flat(f0_real)

        logit, features = module["discriminator"](reference.unsqueeze(1).double())
        out["discriminator_logit"] = flat(logit)
        # Each feature map's first 64 values, sum and sum of squares (the maps are large).
        out["discriminator_features"] = [flat(f)[:64] + [float(f.sum()), float((f * f).sum())] for f in features]

    path = np.zeros((1, 10, 30), dtype=np.int32)
    maximum_path_c(path, np.ascontiguousarray(scores.numpy().astype(np.float32)), np.array([10], dtype=np.int32),
                   np.array([30], dtype=np.int32))
    out["mas_path"] = path[0].tolist()
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--styletts", required=True, help="a checkout of yl4579/StyleTTS")
    parser.add_argument("--write", action="store_true", help="regenerate the fixture (default: verify it)")
    args = parser.parse_args()
    models, ASRCNN, TinyJDC = load(args.styletts)
    mel, reference, scores = inputs()
    if args.write:
        state = {k: v.float().clone().contiguous() for k, v in build(models, ASRCNN, TinyJDC).state_dict().items()}
        module = build(models, ASRCNN, TinyJDC, state).double()
        data = {"model": MODEL, "asr": ASR, "jdc_widths": JDC_WIDTHS, "jdc_lstm": JDC_LSTM, "tokens": TOKENS,
                "input_mel": flat(mel[0].float()), "reference": flat(reference[0].float()), "frames": FRAMES,
                "reference_frames": REFERENCE_FRAMES, "scores": scores[0].tolist(),
                "safetensors_base64": base64.b64encode(save_safetensors(state)).decode("ascii"), **run(module)}
        _fixture.save(FIXTURE, data, __file__)
        print("wrote " + FIXTURE)
        return
    data = _fixture.load(FIXTURE)
    again = run(build(models, ASRCNN, TinyJDC, load_safetensors(base64.b64decode(data["safetensors_base64"]))).double())
    check = _fixture.Comparison("StyleTTS")
    for key in ["text", "style", "duration_encoder", "durations", "f0", "energy", "mel", "aligner_logits", "aligner_energies",
                "aligner_ctc", "pitch", "discriminator_logit"]:
        check.floats(key, data[key], again[key], 1e-12, float)
    for i, (a, b) in enumerate(zip(data["discriminator_features"], again["discriminator_features"])):
        check.floats(f"discriminator feature {i}", a, b, 1e-12, float)
    check.exact("rounded durations", data["rounded_durations"], again["rounded_durations"])
    check.exact("mas path", data["mas_path"], again["mas_path"])
    check.report(exact=False)


if __name__ == "__main__":
    main()
