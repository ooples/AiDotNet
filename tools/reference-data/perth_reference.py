"""Resemble's Perth implicit watermarker reference (resemble-perth 1.0.1, PerthImplicitWatermarker), Chatterbox's output
watermark.

Install the published distribution with ``pip install resemble-perth==1.0.1``. Its internal
``perth.__version__`` string is still 1.0.0; the distribution version identifies the reference.

apply_watermark: the signal resampled to 32 kHz (librosa's soxr_hq), torchaudio's Spectrogram (2048-point periodic Hann,
hop 320, centred, reflect-padded), magnitude in dB normalized over [20·log10(1e-9), +15 dB] and phase, the encoder (1×1,
five 7-wide and a 1×1 convolution with LeakyReLU) adding its residual to the lowest 128 bins on frames louder than 5 % of
the loudest, InverseSpectrogram, resampled back. get_watermark: the decoder's three time scales (linear stretches of
1.25 and 0.75) and their attention-weighted masked means, clipped to [0, 1]. Run in float64 (the release casts the
signal to float32 and builds a float32 window). Records, for a 24 kHz signal (and the same chirp at 32 kHz, which is not resampled), the watermarked signal and the detection of
the signal and of its watermarked form and the decoder's unclipped output on the signal, for a tiny random model (hidden 8) and, with --checkpoint, the released one.
Usage: python tools/reference-data/perth_reference.py [--checkpoint <perth_net_250000.pth.tar>] [--write]   (in .venv-chatterbox)
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fixture  # noqa: E402

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/perth_reference.json"


def patch_float64():
    import perth.perth_net.perth_net_implicit.audio_processor as ap
    real = ap.AudioProcessor.signal_to_magphase

    def signal_to_magphase(self, signal):
        if isinstance(signal, np.ndarray):
            signal = torch.from_numpy(signal.copy())
        spec = self.spectrogram(signal.double())
        return ap.cx_to_magphase(self.hp, spec)

    ap.AudioProcessor.signal_to_magphase = signal_to_magphase
    import perth.perth_net.perth_net_implicit.perth_watermarker as watermarker
    # _to_tensor casts to float32.
    watermarker._to_tensor = lambda x, device: (torch.from_numpy(x.copy()) if isinstance(x, np.ndarray) else x).to(dtype=torch.float64, device=device)
    return real


def build(hidden, checkpoint=None):
    from perth.perth_net.perth_net_implicit.config import PerthConfig
    from perth.perth_net.perth_net_implicit.model.perth_net import PerthNet
    import yaml
    hparams = yaml.safe_load(open(os.path.join(os.path.dirname(__import__("perth").__file__), "perth_net/pretrained/implicit/hparams.yaml")))
    hparams["hidden_size"] = hidden
    net = PerthNet(PerthConfig(**hparams))
    if checkpoint:
        net.load_state_dict(torch.load(checkpoint, map_location="cpu")["model"])
    else:
        torch.manual_seed(0)
        with torch.no_grad():
            for p in net.parameters():
                p.copy_(0.02 * torch.randn_like(p))
    net = net.double().eval()
    # torchaudio builds its windows in float32; rebuild them in float64.
    for module in (net.ap.spectrogram, net.ap.inv_spectrogram):
        module.window = torch.hann_window(2048, dtype=torch.float64)
    return net


def native_signal(signal):
    n = len(signal) * 4 // 3
    t = np.arange(n) / 32000.0
    return 0.3 * np.sin(2 * np.pi * 220 * t * (1 + 0.3 * t)) * np.minimum(1, 4 * t) + 0.05 * np.sin(2 * np.pi * 1700 * t)


def run(net, signal):
    from perth.perth_net.perth_net_implicit.perth_watermarker import PerthImplicitWatermarker
    wm = PerthImplicitWatermarker(run_name=None, perth_net=net)
    with torch.no_grad():
        watermarked = wm.apply_watermark(signal, sample_rate=24000)
        # At the model's own 32 kHz nothing is resampled.
        native = wm.apply_watermark(native_signal(signal), sample_rate=32000)
        detected_plain = wm.get_watermark(signal, sample_rate=24000, round=False)
        detected = wm.get_watermark(watermarked, sample_rate=24000, round=False)
        # The decoder's unclipped output on the 32 kHz signal's magnitude.
        from librosa import resample
        mag, _ = net.ap.signal_to_magphase(resample(signal, orig_sr=24000, target_sr=32000, res_type="polyphase"))
        raw = net.decoder(mag[None])[0]
    return {"raw_decoder": float(raw), "watermarked_32k": np.asarray(native).tolist(), "watermarked": np.asarray(watermarked).tolist(), "detected_plain": float(np.asarray(detected_plain).reshape(-1)[0]),
            "detected": float(np.asarray(detected).reshape(-1)[0])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    patch_float64()
    n = 24000
    t = np.arange(n) / 24000.0
    signal = 0.3 * np.sin(2 * np.pi * 220 * t * (1 + 0.3 * t)) * np.minimum(1, 4 * t) + 0.05 * np.sin(2 * np.pi * 1700 * t)
    tiny = build(8)
    data = {"signal": signal.tolist(), "tiny": run(tiny, signal),
            "tiny_state": {k: v.reshape(-1).tolist() for k, v in tiny.state_dict().items() if v.dtype.is_floating_point and "window" not in k}}
    if args.checkpoint:
        data["released"] = run(build(256, args.checkpoint), signal)
    if args.write:
        _fixture.save(FIXTURE, data, __file__)
        print("wrote", FIXTURE, data["tiny"]["detected_plain"], data["tiny"]["detected"], data.get("released", {}).get("detected"),
              data.get("released", {}).get("detected_plain"))
        return
    stored = _fixture.load(FIXTURE)
    check = _fixture.Comparison("Perth")
    for key in ("tiny", "released"):
        if key in data and key in stored:
            check.floats(key + " watermarked", stored[key]["watermarked"], data[key]["watermarked"], 1e-12, float)
            check.floats(key + " watermarked at 32 kHz", stored[key]["watermarked_32k"], data[key]["watermarked_32k"], 1e-12, float)
            check.floats(key + " detections", [stored[key]["raw_decoder"], stored[key]["detected_plain"], stored[key]["detected"]],
                         [data[key]["raw_decoder"], data[key]["detected_plain"], data[key]["detected"]], 1e-12, float)
    check.report(exact=False)


if __name__ == "__main__":
    main()
