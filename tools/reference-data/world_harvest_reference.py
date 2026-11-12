"""WORLD Harvest reference: pyworld 0.3.5's harvest (WORLD harvest.cpp, Morise 2017) and stonemask, the F0 extractor
CosyVoice's compute_f0 runs for HiFT's training targets.

Records in tests/AiDotNet.Tests/TextToSpeech/ReferenceData/world_harvest_reference.json, for a deterministic test
signal (a gliding harmonic tone with an unvoiced gap, a second voiced section at another pitch, a short voiced blip
and a little noise) at 22.05 kHz, 16 kHz and 8 kHz (decimation 3, 2 and 1), the F0 and times of harvest at frame
periods of 1 ms, 5 ms and CosyVoice's 256 / 22050 s, and stonemask's refinement of each. Default: verify; --write:
regenerate.
"""
import numpy as np
import pyworld

import _fixture

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/world_harvest_reference.json"
RATES = [22050, 16000, 8000]


def signal(fs):
    t = np.arange(int(0.9 * fs)) / fs
    rng = np.random.default_rng(2294)
    x = np.zeros_like(t)
    # 0.05-0.35 s: a tone gliding 140 -> 220 Hz with four harmonics.
    a = (t >= 0.05) & (t < 0.35)
    phase = 2 * np.pi * np.cumsum(np.where(a, 140 + 80 * (t - 0.05) / 0.3, 0)) / fs
    for k, amp in enumerate([0.5, 0.25, 0.12, 0.06], start=1):
        x += np.where(a, amp * np.sin(k * phase), 0)
    # 0.45-0.8 s: 310 Hz with a slow vibrato.
    b = (t >= 0.45) & (t < 0.8)
    phase2 = 2 * np.pi * np.cumsum(np.where(b, 310 + 12 * np.sin(2 * np.pi * 5 * t), 0)) / fs
    x += np.where(b, 0.4 * np.sin(phase2) + 0.15 * np.sin(2 * phase2), 0)
    # 0.84-0.855 s: a blip too short to keep.
    c = (t >= 0.84) & (t < 0.855)
    x += np.where(c, 0.3 * np.sin(2 * np.pi * 180 * t), 0)
    return x + 0.003 * rng.standard_normal(len(t))


def short_sections(fs):
    # A long 300 Hz section, then voiced blips of 8-14 ms at 200-235 Hz 40 ms apart, most of which the fixing steps drop.
    # (ExtendSub's carried mean F0 is kept as WORLD has it; no signal tried distinguishes it from a reset mean.)
    t = np.arange(int(0.75 * fs)) / fs
    x = np.where((t >= 0.02) & (t < 0.22), 0.4 * np.sin(2 * np.pi * 300 * t) + 0.1 * np.sin(4 * np.pi * 300 * t), 0.0)
    start = 0.27
    for length_ms in range(8, 15):
        for pitch in (200, 215, 235):
            blip = (t >= start) & (t < start + length_ms / 1000)
            x = x + np.where(blip, 0.4 * np.sin(2 * np.pi * pitch * (t - start)), 0.0)
            start += length_ms / 1000 + 0.04
            if start > 0.72:
                break
    return x + 0.002 * np.random.default_rng(7).standard_normal(len(t))


def record():
    cases = []
    for fs in RATES:
        x = signal(fs)
        for period in [1.0, 5.0, 256 * 1000 / 22050]:
            f0, t = pyworld.harvest(x, fs, frame_period=period)
            refined = pyworld.stonemask(x, f0, t, fs)
            cases.append({"fs": fs, "frame_period": period, "f0": f0.tolist(), "t": t.tolist(), "stonemask": refined.tolist()})
    x = short_sections(16000)
    for period in [1.0, 5.0]:
        f0, t = pyworld.harvest(x, 16000, frame_period=period)
        cases.append({"fs": 16000, "signal": "short_sections", "frame_period": period, "f0": f0.tolist(), "t": t.tolist(),
                      "stonemask": pyworld.stonemask(x, f0, t, 16000).tolist()})
    signals = {str(fs): signal(fs).tolist() for fs in RATES}
    signals["short_sections"] = short_sections(16000).tolist()
    return {"signals": signals, "cases": cases, "pyworld": pyworld.__version__,
            "compute_f0": compute_f0()}


def compute_f0():
    # CosyVoice's truncate (zero-padded to 24576 samples: 96 mel frames) and compute_f0 at 22.05 kHz, hop 256.
    import torch
    import torch.nn.functional as F
    fs, hop, segment = 22050, 256, 24576
    x = np.concatenate([signal(fs), np.zeros(segment - len(signal(fs)))])
    period = hop * 1000 / fs
    f0, t = pyworld.harvest(x, fs, frame_period=period)
    if sum(f0 != 0) < 5:
        f0, t = pyworld.dio(x, fs, frame_period=period)
    f0 = pyworld.stonemask(x, f0, t, fs)
    frames = segment // hop
    return F.interpolate(torch.from_numpy(f0).view(1, 1, -1), size=frames, mode="linear").view(-1).tolist()


def main():
    args = _fixture.parse_args(__doc__)
    data = record()
    if args.write:
        _fixture.save(FIXTURE, data, __file__)
        print("wrote " + FIXTURE)
        return
    committed = _fixture.load(FIXTURE)
    check = _fixture.Comparison("WORLD Harvest")
    for i, (want, got) in enumerate(zip(committed["cases"], data["cases"])):
        for key in ("f0", "t", "stonemask"):
            check.floats(f"case {i} {key}", want[key], got[key], 0.0, float)
    check.floats("compute_f0", committed["compute_f0"], data["compute_f0"], 0.0, float)
    check.report()


if __name__ == "__main__":
    main()
