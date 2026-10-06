"""Pitch references for FastSpeech 2 and the models built on it.

pyworld_dio_stonemask.json: PyWorld's dio and stonemask (FastSpeech 2's pitch extractor, Ren et al. 2021, App. C.2)
on the shared test signal at 22050 Hz with FastSpeech 2's 256-sample hop and at 16 kHz with WORLD's default 5 ms.

natspeech_pitch_cwt.json: NATSpeech's utils/audio/cwt.py (vendored verbatim in natspeech_cwt.py) on those stonemask
contours: continuous log-F0, the per-utterance mean and standard deviation, the Mexican-hat CWT of the normalized
log-F0 (as FastSpeech 2's binarizer computes it), its inverse and the F0 reconstruction.

Default: verify, to 1e-12 relative (pyworld and pycwt are compiled; builds differ in the last ulps). --write:
regenerate both files.
"""
import numpy as np
import pyworld

import _fixture
import natspeech_cwt
from test_signal import signal

PYWORLD = "tests/AiDotNet.Tests/Audio/Pitch/ReferenceData/pyworld_dio_stonemask.json"
NATSPEECH = "tests/AiDotNet.Tests/Audio/Pitch/ReferenceData/natspeech_pitch_cwt.json"
SETTINGS = {"22050": (1.2, 256 / 22050 * 1000.0), "16000": (1.0, 5.0)}
TOLERANCE = 1e-12


def world(fs, seconds, frame_period):
    x = signal(fs, seconds)
    f0, t = pyworld.dio(x, fs, frame_period=frame_period)
    refined = pyworld.stonemask(x, f0, t, fs)
    return {"seconds": seconds, "frame_period": frame_period, "dio": f0.tolist(), "stonemask": refined.tolist(),
            "t": t.tolist(), "x_head": x[:8].tolist(), "n": len(x)}


def wavelet(f0):
    f0 = np.asarray(f0, dtype=np.float64)
    uv, cont_lf0 = natspeech_cwt.get_cont_lf0(f0)
    mean, std = float(cont_lf0.mean()), float(cont_lf0.std())
    cwt, scales = natspeech_cwt.get_lf0_cwt((cont_lf0 - mean) / std)
    rec = natspeech_cwt.inverse_cwt(cwt[None], scales)[0]
    f0_rec = natspeech_cwt.cwt2f0(cwt[None], np.array([mean]), np.array([std]), scales)[0]
    return {"f0": f0.tolist(), "uv": uv.tolist(), "cont_lf0": cont_lf0.tolist(), "mean": mean, "std": std,
            "cwt": cwt.tolist(), "scales": scales.tolist(), "rec": rec.tolist(), "f0_rec": f0_rec.tolist()}


def main():
    args = _fixture.parse_args(__doc__)
    contours = {fs: world(int(fs), *SETTINGS[fs]) for fs in SETTINGS}
    if args.write:
        import json
        with open(_fixture.fixture_path(PYWORLD), "w", encoding="utf-8", newline="\n") as handle:
            json.dump(contours, handle)
        wavelets = {fs: wavelet(contours[fs]["stonemask"]) for fs in SETTINGS}
        with open(_fixture.fixture_path(NATSPEECH), "w", encoding="utf-8", newline="\n") as handle:
            json.dump(wavelets, handle)
        print("wrote " + PYWORLD + " and " + NATSPEECH)
        return
    committed_world = _fixture.load(PYWORLD)
    committed_cwt = _fixture.load(NATSPEECH)
    failed = False
    for fs in SETTINGS:
        check = _fixture.Comparison(f"PyWorld {fs} Hz")
        want, got = committed_world[fs], contours[fs]
        for key in ("seconds", "frame_period", "n"):
            check.exact(key, want[key], got[key])
        for key in ("x_head", "t"):
            check.floats(key, want[key], got[key], dtype=np.float64)
        for key in ("dio", "stonemask"):
            check.floats(key, want[key], got[key], TOLERANCE, np.float64)
        # The wavelet fixture is computed from the COMMITTED contour, so it is checked independently of the one above.
        cwt_check = _fixture.Comparison(f"NATSpeech CWT {fs} Hz")
        want, got = committed_cwt[fs], wavelet(committed_cwt[fs]["f0"])
        cwt_check.floats("f0 equals the committed stonemask", committed_world[fs]["stonemask"], want["f0"], dtype=np.float64)
        for key in ("uv", "cont_lf0", "mean", "std", "cwt", "scales", "rec", "f0_rec"):
            cwt_check.floats(key, want[key], got[key], TOLERANCE, np.float64)
        for c in (check, cwt_check):
            try:
                c.report(exact=False)
            except SystemExit:
                failed = True
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
