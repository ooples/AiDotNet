"""The synthetic voice the pitch and spectrogram tests share, written identically to the C# `Signal` helpers in
WorldPitchDetectorTests and TacotronSpectrogramTests: harmonics gliding from 120 to 220 Hz, an unvoiced gap between
55 % and 70 % of the clip, and a deterministic LCG noise floor."""
import math

import numpy as np


def signal(fs, seconds):
    n = int(fs * seconds)
    x = np.zeros(n)
    phase = 0.0
    state = 12345
    for i in range(n):
        t = i / fs
        f0 = 120.0 + 100.0 * t / seconds
        phase += 2 * math.pi * f0 / fs
        v = 0.0
        if not (0.55 * seconds <= t < 0.70 * seconds):
            for h in range(1, 6):
                v += math.sin(h * phase) / h
            v *= 0.3
        state = (1103515245 * state + 12345) % 2147483648
        v += 0.01 * ((state / 2147483648.0) - 0.5)
        x[i] = v
    return x
