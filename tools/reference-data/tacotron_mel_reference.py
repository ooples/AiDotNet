"""Tacotron 2 mel spectrogram and FastSpeech 2 frame energy, from librosa.

tests/AiDotNet.Tests/TextToSpeech/ReferenceData/librosa_tacotron_mel.json: on the shared test signal at 22050 Hz,
|librosa.stft(n_fft=1024, hop_length=256, win_length=1024, window='hann', center=True, pad_mode='reflect')|, the mel
basis librosa.filters.mel(sr=22050, n_fft=1024, n_mels=80, fmin=0, fmax=8000) (rows 0, 5, 40 and 79 are stored),
log(max(mel, 1e-5)) for the first 40 frames (Tacotron 2's dynamic-range compression) and each frame's L2 norm of the
magnitude (FastSpeech 2's energy). Default: verify (the basis exactly, the FFT-based values to 1e-12
relative); --write: regenerate.
"""
import librosa
import numpy as np

import _fixture
from test_signal import signal

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/librosa_tacotron_mel.json"
ROWS = (0, 5, 40, 79)


def compute():
    x = signal(22050, 1.2)
    magnitude = np.abs(librosa.stft(x, n_fft=1024, hop_length=256, win_length=1024, window="hann", center=True,
                                    pad_mode="reflect"))                       # [bins, frames]
    basis = librosa.filters.mel(sr=22050, n_fft=1024, n_mels=80, fmin=0, fmax=8000)
    log_mel = np.log(np.maximum(basis @ magnitude, 1e-5))                      # [mels, frames]
    return {"librosa": librosa.__version__, "frames": int(magnitude.shape[1]),
            "energy": np.linalg.norm(magnitude, axis=0).tolist(),
            "mel_first_frames": log_mel[:, :40].T.tolist(),
            "mel_basis_rows_0_5_40_79": basis[list(ROWS)].tolist()}


def main():
    args = _fixture.parse_args(__doc__)
    data = compute()
    if args.write:
        import json
        with open(_fixture.fixture_path(FIXTURE), "w", encoding="utf-8", newline="\n") as handle:
            json.dump(data, handle)
        print("wrote " + FIXTURE)
        return
    want = _fixture.load(FIXTURE)
    check = _fixture.Comparison("librosa Tacotron 2 mel")
    check.exact("librosa", want["librosa"], data["librosa"])
    check.exact("frames", want["frames"], data["frames"])
    check.floats("mel_basis_rows_0_5_40_79", want["mel_basis_rows_0_5_40_79"], data["mel_basis_rows_0_5_40_79"],
                 dtype=np.float64)
    # The STFT goes through scipy's compiled FFT, whose builds differ in the last ulps.
    for key in ("energy", "mel_first_frames"):
        check.floats(key, want[key], data[key], 1e-12, np.float64)
    check.report(exact=False)


if __name__ == "__main__":
    main()
