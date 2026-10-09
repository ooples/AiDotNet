# Reference data generators

The C# tests under `tests/AiDotNet.Tests` compare AiDotNet's implementations against fixed values from the
original implementations: the official codec code, librosa, PyWorld, NATSpeech, Hugging Face T5, pyannote.audio,
espeak-ng and Pheme's own modules. This folder holds the scripts
that produce those values, so you can check them yourself or regenerate them.

| Script | Fixture it produces | Reference implementation |
|---|---|---|
| `dac_reference.py` | `Audio/Codecs/ReferenceData/dac_official_layout_reference.json` | Hugging Face `transformers` `DacModel` |
| `encodec_reference.py` | `Audio/Codecs/ReferenceData/encodec_official_layout_reference.json` | Hugging Face `transformers` `EncodecModel` |
| `speechtokenizer_reference.py` | `Audio/Codecs/ReferenceData/speechtokenizer_official_layout_reference.json` | the authors' `speechtokenizer` package |
| `pitch_reference.py` | `Audio/Pitch/ReferenceData/pyworld_dio_stonemask.json` and `natspeech_pitch_cwt.json` | PyWorld, and NATSpeech's `cwt.py` (vendored unchanged in `natspeech_cwt.py`) |
| `tacotron_mel_reference.py` | `TextToSpeech/ReferenceData/librosa_tacotron_mel.json` | librosa |
| `t5_reference.py` | `TextToSpeech/ReferenceData/t5_reference.json` | Hugging Face `transformers` `T5ForConditionalGeneration` (Pheme's text-to-semantic model) |
| `conformer_reference.py` | `TextToSpeech/ReferenceData/soundstorm_conformer_reference.json` | Pheme's `modules/conformer.py` (needs `--pheme`, a checkout of PolyAI-LDN/pheme) |
| `pheme_s2a_reference.py` | `TextToSpeech/ReferenceData/pheme_s2a_reference.json` | Pheme's `modules/s2a_model.py` `TTSConformer` (needs `--pheme`) |
| `valle_reference.py` | `TextToSpeech/ReferenceData/valle_reference.json` | lifeiteng/vall-e's `VALLE` (needs `--valle`, a checkout of lifeiteng/vall-e) |
| `xvector_reference.py` | `TextToSpeech/ReferenceData/pyannote_xvector_reference.json` | pyannote.audio `XVectorSincNet` |
| `espeak_g2p_reference.py` | `TextToSpeech/ReferenceData/espeak_arctic_phonemes.json` | espeak-ng through `phonemizer` (the front end Pheme was trained with) |
| `g2p_resources.py` | `src/TextToSpeech/FrontEnd/Resources/cmudict.tsv.gz` and `nrl_rules.tsv` (the English G2P's data, not a test fixture) | CMUdict and NRL Report 7948's rules, from pinned sources |

## Set up

You need a 64-bit Python 3.12 or newer (librosa 1.0 requires it, and PyTorch has no 32-bit builds).

```
python -m venv .venv
.venv\Scripts\python -m pip install -r tools/reference-data/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install --no-deps pycwt==0.4.0b0
```

pycwt gets its own step because it declares `numpy<2`, while librosa 1.0 needs numpy 2.1 or newer, so pip refuses to
install the two together. pycwt runs correctly on numpy 2 (the checks below pass with it), and `requirements.txt`
already lists what it needs.

On macOS or Linux the interpreter is `.venv/bin/python`.

## Check the committed fixtures

Run a script with no arguments, from anywhere:

```
.venv\Scripts\python tools/reference-data/dac_reference.py
```

It recomputes every value in its fixture and prints either `matches the reference implementation` or `MISMATCH`
with the values that differ. A mismatch exits with code 1.

- **The codec fixtures must match exactly.** Each one stores its own small random model (the weights, base64-encoded)
  and its input audio. The script loads both into the official implementation and recomputes the latent, the codes
  and the decoded audio.
- **The pitch and spectrogram fixtures are checked to 1e-12, relative.** Their input is the synthetic voice in
  `test_signal.py`, which matches the C# tests' `Signal` helper value for value. PyWorld and the FFT behind librosa
  and pycwt are compiled code, and different builds disagree in the last few digits. That is far tighter than
  the C# tests themselves, which allow 1e-9 to 1e-6.

## Regenerate a fixture

Add `--write`:

```
.venv\Scripts\python tools/reference-data/pitch_reference.py --write
```

For the codec fixtures this keeps the stored weights and input audio and rewrites only the outputs, so the C# tests
keep testing the same model. Run the matching C# tests afterwards, and look at `git diff` before committing.
