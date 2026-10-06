# Text-to-speech: paper-fidelity decisions

Each text-to-speech model, vocoder and neural audio codec listed here implements one published model. This page
records, for each one, where its defaults and behaviour come from, and every place where the code had to choose.

## How the choices were made

1. **The paper wins where it is explicit.** If a paper states a value or an equation, the code uses it, even when
   the official code does something else. Each such case is listed under *Choices* below.
2. **The official implementation fills the gaps.** Papers leave out many details: padding, initialization, feature
   extraction, optimizer settings. Those come from the authors' released code. If the authors released no code, a
   named reproduction is used instead, and the entry says which.
3. **Nothing is invented.** If neither the paper nor any reference states a value, the option is left for you to
   set, and the entry says so.

The full detail, with section, table and equation numbers, is in each model's XML documentation: the `<remarks>` of
the model class and of its options class. This page summarizes it. When the two disagree, the code documentation is
the one to trust, and this page should be corrected.

## Training data a model needs

Most models train from text and a target spectrogram or waveform. These need more, because their paper takes it from
outside the model. Calling plain `Train(tokens, mel)` on them throws; pass a `TtsTrainingSample` instead.

| Model | What it needs, and why |
|---|---|
| FastSpeech, FastSpeech 2, ProDiff | Phoneme durations. The papers take them from a teacher model or a forced aligner (MFA). |
| SpeedySpeech | Phoneme durations, from its teacher network's attention (§3.1). |
| ForwardTacotron | Phoneme durations, from a Tacotron's attention. |
| Non-Attentive Tacotron | Phoneme durations (this is the supervised-duration variant). |
| PortaSpeech | Phoneme durations, summed per word. |
| VoiceFlow | Phoneme durations from a forced alignment (§3.1, §4.1). |
| AdaSpeech, AdaSpeech 2 | Phoneme durations and the speaker's ID. |
| Tacotron, Deep Voice 3 | The recording, or its linear spectrogram, which a mel spectrogram does not contain. |
| SpeechTokenizer | The self-supervised teacher's features for each clip (the HuBERT distillation target). |

Pitch, energy and the mel target are derived from the recording when you don't supply them, the way each paper
derives them.

## Acoustic models

| Model | Paper | Gaps filled from | Choices |
|---|---|---|---|
| FastSpeech | Ren et al. 2019 | — | LJSpeech configuration, §4.2 and Table 5. |
| FastSpeech 2 | Ren et al. 2021 | — | LJSpeech configuration, App. A and Table 7. Pitch is PyWorld DIO + StoneMask, as a CWT pitch spectrogram (App. C.2). |
| AlignTTS | Zeng et al. 2020 | — | Configuration of §4.2. Learns its own alignment with a mix density network. |
| AdaSpeech | Chen et al. 2021 | FastSpeech 2, as the paper says ("other model configurations follow Ren et al. (2021)") | Training phases follow §3: pre-training, joint training, then adaptation of only the speaker embedding and the conditional layer norms. |
| AdaSpeech 2 | Yan et al. 2021 | AdaSpeech | The paper does not say how the mel encoder reads 80-bin frames into 256-wide blocks; a linear projection plus the phoneme encoder's positions does it. Untranscribed speech has no phoneme alignment, so phoneme-level vectors are left out of reconstruction. |
| SpeedySpeech | Vainer & Dušek 2020 | janvainer/speedyspeech | The paper counts single convolutions ("26 encoder blocks, dilations 1, 1, 2, 2, 4, 4"); the reference builds 13 residual blocks of two. Same network. |
| Tacotron | Wang et al. 2017 | keithito/tacotron | Table 1 architecture, Griffin–Lim with 50 iterations on magnitudes raised to 1.2. |
| Tacotron 2 | Shen et al. 2018 | NVIDIA/tacotron2 and Rayhane-mamah/Tacotron-2 | Adam (0.9, 0.999, ε = 1e-6) at 1e-3, decaying after 50k steps (§3.1). The paper gives no decay rate, so it halves every 50k steps down to 1e-5, Rayhane-mamah's reading. Gradient norms are clipped at 1 (NVIDIA's `grad_clip_thresh`). Pre-net dropout stays on at inference (§2.3), and zoneout uses its expectation at inference (Krueger et al. 2017). |
| Transformer TTS | Li et al. 2019 | Vaswani et al. 2017 and Shen et al. 2018, which the paper builds on | Where the paper defers to those works, their values are used: d_model 512, d_ff 2048, Tacotron 2's pre-net and post-net. |
| Glow-TTS | Kim et al. 2020 | jaywalnut310/glow-tts | σ is fixed at 1, as in the reference's released LJSpeech configuration (`mean_only`). Sampling temperature 0.333 (§5.1). |
| Grad-TTS | Popov et al. 2021 | huawei-noah/Speech-Backbones | β_t = 0.05 + 19.95 t; inference τ = 1.5 and 10 reverse steps. |
| Deep Voice 3 | Ping et al. 2018 | r9y9/deepvoice3_pytorch | Single-speaker column of Table 4. Decoding limits (at most 200 steps, at least 10) come from the reference. |
| ForwardTacotron | none: defined by its code | as-ideas/ForwardTacotron | No paper exists, so the reference's single-speaker configuration is the definition. Not to be confused with Non-Attentive Tacotron. |
| Non-Attentive Tacotron | Shen et al. 2020 | — | Table 6 (App. A), supervised-duration variant, reduction factor 2 (§4.1). |
| PortaSpeech | Ren et al. 2021 | NATSpeech | PortaSpeech (normal), Table 6. The KL term is counted only after 10 000 updates, as in the reference. |
| ProDiff | Huang et al. 2022 | Rongjiehuang/ProDiff | A 4-step teacher distilled into a 2-step student (Algorithm 1); variance losses weighted 0.1. |

## Flow and diffusion models

| Model | Paper | Gaps filled from | Choices |
|---|---|---|---|
| Matcha-TTS | Mehta et al. 2024 | shivammehta25/Matcha-TTS | Inference uses 10 Euler steps at temperature 0.667, the reference's defaults; the paper reports 2, 4 and 10 steps. |
| CoMoSpeech | Ye et al. 2023 | zhenye234/CoMoSpeech | EDM preconditioning (Eq. 8–9), a 50-step teacher, consistency distillation with an EMA target at μ = 0.95. |
| E2 TTS | Eskimez et al. 2024 | F5-TTS's reproduction (SWivid/F5-TTS, `E2TTS_Base.yaml`) | The paper has no public code, so its gaps follow the reproduction its successor published. |
| F5-TTS | Chen et al. 2024 | SWivid/F5-TTS | F5-TTS Base (§4, App. B); inference with 32 function evaluations, sway −1, CFG strength 2. |
| VoiceFlow | Guo et al. 2024 | cantabile-kwok/VoiceFlow-TTS | Rectified flow (Algorithm 1) with ground-truth durations; σ = 0.1. |

## End-to-end models

| Model | Paper | Gaps filled from | Choices |
|---|---|---|---|
| VITS | Kim et al. 2021 | jaywalnut310/vits | Generator loss is mel L1 × 45 + KL + duration + adversarial + feature matching (Eq. 9). Synthesis uses duration noise 0.8 and prior noise 0.667. |
| VITS2 | Kong et al. 2023 | p0p4k/vits2_pytorch (community) | The paper has no official code. The duration predictor trains on its own as the last step ("separately trained as the last training step", §2.1). Alignment noise starts at 0.01 and falls by 2·10⁻⁶ per step (§2.2). |
| Piper | none: defined by its code | rhasspy/piper (`piper_train`) | VITS unchanged, at Piper's model sizes and with piper-phonemize's phoneme framing (`^ _ p₁ _ … pₙ _ $`). |
| YourTTS | Casanova et al. 2022 | Coqui TTS (`recipes/vctk/yourtts`) | HiFi-GAN V1 sizes with type-2 residual blocks: the recipe notes the paper's models were trained that way. The speaker-consistency gradient flows through the speaker encoder into the generated audio, per the paper's erratum. |

## Vocoders

| Model | Paper | Gaps filled from | Choices |
|---|---|---|---|
| HiFi-GAN | Kong et al. 2020 | jik876/hifi-gan | V1 (Table 5): λ_fm = 2, λ_mel = 45; the input mel stops at 8 kHz, the loss mel spans the full band. |
| BigVGAN | Lee et al. 2023 | NVIDIA/BigVGAN | The 112M model (Table 6). Snake activations made anti-aliased with the reference's 12-tap Kaiser-windowed sinc filters. Gradient norms clipped at 1000, separately per network. |
| UnivNet | Jang et al. 2021 | maum-ai/univnet | UnivNet-c32; 200k generator-only steps before the adversarial phase. |
| Vocos | Siuzdak 2024 | gemelo-ai/vocos | λ_mel = 45 and the multi-resolution discriminator weighted 0.1, from the reference (the paper gives no weights). The ISTFT uses the reference's centred padding, so T frames give (T − 1) × hop samples. |
| iSTFTNet | Kaneko et al. 2022 | HiFi-GAN's configuration, as the paper says | V1-C8C8I. The reference pads one frame before the output convolution so the waveform has exactly frames × 256 samples. |
| MelGAN | Kumar et al. 2019 | descriptinc/melgan-neurips | Hinge loss and feature matching (λ = 10) only; no audio or spectrogram loss, as §2.3 states. |
| Multi-band MelGAN | Yang et al. 2021 | kan-bayashi/ParallelWaveGAN | λ_adv = 2.5, PQMF cutoff 0.142 and Kaiser β 9 from the reference; 200k generator-only steps. |
| Parallel WaveGAN | Yamamoto et al. 2020 | kan-bayashi/ParallelWaveGAN | λ_adv = 4; discriminator fixed for the first 100k steps; RAdam at 1e-4 and 5e-5. |
| APNet | Ai & Ling 2023 | yangai520/APNet | Negative-cosine anti-wrapping phase losses. The ISTFT is centred, so T frames give (T − 1) × hop samples. |
| APNet2 | Du et al. 2023 | redmist328/APNet2 | **The paper wins:** the hinge GAN terms are averaged equally over both discriminators' sub-discriminators (Eq. 2–3). The reference sums them and weights the multi-resolution ones by 0.1. |
| DiffWave | Kong et al. 2021 | lmnt-com/diffwave | **The paper wins:** the noise loss is L2 (Algorithm 1); the reference code uses L1. BASE with T = 50, plus the 6-step fast schedule (App. B). |
| WaveGrad | Chen et al. 2021 | lmnt-com/wavegrad | Conditioned on the continuous noise level, so one model samples with any schedule. **Left for you to set:** the paper's 6-step schedule came from a grid search and its values were never published, so `InferenceNoiseSchedule` defaults to the training schedule. |
| PriorGrad | Lee et al. 2022 | microsoft/NeuralSpeech | DiffWave with an energy-based diagonal Gaussian prior, clipped below at 0.1 (§4). |
| FreGrad | Nguyen et al. 2024 | kaistmm/fregrad | **Not reproduced:** the reference filters the priors and sampled sub-bands with a biquad; the paper never describes it, so it is not applied. |
| WaveGlow | Prenger et al. 2019 | NVIDIA/waveglow | σ = √0.5 in training, 0.6 at inference. The paper lowers the learning rate by hand when training plateaus; that is left to you. |
| WaveNet | van den Oord et al. 2016 | r9y9/wavenet_vocoder (μ-law preset) | The paper fixes the model's form; the reference supplies the sizes (30 layers in 3 cycles). |
| WaveRNN | Kalchbrenner et al. 2018 | fatchord/WaveRNN (optimizer only) | WaveRNN-896. The paper conditions on linguistic features without saying how; the mel spectrogram is upsampled WaveNet-style and projected into the gates. Weight pruning (§3) is off unless you set `SparsityTarget`. |

## Neural audio codecs

| Model | Paper | Gaps filled from | Choices |
|---|---|---|---|
| EnCodec | Défossez et al. 2022 | facebookresearch/encodec and audiocraft | Defaults are the paper's 24 kHz streamable model. `OfficialCheckpoint24kHz` and `OfficialCheckpoint48kHz` match the released checkpoints, which differ from the paper's text in places. |
| SoundStream | Zeghidour et al. 2021 | EnCodec's reimplementation (Défossez et al. 2022, App. A.2) | No official code exists. Quantizer dropout draws n_q ~ U[1, N_q] per example; FiLM denoising (§III-F). |
| DAC | Kumar et al. 2023 | descriptinc/descript-audio-codec | Factorized 8-dimensional, L2-normalized codes chosen by cosine similarity; loss weights 15 / 2 / 1 / 1 / 0.25. |
| SpeechTokenizer | Zhang et al. 2024 | ZhangXInFD/SpeechTokenizer | The paper's loss forms with the reference's λ values. Training needs the teacher's features (see above). |

Official weights load into the native EnCodec, DAC and SpeechTokenizer models. The tests check them against tiny
reference models; the scripts that produce those references are in `tools/reference-data`.
