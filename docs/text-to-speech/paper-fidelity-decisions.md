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
| Pheme | The recording's SpeechTokenizer codes (or the recording, which it encodes), and for the speaker embedding a recording of the speaker. Synthesis needs a voice: a prompt recording and, optionally, its transcript (`CreateVoice`). |
| VALL-E | The recording's EnCodec codes (or the 24 kHz recording, which it encodes). The paper crops each utterance to a random 10–20 seconds with its aligned phonemes; that needs the alignment, so it is left to you. Synthesis needs a voice: a prompt recording and, optionally, its transcript (`CreateVoice`). |
| VALL-E X | As VALL-E, plus the utterance's language (`LanguageId`) and the codes of another sentence by the same speaker (`PromptCodecTokens`), which the NAR model reads as its prompt (Eq. 2). |
| SoundStorm | The utterance's semantic tokens at the codec's frame rate (w2v-BERT k-means tokens in the paper) followed by its SoundStream codes, or the semantic tokens and the recording, which it encodes. Synthesis with a voice needs the prompt recording and its 25 Hz semantic tokens. |

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
| DiffWave | Kong et al. 2021 | lmnt-com/diffwave | **The paper wins:** the noise loss is L2 (Algorithm 1); the reference code uses L1. BASE with T = 50, plus the 6-step fast schedule (App. B). The paper's other two tasks are built in: `DiffWaveOptions.Unconditional()` (§5.2: 36 layers, C = 256, dilations to 2048, T = 200, β to 0.02, L = 16,000) and `ClassConditional(n)` (§5.3: a shared 128-wide label embedding, projected per layer and added after the dilated convolution). These replace the deleted `Diffusion.Audio.DiffWaveModel`, which paired T = 200 with β up to 0.05 and sampled with DDIM. The class count has no default because it is the dataset's. |
| WaveGrad | Chen et al. 2021 | lmnt-com/wavegrad | Conditioned on the continuous noise level, so one model samples with any schedule. **Left for you to set:** the paper's 6-step schedule came from a grid search and its values were never published, so `InferenceNoiseSchedule` defaults to the training schedule. |
| PriorGrad | Lee et al. 2022 | microsoft/NeuralSpeech | DiffWave with an energy-based diagonal Gaussian prior, clipped below at 0.1 (§4). |
| FreGrad | Nguyen et al. 2024 | kaistmm/fregrad | **Not reproduced:** the reference filters the priors and sampled sub-bands with a biquad; the paper never describes it, so it is not applied. |
| WaveGlow | Prenger et al. 2019 | NVIDIA/waveglow | σ = √0.5 in training, 0.6 at inference. The paper lowers the learning rate by hand when training plateaus; that is left to you. |
| WaveNet | van den Oord et al. 2016 | r9y9/wavenet_vocoder (μ-law preset) | The paper fixes the model's form; the reference supplies the sizes (30 layers in 3 cycles). |
| WaveRNN | Kalchbrenner et al. 2018 | fatchord/WaveRNN (optimizer only) | WaveRNN-896. The paper conditions on linguistic features without saying how; the mel spectrogram is upsampled WaveNet-style and projected into the gates. Weight pruning (§3) is off unless you set `SparsityTarget`. |

## Codec language models

| Model | Paper | Gaps filled from | Choices |
|---|---|---|---|
| Pheme | Budzianowski et al. 2024 | PolyAI-LDN/pheme | Defaults are the paper's small model (Table 5) and its optimizer for both networks (§4.1: AdamW, β = 0.9 / 0.98, 5e-4, 10k warm-up, linear decay over 800k steps; the paper's "decayed from 2×10⁻⁴" contradicts its stated rate, so 5e-4 is the peak). The released checkpoints differ from the paper's text: `OfficialSmallCheckpoint` has a 768-wide acoustic model (the paper says 1024), and `OfficialLargeCheckpoint` has no speaker embedding. Text becomes espeak-style phonemes through the built-in English G2P (`EnglishG2P`: CMUdict, then NRL letter-to-sound rules, mapped to espeak-ng's symbols; 3 % phone error rate against espeak-ng on CMU ARCTIC), and the vocabulary is the reference's. The paper and the code decode the acoustic codebooks in opposite orders. The paper says greedy on the first level and 16 MaskGIT steps on the rest. The default follows the code (MaskGIT on the first level, greedy on the rest), which produced the released outputs. `AcousticDecoding` selects the paper's order. Two reference behaviours are kept: the speaker-embedding dropout stays on at inference, and the last prompt frame is regenerated and returned with the new frames. One is not: the reference's MaskGIT loop feeds the model its initial, fully masked input at every step, so here each step reads the tokens decoded so far, as MaskGIT and SoundStorm do. |
| VALL-E | Wang et al. 2023 | lifeiteng/vall-e (no official code) | The paper's sizes, codec (EnCodec 6 kbps) and optimizer (AdamW, 32k warm-up to 5e-4, linear decay over 800k steps); it names no betas or weight decay, so they are PyTorch's AdamW defaults. The paper's phonemes come from a Kaldi ASR model; this model reads espeak-style phonemes through `EnglishG2P` and the reference's LibriTTS table, as the reference does. The NAR's training prompt is the paper's random 3-second segment of the same utterance (the reference's `prefix_mode 2`, which shortens it to a quarter of an utterance under 12 seconds). The loss is the reference's summed cross-entropy. The reference's sampling (temperature 1, no top-k) and its stop at 16 codes per phoneme token fill what §4.3 leaves open. The reference stops with an error when the AR model ends before its first code; so does this model. A voice without a transcript is an empty enrolled transcript. |
| VALL-E X | Zhang et al. 2023 | VALL-E (which it extends) and Plachtaa/VALL-E-X (no official code) | The paper's sizes and schedule (5e-4 after 8k warm-up steps, 800k steps). It names no optimizer, so it is VALL-E's AdamW. The language embedding is added to the AR model's acoustic-token embeddings, as §3.3 says; `LanguagePlacement` selects the reference's placement (the phoneme embeddings of both models). The NAR model trains on another sentence of the same speaker, as Eq. 2 says, and reads only the text at synthesis (Eq. 4). The paper's phoneme lexicon (BigCiDian) is not public. English goes through `EnglishG2P`, and Mandarin through `MandarinG2P`, which reproduces the reference's front end exactly (jieba, pypinyin and cn2an, all MIT), in one table. Text is split by script, so mixed sentences work. A token's language is taken from its symbols (the nearest symbol only one language uses), because Predict receives only token ids. |
| VALL-E 2 | Chen et al. 2024 | VALL-E (no code or reproduction exists) | Grouped code modeling (§3.1, Eq. 7–8) and repetition-aware sampling (Algorithm 1, K = 10, t_r = 0.1) on VALL-E's Transformer layers. The AR model uses learned positions, an `<eos> <bos>` separator and a fully causal mask (Figure 2). The NAR model uses a code-ID token (Eq. 14). Decoding is by Vocos's EnCodec model (§4.1.1; `VocosEncodecDecoder` matches the vocos package). Searched or unstated values: group size 2 (the paper trains 1, 2, 4, 8); top-p 0 (it searches 0–0.8); the NAR condition at most half the utterance, otherwise a random 3–30 s; the peak rate and step count are VALL-E's (5e-4, 800k); `<eos>`/`<bos>` are text-embedding rows. The text is espeak-style phonemes, as VALL-E's reproduction reads them; the paper's BPE model is not public. Algorithm 1's repetition ratio is read as written, its k = 0 term included, so any repeat in the window switches to random sampling. |
| SoundStorm | Borsos et al. 2023 | lucidrains/soundstorm-pytorch (no official code); Pheme's SoundStorm (PolyAI-LDN/pheme) for initialization | The §4.1 acoustic generator (12-layer 1024-wide Conformer, Q = 12 levels of 1024 codes at 50 Hz, semantic tokens duplicated from 25 Hz) and §5's ByT5-large text-to-semantic model, whose decoder alone trains. Masking follows §3.2 as written: t ~ U{0, T − 1}, level q masked after t with p = cos u, finer levels masked after t only, loss on level q's masked tokens. Decoding is (16, 1, …, 1) with the last iteration of each level greedy (§3.3). **From the reproduction:** a learned mask embedding, no dropout, convolution expansion 2, Adam (0.9, 0.99, ε = 1e-8, no weight decay) at 3e-4 after a linear warm-up from 1e-5, cosine annealing, clipping at 0.5. **Left for you to set:** the warm-up and total steps (the reproduction requires both and defaults neither), and the 24 kHz codec's strides for §5's dialogue setup (the paper gives only its 50 Hz frame rate). The text-to-semantic decoder reuses the acoustic recipe: neither the paper nor a reproduction gives one. |
| w2v-BERT (semantic tokens) | Chung et al. 2021; AudioLM (Borsos et al. 2022) for the tokens | wav2vec 2.0 (quantizer and contrastive task, which §3.2 adopts) and Conformer's convolutional subsampling as ESPnet reproduces it; Google released no code | XL (Table 1): 12 + 12 Conformer blocks, width 1024, 8 heads, kernel 5, no relative attention; one 1024-entry codebook; mask p = 0.065 over 10 steps with random vectors; L_c + L_m; Adam at 2e-3 with the Transformer schedule (25k warm-up). Semantic tokens are the 7th masked-prediction block, standardized, k-means into 1024. **Decided here:** 3×3 subsampling kernels with model-width channels, N(0, 1) mask vectors, the masked-prediction loss summed over masked steps, and a 1e-6 log-mel floor. Meta's w2v-BERT 2.0 is a different network, so its checkpoint does not load. |
| SPEAR-TTS | Kharitonov et al. 2023 | lucidrains/spear-tts-pytorch for S1's procedure; AudioLM's coarse stage (`AudioLmCore`) for S2's sequence (no official code) | Tokens are §7.1's: w2v-BERT layer-7 semantic tokens (512 clusters, 25 Hz, repetitions removed) and the first 3 levels of a 16 kHz SoundStream at 50 Hz, interleaved per frame. S1 is T5-Large as T5 v1.0 defines it (ReLU feed-forward, decoder embedding tied to its output) reading espeak-style phonemes through `EnglishG2P` (§6.3 reads phonemes). Finetuning updates "the encoder embeddings" with "the entire decoder" frozen (§7.2), which is only possible if the encoder reads its own table, so it does, as the reproduction's separate text and speech tables do. A stack's relative position bias table is its first block's, as T5 stores it, so it trains with the lowest layer. Pretraining deletes each token independently with p = 0.6 and reconstructs the whole sequence, as §4.1 says. The reproduction deletes an exact fraction and by default predicts only the deleted tokens. Backtranslation finetunes a copy of the pretrained model (`StartBacktranslationFromPretrained`) the other way. Its decoder's embeddings train with its top and bottom layers, because pretraining saw only semantic tokens and the decoder must learn to emit phonemes. Values the paper searches without reporting the choice: finetuning dropout (from {0.1, 0.3, 0.5}; 0.1), the reading stage's lower layers ({4, 6, 8}, and the lower 4 last; 4), and backtranslation's N ({2, 3, 4, 12}; 2). Beam search (beam 10, §7.4) ranks by summed log-probability without a length penalty, as the reproduction does. An ended beam is kept as it is; the reproduction keeps extending it past its end token. S2 (§7.3 sizes, temperature 0.75) reads `BOS z_prompt z_target SEP y_prompt` and predicts `y_target EOS`; BOS, SEP and EOS are AudioLM's coarse-stage tokens and its 0.1 dropout. Figure 3's two non-overlapping windows are a 3-second prompt (at most half the utterance) at a random place and, as the target, the longer stretch beside it. Adafactor's relative step size is §7.2's inverse square-root decay. Without a voice, synthesis prompts with nothing, which §5 says samples a random voice. `Train(tokens, mel)` throws, because the model trains on codec tokens. |

## Music generation

| Model | Paper | Gaps filled from | Choices |
|---|---|---|---|
| MusicGen | Copet et al. 2023 | facebookresearch/audiocraft | Rebuilt: the copy that survived the duplicate sweep fed its T5 output into the codec embedding. The paper fixes the 32 kHz EnCodec (four codebooks of 2048, 50 Hz), the delay pattern (Eq. 6), pre-norm layers with cross-attention and a ReLU feed-forward block (§2.4), the frozen T5 conditioner, condition merging 0.25, description dropout 0.5, word dropout 0.3, condition dropout 0.2, guidance 3.0, top-k 250 at temperature 1, AdamW (0.9, 0.95) with weight decay 0.1, clipping 1.0, a cosine schedule after 4000 warmup steps, 1M steps, D-Adaptation for the 300M model (`DAdaptAdamOptimizer`, matching audiocraft's step for step) and an EMA of 0.99. The paper says ReLU where audiocraft uses GELU, and condition dropout 0.2 where audiocraft uses 0.3: the paper wins both. From audiocraft: the three sizes' widths and depths, t5-base, bias-free linears, its sinusoidal step embedding, the 1e-4 rate of the AdamW-trained sizes, the EMA every 10 updates (unbiased), keeping each other metadata field with probability 0.5, and 30-second generation. T5 reads t5-base's SentencePiece ids (`T5TextTokenizer`, which matches sentencepiece on 2023 texts). The language model matches audiocraft's `LMModel` in logits, loss and guided greedy generation. Melody conditioning (§2.3, `MelodyConditioning`): the one-hot chromagram (2¹⁴ window, 2¹² hop, argmax) of the clip in training and of the given melody at generation, projected and given with the text as a prefix in place of cross-attention (§2.4); the chroma follows audiocraft's `ChromaExtractor` exactly, and the melody is dropped with probability 0.5 in training (audiocraft's value; the paper is silent). audiocraft separates out drums and bass with Demucs before the chromagram; the paper does not, so a caller wanting that passes the separated stem. Stereo (§4.3, `Stereo`): each channel through EnCodec, the eight codebooks interleaved left, right per level (audiocraft's `InterleaveStereoCompressionModel`), with the stereo partial delay by default (the better rated in Table 3) or the stereo delay. MusicGen has no negative prompts and does not inpaint; both are refused. |

## Neural audio codecs

| Model | Paper | Gaps filled from | Choices |
|---|---|---|---|
| EnCodec | Défossez et al. 2022 | facebookresearch/encodec and audiocraft | Defaults are the paper's 24 kHz streamable model. `OfficialCheckpoint24kHz` and `OfficialCheckpoint48kHz` match the released checkpoints, which differ from the paper's text in places. |
| SoundStream | Zeghidour et al. 2021 | EnCodec's reimplementation (Défossez et al. 2022, App. A.2) | No official code exists. Quantizer dropout draws n_q ~ U[1, N_q] per example; FiLM denoising (§III-F). |
| DAC | Kumar et al. 2023 | descriptinc/descript-audio-codec | Factorized 8-dimensional, L2-normalized codes chosen by cosine similarity; loss weights 15 / 2 / 1 / 1 / 0.25. |
| SpeechTokenizer | Zhang et al. 2024 | ZhangXInFD/SpeechTokenizer | The paper's loss forms with the reference's λ values. Training needs the teacher's features (see above). |

Official weights load into the native EnCodec, DAC and SpeechTokenizer models, and into Pheme's three networks (from their state dictionaries). The tests check them against tiny
reference models; the scripts that produce those references are in `tools/reference-data`.
