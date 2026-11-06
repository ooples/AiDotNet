# PR 2294 option repair evidence

## Ownership

The consumer audit scans 252,235 method bodies with zero unresolved consumers. The old
`VocoderOptions.UpsampleKernelSizes` has three generator consumers: BigVGAN, HiFiGAN
and ISTFTNet. Their options retain independently copied kernel arrays. APNet and
APNet2 operate on spectral frames and do not consume these transposed-convolution
kernels. The removed inherited Kani topology fields are replaced by its existing
LFM2 settings, including `Heads` and `AttentionLayers`.

The SDK assembly compatibility comparison against package 0.234.0 reports exactly
two additional net10.0 diagnostics: removal of the base kernel getter and setter.
All 1,943 preceding net10.0 exceptions still correspond to actual differences.
The two approved removals are acknowledged separately on net10.0, net471 and
net8.0, the baseline package's frameworks. No wildcard compatibility suppression
or package-validation bypass is introduced.

Kani1's `SynthesisSpeakerEmbedding` and `TrainingSpeakerEmbedding` return null;
Kani2 overrides both. Speaker width and cache-position policy therefore belong to
`KaniTTS2Options`. Its defaults and copy constructor preserve both settings.
The explicit ONNX constructor consumes its path argument; the redundant, unread
options property `ModelPath` is removed. `OnnxOptions` remains consumed.

NanoCodec's encoded width is derived as codebooks times FSQ dimensions. Encoder
and decoder rate products must match. Array probes include product-preserving
permutations rather than requiring an invalid codec to build.

FastSpeech2's inherited `DecoderDim` was documented as mel output width but was
unused. It now forwards to `MelChannels`; `HiddenDim` continues to size its FFT
blocks. A scoped comparison preserves initial parameters and mel output exactly,
while setting `DecoderDim` to 12 produces 12 output channels. Copies preserve the
shared value.

## Runtime defects and probe controls

An unloaded speaker embedder previously had random BatchNorm buffers, including
negative variances. They now start at zero mean and unit variance. WavLM's
positional convolution previously normalized an all-zero direction and magnitude,
producing `0/0`. Its reference normal initialization and matching magnitudes now
produce finite hidden states and embeddings before checkpoint loading.

Fitted checkpoint probes measure parameter counts and ordered shapes before
loading incompatible weights. Conditional probes enable LoRA, rotary scales,
frame positions, streaming, scheduler boundaries or reference cropping on both
control and candidate. Candidates start from the prepared control value. Kani2
fixtures exercise the optional speaker path in synthesis and training.
Sample-rate probes prefer double/half rates before one-Hz perturbations: coprime
rates can require gigabytes of polyphase-resampler kernels. The fine-grained
candidates remain available if the simple ratios do not expose an effect.

`ReleaseSpeakerCachePosition` has a documented generic-probe exemption: the tiny
speaking fixture decodes identical audio under both masks. The separate
`SpeakerCachePositionPolicyChangesGeneratedTokens` test requires different token
sequences on the release-reference model with a speaker inserted. It passes;
the existing default-policy reference sequences also pass.

## Validation scope

Local scoped CPU validation passes the Kani prompt, audio and training references;
Kani2 prompt, logits, loss, gradients and both greedy-generation references;
speaker-embedder reference parity; finite unloaded embeddings; and the cache-policy
counterfactual. Thirty-two structural option probes pass. Conditional probes verify
LoRA, scheduling, sampling, frame deduplication, rotary scales, frame positions,
streaming and reference cropping. APNet2's learning rate, decay, epoch and crop
probes pass, as do 29 harness regressions and six span-validation regressions.

On the same CPU and seed, APNet2 (22,752 parameters), APNet (55,890), BigVGAN
(19,581), HiFiGAN (52,310) and ISTFTNet (52,599) match their pre-change binaries
bit for bit in initial parameters, initial waveforms, trained parameters and
trained waveforms. The retained transposed-convolution settings therefore retain
their measured behavior after ownership moves.

The `c7eadaf16` hosted Release build and discovery passed. Running its actual test
assembly locally yielded 234 passes, 26 existing/explained skips and six failures:
four stale insensitive baseline entries, the synchronous cache-policy test's
unsupported timeout attribute, and the ignored FastSpeech2 decoder-width setting.
Those failures are corrected in the follow-up; latest-head reruns remain required.

The scoped builds link existing library binaries and adapt dependency-internal
access for the small source harness. They do not establish a successful full
library build or passing hosted shards. A full local compile was stopped below
1 GB free RAM; subsequent builds use the machine-wide build queue. Hosted build,
framework compatibility, packaging and complete shard validation remain required.

## Recovered sweep follow-up

The partial `d32204019` hosted sweep recorded 378 failed assertions. This is
additional repair work; the focused successes above do not establish passing
paper-fidelity shards. Fifty-four baseline entries whose recorded probes now
report Sensitive were removed individually rather than regenerating a baseline
over unresolved defects.

The scoped harness using the hosted binaries passes all 36 previously failing
audio-duration, frequency-mask-width and time-mask-ratio probes. Duration probes
now exercise the protected preprocessing path used by transcription; masking
probes enable the corresponding augmentation. SPIRAL teacher and contrastive
settings are exercised through actual pretraining over multiple student steps.
Supported 80/128 mel widths and valid probability/moment candidates avoid
rejecting every candidate or manufacturing invalid-domain numerical failures.
All 38 perturbation regressions and six span-validation regressions pass.

ForwardTacotron pitch-bound and clipping probes pass when synthetic pitch crosses
the bound and clipping is active. Its progressive learning-rate schedule had a
real defect: it used the optimizer's default epoch cadence despite defining its
boundary in updates. The model now selects StepPerBatch. Two regression cases
train the changed model source and verify scheduler counts and rates before and
after the configured boundary. The scoped source build passes with zero warnings
or errors. No inherited ForwardTacotron option has been removed in this batch;
its ownership audit remains open, as do the remaining sweep failures.

The subsequent exception-repair batch passes the hosted-binary probes for
MaskGCT FilterThreshold (0.98 to 0.49), FishSpeech RepetitionPenalty (1.2 to 0.6),
and NaturalSpeech CurrentStage (Warmup to Joint). Sampler candidates stay inside
their documented domains; the joint-stage probe initializes its memory bank by
clustering posterior frames, as the public training contract requires.

Wav2Vec2's zero masking probability previously returned a null mask from the
encoder and dereferenced it in negative sampling. Training now represents that
case as an empty mask, and evaluation uses the same zero-target semantics. A
regression verifies zero contrastive/diversity loss, zero evaluation loss and
unchanged parameters after training. The existing reference loss and one-step
SGD comparison also passes. The scoped changed-source build has zero warnings
and errors. Forty candidate-generation regressions and six span regressions pass.

All nine TrainingCfgRate, FlowLoss and FlowAttentionDropoutRate probes for
CosyVoice, CosyVoiceClone and CosyVoice2 pass against hosted binaries after
selecting the public Flow training stage. The original fixture always selected
LanguageModel, whose objective does not run the flow. These options are retained;
their paper paths are active and observably affect training.

Voicebox MaximumAudioFrames passes by observing a single-phone synthesis with a
one-frame phone limit while retaining the longer supervised training utterance;
the one-frame budget changes the actual phone-aligned training crop. It is not
treated as unused merely because an oversized synthesis is correctly rejected.

The FreGrad and MultiBandMelGAN frontend probes now run a deterministic waveform
through the public ComputeMel entry point before observing synthesis. Minimum
and maximum mel frequency and window size pass for both models. MultiBandMelGAN
mean and scale pass with both per-band statistics present, as normalization
requires. FreGrad EnergyMin and its fast inference schedule pass with a
waveform-derived mel and fast sampling enabled, respectively. MultiBandMelGAN's
adversarial weight passes with the discriminator stage active rather than its
200,000-step generator-only warm-up. Longer recordings through typed sample
training make FreGrad CropFrames and MultiBandMelGAN SegmentSize observable.

Neural-audio-codec probes now reconstruct without the fixture's optimizer
override, as TTS probes already do. DAC Beta1 and learning-rate decay pass;
EnCodec and SoundStream learning rates pass, so their two stale baseline entries
are removed. NaturalSpeech's nullable UpdatesPerEpoch is set to one just like
the existing nonnullable form, making epoch decay observable. Its phoneme mask
probe runs phoneme pretraining; duration loss after warm-up, forward KL weight
and memory heads run joint training with an initialized memory bank. All pass.
The K-means iteration probe passes with 128 posterior frames instead of 32, so
one Lloyd iteration no longer reaches the same result as the 20-iteration
control. This clustering probe has a separate cached control group.

IndexTTS2's eight text-to-emotion architecture probes pass after its public
DetectEmotion entry point builds the lazy Qwen module, narrowed for the probe
while retaining the real tokenizer vocabulary. Rotary head-width candidates
prefer an even doubling; the 8-to-16 probe changes parameters before training
instead of counting a rejected odd width as evidence. All 42 perturbation
regressions and six span regressions pass. The SemanticToMel stage makes class
dropout and WaveNet dropout observable; latent fusion also passes when class
dropout is zero so it cannot erase the fusion's conditioning. The emotion stage
with two speaker classes makes the speaker-adversary weight observable; a
single-class classifier has zero cross-entropy.

GLaMM TokenizerPieces passes in its documented build-before-tokenizer mode;
with a tokenizer already supplied, the model correctly derives its vocabulary
from that asset. RegionGroupNormGroups passes when observing the public loss
on a conversation naming a bounding box, which exercises the region encoder
that ordinary image prediction bypasses. TokenizerModel carries an explicit
opaque-asset exemption: arbitrary byte-array mutations corrupt serialized
SentencePiece protobuf. The existing SentencePiece BPE and pinned GLaMM
tokenizer reference tests both pass, including the real model bytes and
tokenization/decoding behavior. The options source compiles without warnings
or errors; no general exception-swallowing rule was added to the harness.

CosyVoice and CosyVoiceClone TopK pass with TopP = 1 and a two-token control,
allowing top-k to act before nucleus filtering makes it redundant. Fish Speech
1.4 and 1.5 ConstrainToSemanticTokens pass by observing the actual generated
frame rows before audio decoding: an untrained unconstrained model may emit
non-speech frames that the codec correctly refuses to decode as audio. Its
parameter fingerprint is included in this observation. FishSpeechV15's
CodebookSize probe passes on its changed parameter count before forward
execution; it no longer tries to decode an untrained, resized vocabulary first.
All five probes pass, as do the 42 perturbation and six span regressions.

CAM++ FrameShiftMilliseconds passes through the same Kaldi preprocessing path
used by ExtractEmbedding on a one-second waveform (10 to 20 ms); precomputed
filterbank inputs cannot exercise frame-shift configuration.

## ForwardTacotron option ownership

The native ForwardTacotron options now derive directly from ModelOptions. The
removed settings are EncoderDim, DecoderDim, HiddenDim, NumEncoderLayers,
NumDecoderLayers, NumHeads and DropoutRate. The native implementation does not
build a transformer from them. Its actual settings remain EmbeddingDim,
SeriesEmbeddingDim, the three predictor convolution/GRU dimensions and dropouts,
PrenetDim/BankSize/Dropout, RnnDim and PostnetChannels/BankSize/Dropout.

This was checked against the upstream [model constructor](https://github.com/as-ideas/ForwardTacotron/blob/2c3d281c4501040bdb844357203504efa7c6614d/models/forward_tacotron.py#L44)
and the `forward_tacotron` section of its [single-speaker configuration](https://github.com/as-ideas/ForwardTacotron/blob/2c3d281c4501040bdb844357203504efa7c6614d/configs/singlespeaker.yaml).
The separate teacher-Tacotron and FastPitch configuration sections are not
evidence for adding their settings to ForwardTacotron's CBHG/LSTM graph.

Every retained default was compared to the hosted options and matches. In the
22,307-parameter narrow fixture, initial parameters, initial output, parameters
after two supervised steps and the resulting output are bit-identical before
and after the option-base change. The progressive-schedule regressions still
pass. The SDK's scoped comparison of this options surface against package
0.234.0 reports the base-type change CP0007 for net10.0, net471 and net8.0;
three exact acknowledgments were added. Full hosted compilation and package
validation are still required.
