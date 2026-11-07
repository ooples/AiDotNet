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
or errors. The later ownership audit below records the inherited ForwardTacotron
settings removed after the runtime checks. The remaining sweep failures stay open.

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
build a transformer from them. The static ownership gate also identified seven
retained legacy no-ops: MaxMelLength, OutputsPerStep, PostnetDim, PostnetLayers,
TextSymbols, UsePostnet and WeightDecay. Those are removed as well, rather than
exempted. The native model always uses its own CBHG post-net and Adam setup.
Its actual settings remain EmbeddingDim,
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

The scoped comparison was repeated after removing the seven legacy no-ops:
each baseline framework still reports only the same CP0007 plus the two
previously acknowledged HighwayDim accessor removals. Initial/trained parameter
and output parity still matches bit for bit. The ONNX loader reads the retained
ModelPath, and the 18 newly owned accessor declarations are registered in
PublicAPI.Unshipped.txt. This corrects the RS0016 and ADNOPT001 failures of the
first pinned build; a new full build remains necessary.

DGCNN DropoutRate and UseDropout pass with dropout enabled at probability 0.5;
the deterministic fixture had disabled it. The now-sensitive UseDropout baseline
entry was removed. E3's default speaker passes with two speaker-table rows, and
TrainingBetaEnd passes with four diffusion steps and three training updates;
a one-step schedule only reads the first beta. E3's actual typed training API
also changes parameters under different padding weights when given a short
recording and the required text. Its actual silence-trimming routine returns
8, 10 and 4 samples under the controlled block-size/threshold counterfactuals.
After releasing the leftover browser processes, the updated observation wrapper
compiled with zero warnings or errors. All five E3 probes pass through that
compiled harness. The full hosted sweep remains required.

CosyVoice2 CodebookSize has an explicit constrained-setting exemption, with a
forwarding accessor that preserves its real value: S3 v2's eight ternary FSQ
dimensions have exactly 6,561 codes. Constructor tests reject 1, 6,560 and 6,562,
and the existing released encoder-state/token reference comparison passes.
Both accessor declarations are registered in the public API manifest. This
setting is retained because the paper model uses that cardinality.

S3 v2 state/head settings carry the same explicit paired-constraint treatment:
64/1 and 128/2 are valid and resize the real encoder, while either isolated
change is rejected by the fixed 64-wide rotary table. Nonpositive state/head
values now throw ArgumentOutOfRangeException before modulo or allocation.
The paired resize, two unpaired failures, four nonpositive guards and the
released states/tokens all pass against the changed tokenizer source. CosyVoice's
single S3 stream is also explicit: counts 0, 2 and 8 are rejected by the real
model constructor. These settings remain because they are used; no general
NoValidPerturbation baseline was introduced. Six new accessor declarations
are included in the API manifest.

CAM++ FrameLengthMilliseconds now uses the same real Kaldi audio frontend as
FrameShiftMilliseconds during option observation. Both probes pass against the
compiled harness: length 25 -> 50 and shift 10 -> 20 change outputs before
training. The scoped probe build completed with zero warnings/errors.

IndexTTS2 EmotionGroupSizes and EmotionMode now load deterministic nonconstant
emotion/speaker rows through LoadEmotionTable and observe the real EmotionTableMix
path used by synthesis. Controls use two rows per label and Paper mode; changing
the first group to three rows or selecting ReleaseNearestSpeaker changes the
actual mixed vector. Both compiled probes pass; E3 silence trimming still passes.
The scoped build has zero warnings/errors.

Upsampling-rate candidates can now transfer a factor of two between stages,
preserving the total waveform ratio even when equal rates cannot be reordered.
CosyVoice and CosyVoiceClone both pass with [8, 8] -> [4, 16], preserving
HopSize while changing the actual generator. All 44 perturbation regressions
pass, including two new waveform-ratio cases; scoped build has zero warnings/errors.

CosyVoice HopSize is retained as an explicitly constrained setting: the HiFT
iSTFT hop is four, so its sample ratio is 4 * product(HiftUpsampleRates).
Two mismatched configurations are rejected by the actual model constructor;
valid rate/hop pairs 2/8 and 4/16 generate 32 and 64 samples from four mel
frames through the actual HiFT generator. All seven Cosy constraint tests and
44 perturbation tests pass together using the matching hosted core/tensors.
The changed-source stage build and its existing regressions also pass. Two
forwarding accessor declarations are included in PublicAPI.Unshipped.txt.

IndexTTS2 DitHidden and WavenetHidden are retained with paired-constraint
explanations. The model rejects both 16/32 and 32/16 before building layers.
The actual IndexTtsDiT velocity estimator runs at paired widths 16 and 32,
produces finite [3,5] outputs, and increases parameter count at width 32.
All three focused constraint tests pass with the matching hosted dependencies.
No single-property candidate can satisfy this enforced equality.

CosyVoice repetition probes now additionally call the real repetition-aware
sampler with settings read from the rebuilt language model's installed
configuration, preserving its original/newer fallback profile. A deterministic
two-draw random source exercises the exact 10 * 0.1 repeat boundary. Compiled
RepetitionWindow probes pass for CosyVoice, CosyVoiceClone and CosyVoice2;
CosyVoice2 RepetitionRatio also passes. The scoped build has zero warnings/errors.

GoogleSpeechV2 MaskNoiseStd now observes its public BEST-RQ PretrainingLoss
with enabled masking, one-frame spans and a reproducible random source.
Its CTC prediction path does not apply pretraining noise. The compiled probe
passes for 0.1 -> 0.2 through the actual pretraining loss; scoped build has
zero warnings/errors.

The same real pretraining-loss observation now covers CodeDim, StackedFrames,
MaskLength and MaskProbability for GoogleSpeechV2, and all five pretraining
settings for BESTRQ. All nine additional compiled probes pass, including span
1 -> 2 and masking probability 0.5 -> 1. These properties affect BEST-RQ's
masked targets/loss rather than CTC decoding; no exemptions were added.

AudioLM's three crop probes now use stage-specific EvaluateTrainingObjective
on eight semantic tokens and sixteen frames of valid codec ids. Controls use
0.08-second crops so the fixture spans the crop boundary. All three compiled
probes pass for 0.08 -> 0.16 through the actual semantic/coarse/fine losses;
the scoped build has zero warnings/errors.

FishSpeechV15 schedule probes now use a four-update cosine schedule and zero
warmup (one for the warmup probe), bringing the schedule into the fixture's
training horizon. TrainingSteps, CosineDecay, FinalLearningRateRatio and
WarmupSteps all change actual post-training parameters/outputs in the compiled
probe. The scoped build completes with zero warnings/errors.

Hosted 7cee slices 0/7 exposed ContextNet's public transcription passing rank-2
features to rank-3 convolution blocks. Transcribe now packs [time,mel] into
[1,mel,time] and removes the batch axis from the [1,time,vocabulary] logits
before CTC. Both constructors now configure the frontend with NumMels instead
of silently using its 128-bin default. A changed-source regression independently
packs features, runs the real encoder, computes greedy CTC and compares the
public transcript. It also verifies 1.25-second input remains untrimmed with
MaxAudioLengthSeconds=1 and that metadata reports 1: this existing field is
explicitly informational, serialized metadata, now explained to the option
harness. The changed-source stage and all previous stage checks pass with
zero build warnings/errors.

CIFDecoder TailThreshold is inference-only: its public Transcribe path calls
the real CIF tail firing, whereas Predict produces training-aligned logits.
The observer now transcribes deterministic recordings at four short lengths.
The compiled probe passes for tail threshold 0.5 -> 0.25 through actual public
transcription; scoped build has zero warnings/errors.
