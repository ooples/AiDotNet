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

DAC QuantizerDropout now uses four quantizers in both control and candidate.
With only one quantizer, U[1,N] always returns one and cannot exercise dropout.
The compiled probe now passes for probability 1 -> 0.5 through real training;
the scoped build has zero warnings/errors.

AudioLM generation probes now observe actual ContinueCodes output before
frozen codec decoding. SemanticTemperature, CoarseTemperature, FineTemperature
and Seed all pass through that public generation path in the compiled harness.
The scoped build has zero warnings/errors; no exemptions were added.

MaskGCT sampling probes now observe the actual semantic and acoustic code
generation methods before waveform decoding, with 32 target frames and the
top-one filter disabled. TextToSemanticSteps, TextToSemanticTemperature,
SemanticToAcousticTemperature and SamplingSeed pass using valid candidates.
Temperature perturbations now exclude negative values and include zero.
All 57 focused perturbation/constraint regressions pass. GuidanceRescale still
fails the original observation and remains unresolved; probing both semantic
and acoustic tokens with active guidance did not distinguish it at test scale.
No baseline or exemption was added for that unresolved case.

DAC Channels is a mono architecture constraint, not a free structural setting:
DacEncoder's first convolution consumes one channel and DacDecoder's final
convolution emits one (DacModules.cs). Both DAC constructors now reject channel
counts other than one before base construction, and CreateCodec repeats the
check for reconstruction/loading. The forwarding setting has an explicit
constraint explanation and two API accessor declarations. Changed-source tests
reject 0, 2 and -1, and a mono model still reconstructs finite 256-sample
audio through the actual encoder, quantizer and decoder. All existing stage
checks continue to pass; scoped build has zero warnings/errors.

MaskGCT duration probes now clear the fixed-duration override, select the
actual predictor and use its Duration training stage. DurationMode changes
TargetLength; DurationSteps, DurationGuidance and DurationHeads are observed
in the predictor's continuous per-phone output before frame rounding. All
four compiled probes pass, with no baseline/exemption additions; scoped build
has zero warnings/errors.

MaskGCT SemanticToAcousticHeads now observes the actual first-acoustic-layer
TrainingLogits on nonconstant tokens/masks and selects that training stage.
Its compiled probe passes for 2 -> 1 heads. GuidanceRescale is observed before
discrete sampling: the unchanged generation-step math was extracted into
GuidedLogits, and Generate calls that same method. A changed-source test proves
rescaling changes its logits by more than 1e-12. Both release-reference T2S and
S2A forwards and iterative token sequences still pass after the extraction.
The harness reads the rebuilt model's private Decoding configuration and calls
that actual step; its end-to-end probe awaits a hosted core containing the new
internal method. Guidance rescale candidates are constrained to [0,1].

MaskGCTOptions now derives directly from ModelOptions and owns the nine
formerly inherited settings that its native/ONNX paths actually reference.
Removed seven legacy no-ops: MelChannels, FftSize, NumEncoderLayers,
NumDecoderLayers, NumHeads, MaxMelLength and DropoutRate. Its separate T2S,
S2A, semantic-encoder and duration heads/layers remain. The upstream config at
open-mmlab/Amphion@26f6883110181f1dbfe95c70a7c7dbaf4de5f42a explicitly
configures those separate networks and Vocos FFT 1920/hop 480; it does not
use the removed shared settings. The pinned reference forwards/token tests pass.

A source-linked full MaskGCT model with 180698 parameters was compared with
the hosted pre-migration model: every retained default, every initial parameter,
the duration objective, one real training update and every resulting parameter
and evaluation loss match exactly. Re-run via MaskParity.Run in the scoped
Stage project; tiny options saved in pr2294-maskgct-tiny-options.json. This is
runtime proof, separate from the API surface stub. Four stale no-op baseline
entries were removed. All 18 new accessor declarations are registered, and SDK
ApiCompat finds exactly CP0007 for MaskGCTOptions on net10.0/net471/net8.0;
three exact base-type acknowledgments were added. Full package validation
remains a hosted check.

MaskGCT's remaining isolated-candidate constraints have positive native tests:
NumQuantizers must vary with the schedule array length; paired counts two
and three increase parameters and generate four target frames with the correct
number of valid acoustic codes. HopSize must vary with the encoder strides;
valid hops 480 and 320 decode four frames into exactly 1920 and 1280 finite
waveform samples. Unpaired changes are rejected. Both settings are retained
with explicit constrained-option explanations, backed by five changed-source
checks; no no-op baselines were added.

The MaskGCT ONNX loader now reads its recorded ModelPath when opening the
graph, keeping the newly owned asset option connected to the load path; scoped
changed-source compilation passes with zero warnings/errors.

AudioPaLM's shared options now own only the settings read by its native TTS,
ASR and shared decoder implementations. Seven unused TtsModelOptions fields
(MelChannels, HopSize, FftSize, NumEncoderLayers, NumDecoderLayers,
MaxMelLength, WeightDecay) are removed; NumHeads and DropoutRate are retained
because AudioPalmCore consumes them. The ONNX loader reads recorded ModelPath.
Scoped API comparisons on net10.0/net471/net8.0 report only the acknowledged
CP0007 base change; public accessors are registered and five stale TTS baseline
rows removed. No baseline was added to hide these failures.

Changed-source native TTS task losses, updated common parent parameters and
every frozen child parameter vector match the hosted implementation exactly.
ASR fitted-tokenizer predictions and post-training predictions match as well.
The scoped assembly has a shorter parent flattened surface (181,912 values),
already present before migration; equality covers its common parent surface
and each child separately, not the complete hosted flattened vector.

ASR public Predict incorrectly inferred an index domain from decoder embeddings
and rejected waveforms before tokenization. Its input domain is now Continuous;
a fitted-tokenizer regression verifies Predict agrees with Transcribe on tones,
with finite confidence and correct duration. Changed-source regression passes.

The 39 historical LLMIntegratedAudioPaLM OOMs allocate the default PaLM through
the generated factory. Its namespace-specific branch now uses the existing
AudioPaLMFixture.Asr factory, fitting the same compact native tokenizer used by
task tests. The generator builds successfully; actual scaffold emission verifies
both ASR and TTS route to their respective compact factories, including float
scaffold conversion. The full changed-source stage passes, including prior
MaskGCT references, codec constraints and native parity checks. Hosted shard
validation remains required.

SpiritLM's option probes now activate the actual Expressive/SpeechProp/SONAR
subnetworks at the existing native task-fixture scale. Each option has its own
control group, so cached observations cannot cross variant or observation
contexts. Architecture changes are detected before replaying fitted HuBERT
centroids and EnCodec codebooks. Twenty-seven of thirty isolated variant probes
passed immediately; the remaining SpeechProp/SONAR head counts now observe
continuous native features/embeddings, and dropout now runs the SONAR
fine-tuning loss that actually applies it. All three then pass.

Fourteen targeted probes pass for HuBERT/style attention, actual generation
budget/context boundaries (EOS forbidden), deterministic word interleaving,
active gradient clipping, SpeechProp optimization and the shared pitch/vocoder
optimizer schedule. Decay candidates remain in [0,1]; 0.999 -> 0.4995 changes
trained parameters. The focused regression suite passes 59 tests.

VocoderTrainingSegment extracts the unchanged native training crop; TrainVocoder
calls it directly. The segment-size probe observes that actual waveform crop.
Three changed-source tests verify code, pitch, style and waveform alignment for
four/eight-frame budgets and a budget longer than the source. The complete
changed-source stage passes with zero compilation warnings/errors. No SpiritLM
production setting was removed in this batch; NumEncoderLayers still needs its
separate ownership audit. Hosted current-head shards remain pending.

SpiritLMOptions now owns the ten shared settings actually consumed by the
native model, retaining all defaults. Removed the six unused inherited fields
MelChannels, HopSize, FftSize, NumEncoderLayers, NumDecoderLayers and MaxMelLength.
The concrete getter IL audit finds reads only for the retained fields; source
construction uses Layers, separate HuBERT/Expressive/SpeechProp/SONAR depths and
the vocoder's own configuration. The ONNX loader consumes recorded ModelPath.

Changed-source Base (80,938 parameters) and Expressive (110,134) match the hosted
model's complete initial and trained vectors and actual TrainText loss exactly.
SpeechProp matches its 286,373-value common parent surface and its complete
frozen EnCodec vector separately before/after the same training step; the scoped
assembly omits the child vector from its parent flattening, as with AudioPaLM.
Every retained default matches. Seven release-reference checks pass: HuBERT
and quantizer, FCPE, pYAAPT, pitch VQ-VAE, both unit vocoders, style classifier
and token strings. Three crop-alignment cases also pass.

Twenty accessor declarations were registered. Scoped API comparisons on all
three frameworks report the existing acknowledged copy-constructor removal
(CP0002) and base change (CP0007); both already have exact SpiritLMOptions
acknowledgments, so no new suppression was needed. Ten stale baseline rows
were removed: absent options and the verified sensitive HiddenDim/MaxTextLength.
No no-op baselines were added. Full hosted compilation/package/shards remain
required; these scoped comparisons are not a full-package proof.

SparkTTS now gets option-specific control groups, replay of lazy speaker-encoder
fitting, annotated mixed-objective training for the ten controllable-speech
fields, BiCodec-stage probes for codec/discriminator losses and architecture,
and continuous mixed XLSR features for Wav2VecHeads. Twenty-one valid-context
probes pass against a2's hosted native models. Four stale baseline rows for
SpeakerEmbeddingDim, MelChannels, LearningRate and WeightDecay were removed.

A discriminator-band candidate previously reached a negative-width tensor slice
only during training. Spark now rejects nonfinite, unordered and out-of-range
band edges before construction. Four negative tests pass; a valid alternative
band partition runs an actual finite BiCodec GAN training step. That regression
also exposed a real fresh-model training bug: the shared training step collected
ECAPA's lazy parameter references before its mel input width was known. Both
ordinary and annotated training now resolve the speaker tokenizer once, without
gradients, before capturing those references. Fresh-model codec training passes
without fixture preparation. Scoped changed-source compilation passes with zero
warnings/errors. AudioPaLM's finite-confidence assertion now also uses APIs
available on net471. Sampling, frontend and paired Spark constraints still need
additional work; no baseline was added for them.

Eleven additional Spark probes now pass through the actual generation loop,
context-limit validation and continuous speaker coordinates before FSQ. Sampling
uses fixed replayable seed sequences and preserves the configured GenerationSeed;
six sampling cases were verified again with that configuration. WaveDecoderRates
now observes actual codec-decoded samples; changing stage rates with the same
product changes the native waveform. No mathematical setting was removed to
make these probes pass. MaxCodecFrames/MaxTextLength stale rows were removed.

The previously ignored shared Seed now controls Qwen and BiCodec initialization;
GenerationSeed remains the sampling seed and the initialization fallback when
Seed is null. A native regression verifies explicit initialization seeds repeat,
change parameters and remain independent of the sampling seed. With Seed null,
all 48,994 initial/trained parameters and the actual LM loss/update match the
hosted native model exactly. Changed-source modules and model compile with zero
warnings/errors, and the fresh-model GAN regression remains green. This retains
the paper model's default random draws while making the shared setting useful.

SparkTTSOptions now owns its consumed settings instead of inheriting unused
shared TTS/codec controls. Removed eight unused fields: FftSize, HiddenDim,
NumEncoderLayers, NumDecoderLayers, MaxMelLength, DropoutRate, TextEncoderDim,
LanguageModelName. Concrete IL reads and source construction confirm the paper
uses Qwen's LLMDim/NumLLMLayers and the separate XLSR/BiCodec/Perceiver/waveform
configuration. The consumed mel hop/channels, speaker width and attention heads
remain; CodebookSize's NotScaled contract and LLMDim/head divisibility remain.
All retained defaults and all 48,994 native initial/trained parameters and
training losses match the hosted model exactly after this ownership change.

Five constrained settings remain explicit, with explanations and positive native
tests: one semantic stream; SampleRate/CodecFrameRate paired with waveform
strides/kernels; FsqLevels paired with GlobalCodebookSize. Three valid waveform
geometries decode two fixed frames into exactly 320, 640 and 160 finite samples.
Both FSQ vocabularies (8 and 16) produce valid global tokens and decode actual
semantic/global tokens to finite waveforms. Five unpaired configurations are
rejected. The single semantic stream is a rank-one token tensor, verified at
runtime. No constrained setting was deleted or recorded as a no-op baseline.

Thirty-four public accessor declarations are registered. Scoped ApiCompat
reports only CP0007 for SparkTTSOptions on each framework, with three exact
acknowledgments added. Six more obsolete/constrained baseline rows were removed.
The ONNX loader consumes recorded ModelPath. Source compilation passes with
zero warnings/errors; native paired geometry/FSQ, seed, fresh GAN training and
complete parameter/training parity checks pass. Full hosted package/shard
validation is still required.

FireRedTTS stage contexts now cover the real flow decoder, acoustic language
model, Mel codec and speech-tokenizer training paths, with a separate control
per option. The dual-decoder weight schedule is compressed to steps 0-2 on
both control and candidate so actual training reaches its transition. Twenty
of twenty-one stage probes pass immediately; CodecAdversarialWeight needed
windows 8/16/32 and discriminator width 8 because the generated 0.4-second
recording cannot fill the production 128-frame window. That actual adversarial
training probe then passes too. No production architecture or default changed.

EMA-decay perturbations remain in [0,1], with two regression rows added. The
focused perturbation/constraint suite passes 61 tests. The source fixture and
helper compile with zero warnings/errors. FireRed's coupled dimensions, vocoder
preset restoration and remaining observation contexts still need separate
checks; no no-op baseline or option exemption was added in this batch.

FireRed paired native tests now verify speaker/tokenizer/codec widths 32/64,
HuBERT/SSL widths 16/32, and codebook/audio vocabulary pairs 4/18 and 8/66 by
running the actual encoder, both tokenizer reconstructions, Mel codec decoder,
product quantizer and GPT output head. Seven unpaired combinations are rejected.
The tokenizer's reconstructed speaker mel is 80 bins; the flow/codec mel is 100.
These distinct paper frontends were verified at runtime, not conflated.

The 48 kHz paper vocoder preset had a real length bug: stride 3 with kernel 6
adds one intermediate sample, propagated into eight extra waveform samples
(two 10 ms frames produced 968 instead of 960). Its stride-3 kernel is now 7,
matching the parity needed by the existing transposed-convolution padding.
Both presets now produce exactly 480 (24 kHz) / 960 (48 kHz) finite samples for
two mel frames. The released 24 kHz preset's complete 4,305,599 initial/trained
parameters and native GPT losses match the hosted model exactly. Source model,
options and training compile with zero warnings/errors. No option exemption or
baseline change was needed for this production vocoder fix.

FireRedTTSOptions now owns its thirteen consumed shared settings, removing
unused FftSize, NumEncoderLayers and MaxMelLength. The concrete getter audit
and construction show the distinct native FFT frontends and network depths;
consumed GPT depth/dropout, mel channels, sample rate and hop remain. Every
retained released-preset default and the complete 4,305,599 initial/trained
parameters and native GPT losses still match the hosted model exactly.

Model-specific coupled settings are retained with explicit constraints backed
by native positive tests: speaker/tokenizer/codec widths, HuBERT/SSL widths,
codebook/audio vocabularies, the vocoder presets and the Mel codec stride
product. Both a single stride of two and a two-stage 1x2 codec encode eight mel
frames to four code frames and decode back to eight finite mel frames. Nine
isolated width/vocabulary/rate/hop changes and an invalid stride product are
rejected. No inactive setting was marked as a test-scale exemption.

Twenty-six accessor declarations are registered. Scoped API comparison on
net10/net471/net8 reports only the existing acknowledged FireRedTTSOptions
CP0007 base change, so no new acknowledgment was needed. Seven obsolete or
constrained baseline rows were removed. The ONNX loader reads recorded ModelPath.
Compilation and native parity/constraints pass; full hosted package/shards
remain required. Four FireRed decoding/text-splitting observers still need work.

FireRed's four remaining decoding/text contexts now observe the actual native
acoustic token generator, semantic GPT candidates and text splitter. Acoustic
sampling temperature/top-k and ReleaseTextSplit pass. The semantic-token mask
needed a control with unrestricted top-k/top-p, temperature one and repetition
penalty one so the start-token exclusion is reachable; its actual GPT candidate
sequences then differ too. All four probes pass, with option-specific controls.
The current source fixture/helper compile cleanly and native full-vector parity,
retained defaults and coupled-geometry tests remain green. All 36 historical
FireRed failures now have local repairs/evidence; current-head hosted shards
still need to validate the complete change.

All 33 historical VALLEXClone failures were OutOfMemoryException during the
generated fixture's default foundation-scale construction. The generator now
uses VallEXCloneFixture, sharing exactly the existing facade paper tests'
16-wide bilingual AR/NAR, three 64-way EnCodec streams at 75 Hz and short
reference enrollment. The production default architecture is unchanged.
Actual scaffold emission selects this factory, and the generator builds.

The shared factory has 113,168 parameters. Nine native facade paper checks
pass: canonical graph parity, transcript requirement, actual variable-length
codec speaker representation, cross-lingual AR/NAR inference, both native
training stages, inherited checkpoint loader and deep copy of codec/facade
settings. The paper tests now reuse the same options rather than duplicating
them. Hosted generated invariants and newly exposed option outcomes still
need verification; this establishes that the native factory runs without the
foundation-model allocation.

VALLEXClone also needs the canonical codec-TTS scaffold: framed phoneme input,
16 codec frames as the target and teacher-forced text-sensitivity checks. It is
now routed through the same template as VALLEX, rather than the generic mel
input template. Actual scaffold emission verifies the factory and shapes.

Native option probes exposed invalid Transformer widths being accepted by the
public constructor and failing later during inference; the harness could then
mistake rejection for useful sensitivity. VallEModelBase now rejects nonpositive
widths/depths and incompatible hidden/head widths before network construction.
Four changed-source negative cases plus the nine existing native facade paper
checks pass. Valid graph behavior remains covered by the canonical graph,
AR/NAR training and cross-lingual inference checks. Fresh generated option
outcomes are being audited after the OOM fix; several previously unreachable
controls still need stage/frontend/ownership work.

Fresh VALL-E facade probes exposed two real shared-base defects. The linear
warmup/decay scheduler inherited StepPerEpoch, so Train(sample) never advanced
it. Its native stage optimizers now step per batch. Two changed-source tests
show WarmupSteps and TrainingSteps change actual native parameter updates over
two batches; these were bit-identical in the old hosted probes. The shared
Seed also now controls AR/NAR and codec initialization, ahead of architecture
and sampling-seed fallbacks. Native seed tests verify repeatability, changed
parameters for a changed seed and independence from SamplingSeed. Null shared
Seed preserves the existing initialization fallback.

The four negative Transformer-geometry tests, nine existing native facade paper
checks and three new schedule/seed cases all pass with source VallEModelBase,
VALLEX and VALLEXClone linked (zero compile warnings/errors). In this source
assembly the parent's flattened surface excludes the separately owned codec,
so its parent count is 65,857 versus the hosted full 113,168; those counts are
not a parameter-loss claim. Current option outcomes still need the new hosted
model metadata, and remaining frontend/task-specific ownership is under audit.

VALL-E option ownership now follows the actual native consumers. The common
VallEModelOptions retains both Transformer depths and all 25 consumed shared
properties. VALLE alone retains PromptSeconds: its NAR samples the same-utterance
prompt using that duration. VALLEX/Clone use an explicit second-utterance prompt
(NarLossWithPrompt), while VALLE2 uses MinPromptSeconds/MaxPromptSeconds. The
public configurable VocabSize was overwritten by the selected phone table;
it is now internal computed state. MelChannels/FftSize/MaxMelLength and the old
codec facade settings have no native consumers and are absent.

Changed-source family compilation passes with zero warnings/errors. Every
retained default on VALLEOptions, VALLEXOptions, VALLE2Options and
VALLEXCloneOptions matches the old hosted configuration, including nested codec
and ONNX settings. All 16 prior native facade cases still pass. Two positive
paired-depth cases (depth 1 and 2) train both real AR/NAR objectives and change
parameters. API comparison against release 0.234.0 for net10.0, net8.0 and net471
reports only the three already acknowledged CP0007 base changes for these option
types. The newly introduced protected base signatures and moved declarations
are updated in PublicAPI.Unshipped.txt. Removed baseline rows correspond only
to properties absent from the new public configuration; retained behavioral
probes still need their real task contexts.

Five VALL-E X/Clone task contexts now pass actual generated sensitivity probes
against the hosted 703 core: MaxTextLength truncation (4 to 5), native community
phone tokenization, SynthesisLanguage in community mode with a three-language
table, minimum enrollment acceptance at 320 samples, and EOS stopping. The EOS
fixture temporarily fits the actual bias-free AR head so EOS is most likely
while a controlled native sampling draw selects a real code; it verifies that
boundary and restores every head weight in finally. It observes native ArGenerate,
including its actual early-end exception, rather than computing an option-derived
answer. Per-option control groups prevent sharing incompatible observations.

All five contexts also pass changed-source tests; each proves observation leaves
the original parameter vector unchanged. Positive paired codec tests run actual
encoding/decoding at rates 16/24 kHz, hops 160/320, quantizer counts 2/3 and
vocabularies 64/128, verifying code bounds, finite audio and decoded sample count.
Together with paired-depth tests these justify constrained-property annotations;
they do not remove any consumed setting. The complete changed-source facade run
now covers 25 native cases and the four-family retained-default comparison.

GPT-SoVITS: the previous bounded v2ProPlus fixture still had 63,936,119
parameters, dominated by the actual GAN discriminators; repeated full snapshots
exhausted a 2 GB local probe budget. The generated fixture now uses a complete
native v3 graph with 574,474 parameters (semantic GPT, CFM synthesizer, BigVGAN,
HuBERT and BERT). GAN-only option contexts select the real v2 GAN explicitly.
The actual generator emitter confirms both float/double factory substitution.
This changes test configuration, not the released model defaults.

Eight structural probes and ten task/stage probes pass against the hosted core:
HiddenDim, DitDim/Depth/Heads/HeadDim/ConvLayers, BridgeChannels,
VocoderInitialChannels; CfgRate in V5Dev, SampleSteps, Cut, HubertHeads,
BertHeads, Beta1/Beta2/Epsilon, SynthesizerLearningRate and UpdatesPerEpoch.
Two further vocoder stride/kernel probes pass using actual native vocoder audio.
Inference-only flow settings observe native SynthesizeMel, frontend heads observe
actual continuous features, and s2 optimizer settings train the native CFM
objective. Per-option controls prevent cross-context cache reuse. Unused inherited
settings and remaining GAN/task constraints are still under audit.

The v3 factory exposed a real existing persistence defect: CreateVoice stored its
transcript and original sample rate only in a ConditionalWeakTable. Clones lost
them and refused inference. The model now declares the active reference's source
samples/rate/text as state, restores them independently, and lazily recomputes
features with restored weights. Changed-source v3/V5Dev tests preserve actual
waveforms through cloning AND serialization with a 16 kHz reference, despite
24/48 kHz model outputs. The console serialization run uses the existing isolated
trial-file test hook, as xUnit's assembly-wide trial-isolation attribute does.

Paired native GAN upsampling tests decode six frames to exactly 1,920/3,840
samples at hops 320/640. They exposed an eight-sample overrun for an odd stride
paired with an even kernel; constructors now reject kernel/stride pairs whose
difference is odd, mismatched arrays and invalid vocoder upsampling products.
Residual kernels must be positive odd values with matching positive dilation
arrays. Native v3 inference passes with residual kernels 3 and 5, and invalid
GAN/CFM residual kernels are rejected before inference. The eight permanent
native cases pass with zero compilation warnings/errors. All 574,474 initial
and trained parameters and semantic losses still exactly match the old hosted
model under the same valid v3 configuration.

GPT-SoVITS shared-option ownership is now explicit. Source and compiled IL show
no consumers for FftSize, NumEncoderLayers, NumDecoderLayers, VocabSize,
MaxMelLength, DropoutRate or LearningRate. Native FFT/spec geometry is the
release's fixed 2048/1025; native depths are T2SLayers, SynthesizerLayers and
DitDepth; phone vocabularies come from the version's symbol table; semantic-GPT
embedding/layer dropout follows its own release configuration; native optimizers
read T2SLearningRate and SynthesizerLearningRate. All seven obsolete inherited
properties are removed. The retained nine shared properties have their original
defaults. SampleRate is now derived from Version rather than accepting a setter
that ConfigureBase immediately overwrote; default behavior and all output rates
are unchanged. HopSize/UpsampleRates remain real constrained controls, supported
by the positive native paired-geometry tests rather than removed.

Every retained default matches hosted options, and the complete 574,474 initial
and trained v3 parameter vectors plus semantic losses still match exactly after
the ownership change. The eight native persistence/geometry/kernel cases pass.
The scoped API surface builds without warnings. SDK release-0.234.0 comparisons
for net10.0, net8.0 and net471 report only the already acknowledged GPTSoVITSOptions
CP0007 base change; no new suppression was needed. New property declarations are
registered in PublicAPI.Unshipped.txt. Remaining GAN-specific option probes and
TextLanguage still need fresh native validation.

The final five historical GPT-SoVITS contexts now pass with source GPTSoVITS and
its owned options: SynthesizerDropout observes the real training-mode native
encoder; SegmentSize observes native GAN cropping/loss; UpsampleKernels now has
valid positive candidates after early geometry validation (16 to 18 rather than
accepting an invalid odd kernel); TextLowLearningRate changes actual full GAN
parameter updates; TextLanguage exercises the real frontend's English versus
asset-requiring language modes. The language boundary test checks the precise
missing-fastText requirement, not multilingual phonetic accuracy without assets.

Full-width GAN training exceeded even a 4 GB heap. DiscriminatorWidthDivisor now
exposes the native discriminator module's existing width scaling, defaulting to
1 (the released architecture). The fixture uses 16 while retaining every period
and scale discriminator, both optimizer steps and the complete adversarial,
feature-matching, mel and KL objectives. Changed-source positive tests run the
whole GAN step at divisors 16 and 32; zero/negative divisors fail before allocation.
A separate native comparison proves default 1 preserves all 63,301,658 initial
GAN discriminator parameters, bit for bit, across all 49 layers versus the older
hosted v2ProPlus graph. The API surface remains additive for this new control.

All 27 active probes now pass locally: the 26 retained historical knobs plus the
new divisor. The other six historical failures correspond to three removed
unused fields, the formerly overwritten sample-rate setter and the two documented
coupled-hop/stride settings. DiT/bridge settings observe actual native mel
sampling rather than relying on parent parameter snapshots: the source-linked
hybrid assembly omitted those child-owned changes from its parent view. Six
positive source tests independently verify the real mel changes. There are now
20 permanent native GPT-SoVITS cases, all passing, alongside retained-default,
574,474 full v3 parameter/train-loss parity, and full-width discriminator parity.
The temporary generic console substitutes only public Shape.ToArray() for an
inaccessible internal _shape read in a training-source copy; the native test
project links unchanged production training source. All scoped compilations have
zero warnings/errors. Latest-head hosted validation remains required.

Zonos fresh probes confirm a stale HiddenDim baseline (32 to 64 is sensitive),
while Backbone, Seed, MaxGradNorm and ConditionerDropout remain insensitive in
the old generic context. The first two are real production routing defects:
NetworkConfiguration never consulted Backbone, so selecting Hybrid alone kept
all attention layers; initialization ignored the shared Seed. Hybrid now selects
attention every fourth layer when no explicit AttentionLayers is supplied, while
an explicit list remains authoritative. Shared Seed now precedes architecture
and sampling fallbacks for the native backbone and the frozen DAC initializer.

Four changed-source native tests pass: implicit hybrid selection builds and trains
an actual Mamba layer; explicit hybrid layout is retained; seeds repeat and change
both backbone and codec independently of SamplingSeed; null shared Seed matches
the architecture fallback in both vectors. The bounded fixture is used by these
native tests; the generated Zonos factory has not yet changed. No consumed setting
was removed. Source compilation has zero warnings/errors.

The old d322 hosted binary is not yet a trustworthy Zonos initialization parity
baseline: identical visible network configurations and the complete matching codec
vector still produce different initial backbone values and losses. This predates
any established source-before/source-after comparison, so it is not labeled a
regression or used as proof of parameter loss. Newer hosted metadata or the pinned
release reference tests are needed for that comparison. Remaining 25 historical
Zonos probes and the wider shard inventory remain unfinished.

All 28 historical Zonos failures are now locally addressed. Twenty active probes
pass after the ownership change: the earlier 19 native contexts plus HiddenDim.
Hybrid-only Mamba settings use real hybrid layers; quality targets use the actual
hybrid conditioning prefix; repetition/unified-sampler knobs observe native
SampleFrame with controlled logits and history. ChunkSize observes the actual
continuous backbone output before code selection hides floating-point differences.
ConditionerDropout trains with an optional language label, and MaxGradNorm trains
at a clipping boundary. Every option has its own control group. The actual scaffold
emitter confirms the shared bounded transformer/DAC factory.

The other eight historical cases are four unused inherited properties and four
coupled codec contracts. Source and compiled getter audit prove nine inherited
settings unused overall: NumHeads (the real property is Heads), encoder/decoder
depths (the real property is Layers), VocabSize (the phone table fixes it),
DropoutRate (conditioner dropout is separate), MelChannels, FftSize, MaxMelLength,
and nominal HopSize (the model derives its real hop from DAC strides). These are
removed; all seven consumed shared properties and their defaults are retained.
The existing reference fixture stops assigning the unused nominal hop.

Positive native tests pair 8/16 kHz, two/three codebooks and 32/64-code vocabularies
with the real codec bandwidth and end/mask sentinels. They encode, verify code
bounds, decode finite correctly sized waveforms, and train the actual delayed-code
objective with changed parameters. Negative unpaired sample-rate/codebook/end/mask
changes are rejected. Sentinel locations are now validated before inference rather
than allowing real DAC codes to collide with them. These positive tests justify
four constrained-option annotations, preserving the actual settings.

Every retained default, including complete nested codec/ONNX values, matches the
hosted option defaults after order-independent JSON comparison. PublicAPI's 14 new
accessors are registered; SDK comparisons against release 0.234.0 on net10.0,
net8.0 and net471 report only the existing ZonosOptions CP0007 acknowledgment.
Twenty permanent native option cases pass with zero native compilation warnings.
Both pinned release backbones match their conditioning, guided logits, generated
codes and training losses; released speaker, number-normalization and kanjize
references pass too. This supplies actual reference behavior independently of the
unresolved old-artifact initialization-comparison limitation. Current-head hosted
build/shards and all remaining model groups are still required.

### Chatterbox native training and initialization

Chatterbox has inference checkpoints but no declared released stage optimizer recipe. Its native flow stage previously threw before any update, while T3 silently used AdamW defaults. A component optimizer hook now retains declared CosyVoice recipes and gives Chatterbox configured AdamW with parameter masks and per-batch linear warmup from zero. This is an explicit training policy, not a claim about the release optimizer. The shared Seed now reaches CosyVoice initialization and Chatterbox's frozen voice encoder.

Scoped source validation: ChatterboxOptionContextTests passed native flow updates with unchanged LM/frozen/voice weights, native T3 parameter updates with differing warmup trajectories, and repeatable backbone/voice initialization independent of SamplingSeed. The warmup issue was configuration (initial rate equalled peak rate), not a fused-scheduler defect. Full hosted validation remains pending.
All four ChatterboxPaperTests also passed in the same source scope: released defaults, native component construction, punctuation/BPE, and bounded S3Gen versus full T3 reference conditioning.

### Standard CosyVoice sampler ownership and Chatterbox partial windows

TopP, TopK, RepetitionWindow and RepetitionRatio are consumed by the native CosyVoice/CosyVoice2/CosyVoice3 LM configurations (CosyVoice.cs:184, CosyVoice2.cs:191, CosyVoice3.cs:147). Chatterbox uses T3's GenerationTopP, MinP and RepetitionPenalty instead. The four standard controls now belong to CosyVoiceAutoregressiveOptions, inherited by all three standard models but not Chatterbox. All retained serialized defaults match the earlier hosted core for all four options classes; copies preserve nondefault sampler settings, and a constructed native CosyVoice LM receives them. Chatterbox's four release checks and native training checks remain passing.

VoiceEncoderPartialRate was saturated by the tiny fixture's four-frame windows. The option probe now uses the released 160-frame partial size and observes the actual voice embedding for a two-second reference. The permanent regression verifies different rates change embeddings while initialization remains equal. The reference get_frame_step formula was checked against resemble-ai/chatterbox src/chatterbox/models/voice_encoder/voice_encoder.py (blob d986f17fd6afab59364863b5e92fd56eec21236b); no production formula was changed.

Three permanent flow counterfactuals also pass: TrainingCfgRate, FlowLoss and FlowAttentionDropoutRate each change finite native losses and updated flow weights.

### Hosted build blockers and Chatterbox codec contracts

Pinned run 42343819680 failed compilation before running shards: VALLE2Options had a stale inherited NumCodebooks cref; ADNOPT001 reported three ModelPath properties with writes but no reads. The cref now names VallEModelOptions. The VALL-E/GPT-SoVITS/Zonos ONNX constructors now pass the stored validated ModelPath into the ONNX runtime, keeping their explicit path behavior.

Chatterbox exposes S3-v2 cardinality and paired rotary-width contracts explicitly. Permanent tests pass native reference encoding at state/head geometries 64/1 and 128/2 and reject isolated codebook cardinality, state or head changes. The guarded source build passes with zero warnings/errors.

### Common CosyVoice options and v1 encoder ownership

The common options now derive from ModelOptions and explicitly retain consumed frontend, codec, flow, language-model, ONNX-runtime and training settings. TextEncoderDim and NumEncoderLayers belong to CosyVoiceOptions alone: CosyVoice.cs uses both to build its v1 text encoder; independent native changes to either alter its parameter count. They are not used by Chatterbox, CosyVoice2 or CosyVoice3. HiddenDim, NumDecoderLayers, MaxMelLength, MaxCodecFrames and LanguageModelName have no option consumers in the 23 CosyVoice/Chatterbox source files; model HiddenDim is assigned from LLMDim. The old ModelPath option was written only after explicit-path ONNX construction, never read; explicit ONNX constructors still load their path argument.

All retained serialized defaults match the hosted core for all four concrete option classes after removing only the deliberately deleted keys. V1/v2/v3 copies retain their settings. Guarded source compilation passes with zero warnings/errors. Native component, voice, flow-training, streaming-chunk, codec and all four Chatterbox release-reference checks pass. The generic sensitivity harness additionally reports Sensitive for DropoutRate and StaticChunkSize with the new task contexts. API option-surface comparisons against the published package report CP0007 for the four concrete options on net10.0/net8.0/net471; the scope omits runtime classes, so its unrelated CP0001 entries are not package-build evidence.

### Completing Chatterbox's historical option inventory locally

The remaining vocoder option now observes the real VocoderTargets crop, mel/pitch targets and decoded native HiFT waveform, rather than unrelated T3 training. Valid 32/64-sample targets and paired 2/4 upsampling rates pass permanent regressions. These target/decoder checks do not claim full GAN optimizer parity.

A new optional-frontend contract test first failed because Chatterbox never called InitializeReleasedFrontends. Both constructors now initialize the configured original ONNX frontends. A small explicit contract graph verifies the same two-second recording is rejected with MaxReferenceSeconds=1 and executes ONNX Runtime with MaxReferenceSeconds=2; it is not a released tokenizer fidelity fixture. Local execution required copying matching ONNX Runtime 1.31.1 native DLL beside the scoped executable: Windows otherwise found system ORT1.17.1, which lacks managed API31. Chatterbox's released-reference and native training/codec/vocoder checks still pass after initialization wiring.

All 27 historical Chatterbox option failures have local evidence: 14 active observations, six removed irrelevant options, six paired/fixed codec-vocoder constraints and the optional-frontend duration contract. This is not a claim that current-head hosted shards are green.
