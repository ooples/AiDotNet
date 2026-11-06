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
