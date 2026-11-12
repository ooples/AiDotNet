# Released TTS checkpoint audit

This executable compares AiDotNet computations against independently generated author outputs using original pretrained weights. It requires the original weights locally; it does not download, modify or redistribute them. Failures throw and return a nonzero exit code.

The `tortoise-sampling <checkpoint-directory> <reference-json>` mode additionally executes the complete released diffusion checkpoint with the original standard 200-step DDPM schedule and replayed Gaussian noise. Generate its reference with `python tools/reference-data/tortoise_sampling_reference.py <author-checkout> <checkpoint-directory> <reference-json>`. The author checkout is pinned to `8a2563ecabe93c4fb626f876dd0c52c966edef2f`; the reference records and verifies the original checkpoint SHA-256. See [the complete sampler review](../../docs/text-to-speech/tortoise-sampling-review.md) for the reduced regression mode and scope limits.

## Independent paper-equation tests

SparkTTS's inference policy has a separate pinned reference:

```text
python tools/reference-data/spark_sampling_reference.py <Spark-TTS-checkout> <transformers-4.46.2-py3-none-any.whl>
```

The generator executes the original processor bodies from the uninstalled wheel, verifies their order in the original generation pipeline, and records source hashes. `SparkSamplingReferenceTests` compares nine distributions and the actual reduced LM generation loop. See [the Spark sampling review](../../docs/text-to-speech/spark-sampling-review.md) for failures reproduced before the fix and the limits of this controlled reference.

UniAudio's controller reference runs the original search loop with controlled neural logits:

```text
python tools/reference-data/uniaudio_generation_reference.py <UniAudio-checkout> tests/AiDotNet.Tests/TextToSpeech/ReferenceData/uniaudio_generation_reference.json
```

`UniAudioGenerationReferenceTests` checks these length and termination results through the public synthesis API. See [the generation review](../../docs/text-to-speech/uniaudio-generation-review.md), including the native empty-output boundary.

XTTS-v2's policy and EOS latent-replay reference uses original Coqui methods and its supported HF lower bound:

```text
python tools/reference-data/xtts_sampling_reference.py <coqui-TTS-checkout> <transformers-4.33.0-py3-none-any.whl> tests/AiDotNet.Tests/TextToSpeech/ReferenceData/xtts_sampling_reference.json
```

The uninstalled wheel supplies original sampling processors. Original Coqui prefix construction, generated-ID slicing and padded latent replay run with controlled/tagged neural outputs. `XttsSamplingReferenceTests` checks distributions, actual reduced GPT generation and the option boundary; see [the XTTS sampling review](../../docs/text-to-speech/xtts-sampling-review.md).

The community VALL-E X controller reference runs the original AR loop and sampler helpers:

```text
python tools/reference-data/vallex_generation_reference.py <VALL-E-X-checkout> tests/AiDotNet.Tests/TextToSpeech/ReferenceData/vallex_generation_reference.json
```

`VALLEXGenerationReferenceTests` compares the canonical cloning generation path and sampler support with these controlled outputs. See [the generation review](../../docs/text-to-speech/vallex-generation-review.md) for the BOS-dependent cap, zero-mass sampling regression and community/Microsoft distinction.

DiTTo's speech-length reference uses standard PyTorch transformer-decoder operators:

```text
python tools/reference-data/ditto_length_reference.py
```

It generates complete BOS/prompt decoder outputs and all 59 learned-parameter gradients, loss/gradient stress cases, and expected-length boundary cases. `DiTToLengthReferenceTests` and `DiTToLengthBoundaryReferenceTests` verify the native decoder and actual synthesis length handling. See [the length review](../../docs/text-to-speech/ditto-length-review.md) for the reproduced failures and the private-implementation limits.

These smaller fixtures complement the original-checkpoint modes. They use independent PyTorch operators and autograd, with reduced dimensions and all disclosed branches retained. They are explicitly not unreleased author checkpoints.

```text
python tools/reference-data/fastspeech2_acoustic_reference.py
python tools/reference-data/adaspeech_acoustic_reference.py
python tools/reference-data/aligntts_acoustic_reference.py
python tools/reference-data/e3_unet_reference.py
python tools/reference-data/voicebox_network_reference.py
python tools/reference-data/seed_diffusion_reference.py
python tools/reference-data/ditto_diffusion_reference.py <original-DiT-models.py>
```

The FastSpeech/AdaSpeech/AlignTTS generators require NumPy, SciPy, and pycwt 0.5.0b0 for the vendored original NATSpeech CWT routine. AdaSpeech and AlignTTS reuse the independent backbone and also regenerate its stable fixtures. Other new network fixtures require PyTorch (audited with CPU PyTorch 2.6). DiTTo executes the unmodified original `TimestepEmbedder` from `facebookresearch/DiT@ed81ce2229091fd4ecc9a223645f95cf379d582b`, records the source SHA-256, and retains the original float32 timestep result separately from explicitly promoted double-operator precision.

The repository tests are `FastSpeech2AcousticReferenceTests` (including AdaSpeech), `AlignTtsAcousticReferenceTests`, `E3UnetReferenceTests`, `VoiceboxNetworkReferenceTests`, and `SeedDiffusionReferenceTests`/`DiTToDiffusionReferenceTests`. Each accounts for every learned tensor and checks complete outputs, objectives and gradients; explicit unconditioned branches require zero gradients for dropped inputs. Research/source boundaries and the failures found are recorded under `docs/text-to-speech/*-review.md`.

## Original CosyVoice and CosyVoice 2

The original `CosyVoiceClone` wraps the 300M CosyVoice model. Use author source `8555549e882236e6541748b1042d95693caa82ba` (tag v2.0), weights `FunAudioLLM/CosyVoice-300M@24c40509c3c5ea6fe06b5f8790ff99e3714a6bee`, and the original `llm.pt`, `flow.pt`, `hift.pt`, `speech_tokenizer_v1.onnx`, and `campplus.onnx`. Modern legacy frontend code hardcodes a different feature sample rate; it must not be substituted for the original 300M oracle. Generate the following files in one reference directory:

```text
python tools/reference-data/cosyvoice_pretrained_reference.py <v2-author-checkout> <weights> <references>/cosyvoice1-pretrained-reference.json
python tools/reference-data/cosyvoice_flow_pretrained_reference.py <v2-author-checkout> <matcha-checkout> <weights> <references>/cosyvoice1-flow-pretrained-reference.json
python tools/reference-data/cosyvoice_hift_pretrained_reference.py <v2-author-checkout> <weights> <references>/cosyvoice1-flow-pretrained-reference.json <references>/cosyvoice1-hift-pretrained-reference.json
python tools/reference-data/cosyvoice_released_frontend_reference.py <v2-author-checkout> <weights> <references>/cosyvoice1-frontend-reference.json 1 <matcha-checkout>
dotnet run --project tools/TtsCheckpointAudit -- cosyvoice1-all <weights> <references>
dotnet run --project tools/TtsCheckpointAudit -- cosyvoice1-frontend <weights> <references>/cosyvoice1-frontend-reference.json
```

The complete audit constructs the original default cloning facade and loads every original learned checkpoint and both frozen ONNX graphs through its public loader. It checks public prompted flow and actual stochastic waveform generation. Only random draws are replayed, including harmonic phase and source noise; no model forward is substituted. The strict double comparison explicitly isolates nonlearned derived buffers and forced float32 casts. Original float32 outputs remain in the reference. Frontend generation requires the author's Whisper 20231117, NumPy 1.26, and ORT 1.18; small CAM++ differences between ORT versions are checked separately.

For canonical CosyVoice 2 use author source `074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc` and `FunAudioLLM/CosyVoice2-0.5B@eec1ae6c79877dbd9379285cf8789c9e0879293d`. Its component modes are `cosyvoice2-lm <weights> <reference-json>` and `cosyvoice2-frontend <weights> <reference-json>`. The frontend generator above takes variant `2`; its release uses `speech_tokenizer_v2.onnx`, the 151,663-token BlankEN tokenizer, and a 24 kHz mel frontend with a 12 kHz cutoff. A component result must not be described as a complete CosyVoice 2 facade result.

UniAudio uses the original `Dongchao/UniAudio` revision `835f07dcf1bdf586b0f427be59e4d48d0320f00e`: `uniaudio_v1.zip` contains `config.yaml` and `ep5-iter340000.checkpoint`; `universal_model.zip` contains the codec's `config.yaml` and `model.pth`. Preserve the latter pair under `universal_model/`. The pinned author source is `c1146a8cecf84f1c2e877170d84998393b7e1e78`.

```text
python tools/reference-data/uniaudio_pretrained_reference.py <author-checkout> <weight-directory> <uniaudio-reference.json>
python tools/reference-data/uniaudio_codec_pretrained_reference.py <author-checkout> <weight-directory> <uniaudio-codec-reference.json>
dotnet run --project tools/TtsCheckpointAudit -- uniaudio <weight-directory> <uniaudio-reference.json> <uniaudio-codec-reference.json>
```

`UniAudioOptions.OfficialV1()` follows the actual archive rather than the README announcement: 24 global and eight local layers, width 1,536, 16 heads, 4,215 tokens, 8,000 global positions, causal attention, and a 256-dimensional eight-codebook codec selecting three streams at 1.5 kbps. The original model and codec wrappers are imported directly; optimizer storages are skipped before allocation. The source generator retains the original float32 outputs and a double run with the author's forced-float32 normalization and softmax. A separately labeled operator-precision run uses TorchDispatchMode solely to retain double precision at those explicit casts; it replaces no author neural forward method or learned tensor and must meet the strict double tolerance. Original cross-precision deltas and every argmax remain reported/checked separately. The independent frontend/objective generator executes the original Kaldi lexicon method, alignment-phone tokenizer, loss and autograd; its small fixtures ship with the tests.

VoiceCraft uses `pyp1/VoiceCraft` revision `52b0f909d652ea016966804767320f13ede39abd`, including the original `giga830M.pth` and `encodec_4cb2048_giga.th`. The independent generators execute the pinned VoiceCraft source (`f0a7a971363905883a21c466a3aa4ea2d4c0b690`) and AudioCraft 1.3.0 codec source:

```text
python tools/reference-data/voicecraft_pretrained_reference.py <voicecraft-checkout> <weight-directory> <voicecraft-reference.json>
python tools/reference-data/voicecraft_codec_pretrained_reference.py <audiocraft-source-root> <weight-directory> <voicecraft-codec-reference.json>
dotnet run --project tools/TtsCheckpointAudit -- voicecraft <weight-directory> <voicecraft-reference.json> <voicecraft-codec-reference.json>
```

The public loader reads the original Transformer `model` wrapper and the codec's `best_state.model`, rather than optimizer or last-step weights. Every phone ID and prediction-head argmax is checked, along with all logits, exact four-codebook encoding and waveform decoding. The released codec uses 128 latent channels, 64 base filters, noncausal weight-normalized convolutions, constant padding, identity residual skips and residual kernels `[3,1]`. The original float32 and float64 network outputs are retained. Because the author's positional tables are constructed explicitly in float32 and then upcast, a separately identified comparison calculates only those non-learned tables in float64; all author forward methods and learned weights remain unchanged. That additional comparison must meet the strict float64 tolerance; the original upcast-table delta is reported separately.

Spark-TTS uses `SparkAudio/Spark-TTS-0.5B` revision `642071559bfc6346c2359d19dcb6be3f9dd8a05d`, retaining the original `LLM`, `BiCodec`, and `wav2vec2-large-xlsr-53` directory layout and configuration/tokenizer files. Generate independent outputs with the original SparkAudio source and its dependencies, then run the native comparison:

```text
python tools/reference-data/spark_pretrained_reference.py <author-checkout> <weight-directory> <spark-reference.json>
python tools/reference-data/spark_xlsr_precision_reference.py <weight-directory> <spark-reference.json>
dotnet run --project tools/TtsCheckpointAudit -- spark <weight-directory> <spark-reference.json>
```

The generator executes complete original pretrained Qwen, XLSR, and BiCodec networks. It caches completed stages outside the repository with source, script, and checkpoint fingerprints. The comparison checks tokenizer IDs, full-vocabulary next-token logits, XLSR feature mixing, semantic encoding and discrete IDs, speaker tokens, conditional prenet/postnet, and waveform reconstruction. Controlled token prefixes isolate arithmetic from stochastic sampling. A successful comparison does not certify free-running generation or reproduce the original training run.

Spark's full comparison retains the original upcast-float32 mel-buffer outputs and separately generates the original frontend with float64 buffers. Both numerical differences are reported. The `spark-speaker` mode isolates ECAPA, Perceiver, FSQ, and speaker reconstruction to diagnose differences without initializing the full Qwen/XLSR networks; it also enforces numerical thresholds and exact global IDs.

The separate XLSR precision generator preserves the original float32 extraction results and additionally executes the unchanged HF normalization function and original complete backbone in float64. The native comparison enforces strict float64 normalization/feature agreement and independently enforces exact discrete codes and waveform agreement with the original float32-conditioned pipeline. The cross-precision hidden-feature delta is reported explicitly, rather than certified as float64 arithmetic equivalence.

CosyVoice3's full pretrained LM comparison uses `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` revision `29e01c4e8d000f4bcd70751be16fa94bf3d85a18`, with its original `llm.pt` and `CosyVoice-BlankEN` configuration/tokenizer files:

```text
python tools/reference-data/cosyvoice3_pretrained_reference.py <author-checkout> <weight-directory> <cosy-reference.json>
dotnet run --project tools/TtsCheckpointAudit -- cosyvoice3-lm <weight-directory> <cosy-reference.json>
```

This mode calls the production `LoadPretrainedWeights` on the original directory: `llm.pt`, `flow.pt`, `hift.pt`, `speech_tokenizer_v3.onnx`, and `campplus.onnx`. Converted safetensors remain supported. It checks tokenizer IDs, exact autoregressive targets, input embeddings, complete speech logits, and normalized loss. When `cosyvoice3-frontend-pretrained-reference.json` is present beside the LM reference, it also compares public `CreateVoice` with the original `frontend_zero_shot` preparation.

The complete original pretrained CosyVoice3 flow has a separate controlled-noise comparison. The 56-frame fixture spans the released 50-frame streaming chunk boundary. It executes the authors' full 22-block DiT and unchanged ten-step guided Euler solver in both offline and streaming modes:

```text
python tools/reference-data/cosyvoice3_flow_pretrained_reference.py <author-checkout> <weight-directory> <cosy-flow-reference.json>
dotnet run --project tools/TtsCheckpointAudit -- cosyvoice3-flow <weight-directory> <cosy-flow-reference.json>
```

The flow comparison retains the original float32 and upcast-float64 rotary buffers. It separately executes every unchanged author method with a float64 reconstruction of the derived rotary-frequency buffer, matching native double arithmetic. Both results are reported. A read-only Python return trace captures the solver's final value before its mandatory float32 output cast; no solver or neural forward method is replaced.

The frozen prompt frontend and pretrained waveform decoder have separate checks:

```text
python tools/reference-data/cosyvoice3_frontend_pretrained_reference.py <author-checkout> <weight-directory> <reference-directory>/cosyvoice3-frontend-pretrained-reference.json <matcha-checkout>
dotnet run --project tools/TtsCheckpointAudit -- cosyvoice3-frontend <weight-directory> <reference-directory>/cosyvoice3-frontend-pretrained-reference.json
python tools/reference-data/cosyvoice3_hift_pretrained_reference.py <author-checkout> <weight-directory> <cosy-flow-reference.json> <cosy-hift-reference.json>
dotnet run --project tools/TtsCheckpointAudit -- cosyvoice3-vocoder <weight-directory> <cosy-hift-reference.json>
```

The frontend generator uses the author's pinned NumPy 1.26.4, ONNX Runtime 1.18.0, and openai-whisper 20231117. It executes the original speech-token, speaker, mel, and zero-shot preparation methods for 16/24 kHz inputs. Original float32 frontend outputs remain intact; separately labeled float64-derived Kaldi buffers isolate native arithmetic from frontend rounding. Native ONNX runtime differences are measured by feeding identical features to the exact same checkpoint. CausalHiFT checks F0 and waveform generation using identical original stochastic excitation.

For the original English Chatterbox release, obtain `tokenizer.json`, `ve.safetensors`, `t3_cfg.safetensors` and `s3gen.safetensors` from `ResembleAI/chatterbox` revision `5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18`. Use author source revision `5de7a54aa4e5e2baadb0182dde554908b48b85c2` and its dependencies to generate reference outputs:

```text
python tools/reference-data/chatterbox_pretrained_reference.py <author-checkout> <weight-directory> <t3-reference.json>
python tools/reference-data/chatterbox_s3gen_pretrained_reference.py <author-checkout> <weight-directory> <s3gen-reference.json>
dotnet run --project tools/TtsCheckpointAudit -- chatterbox <weight-directory> <t3-reference.json>
dotnet run --project tools/TtsCheckpointAudit -- chatterbox-mel unused <s3gen-reference.json>
python tools/reference-data/chatterbox_mel_precision_reference.py <author-checkout> <s3gen-reference.json>
python tools/reference-data/chatterbox_cam_precision_reference.py <author-checkout> <weight-directory> <s3gen-reference.json>
python tools/reference-data/chatterbox_s3gen_pretrained_reference.py <author-checkout> <weight-directory> <s3gen-reference.json>
dotnet run --project tools/TtsCheckpointAudit -- chatterbox-cam <weight-directory> <s3gen-reference.json>
dotnet run --project tools/TtsCheckpointAudit -- chatterbox-s3gen <weight-directory> <s3gen-reference.json>
```

The T3 comparison covers the original tokenizer, complete pretrained voice encoder, full 520M transformer and the native waveform-derived speaker conditioning chain. It checks numerical logits and exact speech argmax IDs. S3Gen separately compares reference token IDs, mel features, CAM++ embeddings, complete guided flow integration, F0 prediction and HiFT decoding. The author flow runs in its supported float32 precision. Flow noise and the original stochastic vocoder excitation are supplied identically to both implementations so different random number generators cannot hide or create a mismatch.

Checkpoint and generated fixture files are several gigabytes and stay outside Git. The executable uses the existing `AiDotNetTests` friend assembly access to inspect neural components without adding public diagnostic APIs. Running these comparisons does not certify other TTS models; see the 18-model review register in `docs/text-to-speech/pr2294-adversarial-review.md`.

The full-model audit disables automatic weight streaming before changing execution mode: inference-store quantization would change the checkpoint precision and invalidate a strict arithmetic comparison. The original public checkpoint loader and public reference-preparation API still run unchanged.

The original S3Gen frontend uses float32 Hann/filterbank buffers, FFT arithmetic and forced float32 CAM++ features. On the deliberately low-energy reference, native double arithmetic differs by 0.00492 in log mel, 0.000711 in the speaker embedding and 0.000240 in the subsequent flow output. These original-release differences are always reported. Separate, explicitly labeled author-function variants verify native double frontend computations at matching precision; the additional original float32 flow run uses those independently generated features to isolate neural arithmetic from the frontend difference. The original float32 waveform, features and flow results remain present and are not overwritten by the variants.

## TorToiSe original inference checkpoints

Obtain `.models/autoregressive.pth`, `.models/clvp2.pth`, `.models/diffusion_decoder.pth` and `.models/vocoder.pth` from the author's `jbetker/tortoise-tts-v2` revision `b20a372926b4d6132bcec0a7087f5dd14c8d9e10`, preserving the `.models` subdirectory. The reference uses the original `neonbjb/tortoise-tts` classes and records the checkout revision in its JSON.

```text
python tools/reference-data/tortoise_pretrained_reference.py <author-checkout> <weight-directory> <tortoise-reference.json>
dotnet run --project tools/TtsCheckpointAudit -- tortoise <weight-directory> <tortoise-reference.json>
```

The audit imports the original PyTorch files with `WeightLoader`, validates requested tensor axes, and compares full AR conditioning/logits/latents, CLVP scores, diffusion conditioning/alignment/epsilon/variance and UnivNet samples. The original AR `GroupNorm32` forcibly casts its input to float32; the labeled double reference retains those normalization parameters in float32 so the unchanged author method can run. This comparison covers the four released inference networks; it does not claim a pretrained dVAE training-checkpoint comparison or uncontrolled random-generation equivalence.

## XTTS-v2 original checkpoint

Use the original `model.pth`, `dvae.pth`, `vocab.json` and `mel_stats.pth` from `coqui/XTTS-v2` revision `6c2b0d75eae4b7047358e3b6bd9325f857d43f77`, with Coqui source revision `dbf1a08a0d4e47fdad6172e433eeb34bc6b13b4e`.

```text
python tools/reference-data/xtts_pretrained_reference.py <author-checkout> <weight-directory> <xtts-reference.json>
dotnet run --project tools/TtsCheckpointAudit -- xtts <weight-directory> <xtts-reference.json>
```

The native audit instantiates the cloning facade and calls its inherited public `LoadAuthorWeights` on the original wrapped files. It compares actual English/Hindi tokenizer IDs, full GPT conditioning/hidden states/logits, dVAE code IDs and reconstruction, H/ASP waveform speaker embeddings and speaker-conditioned HiFi-GAN waveforms. Fixed text/audio token streams isolate neural correctness from different sampling RNGs. The reference labels its original forced-float32 GroupNorm behavior; the dVAE's three absent entropy-diagnostic accumulator buffers are the only allowed missing dVAE state entries, with all original neural/EMA tensors required. Trainer-config objects are inert containers during restricted checkpoint deserialization and never replace neural modules.
