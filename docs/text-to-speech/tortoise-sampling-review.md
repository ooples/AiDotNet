# TorToiSe complete diffusion sampler review

The primary implementation is [the author's diffusion process](https://github.com/neonbjb/tortoise-tts/blob/8a2563ecabe93c4fb626f876dd0c52c966edef2f/tortoise/utils/diffusion.py), at the same revision as the existing four-checkpoint audit. This check executes `SpacedDiffusion.p_sample_loop_progressive` and the complete original `DiffusionTts` network. It does not replace a neural forward or substitute independently computed denoising statistics.

`tools/reference-data/tortoise_sampling_reference.py` records the original initial noise and every per-step Gaussian draw, including the final unused draw. The native paired Box-Muller transform receives inverse-transformed uniform draws. Thus the actual native network, conditioning normalization, frame-rate conversion, timestep spacing, learned-range variance, clipping, classifier-free guidance ramp, stochastic transitions and mel denormalization execute normally. The reference stores every original intermediate sample and the final mel output.

The two reduced tests cover three denoising steps with guidance coefficients zero and two. Both pass against the production assembly within 2e-4 of the original float32 author output, in the actual repository test project on both `net8.0` and `net471` (zero failures or skips). Native computation uses double precision, so the comparison explicitly retains the original float32 operator boundary. These reduced cases alone are not a claim of trained checkpoint or perceptual quality equivalence.

The released mode imports the complete original `diffusion_decoder.pth` from `jbetker/tortoise-tts-v2`, revision `b20a372926b4d6132bcec0a7087f5dd14c8d9e10`, and uses the standard 200-step schedule. Both original and native full-checkpoint runs pass. The final mel maximum error is `6.958992046968149e-5` against original float32 author computation, below the predeclared accumulated-trajectory tolerance of `2e-3`. All 261,300 replayed uniform draws are consumed exactly. The production Build44 assembly executes the native sampler. The checkpoint SHA-256, verified before comparison, is `ea776fc354eabb70cfae145777153483fad72e3e0c5ea345505ded2231a90ce1`. The result closes the complete trained multi-step diffusion-sampling gap; it does not infer full stochastic autoregressive generation or corpus training equivalence.

Reproduce the reduced fixture with:

```powershell
python tools/reference-data/tortoise_sampling_reference.py <original-author-checkout>
dotnet test tests/AiDotNet.Tests/AiDotNetTests.csproj -f net8.0 --filter FullyQualifiedName~TortoiseSamplingReferenceTests
```

For the complete released checkpoint:

```powershell
python tools/reference-data/tortoise_sampling_reference.py <original-author-checkout> <checkpoint-directory> <output-json>
dotnet run --project tools/TtsCheckpointAudit -- tortoise-sampling <checkpoint-directory> <output-json>
```

The checkpoint directory must contain `.models/diffusion_decoder.pth`. Reference inputs are three autoregressive latent frames and twelve speaker-mel frames, yielding thirteen output frames at the release's 24 kHz / 22.05 kHz conversion. These controlled inputs test the complete diffusion sampler and do not certify full free-running autoregressive candidate generation, ranking or perceptual quality.
