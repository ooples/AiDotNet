Full HiFi-GAN V1 equivalence is checked separately from timing and allocation measurements.
The exporter reuses `benchmark.py`'s model and actual `train_step`, with all84,660,721 learned
parameters,78 generator layers and54 discriminator layers. It compiles the same generator and
discriminators that the benchmark compiles. `--eager` produces a diagnostic reference that the
completion verifier deliberately refuses.

After building the harness against the candidate AiDotNet and Tensors sources, run these commands
on the idle benchmark machine, once for `cpu` and once for `cuda`:

```powershell
python benchmarks/AiDotNet.PyTorchParity/pytorch/export_hifigan_reference.py --device cuda --out ../perf2561/hifigan-reference-cuda
dotnet benchmarks/AiDotNet.PyTorchParity/bin/Release/net10.0/AiDotNet.PyTorchParity.dll --device cuda --verify-hifigan ../perf2561/hifigan-reference-cuda --output ../perf2561/hifigan-equivalence-cuda.json
```

The reference contains fixed input mels and audio, matching initial weights and spectral u/v,
generator inference at batches1/8/16 with the benchmark's full256 frames, isolated discriminator
and generator objectives with every learned tensor's gradient, separate adversarial/feature
matching/mel terms, and every common weight/state tensor after3 actual GAN iterations. Each
iteration supplies exactly32 mel frames and8192 audio samples, so both normal training entry
points crop at frame0. The input changes each iteration. Isolated loss/gradient checks restore
the initial state between objectives; the trajectory checks run the actual production `Train`
and the actual reference `train_step`, retaining each optimizer's history throughout.

The verifier imports through the existing streamed checkpoint source and checks exact field
coverage, tensor shapes, full parameter count, reference file hashes and device. It requires
all6 discriminator/generator updates to remain fused with0 eager steps. Every tensor must
meet both maximum-error and relative-L2 checks. CPU tolerances are absolute2e-5/relative1e-3;
CUDA tolerances are absolute2e-4/relative3e-3. In addition, aggregate gradient relative error
and norm-ratio error must be at most1% CPU/3% CUDA, with cosine at least0.999/0.995. These
budgets are explicit in the receipt. Reference TF32 is disabled; candidate CUDA kernels retain
their actual precision policy. Tolerances must not be widened merely to turn a failed run green.

Exit0 and a receipt with `pass: true` require every check. The receipt records individual
errors, aggregate gradient checks, fused/eager counts, reference-manifest hash, engine identity
and candidate assembly hashes. It provides numerical evidence; speed, allocations and peak
memory still have to pass the independent repeated performance protocol.
