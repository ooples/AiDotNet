# #1662 Backward-pass Optimizations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut per-step training peak-RSS and allocation on the CPU backward path, and make AiDotNet *handily beat* PyTorch CPU on per-step time / peak RSS / allocation, via three levers: backward buffer-reuse audit (#4), FlashAttention-style tiled backward (#3), and promoting single-pass fused optimizer-in-backward to the common-case default (#1).

**Architecture:** Cross-repo, Tensors-first. Levers #4 and #3 are changes in **AiDotNet.Tensors** (`BackwardFunctions.cs`, `FusedAttention.cs`, `ConvParallelProbe`) shipped in a new Tensors release (next after v0.101.6, e.g. **v0.102.0**). Lever #1 is an **AiDotNet** change in `NeuralNetworkBase` that bumps the Tensors pin and changes *when* the existing `streamingOptimizer.Apply` path engages, plus an opt-in fast-clip and a PyTorch-comparison proof harness. Lever #2 (checkpointing) is explicitly out of scope — owned by open PR #1633.

**Tech Stack:** C# (net10.0 + net471), xUnit, `AiDotNet.Tensors` autodiff (`GradientTape`, `FusedAttention`, `BackwardFunctions`), `ConvParallelProbe` CLI, Python+PyTorch (CPU) for the baseline harness.

**Worktrees (already created):**
- Tensors: `C:\Users\yolan\source\repos\AiDotNet.Tensors\.claude\worktrees\pr1662` on `perf/1662-backward` (off `main`).
- AiDotNet: `C:\Users\yolan\source\repos\AiDotNet\.claude\worktrees\pr1662` on `perf/1662-optimizer-in-backward` (off `master`).

**Spec:** `docs/superpowers/specs/2026-06-22-1662-backward-pass-optimizations-design.md`

**Build/test commands (run from the relevant worktree):**
- Tensors build: `dotnet build src/AiDotNet.Tensors/AiDotNet.Tensors.csproj --no-restore`
- Tensors tests: `dotnet test tests/AiDotNet.Tensors.Tests/AiDotNet.Tensors.Tests.csproj --no-build --filter "FullyQualifiedName~<Name>"`
- AiDotNet build: `dotnet build --no-restore`
- AiDotNet tests: `dotnet test tests/AiDotNet.Tests/AiDotNetTests.csproj --no-build --filter "FullyQualifiedName~<Name>"`

---

## Phase 1 — Lever #4: backward buffer-reuse audit (Tensors repo)

Cheapest, immediate signal. All tasks in the **Tensors** worktree. Produces the `--trainbench` probe (also the #1 proof harness), arena-bypass fixes, a guard test, and the fused grad-norm reduction helper used by #1's clipped path.

### Task 1.1: `--trainbench` probe mode (measurement first)

**Files:**
- Modify: `tools/ConvParallelProbe/Program.cs` (add dispatch + `RunTrainbench`)

- [ ] **Step 1: Add the dispatch line.** In `Main`, alongside the existing `--attnblock` dispatch (line ~28), add:

```csharp
if (args.Length > 0 && args[0] == "--trainbench") return RunTrainbench(eng, args);
```

- [ ] **Step 2: Implement `RunTrainbench`.** Add this method next to `RunAttnBlock`. It builds a small transformer block, runs `forward → backward → optimizer-step` in a loop using the tape, and reports the three proof metrics. Use the same arg helpers (`ArgI`, `HasFlag`) already in the file.

```csharp
// #1662 lever #4/#1 proof harness: per-step TRAINING cost (fwd+bwd+step) for a
// transformer block. Emits machine-readable metrics so trainbench_torch.py (same
// shape/optimizer) can be diffed against it. Mirrors --attnblock's shape knobs.
private static int RunTrainbench(CpuEngine eng, string[] a)
{
    int maxdop = ArgI(a, "--maxdop", Environment.ProcessorCount);
    int S = ArgI(a, "--s", 128);       // sequence length
    int D = ArgI(a, "--d", 384);       // model dim (SimCSE acceptance shape)
    int H = ArgI(a, "--h", 6);         // heads
    int layers = ArgI(a, "--layers", 10);
    int reps = ArgI(a, "--reps", 20);
    int warmup = ArgI(a, "--warmup", 5);
    if (maxdop < 1 || S < 1 || D < 1 || H < 1 || layers < 1 || reps < 1)
    {
        Console.Error.WriteLine("trainbench: --maxdop,--s,--d,--h,--layers,--reps must be >= 1.");
        return 1;
    }
    if (D % H != 0) { Console.Error.WriteLine($"trainbench: --d ({D}) must be a multiple of --h ({H})."); return 1; }
    CpuParallelSettings.MaxDegreeOfParallelism = maxdop;

    var rng = new Random(0);
    // Build `layers` worth of weights once (W_qkv, W_o, W_ff1, W_ff2 per layer).
    var step = new TrainbenchStep(eng, S, D, H, layers, rng);

    for (int i = 0; i < warmup; i++) step.RunStep();

    var times = new double[reps];
    long allocStart = GC.GetAllocatedBytesForCurrentThread();
    long peakWs = 0;
    for (int i = 0; i < reps; i++)
    {
        var sw = Stopwatch.StartNew();
        step.RunStep();
        sw.Stop();
        times[i] = sw.Elapsed.TotalMilliseconds;
        peakWs = Math.Max(peakWs, Process.GetCurrentProcess().WorkingSet64);
    }
    long allocTotal = GC.GetAllocatedBytesForCurrentThread() - allocStart;
    Array.Sort(times);
    double perStepAllocMB = allocTotal / (double)reps / (1024.0 * 1024.0);
    // Machine-readable single line for diffing against trainbench_torch.py.
    Console.WriteLine(
        $"TRAINBENCH engine=aidotnet S={S} D={D} H={H} layers={layers} maxdop={maxdop} " +
        $"median_ms={times[reps / 2]:F3} min_ms={times[0]:F3} " +
        $"alloc_mb_per_step={perStepAllocMB:F3} peak_ws_mb={peakWs / (1024.0 * 1024.0):F1}");
    return 0;
}
```

- [ ] **Step 3: Implement `TrainbenchStep`** as a private nested class in `Program.cs` that owns the weights and runs one `forward → backward → SGD-step` using the public tape API. Keep it a *real* transformer block (MHA via `FusedAttention.Forward`/`Backward` + a 2-layer MLP), reusing buffers across steps so only genuine per-step churn shows up.

```csharp
// Holds layer weights and runs one training step on the tape. SGD (not Adam) so
// the optimizer step itself contributes ~no allocation — the metric isolates the
// fwd+bwd allocation, which is what lever #4 targets.
private sealed class TrainbenchStep
{
    private readonly CpuEngine _eng;
    private readonly int _s, _d, _h;
    private readonly Tensor<float>[] _wQkv, _wO, _wF1, _wF2; // per layer
    private readonly Tensor<float> _x;
    private const float Lr = 1e-3f;

    public TrainbenchStep(CpuEngine eng, int s, int d, int h, int layers, Random rng)
    {
        _eng = eng; _s = s; _d = d; _h = h;
        _x = Rand(new[] { 1, s, d }, rng);
        _wQkv = new Tensor<float>[layers]; _wO = new Tensor<float>[layers];
        _wF1 = new Tensor<float>[layers]; _wF2 = new Tensor<float>[layers];
        for (int l = 0; l < layers; l++)
        {
            _wQkv[l] = Rand(new[] { d, 3 * d }, rng);
            _wO[l]   = Rand(new[] { d, d }, rng);
            _wF1[l]  = Rand(new[] { d, 4 * d }, rng);
            _wF2[l]  = Rand(new[] { 4 * d, d }, rng);
        }
    }

    public void RunStep()
    {
        using var tape = new GradientTape<float>();
        tape.Watch(_wQkv); tape.Watch(_wO); tape.Watch(_wF1); tape.Watch(_wF2);
        var hOut = _x;
        for (int l = 0; l < _wQkv.Length; l++)
            hOut = TransformerBlockForward(hOut, l);
        // Scalar MSE-to-zero loss so the graph is fully connected.
        var loss = _eng.TensorMultiply(hOut, hOut);
        loss = _eng.Sum(loss);
        var sources = new List<Tensor<float>>();
        for (int l = 0; l < _wQkv.Length; l++) { sources.Add(_wQkv[l]); sources.Add(_wO[l]); sources.Add(_wF1[l]); sources.Add(_wF2[l]); }
        tape.ComputeGradientsStreaming(loss, sources, (src, g) =>
        {
            if (g is null || g.Length == 0) return;
            var sp = src.Data.Span; var gs = g.Data.Span;
            for (int i = 0; i < sp.Length; i++) sp[i] -= Lr * gs[i]; // SGD
        });
    }

    private Tensor<float> TransformerBlockForward(Tensor<float> x, int l)
    {
        // qkv = x @ W_qkv ; split to q,k,v ; FusedAttention ; @ W_o ; + residual ; MLP ; + residual
        var qkv = _eng.TensorMatMul(x, _wQkv[l]);
        var (q, k, v) = SplitQkv(qkv);
        var attn = FusedAttention<float>.Forward(q, k, v, new FlashAttentionConfig(), null, _eng);
        var o = _eng.TensorMatMul(attn, _wO[l]);
        var res1 = _eng.TensorAdd(x, o);
        var f = _eng.TensorMatMul(res1, _wF1[l]);
        f = _eng.TensorGelu(f);
        f = _eng.TensorMatMul(f, _wF2[l]);
        return _eng.TensorAdd(res1, f);
    }

    private (Tensor<float>, Tensor<float>, Tensor<float>) SplitQkv(Tensor<float> qkv)
    {
        // [1,S,3D] -> three [1,H,S,Dh]; use engine slice + reshape (tape-tracked).
        int dh = _d / _h;
        var q = ReshapeHeads(_eng.SliceLastDim(qkv, 0, _d), dh);
        var k = ReshapeHeads(_eng.SliceLastDim(qkv, _d, _d), dh);
        var vv = ReshapeHeads(_eng.SliceLastDim(qkv, 2 * _d, _d), dh);
        return (q, k, vv);
    }

    private Tensor<float> ReshapeHeads(Tensor<float> t, int dh) =>
        _eng.Transpose(_eng.Reshape(t, new[] { 1, _s, _h, dh }), new[] { 0, 2, 1, 3 });
}
```

> **Note for executor:** the exact engine method names (`SliceLastDim`, `TensorGelu`, `Transpose`, `Sum`, `Watch`) must be confirmed against the current `IEngine`/`CpuEngine`/`GradientTape` surface in this Tensors version; substitute the real names. The point of this task is a *running* probe, not these exact calls. If a tape-tracked slice helper is missing, fold the QKV projection into three separate `W_q/W_k/W_v` matmuls instead of one `W_qkv` + slice.

- [ ] **Step 4: Build and run the probe.**

Run: `dotnet run -c Release --project tools/ConvParallelProbe -- --trainbench --s 128 --d 384 --h 6 --layers 10`
Expected: one `TRAINBENCH engine=aidotnet ... median_ms=... alloc_mb_per_step=... peak_ws_mb=...` line. **Record this baseline number** — it is the before-audit figure.

- [ ] **Step 5: Commit.**

```bash
git add tools/ConvParallelProbe/Program.cs
git commit -m "perf(#1662): --trainbench probe — per-step training time/alloc/peak-RSS"
```

### Task 1.2: Backward arena-bypass audit (iterative, measurement-driven)

**Files:**
- Modify: `src/AiDotNet.Tensors/Engines/Autodiff/BackwardFunctions.cs`
- Modify: CpuEngine `*Backward` ops as found

- [ ] **Step 1: Enumerate bypass candidates.** Run these greps and record the hit list:

```bash
grep -nE "new Tensor<" src/AiDotNet.Tensors/Engines/Autodiff/BackwardFunctions.cs
grep -nE "new T\[|new float\[|new double\[|new Vector<" src/AiDotNet.Tensors/Engines/Autodiff/BackwardFunctions.cs
grep -rnE "RentOrAllocate" src/AiDotNet.Tensors/Engines/Autodiff/
```

- [ ] **Step 2: For the single highest-frequency bypass** (the one on the hottest backward op — matmul-backward / softmax-backward / gelu-backward), route its temporary through the per-step arena `Rent` instead of `new`. Follow the existing arena-`Rent` pattern already used elsewhere in the same file (find one with `grep -n "Rent(" src/AiDotNet.Tensors/Engines/Autodiff/BackwardFunctions.cs`). Fix exactly ONE op in this step.

- [ ] **Step 3: Re-run the probe to confirm the allocation dropped.**

Run: `dotnet run -c Release --project tools/ConvParallelProbe -- --trainbench --s 128 --d 384 --h 6 --layers 10`
Expected: `alloc_mb_per_step` is strictly lower than the Task 1.1 baseline, `median_ms` not worse.

- [ ] **Step 4: Run the autodiff suite to confirm grads unchanged.**

Run: `dotnet test tests/AiDotNet.Tensors.Tests/AiDotNet.Tensors.Tests.csproj --filter "FullyQualifiedName~Autodiff"`
Expected: all pass (grads bit-identical — arena `Rent` returns zeroed/owned buffers of the same shape).

- [ ] **Step 5: Commit, then repeat Steps 2-5 for the next-hottest bypass** until `alloc_mb_per_step` stops dropping materially (diminishing returns < ~2% per fix). Each fix is its own commit:

```bash
git add -A && git commit -m "perf(#1662): route <op>Backward temp through step arena (alloc -N%)"
```

- [ ] **Step 6: Log any deliberately-skipped allocations** (e.g. a temp that legitimately can't be pooled). The spec forbids silent caps — add a one-line `// #1662: not arena-pooled because <reason>` comment at each skipped site.

### Task 1.3: `BackwardArenaTests` guard (lock in the win)

**Files:**
- Create: `tests/AiDotNet.Tensors.Tests/Engines/Autodiff/BackwardArenaTests.cs`

- [ ] **Step 1: Write the failing test.** Model it on the AiDotNet-side `InferenceArenaForwardTests` (asserts per-step allocation stays under a bound after warmup). Here it asserts the per-step *backward* allocation of a fixed block is below a threshold derived from the post-audit number.

```csharp
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

public class BackwardArenaTests
{
    [Fact]
    public void StreamingBackward_PerStepAllocation_StaysBounded()
    {
        var eng = new CpuEngine();
        var rng = new Random(0);
        var w = Tensor<float>.FromArray(RandArray(256 * 256, rng), new[] { 256, 256 });
        var x = Tensor<float>.FromArray(RandArray(32 * 256, rng), new[] { 32, 256 });

        // Warm up arena + JIT.
        for (int i = 0; i < 5; i++) OneStep(eng, w, x);

        long start = GC.GetAllocatedBytesForCurrentThread();
        const int reps = 10;
        for (int i = 0; i < reps; i++) OneStep(eng, w, x);
        long perStep = (GC.GetAllocatedBytesForCurrentThread() - start) / reps;

        // Threshold = post-audit measured per-step backward alloc + 20% headroom.
        // Set THRESHOLD_BYTES from the Task 1.2 final number.
        const long thresholdBytes = THRESHOLD_BYTES;
        Assert.True(perStep < thresholdBytes,
            $"per-step backward alloc {perStep} >= threshold {thresholdBytes}");
    }

    private static void OneStep(CpuEngine eng, Tensor<float> w, Tensor<float> x)
    {
        using var tape = new GradientTape<float>();
        tape.Watch(w);
        var y = eng.TensorMatMul(x, w);
        var loss = eng.Sum(eng.TensorMultiply(y, y));
        tape.ComputeGradientsStreaming(loss, new[] { w }, (_, __) => { });
    }

    private static float[] RandArray(int n, Random r)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(r.NextDouble() - 0.5);
        return a;
    }
}
```

- [ ] **Step 2: Replace `THRESHOLD_BYTES`** with the measured post-audit per-step number (from Task 1.2) plus 20% headroom. Confirm method names (`Tensor<float>.FromArray`, `tape.Watch`, `eng.Sum`) against the current API.

- [ ] **Step 3: Run the test.**

Run: `dotnet test tests/AiDotNet.Tensors.Tests/AiDotNet.Tensors.Tests.csproj --filter "FullyQualifiedName~BackwardArenaTests"`
Expected: PASS.

- [ ] **Step 4: Sanity-check it fails when regressed** — temporarily set the threshold to a tiny value, run, confirm FAIL, then restore.

- [ ] **Step 5: Commit.**

```bash
git add tests/AiDotNet.Tensors.Tests/Engines/Autodiff/BackwardArenaTests.cs
git commit -m "test(#1662): BackwardArenaTests — guard per-step backward allocation"
```

### Task 1.4: Fused grad-norm reduction helper (for #1's clipped path)

**Files:**
- Modify: `src/AiDotNet.Tensors/Engines/Autodiff/GradientTape.cs` (overload of `ComputeGradientsStreaming` that also yields each source grad's sum-of-squares)

- [ ] **Step 1: Write the failing test.** A new test asserting the fused-norm overload produces the same global sum-of-squares as a manual span loop over the grads.

```csharp
[Fact]
public void ComputeGradientsStreaming_FusedNorm_MatchesManualSumOfSquares()
{
    var eng = new CpuEngine();
    var rng = new Random(1);
    var w = Tensor<float>.FromArray(RandArray(64 * 64, rng), new[] { 64, 64 });
    var x = Tensor<float>.FromArray(RandArray(8 * 64, rng), new[] { 8, 64 });

    double manual = 0, fused = 0;
    using (var tape = new GradientTape<float>())
    {
        tape.Watch(w);
        var loss = eng.Sum(eng.TensorMultiply(eng.TensorMatMul(x, w), eng.TensorMatMul(x, w)));
        tape.ComputeGradientsStreaming(loss, new[] { w }, (src, g) =>
        {
            var sp = g.Data.Span;
            for (int i = 0; i < sp.Length; i++) manual += (double)sp[i] * sp[i];
        });
    }
    using (var tape = new GradientTape<float>())
    {
        tape.Watch(w);
        var loss = eng.Sum(eng.TensorMultiply(eng.TensorMatMul(x, w), eng.TensorMatMul(x, w)));
        // New overload: 4th arg receives (source, sumOfSquaresOfThatGrad).
        tape.ComputeGradientsStreaming(loss, new[] { w },
            (src, g) => { },
            (src, ssq) => { fused += ssq; });
    }
    Assert.Equal(manual, fused, 4);
}
```

- [ ] **Step 2: Run it to confirm it fails** (overload doesn't exist).

Run: `dotnet test ... --filter "FullyQualifiedName~FusedNorm_MatchesManual"`
Expected: FAIL (no matching overload).

- [ ] **Step 3: Add the overload.** In `GradientTape.cs`, add a `ComputeGradientsStreaming(loss, sources, onSourceGradient, onSourceGradientSumSq)` overload. Reuse the existing single-arg body; just after each source gradient is produced (before release), compute its sum-of-squares in the same loop that already touches the grad buffer (it is hot in cache) and invoke `onSourceGradientSumSq(source, ssq)`.

- [ ] **Step 4: Run the test.** Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add -A && git commit -m "feat(#1662): ComputeGradientsStreaming fused sum-of-squares overload"
```

### Task 1.5: Release Tensors v0.102.0

- [ ] **Step 1:** Bump the Tensors package version (find the `<Version>` in `src/AiDotNet.Tensors/AiDotNet.Tensors.csproj` or `Directory.Build.props`) to `0.102.0` with a changelog line referencing #1662 levers #4 + #3 (after Phase 2 lands) + the fused-norm overload.
- [ ] **Step 2:** Push `perf/1662-backward`, open the Tensors PR, get it merged, and publish the NuGet release per the repo's release process. **Phase 3 depends on this version being live.**

---

## Phase 2 — Lever #3: FlashAttention-style tiled backward (Tensors repo)

Largest/most novel. All tasks in the **Tensors** worktree, on `perf/1662-backward` (before the Task 1.5 release so it ships in v0.102.0).

### Task 2.1: Parity test for tiled backward (TDD — write first)

**Files:**
- Create: `tests/AiDotNet.Tensors.Tests/Engines/Autodiff/FusedAttentionTiledBackwardTests.cs`

- [ ] **Step 1: Write the parity test.** It pins the *new* tiled backward against the *current* stored-P backward across seq lengths / head counts. Since we are replacing `Backward` in place, the test compares the tiled output to an analytic/reference value captured before the rewrite — so first capture the current outputs as the golden, then assert the rewrite still matches.

```csharp
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

public class FusedAttentionTiledBackwardTests
{
    [Theory]
    [InlineData(1, 2, 16, 8)]    // tiny
    [InlineData(2, 4, 64, 16)]   // medium
    [InlineData(1, 8, 256, 32)]  // long-ish sequence (where O(S) matters)
    public void TiledBackward_MatchesReference_dQdKdV(int B, int H, int S, int Dh)
    {
        var eng = new CpuEngine();
        var rng = new Random(7);
        var q  = Rand(new[] { B, H, S, Dh }, rng);
        var k  = Rand(new[] { B, H, S, Dh }, rng);
        var v  = Rand(new[] { B, H, S, Dh }, rng);
        var dO = Rand(new[] { B, H, S, Dh }, rng);
        var cfg = new FlashAttentionConfig();

        // Reference: a straightforward (non-tiled) analytic backward computed here
        // in the test from P = softmax(QK^T*scale), independent of the kernel.
        var (refQ, refK, refV) = ReferenceBackward(eng, dO, q, k, v, cfg);
        var (gQ, gK, gV) = FusedAttention<float>.Backward(dO, q, k, v, cfg, null, eng);

        AssertClose(refQ, gQ, 1e-4f);
        AssertClose(refK, gK, 1e-4f);
        AssertClose(refV, gV, 1e-4f);
    }

    // ReferenceBackward, Rand, AssertClose helpers: implement the dV=P^T dO,
    // dP=dO V^T, dS=P*(dP-rowsum(dP*P)), dQ=dS K*scale, dK=dS^T Q*scale chain
    // directly with engine batch-matmuls (the math in FusedAttention.Backward's
    // doc-comment). This is the "stored-P" formulation — the golden the tiled
    // kernel must reproduce.
}
```

- [ ] **Step 2: Implement `ReferenceBackward`/`Rand`/`AssertClose`** in the test using the exact formula from the current `FusedAttention.Backward` doc comment (lines 461-469). This makes the test self-contained and independent of the kernel under test.

- [ ] **Step 3: Run it against the CURRENT (stored-P) `Backward`.**

Run: `dotnet test ... --filter "FullyQualifiedName~FusedAttentionTiledBackward"`
Expected: PASS (current backward already matches the reference math). This proves the test + reference are correct *before* we change the kernel.

- [ ] **Step 4: Commit the test.**

```bash
git add tests/AiDotNet.Tensors.Tests/Engines/Autodiff/FusedAttentionTiledBackwardTests.cs
git commit -m "test(#1662): parity test for FusedAttention tiled backward"
```

### Task 2.2: Rewrite `FusedAttention.Backward` to tile over K (no stored S×S)

**Files:**
- Modify: `src/AiDotNet.Tensors/Engines/Autodiff/FusedAttention.cs:482-607`

- [ ] **Step 1: Add a `--backward-mem` measurement to the probe** (optional but recommended): extend `--attnblock` or add a tiny harness that reports peak working set during `FusedAttention.Backward` at `S=2048`. Record the current (stored-P) peak as the baseline.

- [ ] **Step 2: Replace the body of `Backward`** (keep the signature, the 3D-promote/demote wrapper, and the `EnsureSupportedElementType`/null guards). New algorithm — tile the keys into blocks of `Bk` (e.g. 128) and accumulate `dQ/dK/dV` without ever holding the full `[Sq, Sk]` matrix:

```csharp
// FlashAttention-2 style backward. Recompute per-(query-row, key-tile) scores,
// reuse the forward softmax normalizer so P is reconstructed tile-local, and
// accumulate dQ/dK/dV. Peak extra memory is O(Bq*Bk) per tile, never O(Sq*Sk).
//
// First pass over key-tiles per query-block computes the softmax denominator
// (online max + sumexp) AND the row-wise correction term
//   D_i = sum_k P_ik * dP_ik = rowsum(dO . O)            (cheap, O(S*Dh))
// so the second pass can form dS_ik = P_ik * (dP_ik - D_i) tile-locally.
//
// Concretely (per head, per query block i):
//   1. D_i = rowsum(dO_i * O_i)                         // needs O_i = forward output
//   2. for each key tile j:
//        S_ij = scale * Q_i @ K_j^T  (+ bias/causal)
//        P_ij = exp(S_ij - m_i) / l_i                    // m_i,l_i from forward stats
//        dV_j += P_ij^T @ dO_i
//        dP_ij = dO_i @ V_j^T
//        dS_ij = P_ij * (dP_ij - D_i)                    // D_i broadcast over the tile
//        dQ_i += scale * dS_ij @ K_j
//        dK_j += scale * dS_ij^T @ Q_i
```

Implementation notes for the executor:
- `O_i` (forward output) and the softmax stats `m_i`/`l_i` are needed. The cheapest correct approach that avoids storing S×S is: recompute `O` and the row-normalizers in a forward micro-pass per query block (FlashAttention-2 does exactly this). Reuse the existing `Forward` tiling helpers in this same file if present (`grep -n "private static" FusedAttention.cs` for tile helpers).
- Use the existing `engine.TensorBatchMatMul`, `TransposeLast2D`, `Reshape`, `TensorMultiplyScalar` already used in the current body — only the *materialization* changes (tile-local instead of full `P`).
- Keep `SoftmaxBackward`'s math (`dS = P*(dP - rowsum)`), but apply it tile-locally using the precomputed `D_i` instead of a full-matrix rowsum.

- [ ] **Step 3: Build.**

Run: `dotnet build src/AiDotNet.Tensors/AiDotNet.Tensors.csproj --no-restore`
Expected: builds clean.

- [ ] **Step 4: Run the Task 2.1 parity test.**

Run: `dotnet test ... --filter "FullyQualifiedName~FusedAttentionTiledBackward"`
Expected: PASS — dQ/dK/dV still bit-close (1e-4) to the reference across all three shapes.

- [ ] **Step 5: Run the full FusedAttention suite** (existing tests must not regress).

Run: `dotnet test ... --filter "FullyQualifiedName~FusedAttention"`
Expected: all PASS.

- [ ] **Step 6: Re-measure peak memory** (Step 1 harness) at `S=2048`. Expected: peak working set during backward drops materially vs the stored-P baseline (the O(S²)→O(S) win).

- [ ] **Step 7: Update the doc comment** — remove the "future work" note (lines 472-476) and describe the tiled algorithm. Commit.

```bash
git add src/AiDotNet.Tensors/Engines/Autodiff/FusedAttention.cs
git commit -m "perf(#1662): tiled FlashAttention backward — O(S) memory, no stored SxS"
```

> After Phase 2, complete Task 1.5 (release v0.102.0 carrying #4 + #3 + fused-norm).

---

## Phase 3 — Lever #1: optimizer-in-backward as the common-case default (AiDotNet repo)

All tasks in the **AiDotNet** worktree on `perf/1662-optimizer-in-backward`. Depends on Tensors v0.102.0 being published (Task 1.5).

### Task 3.1: Bump the Tensors pin

**Files:**
- Modify: `Directory.Packages.props`

- [ ] **Step 1:** Change `<PackageVersion Include="AiDotNet.Tensors" Version="0.94.2" />` to `0.102.0` and add a changelog comment referencing #1662.
- [ ] **Step 2:** Restore + build.

Run: `dotnet restore && dotnet build --no-restore`
Expected: builds against v0.102.0 (the fused-norm overload + tiled backward are available). If the package isn't published yet, this stays red — acceptable for the draft (note it in the PR body).

- [ ] **Step 3: Commit.**

```bash
git add Directory.Packages.props && git commit -m "deps(#1662): bump AiDotNet.Tensors 0.94.2 -> 0.102.0"
```

### Task 3.2: Full-precision streaming optimizer (bit-identical prerequisite)

**Problem:** the existing streaming path uses **8-bit quantized** Adam state (`StreamingAdam`), which is NOT bit-identical to the classic full-precision `opt.Step`. §5a's common-case default must be bit-identical, so the engaged-for-fitting-models path needs a full-precision streaming optimizer wrapper.

**Files:**
- Modify: `src/NeuralNetworks/NeuralNetworkBase.cs` (`GetOrCreateStreamingOptimizer`)
- Inspect: `src/Training/` streaming optimizer implementations

- [ ] **Step 1: Locate the streaming optimizer + its quantization.**

```bash
grep -rn "IStreamingOptimizer\|StreamingAdam\|class.*Streaming.*Optimizer" src/Training/ src/NeuralNetworks/NeuralNetworkBase.cs
```

- [ ] **Step 2: Write the failing bit-identical test** (Task 3.3 below is the full gate; here a minimal unit): one training step via the full-precision streaming optimizer must produce parameters bit-identical to one step via classic `opt.Step` on a 2-layer model. (Code shown in Task 3.3.)

- [ ] **Step 3: Add a full-precision mode** to `GetOrCreateStreamingOptimizer` so that when engaged for a fitting model (not OOM), it keeps fp32 Adam `m`/`v` (no int8 block quantization). Gate the quantized variant to the OOM/ForceOn-memory case only. The per-parameter `Apply(source, grad)` must advance `m`/`v` with the identical update rule and ordering as the classic whole-vector `opt.Step`.

- [ ] **Step 4: Build + run the minimal parity test.** Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add -A && git commit -m "feat(#1662): full-precision streaming optimizer for bit-identical fused-in-backward"
```

### Task 3.3: Engage single-pass fused optimizer-in-backward for unclipped fitting models

**Files:**
- Modify: `src/NeuralNetworks/NeuralNetworkBase.cs` (`ShouldUseStreamingTraining` ~L6166)
- Create: `tests/AiDotNet.Tests/IntegrationTests/NeuralNetworks/FusedOptimizerInBackwardParityTests.cs`

- [ ] **Step 1: Write the bit-identical gate test FIRST.** Train two identical models from the same seed for N steps — one forced onto the fused-in-backward path, one on the classic path — and assert parameters + loss trajectory are bit-identical, on the acceptance shape (dim=384, 10 layers, unclipped, fp32 Adam).

```csharp
using AiDotNet.Enums;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

public class FusedOptimizerInBackwardParityTests
{
    [Fact(Timeout = 120000)]
    public void FusedInBackward_Unclipped_BitIdenticalToClassic()
    {
        const int steps = 20;
        var classic = BuildAcceptanceModel(seed: 42);
        classic.StreamingTraining = StreamingTrainingMode.ForceOff;     // classic collect-then-step
        var fused = BuildAcceptanceModel(seed: 42);
        fused.StreamingTraining = StreamingTrainingMode.ForceOn;        // fused optimizer-in-backward

        var (x, y) = MakeBatch(seed: 7);
        for (int i = 0; i < steps; i++)
        {
            classic.TrainPublic(x, y);
            fused.TrainPublic(x, y);
            Assert.Equal(classic.LastLoss, fused.LastLoss, 6); // bit-close per step
        }
        // Final parameter vectors bit-identical.
        var pc = classic.GetParameters(); var pf = fused.GetParameters();
        Assert.Equal(pc.Length, pf.Length);
        for (int i = 0; i < pc.Length; i++)
            Assert.Equal((double)Convert.ToDouble(pc[i]), (double)Convert.ToDouble(pf[i]), 5);
    }

    // BuildAcceptanceModel: SimCSE<float> dim=384, 10 layers (or the nearest
    // FusedOptimizerIntegrationTests harness model); MaxGradNorm = 0 (unclipped);
    // Adam, fp32. MakeBatch: deterministic input/target. TrainPublic: the existing
    // test-shim seam used by FusedOptimizerIntegrationTests.
}
```

- [ ] **Step 2: Run it — expect FAIL** (today `ForceOn` uses 8-bit streaming Adam, so params diverge from classic fp32).

Run: `dotnet test ... --filter "FullyQualifiedName~FusedOptimizerInBackwardParity"`
Expected: FAIL on the per-step loss or final-param assertion.

- [ ] **Step 3: Make `ForceOn` use the Task 3.2 full-precision streaming optimizer** for the bit-identical path. Re-run.
Expected: PASS (now bit-identical).

- [ ] **Step 4: Change the `Auto` engagement** so unclipped training on fitting models engages the fused path. In `ShouldUseStreamingTraining` Auto branch, return `true` when `MaxGradNormValue <= 0` (unclipped) — the pure double-win case — keeping the `footprint > 0.5*available` rule for the clipped case (clipped streaming is 2× backward, only worth it under memory pressure). Add a clear comment citing #1662 §5a/§5b.

- [ ] **Step 5: Run `StreamingTrainingTests` + `FusedOptimizerIntegrationTests`** to confirm no regression.

Run: `dotnet test ... --filter "FullyQualifiedName~StreamingTraining|FullyQualifiedName~FusedOptimizer"`
Expected: all PASS, including `Auto_OnSmallModel_TrainsViaClassicPath` — **update that test** if its assertion assumed small unclipped models stay classic (they now use fused-in-backward); the correct new invariant is "small CLIPPED models stay classic." Adjust the test's model to be clipped, or rename to reflect the new contract — do NOT delete the invariant.

- [ ] **Step 6: Commit.**

```bash
git add -A && git commit -m "perf(#1662): default unclipped fitting models to fused optimizer-in-backward"
```

### Task 3.4: Opt-in fast-clip (single-pass clipped, OFF by default)

**Files:**
- Modify: `src/Enums/` or `NeuralNetworkBase.cs` — add a `FastApproxGradClip` opt-in flag (bool, default false)
- Modify: `src/NeuralNetworks/NeuralNetworkBase.cs` (`TrainWithTapeStreaming` clipped branch ~L6317)
- Create: `tests/AiDotNet.Tests/IntegrationTests/NeuralNetworks/FastClipConvergenceTests.cs`

- [ ] **Step 1: Add the opt-in property** with XML docs stating it is an approximation (running/EMA grad-norm), OFF by default, NOT bit-identical:

```csharp
/// <summary>
/// #1662 §5c — opt-in single-pass approximate gradient clipping. When true, the
/// fused optimizer-in-backward path clips using an EMA of the previous step's
/// global grad-norm instead of the current step's exact norm, so even clipped
/// training runs in ONE backward pass (no 2x norm pass). This is an approximation
/// (NFNet-style adaptive clipping, Brock et al. 2021): it is NOT bit-identical to
/// exact clip_grad_norm_ and changes the trajectory. Default: false (exact two-pass).
/// </summary>
public bool FastApproxGradClip { get; set; } = false;
```

- [ ] **Step 2: Write the convergence test FIRST.** Fast-clip training must reduce loss and stay finite over N steps (it need not match exact clipping).

```csharp
[Fact(Timeout = 120000)]
public void FastClip_ReducesLoss_AndStaysFinite()
{
    var m = BuildAcceptanceModel(seed: 42);
    m.MaxGradNorm = 1.0;                 // clipping ON
    m.FastApproxGradClip = true;         // single-pass approximate
    m.StreamingTraining = StreamingTrainingMode.ForceOn;
    var (x, y) = MakeBatch(seed: 7);
    double first = double.NaN, last = 0;
    for (int i = 0; i < 30; i++)
    {
        m.TrainPublic(x, y);
        if (i == 0) first = Convert.ToDouble(m.LastLoss);
        last = Convert.ToDouble(m.LastLoss);
        Assert.False(double.IsNaN(last) || double.IsInfinity(last));
    }
    Assert.True(last < first, $"loss did not decrease: {first} -> {last}");
}
```

- [ ] **Step 3: Run — expect FAIL** (flag does nothing yet; with ForceOn+clip it runs the two-pass path).

- [ ] **Step 4: Implement fast-clip** in the clipped branch of `TrainWithTapeStreaming`: when `FastApproxGradClip`, skip pass 1; compute the clip scale from a stored EMA of the previous step's `totalNorm` (seed the EMA on step 0 with a single norm pass, or with no clip on step 0), and run a SINGLE streaming pass that folds that scale and applies the optimizer. Update the EMA with this step's norm computed via the Task 1.4 fused-norm overload (free, in the same pass).

- [ ] **Step 5: Run the convergence test.** Expected: PASS. Also confirm the Task 3.3 bit-identical gate still PASSES with `FastApproxGradClip=false` (default unchanged).

- [ ] **Step 6: Commit.**

```bash
git add -A && git commit -m "feat(#1662): opt-in single-pass fast approximate grad-clip (OFF by default)"
```

### Task 3.5: PyTorch-comparison proof (handily beat on ALL metrics)

**Files:**
- Create: `tools/ConvParallelProbe/trainbench_torch.py` (Tensors repo — same dir as the probe) OR `benchmarks/trainbench_torch.py` (AiDotNet repo). Place it next to wherever `--trainbench` lives so the two run the same shape.
- Create: `docs/superpowers/specs/1662-pytorch-comparison.md` (results table)

- [ ] **Step 1: Write `trainbench_torch.py`** — a torch CPU script building the *same* transformer block (S=128, D=384, H=6, 10 layers), running fwd→bwd→SGD for the same reps, and printing a matching `TRAINBENCH engine=torch ... median_ms=... alloc_mb_per_step=... peak_ws_mb=...` line (use `tracemalloc` / `resource.getrusage` for peak RSS). Pin `torch.set_num_threads` to match `--maxdop`.

- [ ] **Step 2: Run both and capture numbers.**

```bash
dotnet run -c Release --project tools/ConvParallelProbe -- --trainbench --s 128 --d 384 --h 6 --layers 10 --maxdop 8
python trainbench_torch.py --s 128 --d 384 --h 6 --layers 10 --threads 8
```

- [ ] **Step 3: Record the head-to-head** in `1662-pytorch-comparison.md` as a table (AiDotNet vs torch: median_ms, alloc_mb_per_step, peak_ws_mb). **Acceptance: AiDotNet wins on all three.** If it does not win on any metric, that metric is a bug to fix (return to Phase 1 audit or the engagement logic) — do NOT weaken the claim. Log it honestly.

- [ ] **Step 4: Commit the harness + results.**

```bash
git add trainbench_torch.py docs/superpowers/specs/1662-pytorch-comparison.md
git commit -m "test(#1662): PyTorch CPU comparison harness + results (beat on time/alloc/RSS)"
```

### Task 3.6: Acceptance-shape memory/allocation gate

**Files:**
- Create: `tests/AiDotNet.Tests/IntegrationTests/NeuralNetworks/FusedInBackwardMemoryTests.cs`

- [ ] **Step 1: Write the test** — on the acceptance SimCSE shape, assert per-step managed allocation under the fused path is materially below the classic path (the memory win), and loss trajectory bit-close (already covered by 3.3; here assert the allocation delta).

```csharp
[Fact(Timeout = 180000)]
public void FusedInBackward_PerStepAllocation_BelowClassic()
{
    long Measure(StreamingTrainingMode mode)
    {
        var m = BuildAcceptanceModel(seed: 42);
        m.StreamingTraining = mode;
        var (x, y) = MakeBatch(seed: 7);
        for (int i = 0; i < 3; i++) m.TrainPublic(x, y);      // warmup
        long s = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 5; i++) m.TrainPublic(x, y);
        return (GC.GetAllocatedBytesForCurrentThread() - s) / 5;
    }
    long classic = Measure(StreamingTrainingMode.ForceOff);
    long fused = Measure(StreamingTrainingMode.ForceOn);
    Assert.True(fused < classic,
        $"fused per-step alloc {fused} not below classic {classic}");
}
```

- [ ] **Step 2: Run.** Expected: PASS.
- [ ] **Step 3: Commit.**

```bash
git add -A && git commit -m "test(#1662): fused-in-backward per-step allocation below classic on acceptance shape"
```

### Task 3.7: Open the draft PR

- [ ] **Step 1:** Push `perf/1662-optimizer-in-backward`.
- [ ] **Step 2:** Open a **draft** PR titled `perf(#1662): backward-pass optimizations — alloc audit (#4), tiled flash backward (#3), fused optimizer-in-backward default (#1)`. Body must: link #1662 + the Tensors PR; state it depends on Tensors v0.102.0 (red until published); paste the §3.5 PyTorch-comparison table; explicitly note lever #2 is out of scope (owned by #1633); list the bit-identical gate + convergence + memory tests.

```bash
gh pr create --draft --repo ooples/AiDotNet --base master --head perf/1662-optimizer-in-backward \
  --title "perf(#1662): backward-pass optimizations (#4 alloc audit, #3 tiled flash backward, #1 optimizer-in-backward default)" \
  --body-file docs/superpowers/specs/1662-pytorch-comparison.md
```

---

## Self-Review

**Spec coverage:**
- §3 lever #4 (probe, audit, guard test, fused-norm helper) → Tasks 1.1–1.4. ✅
- §4 lever #3 (tiled backward, parity gate) → Tasks 2.1–2.2. ✅
- §5a promote single-pass fused to common-case default (unclipped) → Task 3.3. ✅
- §5b clipped stays two-pass (no regression) → preserved; Task 3.3 Step 4 keeps clipped on the memory gate. ✅
- §5c opt-in fast-clip → Task 3.4. ✅
- §5d PyTorch proof harness → Tasks 1.1 (metrics) + 3.5. ✅
- §1 acceptance (peak-RSS + alloc on SimCSE shape, trajectory unchanged) → Tasks 3.3 (trajectory), 3.6 (alloc). ✅
- Full-precision-streaming prerequisite (8-bit-state nuance) → Task 3.2. ✅
- Cross-repo release ordering → Task 1.5 + Task 3.1. ✅

**Placeholder scan:** `THRESHOLD_BYTES` (Task 1.3) and `BuildAcceptanceModel`/`MakeBatch`/`TrainPublic` helpers are intentionally derived-at-execution (the first from a measured number, the latter from the existing `FusedOptimizerIntegrationTests` shim) — each is called out with how to source it. Engine method names in Tasks 1.1/1.4/2.x are flagged for confirmation against the live API. No silent TODOs.

**Type consistency:** `StreamingTrainingMode.{Auto,ForceOn,ForceOff}` used consistently (verified in `src/Enums/StreamingTrainingMode.cs`); `FastApproxGradClip` and `MaxGradNorm`/`MaxGradNormValue` used consistently; `ComputeGradientsStreaming` 3-arg vs new 4-arg overload distinguished (Task 1.4). `FusedAttention<float>.Backward` signature matches the current file.

**Known risk:** the bulk-risk task is 2.2 (novel tiled kernel) — fully gated by the 2.1 parity test written first. The 3.3 engagement change is the broad-blast-radius task — gated by the bit-identical test and the preserved `StreamingTrainingTests` invariants.
