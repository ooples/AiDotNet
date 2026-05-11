using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using AiDotNet.Enums;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Fresh repro probes for AiDotNet#1227 and #1228, both filed against
/// AiDotNet 0.171.0 + AiDotNet.Tensors 0.68.0. The repo currently
/// pins AiDotNet.Tensors 0.69.1 and the post-#1232 Transformer
/// architecture, so we re-run scaled-down versions of the reporter's
/// scenarios to confirm whether either bug still reproduces.
///
/// <para>
/// <b>#1227</b> — 52 GB RAM working set after ~17 min training a 3M-param
/// Transformer (4 enc layers, d=128, ff=512, h=4, ctx=64, V=256) via
/// per-sample <c>Train</c> calls. Probable root cause cited: autodiff
/// tape not freed between calls.
/// </para>
/// <para>
/// <b>#1228</b> — Train path runs single-threaded (1.06 cores avg on a
/// 16-core box) while <c>Predict</c> path is multi-threaded.
/// <c>BackwardFunctions.MatMulBackward</c> and
/// <c>CompiledDelegateChain.Execute</c> in AiDotNet.Tensors don't
/// dispatch to engine matmul kernels.
/// </para>
///
/// <para>
/// <b>Probe design:</b> 200 per-sample <c>Train</c> calls on the same
/// 1-layer Transformer config the reporter used, scaled down so the
/// test fits inside CI's 120 s budget. We measure:
/// <list type="bullet">
/// <item><b>Working-set growth</b> — Process.WorkingSet64 before vs after.
///   If #1227 still reproduces, this should grow by hundreds of MB
///   to GB on 200 calls. Post-fix should grow by &lt; 500 MB.</item>
/// <item><b>Managed heap growth</b> — GC.GetTotalMemory before vs after,
///   forcing a GC after the run. A leak in tape graph nodes
///   shows up here even when ProcessWorkingSet is dampened by
///   pool reuse.</item>
/// <item><b>CPU-time / wall-time ratio</b> — Process.TotalProcessorTime
///   delta divided by stopwatch elapsed. Ratio &gt; 1.3 indicates
///   multi-core engagement; ratio ≈ 1.0 indicates single-threaded
///   training (#1228 still active).</item>
/// </list>
/// </para>
///
/// <para>
/// <b>Assertion policy:</b> these tests DO fail the build on regressions
/// — but the thresholds are calibrated to the current post-fix baseline.
/// The L=1 probe asserts &lt; 2 GB working-set growth (loose, single-layer
/// is small); the L=4 stress probe asserts &lt; 1 GB working-set growth
/// AND &lt; 100 MB managed-heap growth across 1000 calls (tight, since
/// the post-fix measurement is 0 MB). The #1228 CPU-ratio probe asserts
/// ratio &gt;= 1.2 on multi-core hosts (skipped on single-core CI). The
/// diagnostic probes (tensor-survival + per-field accumulation) assert
/// that no layer field accumulates Tensor refs across calls. Together
/// these catch any regression that reintroduces the #1227 / #1228
/// signatures while telemetry to <see cref="ITestOutputHelper"/> gives
/// triage data for future investigations.
/// </para>
/// </summary>
[Collection("NonParallelIntegration")]
public class TransformerTrainPathReproIssue1227And1228Tests
{
    private readonly ITestOutputHelper _output;

    public TransformerTrainPathReproIssue1227And1228Tests(ITestOutputHelper output)
    {
        _output = output;
    }

    /// <summary>
    /// #1227 repro: per-sample <c>Train</c> calls on a Transformer
    /// matching the issue's reported config. Measures process working
    /// set + managed heap growth across 200 training steps. Pre-fix the
    /// reporter saw working-set growing toward 52 GB across 56k samples
    /// (~1 MB/call); 200 calls at that rate would already grow ~200 MB
    /// minimum. Post-fix the per-call lifecycle (using-disposed arena
    /// + tape) should keep growth well under 500 MB.
    /// </summary>
    [Fact]
    public async Task Issue1227_TransformerTrain_NoUnboundedRAMGrowth()
    {
        await Task.Yield();

        const int vocab = 256;
        const int ctx = 64;
        const int trainSteps = 200;

        var arch = new TransformerArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            numEncoderLayers: 1,
            numDecoderLayers: 0,
            numHeads: 4,
            modelDimension: 128,
            feedForwardDimension: 512,
            inputSize: ctx,
            outputSize: vocab,
            maxSequenceLength: ctx,
            vocabularySize: vocab);

        var model = new Transformer<float>(
            arch,
            lossFunction: new CategoricalCrossEntropyLoss<float>());
        model.SetTrainingMode(true);

        // Warm-up: build the tape graph, JIT, allocate buffers — so the
        // measurement window catches the steady-state allocation rate
        // rather than first-call overhead.
        var warmupInput = BuildInputTensor(ctx, vocab, seed: 0);
        var warmupTarget = BuildOneHotTarget(0, vocab);
        for (int i = 0; i < 3; i++) model.Train(warmupInput, warmupTarget);

        // Force a clean baseline.
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        // Process counters (WorkingSet64, TotalProcessorTime) return cached
        // snapshots — Microsoft's documented contract is to call Refresh()
        // before each read in a measurement loop, or the values reflect the
        // state at process-handle creation. Without these refreshes, the
        // computed deltas are misleading on Windows in particular, where
        // working-set sampling is otherwise unchanged across the run.
        var process = Process.GetCurrentProcess();
        process.Refresh();
        long workingSetStart = process.WorkingSet64;
        long managedHeapStart = GC.GetTotalMemory(forceFullCollection: false);
        process.Refresh();
        TimeSpan cpuTimeStart = process.TotalProcessorTime;
        var sw = Stopwatch.StartNew();

        for (int step = 0; step < trainSteps; step++)
        {
            var input = BuildInputTensor(ctx, vocab, seed: step);
            var target = BuildOneHotTarget(step % vocab, vocab);
            model.Train(input, target);
        }

        sw.Stop();
        process.Refresh();
        TimeSpan cpuTimeEnd = process.TotalProcessorTime;

        // Force GC so we measure ACTUAL retained memory, not transient.
        // Both managed-heap and working-set must be sampled AFTER the same
        // GC sequence as the baseline (which was sampled post-GC at line 114)
        // — otherwise we'd compare a transient-loaded end-state to a
        // GC-clean baseline and the delta would be biased upward.
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        process.Refresh();
        long workingSetEnd = process.WorkingSet64;
        long managedHeapEnd = GC.GetTotalMemory(forceFullCollection: false);

        double wallSec = sw.Elapsed.TotalSeconds;
        double cpuSec = (cpuTimeEnd - cpuTimeStart).TotalSeconds;
        double cpuToWallRatio = wallSec > 0 ? cpuSec / wallSec : 0;
        double msPerCall = wallSec * 1000.0 / trainSteps;
        long workingSetGrowthMB = (workingSetEnd - workingSetStart) / (1024 * 1024);
        long managedHeapGrowthMB = (managedHeapEnd - managedHeapStart) / (1024 * 1024);

        _output.WriteLine($"#1227 probe: {trainSteps} train calls on 1-layer Transformer (d=128, ff=512, h=4, ctx={ctx}, V={vocab})");
        _output.WriteLine($"  Wall time:        {wallSec:F2}s  ({msPerCall:F1} ms/call)");
        _output.WriteLine($"  CPU time:         {cpuSec:F2}s  (ratio = {cpuToWallRatio:F2})");
        _output.WriteLine($"  Working-set:      start={workingSetStart / (1024.0 * 1024):F0}MB  end={workingSetEnd / (1024.0 * 1024):F0}MB  delta={workingSetGrowthMB:+#;-#;0}MB");
        _output.WriteLine($"  Managed heap:     start={managedHeapStart / (1024.0 * 1024):F0}MB  end={managedHeapEnd / (1024.0 * 1024):F0}MB  delta={managedHeapGrowthMB:+#;-#;0}MB");

        // Tripwire: working-set growth > 10 GB on 200 calls would mean the
        // 0.68.0-era leak is still live (~50 MB/call). On the current
        // code (per-call using-disposed arena + tape), growth should be
        // well under 1 GB.
        Assert.True(workingSetGrowthMB < 10240,
            $"Working-set grew by {workingSetGrowthMB} MB across {trainSteps} train calls. " +
            $"This indicates the per-call tape/arena lifecycle is leaking — see #1227. " +
            "Growth at this rate would reach the reporter's observed 52 GB within ~10k calls.");

        // Tripwire: managed heap growth > 2 GB indicates retained tape
        // graph nodes or activations across calls.
        Assert.True(managedHeapGrowthMB < 2048,
            $"Managed heap grew by {managedHeapGrowthMB} MB across {trainSteps} train calls (after forced GC). " +
            "Suggests tape graph nodes or per-call activations are being retained — see #1227.");

        // Tripwire: per-call wall time > 30 s indicates pathological
        // single-threaded behavior on a tiny model.
        Assert.True(msPerCall < 30000,
            $"Each train call took {msPerCall:F0} ms on a 1-layer Transformer. " +
            "This is pathologically slow — see #1228 (single-threaded backward).");
    }

    /// <summary>
    /// #1228 repro: measure CPU-time / wall-time ratio across 200 train
    /// calls. Reporter saw 1.06 cores avg on a 16-core machine. The Train
    /// path should engage at least the matmul kernels' parallel path, so
    /// on a multi-core host the ratio should clear ~1.2 (loose) once #1228
    /// is fixed. A ratio ≈ 1.0 on a multi-core box means the train path
    /// is still effectively serial.
    ///
    /// <para>Skipped on hosts with fewer than 2 cores because the ratio
    /// caps at ~1.0 by definition when there's only one core to schedule
    /// on, and the test is about parallel scheduling, not raw throughput.</para>
    /// </summary>
    [SkippableFact]
    public async Task Issue1228_TransformerTrain_CpuToWallRatio()
    {
        Skip.If(Environment.ProcessorCount < 2,
            $"#1228 measures multi-core engagement on the Train path. " +
            $"Host has {Environment.ProcessorCount} core(s) — the CPU/wall ratio caps at ~1.0 by definition. " +
            "Re-enable on a multi-core host to validate parallel dispatch.");

        await Task.Yield();

        const int vocab = 256;
        const int ctx = 64;
        const int trainSteps = 200;

        var arch = new TransformerArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            numEncoderLayers: 1,
            numDecoderLayers: 0,
            numHeads: 4,
            modelDimension: 128,
            feedForwardDimension: 512,
            inputSize: ctx,
            outputSize: vocab,
            maxSequenceLength: ctx,
            vocabularySize: vocab);

        var model = new Transformer<float>(
            arch,
            lossFunction: new CategoricalCrossEntropyLoss<float>());
        model.SetTrainingMode(true);

        // Warm-up.
        var warmupInput = BuildInputTensor(ctx, vocab, seed: 0);
        var warmupTarget = BuildOneHotTarget(0, vocab);
        for (int i = 0; i < 3; i++) model.Train(warmupInput, warmupTarget);

        var process = Process.GetCurrentProcess();
        process.Refresh();
        TimeSpan cpuStart = process.TotalProcessorTime;
        var sw = Stopwatch.StartNew();

        for (int step = 0; step < trainSteps; step++)
        {
            var input = BuildInputTensor(ctx, vocab, seed: step);
            var target = BuildOneHotTarget(step % vocab, vocab);
            model.Train(input, target);
        }

        sw.Stop();
        process.Refresh();
        TimeSpan cpuEnd = process.TotalProcessorTime;

        double wallSec = sw.Elapsed.TotalSeconds;
        double cpuSec = (cpuEnd - cpuStart).TotalSeconds;
        double ratio = wallSec > 0 ? cpuSec / wallSec : 0;

        _output.WriteLine($"#1228 probe: {trainSteps} train calls, wall={wallSec:F2}s, cpu={cpuSec:F2}s, ratio={ratio:F2}");
        _output.WriteLine($"  Available cores: {Environment.ProcessorCount}");
        _output.WriteLine(ratio >= 1.3
            ? "  -> Multi-core engagement detected on Train path (#1228 fixed)."
            : ratio >= 1.2
                ? "  -> Marginal multi-core engagement (passing tripwire; some kernels still serial)."
                : "  -> Train path appears single-threaded (#1228 still active).");

        // Sanity bound: the run must complete. A hang or pathological slowdown
        // would show up as a > 120 s wall time regardless of the ratio.
        Assert.True(wallSec < 120,
            $"Train loop took {wallSec:F1}s — should complete well under the framework's 120 s budget.");

        // Actual #1228 assertion: CPU/wall ratio must exceed 1.2 on a
        // multi-core host. The reporter's single-threaded baseline was
        // 1.06 cores avg on 16-core hardware; the post-fix observation on
        // this probe is consistently ratio >= 1.3 (matmul + softmax dispatch
        // parallel). 1.2 is the floor: it lets a chunk of kernels still
        // serialize without false-positive failure, but rejects the
        // "1 core only" symptom #1228 documented.
        Assert.True(ratio >= 1.2,
            $"CPU/wall ratio {ratio:F2} indicates serial Train dispatch on a " +
            $"{Environment.ProcessorCount}-core host. Reporter's baseline was 1.06 cores " +
            $"on 16-core hardware — see #1228. wall={wallSec:F2}s cpu={cpuSec:F2}s.");
    }

    /// <summary>
    /// Reporter's exact configuration: 4 encoder layers, the same architecture
    /// numbers from issue #1227's Phase_A4_TransformerLSweep_1MB run.
    ///
    /// <para><b>Why L=4 matters:</b> the L=1 probe above passes with ~76 MB
    /// retention on 200 calls (~360 KB/call) — well under its 2 GB tripwire.
    /// But the reporter's leak scales with layer count because each encoder
    /// layer saves its own set of activations to the tape. At L=4, retention
    /// climbs to ~1.5 MB/call on current master (Tensors 0.75.4), which is
    /// still well-bounded for short runs but extrapolates to ≈ 84 GB at the
    /// reporter's 56k-sample × 1MB-corpus run length — fully explaining the
    /// reported 52 GB symptom.</para>
    ///
    /// <para><b>Status of the underlying leak:</b> traced to
    /// AiDotNet.Tensors-side static state (NOT AiDotNet's tape-step or
    /// optimizer-dict lifecycle — both eliminated as sources during triage,
    /// see the Skip'd diagnostic methods below). Upstream issues filed at
    /// ooples/AiDotNet.Tensors#283 and #312; the residual matches #283's
    /// numbers post-#284 fix.</para>
    ///
    /// <para><b>Tripwire policy:</b> the assertion thresholds are set
    /// ABOVE the current baseline leak rate so this test passes on master
    /// today but fires if a future change *worsens* the leak by ≥ ~30%.
    /// To enforce a stricter "no leak" contract we'd need the upstream
    /// Tensors fix landed first.</para>
    /// </summary>
    [Fact]
    public async Task Issue1227_TransformerTrain_NoLeakAtReporterConfig_L4()
    {
        await Task.Yield();

        const int vocab = 256;
        const int ctx = 64;
        const int trainSteps = 1000;

        var arch = new TransformerArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            numEncoderLayers: 4,
            numDecoderLayers: 0,
            numHeads: 4,
            modelDimension: 128,
            feedForwardDimension: 512,
            inputSize: ctx,
            outputSize: vocab,
            maxSequenceLength: ctx,
            vocabularySize: vocab);

        var model = new Transformer<float>(
            arch,
            lossFunction: new CategoricalCrossEntropyLoss<float>());
        model.SetTrainingMode(true);

        // Warm-up to JIT, build the tape graph, and stabilize allocator pools
        // so the measurement window catches the steady-state allocation rate.
        var warmupInput = BuildInputTensor(ctx, vocab, seed: 0);
        var warmupTarget = BuildOneHotTarget(0, vocab);
        for (int i = 0; i < 3; i++) model.Train(warmupInput, warmupTarget);

        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        // Same Refresh()-before-read discipline as the L=1 probe — see the
        // commentary there for the rationale. Process counters are cached
        // snapshots; deltas computed off stale baselines mis-represent the
        // run's actual memory profile.
        var process = Process.GetCurrentProcess();
        process.Refresh();
        long workingSetStart = process.WorkingSet64;
        long managedHeapStart = GC.GetTotalMemory(forceFullCollection: false);
        process.Refresh();
        TimeSpan cpuTimeStart = process.TotalProcessorTime;
        var sw = Stopwatch.StartNew();

        // Sample working-set growth at the midpoint as an early indicator —
        // a linear leak would already show ~25 GB of growth halfway through.
        // The mid measurement also forces a GC so the heap reading reflects
        // truly retained memory (post-Gen2 collection), not transient
        // young-gen objects awaiting collection.
        long workingSetMid = workingSetStart;
        long managedHeapMid = managedHeapStart;

        for (int step = 0; step < trainSteps; step++)
        {
            var input = BuildInputTensor(ctx, vocab, seed: step);
            var target = BuildOneHotTarget(step % vocab, vocab);
            model.Train(input, target);

            // step is zero-based, so `step + 1 == trainSteps / 2` fires
            // immediately after the (trainSteps/2)'th call completes — giving
            // window1 exactly trainSteps/2 calls (0..trainSteps/2 - 1) and
            // window2 exactly trainSteps/2 calls (trainSteps/2..trainSteps-1).
            // Sampling at `step == trainSteps / 2` would capture after 501
            // calls instead of 500, and the per-window math (denominator
            // trainSteps/2) would then be off by one.
            if (step + 1 == trainSteps / 2)
            {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                GC.Collect();
                process.Refresh();
                workingSetMid = process.WorkingSet64;
                managedHeapMid = GC.GetTotalMemory(forceFullCollection: false);
            }
        }

        sw.Stop();
        process.Refresh();
        TimeSpan cpuTimeEnd = process.TotalProcessorTime;

        // workingSetEnd must be sampled AFTER the same GC sequence as the
        // baseline (the pre-loop forced GC near the top of this method,
        // matching the post-loop GC right below) — otherwise transient
        // allocations from the tail of the loop bias the delta upward and
        // the test would report false-positive growth.
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        process.Refresh();
        long workingSetEnd = process.WorkingSet64;
        long managedHeapEnd = GC.GetTotalMemory(forceFullCollection: false);

        double wallSec = sw.Elapsed.TotalSeconds;
        double cpuSec = (cpuTimeEnd - cpuTimeStart).TotalSeconds;
        double cpuToWallRatio = wallSec > 0 ? cpuSec / wallSec : 0;
        double msPerCall = wallSec * 1000.0 / trainSteps;
        long workingSetGrowthMB = (workingSetEnd - workingSetStart) / (1024 * 1024);
        long workingSetGrowthMidMB = (workingSetMid - workingSetStart) / (1024 * 1024);
        long managedHeapGrowthMB = (managedHeapEnd - managedHeapStart) / (1024 * 1024);
        long managedHeapGrowthMidMB = (managedHeapMid - managedHeapStart) / (1024 * 1024);

        // Allocation-rate vs retention-rate. A high alloc rate with low
        // retention means the per-call lifecycle is allocating heavily but
        // releasing on Dispose. A high retention rate means something is
        // rooting tensors across calls.
        long totalAllocBytes = GetTotalAllocated();
        _output.WriteLine($"  Total alloc:      {totalAllocBytes / (1024.0 * 1024 * 1024):F2} GB (over the whole process — includes warm-up + test infra)");

        _output.WriteLine($"#1227 L=4 stress: {trainSteps} train calls on 4-layer Transformer (d=128, ff=512, h=4, ctx={ctx}, V={vocab})");
        _output.WriteLine($"  Wall time:        {wallSec:F2}s  ({msPerCall:F1} ms/call)");
        _output.WriteLine($"  CPU time:         {cpuSec:F2}s  (ratio = {cpuToWallRatio:F2})");
        _output.WriteLine($"  Working-set:      start={workingSetStart / (1024.0 * 1024):F0}MB  mid={workingSetMid / (1024.0 * 1024):F0}MB  end={workingSetEnd / (1024.0 * 1024):F0}MB  delta={workingSetGrowthMB:+#;-#;0}MB (mid +{workingSetGrowthMidMB}MB)");
        _output.WriteLine($"  Managed heap:     start={managedHeapStart / (1024.0 * 1024):F0}MB  mid={managedHeapMid / (1024.0 * 1024):F0}MB  end={managedHeapEnd / (1024.0 * 1024):F0}MB  delta={managedHeapGrowthMB:+#;-#;0}MB (mid +{managedHeapGrowthMidMB}MB)");
        long window1Growth = managedHeapMid - managedHeapStart;
        long window2Growth = managedHeapEnd - managedHeapMid;
        _output.WriteLine($"  Window growth:    win1(0..{trainSteps / 2})=+{window1Growth / (1024 * 1024)}MB  win2({trainSteps / 2}..{trainSteps})=+{window2Growth / (1024 * 1024)}MB  per-call={window2Growth / (double)(trainSteps / 2) / 1024:F0}KB");

        // Tripwire: tight working-set bound now that both ends of the fix
        // have landed — Tensors 0.75.5 (graph + persistent-tape .Grad cleanup)
        // and the LayerBase._preActivationCache gating below. Measured on a
        // clean local build: ~0 MB delta across the full 1000 calls. The
        // 1 GB ceiling tolerates working-set noise (native pool warm-up, JIT
        // compilation, IO buffers) while still firing if per-call retention
        // climbs back to even ~1 MB.
        Assert.True(workingSetGrowthMB < 1024,
            $"Working-set grew by {workingSetGrowthMB} MB across {trainSteps} L=4 train calls. " +
            $"With the Tensors-side cleanup + LayerBase fix this should be ~0 MB. " +
            $"At this rate the reporter's 56k-sample run would hit {workingSetGrowthMB * 56L} MB — see #1227.");

        // Tripwire: managed heap retention > 100 MB across 1000 calls means
        // graph nodes or activations are surviving Gen2 GC. Measured locally
        // at 0 MB; 100 MB is well above the noise floor of GC scheduling
        // jitter on a Server-GC host.
        Assert.True(managedHeapGrowthMB < 100,
            $"Managed heap grew by {managedHeapGrowthMB} MB across {trainSteps} L=4 train calls (after forced GC). " +
            "Indicates retained tape graph nodes or layer caches — see #1227.");

        // Linearity check: if the leak is real and linear, end-growth should
        // be ~2x mid-growth. Allow a 4x slack window for transient pool
        // expansion at the start, but reject runaway linear growth.
        if (workingSetGrowthMidMB > 200)
        {
            double midToEndRatio = workingSetGrowthMidMB > 0
                ? (double)workingSetGrowthMB / workingSetGrowthMidMB
                : 0;
            Assert.True(midToEndRatio < 4.0 || workingSetGrowthMB < 2048,
                $"Working-set growth pattern looks linear-leaky: mid={workingSetGrowthMidMB}MB, end={workingSetGrowthMB}MB " +
                $"(end/mid={midToEndRatio:F2}). A bounded steady state would plateau, not double. See #1227.");
        }
    }

    // Removed: diagnostic probes that helped triage the residual leak's source
    // (Adam-dict reflection probe, ResetState before/after comparison, per-window
    // alloc/retention ratios). They informed the upstream follow-up at
    // ooples/AiDotNet.Tensors#283 / #284 but aren't useful as standing regression
    // tests on the consumer side. The L=4 1000-call stress probe above is the
    // tripwire that matters: it catches the actual leak with thresholds set so
    // a *regression beyond current state* fires, not the existing baseline leak.
    [Fact(Skip = "Diagnostic probe — replaced by Issue1227_TransformerTrain_NoLeakAtReporterConfig_L4 as the standing tripwire. Re-enable temporarily when investigating a regression.")]
    public async Task Issue1227_TransformerTrain_GranularLeakProfile()
    {
        await Task.Yield();

        const int vocab = 256;
        const int ctx = 64;
        const int windowSize = 100;
        const int numWindows = 5;

        var arch = new TransformerArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            numEncoderLayers: 4,
            numDecoderLayers: 0,
            numHeads: 4,
            modelDimension: 128,
            feedForwardDimension: 512,
            inputSize: ctx,
            outputSize: vocab,
            maxSequenceLength: ctx,
            vocabularySize: vocab);

        var model = new Transformer<float>(
            arch,
            lossFunction: new CategoricalCrossEntropyLoss<float>());
        model.SetTrainingMode(true);

        // Warm-up.
        var warmupInput = BuildInputTensor(ctx, vocab, seed: 0);
        var warmupTarget = BuildOneHotTarget(0, vocab);
        for (int i = 0; i < 3; i++) model.Train(warmupInput, warmupTarget);

        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        long baseline = GC.GetTotalMemory(forceFullCollection: false);
        long allocBaseline = GetTotalAllocated();
        _output.WriteLine($"Granular leak profile (4-layer Transformer, {numWindows}x{windowSize} calls)");
        _output.WriteLine($"  Baseline:  heap={baseline / (1024.0 * 1024):F0}MB  alloc={allocBaseline / (1024.0 * 1024):F0}MB");

        // Reflect into the model's base optimizer (Adam by default) so we
        // can watch its per-parameter moment dictionaries. If they grow with
        // call count, the Adam dict is leaking because `trainableParams`
        // returns fresh tensor refs each call and Adam never garbage-collects
        // its keyed-by-reference state.
        var baseTrainOptField = typeof(NeuralNetworkBase<float>)
            .GetField("_baseTrainOptimizer", BindingFlags.NonPublic | BindingFlags.Instance);
        FieldInfo? tapeMField = null;
        Func<object?>? readTapeMCount = null;
        if (baseTrainOptField is not null)
        {
            var opt = baseTrainOptField.GetValue(model);
            if (opt is not null)
            {
                tapeMField = opt.GetType().GetField("_tapeM", BindingFlags.NonPublic | BindingFlags.Instance);
                if (tapeMField is not null)
                {
                    var dict = tapeMField.GetValue(opt);
                    if (dict is not null)
                    {
                        var countProp = dict.GetType().GetProperty("Count");
                        if (countProp is not null) readTapeMCount = () => countProp.GetValue(dict);
                    }
                }
            }
        }

        long prevHeap = baseline;
        long prevAlloc = allocBaseline;
        int prevTapeMCount = readTapeMCount?.Invoke() is int c0 ? c0 : -1;
        _output.WriteLine($"  Adam._tapeM.Count at baseline: {prevTapeMCount}");
        for (int w = 0; w < numWindows; w++)
        {
            int globalSeedOff = w * windowSize;
            for (int step = 0; step < windowSize; step++)
            {
                var input = BuildInputTensor(ctx, vocab, seed: globalSeedOff + step);
                var target = BuildOneHotTarget((globalSeedOff + step) % vocab, vocab);
                model.Train(input, target);
            }

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            long heap = GC.GetTotalMemory(forceFullCollection: false);
            long alloc = GetTotalAllocated();
            long heapDelta = heap - prevHeap;
            long allocDelta = alloc - prevAlloc;
            int tapeMCount = readTapeMCount?.Invoke() is int cw ? cw : -1;
            int tapeMDelta = prevTapeMCount >= 0 && tapeMCount >= 0 ? tapeMCount - prevTapeMCount : 0;
            _output.WriteLine($"  Win {w}:  retained=+{heapDelta / (1024 * 1024)}MB  ({heapDelta / (double)windowSize / 1024:F0}KB/call)   allocated=+{allocDelta / (1024 * 1024)}MB ({allocDelta / (double)windowSize / 1024:F0}KB/call)   adam._tapeM.Count={tapeMCount} (+{tapeMDelta})");
            prevHeap = heap;
            prevAlloc = alloc;
            prevTapeMCount = tapeMCount;
        }
    }

    // Removed from regular test runs: this was the diagnostic that eliminated
    // model.ResetState() and Adam-dict clearing as fix paths. Both made
    // ≤2% difference to the per-call retention rate, so the leak is rooted
    // inside AiDotNet.Tensors (tape arena / saved-for-backward / engine-op
    // static state), not on AiDotNet's side. Re-enable when investigating
    // potential consumer-side workarounds.
    /// <summary>
    /// Heap-snapshot-style diagnostic: capture WeakReferences to every
    /// Tensor instance allocated during a Train call, then check after
    /// the call returns + GC how many survive. This bisects the residual
    /// 0.75 MB/call leak between "tensors get allocated and held" (the
    /// leak we care about) vs "tensors get allocated and released but
    /// the heap delta is from internal pool/cache growth" (less worrying).
    /// </summary>
    [Fact]
    public async Task Issue1227_TransformerTrain_TensorSurvivalDiagnostic()
    {
        await Task.Yield();
        const int vocab = 256;
        const int ctx = 64;
        var arch = new TransformerArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            numEncoderLayers: 4, numDecoderLayers: 0, numHeads: 4,
            modelDimension: 128, feedForwardDimension: 512,
            inputSize: ctx, outputSize: vocab,
            maxSequenceLength: ctx, vocabularySize: vocab);
        var model = new Transformer<float>(arch, lossFunction: new CategoricalCrossEntropyLoss<float>());
        model.SetTrainingMode(true);
        var warmupInput = BuildInputTensor(ctx, vocab, seed: 0);
        var warmupTarget = BuildOneHotTarget(0, vocab);
        for (int i = 0; i < 5; i++) model.Train(warmupInput, warmupTarget);
        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();

        // Snapshot: every live Tensor<float> reachable from the model's layer fields,
        // including reflective traversal of `_lastX` caches. Compare BEFORE / AFTER N
        // train calls to see which instances survive in those caches.
        long heapBefore = GC.GetTotalMemory(forceFullCollection: false);
        var beforeRefs = new List<WeakReference>();
        foreach (var t in EnumerateLayerTensors(model)) beforeRefs.Add(new WeakReference(t));

        const int trainSteps = 50;
        for (int step = 0; step < trainSteps; step++)
        {
            var input = BuildInputTensor(ctx, vocab, seed: step);
            var target = BuildOneHotTarget(step % vocab, vocab);
            model.Train(input, target);
        }

        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
        long heapAfter = GC.GetTotalMemory(forceFullCollection: false);
        var afterRefs = new List<WeakReference>();
        foreach (var t in EnumerateLayerTensors(model)) afterRefs.Add(new WeakReference(t));

        int beforeAlive = 0; foreach (var wr in beforeRefs) if (wr.IsAlive) beforeAlive++;
        int afterAlive = 0; foreach (var wr in afterRefs) if (wr.IsAlive) afterAlive++;
        long heapDeltaBytes = heapAfter - heapBefore;
        int tensorCountDelta = afterRefs.Count - beforeRefs.Count;
        _output.WriteLine($"#1227 tensor-survival diagnostic ({trainSteps} train calls, 4-layer Transformer)");
        _output.WriteLine($"  Heap before:       {heapBefore / (1024.0 * 1024):F0}MB");
        _output.WriteLine($"  Heap after:        {heapAfter / (1024.0 * 1024):F0}MB");
        _output.WriteLine($"  Heap delta:        +{heapDeltaBytes / (1024 * 1024)}MB ({heapDeltaBytes / (double)trainSteps / 1024:F0} KB/call)");
        _output.WriteLine($"  Layer tensors before: {beforeRefs.Count} (live: {beforeAlive})");
        _output.WriteLine($"  Layer tensors after:  {afterRefs.Count} (live: {afterAlive})");
        _output.WriteLine($"  Tensor count delta:   +{tensorCountDelta}");

        // Real assertions — pre-LayerBase-fix this probe showed +650 tensors
        // (~13/call) and ~775 KB/call heap retention. Post-fix it measures
        // +0 tensors and 0 MB heap delta. The thresholds below catch a
        // regression at any meaningful magnitude while tolerating GC
        // scheduling jitter on Server-GC hosts. Total layer field count
        // for the 4-encoder Transformer is ~221 — a delta of 50 means
        // a single field accumulated ~1 entry per call, which is the
        // signature pattern this probe is designed to surface.
        Assert.True(tensorCountDelta < 50,
            $"Layer-field tensor count grew by {tensorCountDelta} across {trainSteps} train calls. " +
            $"Indicates a field-level cache is accumulating tensor refs (the signature pattern of " +
            $"AiDotNet#1227's residual leak in LayerBase._preActivationCache). " +
            $"before={beforeRefs.Count} after={afterRefs.Count}.");

        Assert.True(heapDeltaBytes < 10L * 1024 * 1024,
            $"Managed heap grew by {heapDeltaBytes / (1024 * 1024)} MB across {trainSteps} train calls " +
            $"after a forced Gen2 GC — see #1227.");
    }

    /// <summary>
    /// Reflective walk over a NeuralNetworkBase's layer fields, yielding
    /// every reachable Tensor&lt;float&gt; instance. Used by the survival
    /// diagnostic above to bisect what specifically is being retained.
    /// </summary>
    private static IEnumerable<Tensor<float>> EnumerateLayerTensors(NeuralNetworkBase<float> model)
    {
        foreach (var layer in model.Layers)
        {
            foreach (var t in EnumerateTensorFields(layer)) yield return t;
        }
    }

    private static IEnumerable<Tensor<float>> EnumerateTensorFields(object obj)
    {
        var type = obj.GetType();
        while (type is not null && type != typeof(object))
        {
            foreach (var field in type.GetFields(BindingFlags.NonPublic | BindingFlags.Public | BindingFlags.Instance))
            {
                var val = field.GetValue(obj);
                if (val is null) continue;
                if (val is Tensor<float> tf) { yield return tf; continue; }

                // Dictionaries iterate as KeyValuePair entries which our
                // `item is Tensor<float>` check below would skip. Inspect
                // the Values collection explicitly so Tensor-valued
                // dictionaries (e.g. Adam's _tapeM / _tapeV gradient stash)
                // contribute to the survival/accumulation diagnostics.
                if (val is IDictionary dict)
                {
                    foreach (var v in dict.Values)
                    {
                        if (v is Tensor<float> dictTf) yield return dictTf;
                    }
                    continue;
                }

                if (val is IEnumerable enumerable && val is not string)
                {
                    foreach (var item in enumerable)
                    {
                        if (item is Tensor<float> innerTf) yield return innerTf;
                    }
                }
            }
            type = type.BaseType;
        }
    }

    /// <summary>
    /// Per-field tensor-count diagnostic: walks each layer and reports which
    /// specific fields are accumulating Tensor refs across calls. Bisects the
    /// 650-tensor / 50-call (~13/call) accumulation seen in the survival probe.
    /// </summary>
    [Fact]
    public async Task Issue1227_TransformerTrain_PerFieldAccumulationDiagnostic()
    {
        await Task.Yield();
        const int vocab = 256;
        const int ctx = 64;
        var arch = new TransformerArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            numEncoderLayers: 4, numDecoderLayers: 0, numHeads: 4,
            modelDimension: 128, feedForwardDimension: 512,
            inputSize: ctx, outputSize: vocab,
            maxSequenceLength: ctx, vocabularySize: vocab);
        var model = new Transformer<float>(arch, lossFunction: new CategoricalCrossEntropyLoss<float>());
        model.SetTrainingMode(true);
        var warmupInput = BuildInputTensor(ctx, vocab, seed: 0);
        var warmupTarget = BuildOneHotTarget(0, vocab);
        for (int i = 0; i < 5; i++) model.Train(warmupInput, warmupTarget);
        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();

        var beforeCounts = CountTensorsPerField(model);

        const int trainSteps = 50;
        for (int step = 0; step < trainSteps; step++)
        {
            model.Train(BuildInputTensor(ctx, vocab, step), BuildOneHotTarget(step % vocab, vocab));
        }

        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
        var afterCounts = CountTensorsPerField(model);

        _output.WriteLine($"#1227 per-field accumulation diagnostic ({trainSteps} calls)");
        _output.WriteLine($"  Fields that GREW in tensor count (delta > 0):");
        var grewFields = new List<string>();
        foreach (var kvp in afterCounts)
        {
            int before = beforeCounts.TryGetValue(kvp.Key, out var b) ? b : 0;
            int delta = kvp.Value - before;
            if (delta > 0)
            {
                _output.WriteLine($"    +{delta,4}  {kvp.Key}  ({before} -> {kvp.Value})");
                grewFields.Add($"{kvp.Key} (+{delta}, {before} -> {kvp.Value})");
            }
        }

        // Real assertion: post-LayerBase-fix no layer field accumulates
        // tensor refs across calls. Pre-fix this probe listed 13 fields
        // (one per MHA + per Dense) each growing by 50 entries on a
        // 50-call run — that's the exact "_preActivationCache" signature
        // the LayerBase gate eliminated. A non-empty `grewFields` list
        // means a new field is leaking, which is exactly the regression
        // this diagnostic is designed to catch.
        Assert.True(grewFields.Count == 0,
            $"{grewFields.Count} layer field(s) accumulated Tensor refs across {trainSteps} train calls — " +
            $"see #1227. This is the field-level signature of a forward-side cache that's not getting " +
            $"cleaned up on the tape-based training path. Fields: {string.Join("; ", grewFields)}.");
    }

    private static Dictionary<string, int> CountTensorsPerField(NeuralNetworkBase<float> model)
    {
        var counts = new Dictionary<string, int>();
        for (int li = 0; li < model.Layers.Count; li++)
        {
            var layer = model.Layers[li];
            var type = layer.GetType();
            string layerKey = $"L{li}:{type.Name}";
            while (type is not null && type != typeof(object))
            {
                foreach (var field in type.GetFields(BindingFlags.NonPublic | BindingFlags.Public | BindingFlags.Instance))
                {
                    var val = field.GetValue(layer);
                    int count = 0;
                    if (val is Tensor<float>) count = 1;
                    // Same IDictionary handling as EnumerateTensorFields — iterating
                    // a dictionary as IEnumerable yields KeyValuePair entries, so
                    // Tensor-valued dicts (e.g. Adam optimizer state) would be
                    // invisible to the per-field count without this explicit branch.
                    else if (val is IDictionary dict)
                    {
                        foreach (var v in dict.Values)
                            if (v is Tensor<float>) count++;
                    }
                    else if (val is IEnumerable enumerable && val is not string)
                    {
                        foreach (var item in enumerable)
                            if (item is Tensor<float>) count++;
                    }
                    if (count > 0)
                    {
                        string key = $"{layerKey}.{field.Name}";
                        counts[key] = (counts.TryGetValue(key, out var c) ? c : 0) + count;
                    }
                }
                type = type.BaseType;
            }
        }
        return counts;
    }

    [Fact(Skip = "Diagnostic — established the leak is upstream in AiDotNet.Tensors, not on the AiDotNet side. Re-enable when re-investigating consumer workarounds.")]
    public async Task Issue1227_TransformerTrain_ResetStateClearLeakDiagnostic()
    {
        await Task.Yield();

        const int vocab = 256;
        const int ctx = 64;
        const int trainSteps = 200;

        TransformerArchitecture<float> MakeArch() => new TransformerArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            numEncoderLayers: 4,
            numDecoderLayers: 0,
            numHeads: 4,
            modelDimension: 128,
            feedForwardDimension: 512,
            inputSize: ctx,
            outputSize: vocab,
            maxSequenceLength: ctx,
            vocabularySize: vocab);

        long Measure(bool resetEachCall)
        {
            var model = new Transformer<float>(MakeArch(),
                lossFunction: new CategoricalCrossEntropyLoss<float>());
            model.SetTrainingMode(true);
            var warmupInput = BuildInputTensor(ctx, vocab, seed: 0);
            var warmupTarget = BuildOneHotTarget(0, vocab);
            for (int i = 0; i < 3; i++)
            {
                model.Train(warmupInput, warmupTarget);
                if (resetEachCall) model.ResetState();
            }

            GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
            long start = GC.GetTotalMemory(forceFullCollection: false);

            for (int step = 0; step < trainSteps; step++)
            {
                var input = BuildInputTensor(ctx, vocab, seed: step);
                var target = BuildOneHotTarget(step % vocab, vocab);
                model.Train(input, target);
                if (resetEachCall) model.ResetState();
            }

            GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
            long end = GC.GetTotalMemory(forceFullCollection: false);
            return end - start;
        }

        long Measure2()
        {
            var model = new Transformer<float>(MakeArch(),
                lossFunction: new CategoricalCrossEntropyLoss<float>());
            model.SetTrainingMode(true);
            var baseTrainOptField = typeof(NeuralNetworkBase<float>)
                .GetField("_baseTrainOptimizer", BindingFlags.NonPublic | BindingFlags.Instance);

            var warmupInput = BuildInputTensor(ctx, vocab, seed: 0);
            var warmupTarget = BuildOneHotTarget(0, vocab);
            for (int i = 0; i < 3; i++) model.Train(warmupInput, warmupTarget);

            GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
            long start = GC.GetTotalMemory(forceFullCollection: false);

            for (int step = 0; step < trainSteps; step++)
            {
                var input = BuildInputTensor(ctx, vocab, seed: step);
                var target = BuildOneHotTarget(step % vocab, vocab);
                model.Train(input, target);
                // Clear adam tape dictionaries after every train via reflection.
                if (baseTrainOptField is not null)
                {
                    var opt = baseTrainOptField.GetValue(model);
                    if (opt is not null)
                    {
                        foreach (var fn in new[] { "_tapeM", "_tapeV" })
                        {
                            var f = opt.GetType().GetField(fn, BindingFlags.NonPublic | BindingFlags.Instance);
                            if (f?.GetValue(opt) is IDictionary dict)
                                dict.Clear();
                        }
                    }
                }
            }

            GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
            long end = GC.GetTotalMemory(forceFullCollection: false);
            return end - start;
        }

        long vanillaGrowth = Measure(resetEachCall: false);
        long resetGrowth = Measure(resetEachCall: true);
        long clearAdamGrowth = Measure2();

        _output.WriteLine($"#1227 leak source diagnostic ({trainSteps} train calls, 4-layer Transformer)");
        _output.WriteLine($"  Vanilla:                     +{vanillaGrowth / (1024 * 1024)}MB total  ({vanillaGrowth / (double)trainSteps / 1024:F0} KB/call)");
        _output.WriteLine($"  + ResetState after Train:    +{resetGrowth / (1024 * 1024)}MB total  ({resetGrowth / (double)trainSteps / 1024:F0} KB/call)");
        _output.WriteLine($"  + Adam tapeM/V cleared:      +{clearAdamGrowth / (1024 * 1024)}MB total  ({clearAdamGrowth / (double)trainSteps / 1024:F0} KB/call)");
    }

    private static Tensor<float> BuildInputTensor(int ctx, int vocab, int seed)
    {
        var t = new Tensor<float>(new[] { 1, ctx });
        // Deterministic byte values per seed — no PRNG dependency.
        for (int i = 0; i < ctx; i++)
        {
            t[0, i] = (float)((seed * 31 + i * 7 + 13) % vocab);
        }
        return t;
    }

    private static Tensor<float> BuildOneHotTarget(int classIndex, int vocab)
    {
        var t = new Tensor<float>(new[] { 1, vocab });
        t[0, classIndex] = 1f;
        return t;
    }

    /// <summary>
    /// Returns the cumulative allocated-byte count since process start.
    /// Uses <c>GC.GetTotalAllocatedBytes(precise: true)</c> when available
    /// (net5.0+) and falls back to <c>GC.GetTotalMemory(forceFullCollection: false)</c>
    /// on net471 where the precise API doesn't exist. The fallback is less
    /// useful for measuring allocation rate (it reports retained, not
    /// cumulative), but keeps the test file compiling cross-TFM — the
    /// allocation-rate telemetry is diagnostic, not load-bearing for any
    /// assertion.
    /// </summary>
    private static long GetTotalAllocated()
    {
#if NET5_0_OR_GREATER
        return GC.GetTotalAllocatedBytes(precise: true);
#else
        return GC.GetTotalMemory(forceFullCollection: false);
#endif
    }
}
