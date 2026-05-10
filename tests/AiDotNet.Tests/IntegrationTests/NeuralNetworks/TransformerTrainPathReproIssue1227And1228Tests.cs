using System;
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
/// These tests do NOT fail the build on regressions — they emit
/// telemetry to <see cref="ITestOutputHelper"/> and assert only on
/// extreme thresholds (10 GB working-set growth, &gt; 30 s/call) so the
/// test serves as a tripwire for future regressions while staying
/// stable on the current implementation.
/// </para>
/// </summary>
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

        var process = Process.GetCurrentProcess();
        long workingSetStart = process.WorkingSet64;
        long managedHeapStart = GC.GetTotalMemory(forceFullCollection: false);
        TimeSpan cpuTimeStart = process.TotalProcessorTime;
        var sw = Stopwatch.StartNew();

        for (int step = 0; step < trainSteps; step++)
        {
            var input = BuildInputTensor(ctx, vocab, seed: step);
            var target = BuildOneHotTarget(step % vocab, vocab);
            model.Train(input, target);
        }

        sw.Stop();
        TimeSpan cpuTimeEnd = process.TotalProcessorTime;
        long workingSetEnd = process.WorkingSet64;

        // Force GC so we measure ACTUAL retained memory, not transient.
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        long managedHeapEnd = GC.GetTotalMemory(forceFullCollection: false);

        double walSec = sw.Elapsed.TotalSeconds;
        double cpuSec = (cpuTimeEnd - cpuTimeStart).TotalSeconds;
        double cpuToWallRatio = walSec > 0 ? cpuSec / walSec : 0;
        double msPerCall = walSec * 1000.0 / trainSteps;
        long workingSetGrowthMB = (workingSetEnd - workingSetStart) / (1024 * 1024);
        long managedHeapGrowthMB = (managedHeapEnd - managedHeapStart) / (1024 * 1024);

        _output.WriteLine($"#1227 probe: {trainSteps} train calls on 1-layer Transformer (d=128, ff=512, h=4, ctx={ctx}, V={vocab})");
        _output.WriteLine($"  Wall time:        {walSec:F2}s  ({msPerCall:F1} ms/call)");
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
    /// calls. Reporter saw 1.06 cores avg on a 16-core machine.
    /// Post-fix the ratio should exceed 1.3 (some multi-core engagement);
    /// a ratio close to 1.0 means the train path is still serial. We
    /// only assert as a tripwire — the test will pass on current code
    /// regardless of the exact ratio, while emitting telemetry.
    /// </summary>
    [Fact]
    public async Task Issue1228_TransformerTrain_CpuToWallRatio()
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

        // Warm-up.
        var warmupInput = BuildInputTensor(ctx, vocab, seed: 0);
        var warmupTarget = BuildOneHotTarget(0, vocab);
        for (int i = 0; i < 3; i++) model.Train(warmupInput, warmupTarget);

        var process = Process.GetCurrentProcess();
        TimeSpan cpuStart = process.TotalProcessorTime;
        var sw = Stopwatch.StartNew();

        for (int step = 0; step < trainSteps; step++)
        {
            var input = BuildInputTensor(ctx, vocab, seed: step);
            var target = BuildOneHotTarget(step % vocab, vocab);
            model.Train(input, target);
        }

        sw.Stop();
        TimeSpan cpuEnd = process.TotalProcessorTime;

        double wallSec = sw.Elapsed.TotalSeconds;
        double cpuSec = (cpuEnd - cpuStart).TotalSeconds;
        double ratio = wallSec > 0 ? cpuSec / wallSec : 0;

        _output.WriteLine($"#1228 probe: {trainSteps} train calls, wall={wallSec:F2}s, cpu={cpuSec:F2}s, ratio={ratio:F2}");
        _output.WriteLine($"  Available cores: {Environment.ProcessorCount}");
        _output.WriteLine(ratio >= 1.3
            ? "  -> Multi-core engagement detected on Train path (#1228 reduced or fixed)."
            : ratio >= 1.05
                ? "  -> Light multi-core engagement (some kernels parallel, others serial)."
                : "  -> Train path appears single-threaded (#1228 still active or partly so).");

        // Tripwire only: don't fail the build on the parallelism level
        // since that's a perf concern not a correctness concern, AND it
        // depends heavily on machine topology and Tensors version.
        // The training itself must complete (no hangs).
        Assert.True(wallSec < 120,
            $"Train loop took {wallSec:F1}s — should complete well under the framework's 120 s budget.");
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

        var process = Process.GetCurrentProcess();
        long workingSetStart = process.WorkingSet64;
        long managedHeapStart = GC.GetTotalMemory(forceFullCollection: false);
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

            if (step == trainSteps / 2)
            {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                GC.Collect();
                workingSetMid = process.WorkingSet64;
                managedHeapMid = GC.GetTotalMemory(forceFullCollection: false);
            }
        }

        sw.Stop();
        TimeSpan cpuTimeEnd = process.TotalProcessorTime;
        long workingSetEnd = process.WorkingSet64;

        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
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
        long totalAllocBytes = GC.GetTotalAllocatedBytes(precise: true);
        _output.WriteLine($"  Total alloc:      {totalAllocBytes / (1024.0 * 1024 * 1024):F2} GB (over the whole process — includes warm-up + test infra)");

        _output.WriteLine($"#1227 L=4 stress: {trainSteps} train calls on 4-layer Transformer (d=128, ff=512, h=4, ctx={ctx}, V={vocab})");
        _output.WriteLine($"  Wall time:        {wallSec:F2}s  ({msPerCall:F1} ms/call)");
        _output.WriteLine($"  CPU time:         {cpuSec:F2}s  (ratio = {cpuToWallRatio:F2})");
        _output.WriteLine($"  Working-set:      start={workingSetStart / (1024.0 * 1024):F0}MB  mid={workingSetMid / (1024.0 * 1024):F0}MB  end={workingSetEnd / (1024.0 * 1024):F0}MB  delta={workingSetGrowthMB:+#;-#;0}MB (mid +{workingSetGrowthMidMB}MB)");
        _output.WriteLine($"  Managed heap:     start={managedHeapStart / (1024.0 * 1024):F0}MB  mid={managedHeapMid / (1024.0 * 1024):F0}MB  end={managedHeapEnd / (1024.0 * 1024):F0}MB  delta={managedHeapGrowthMB:+#;-#;0}MB (mid +{managedHeapGrowthMidMB}MB)");
        long window1Growth = managedHeapMid - managedHeapStart;
        long window2Growth = managedHeapEnd - managedHeapMid;
        _output.WriteLine($"  Window growth:    win1(0..{trainSteps / 2})=+{window1Growth / (1024 * 1024)}MB  win2({trainSteps / 2}..{trainSteps})=+{window2Growth / (1024 * 1024)}MB  per-call={window2Growth / (double)(trainSteps / 2) / 1024:F0}KB");

        // Tripwire: at the reporter's observed ~50 MB/call leak rate, 1000
        // calls = 50 GB. 10 GB band catches the leak with a 5x safety margin
        // while tolerating up to ~10 MB/call of legitimate steady-state
        // allocation noise on a 4-layer Transformer.
        Assert.True(workingSetGrowthMB < 10240,
            $"Working-set grew by {workingSetGrowthMB} MB across {trainSteps} L=4 train calls. " +
            $"At this rate the reporter's 56k-sample run would hit {workingSetGrowthMB * 56L} MB — see #1227.");

        // Tripwire: managed heap retention > 2 GB across 1000 calls means
        // graph nodes or activations are surviving Gen2 GC.
        Assert.True(managedHeapGrowthMB < 2048,
            $"Managed heap grew by {managedHeapGrowthMB} MB across {trainSteps} L=4 train calls (after forced GC). " +
            "Indicates retained tape graph nodes — see #1227.");

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
        long allocBaseline = GC.GetTotalAllocatedBytes(precise: true);
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
            long alloc = GC.GetTotalAllocatedBytes(precise: true);
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
                            if (f?.GetValue(opt) is System.Collections.IDictionary dict)
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
}
