using System;
using System.Diagnostics;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNetTests.IntegrationTests.NeuralNetworks;

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
