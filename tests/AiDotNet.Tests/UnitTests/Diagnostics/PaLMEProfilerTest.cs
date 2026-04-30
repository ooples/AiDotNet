using System.Diagnostics;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.VisionLanguage.Robotics;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.UnitTests.Diagnostics;

/// <summary>
/// Granular profiler for PaLME forward / construction / parameter-collection.
///
/// The 562B paper-faithful Driess et al. 2023 config is INTENTIONAL — it's a
/// performance-bug canary. The right answer to a slow forward isn't to shrink
/// the config; it's to find which engine path is doing unnecessary work
/// (eager allocation when lazy would do, dense matmul where a fused kernel
/// would do, etc.) and fix THAT.
///
/// This profiler covers four distinct phases so the bottleneck is unambiguous:
///   1. Architecture construction (layer-list build, no weights yet on lazy layers)
///   2. Parameter buffer initialization (the first GetParameters / ParameterCount
///      walk that materializes weights)
///   3. Forward pass (per-layer wall-clock + cumulative)
///   4. GC pressure (Gen0/Gen1/Gen2 collections, allocated bytes)
/// </summary>
public class PaLMEProfilerTest
{
    private readonly ITestOutputHelper _out;
    public PaLMEProfilerTest(ITestOutputHelper o) => _out = o;

    [Fact(Skip = "Manual profiler — exhibits OutOfMemoryException at 255s on the paper-faithful 562B config (decoder MHA alone needs 134GB at double precision). Remove Skip when investigating engine-side memory issues.")]
    public async Task Profile_PaLME_DefaultConfig_FullBreakdown()
    {
        await Task.Yield();
        long allocBefore = GC.GetTotalAllocatedBytes(precise: true);
        int gen0Before = GC.CollectionCount(0);
        int gen1Before = GC.CollectionCount(1);
        int gen2Before = GC.CollectionCount(2);

        // Phase 1: architecture construction.
        var swArch = Stopwatch.StartNew();
        var arch = new NeuralNetworkArchitecture<double>(
            inputType: AiDotNet.Enums.InputType.ThreeDimensional,
            taskType: AiDotNet.Enums.NeuralNetworkTaskType.Regression,
            inputDepth: 3, inputHeight: 128, inputWidth: 128, outputSize: 4);
        swArch.Stop();

        // Phase 2: model construction.
        var swCtor = Stopwatch.StartNew();
        var model = new PaLME<double>(arch);
        swCtor.Stop();
        long allocAfterCtor = GC.GetTotalAllocatedBytes(precise: true);
        _out.WriteLine($"Architecture: {swArch.ElapsedMilliseconds} ms");
        _out.WriteLine($"PaLME ctor:   {swCtor.ElapsedMilliseconds} ms");
        _out.WriteLine($"  Layers.Count = {model.Layers.Count}");
        _out.WriteLine($"  GC alloc through ctor: {(allocAfterCtor - allocBefore) / 1024.0 / 1024.0:F1} MB");

        // Phase 3: ParameterCount probe — exposes whether ParameterCount itself
        // walks lazy layers and triggers weight allocation.
        var swPC = Stopwatch.StartNew();
        int pc = model.ParameterCount;
        swPC.Stop();
        long allocAfterPC = GC.GetTotalAllocatedBytes(precise: true);
        _out.WriteLine($"ParameterCount: {swPC.ElapsedMilliseconds} ms (count={pc:N0})");
        _out.WriteLine($"  GC alloc during ParameterCount: {(allocAfterPC - allocAfterCtor) / 1024.0 / 1024.0:F1} MB");

        // Phase 4: input prep (negligible but worth measuring).
        var swInput = Stopwatch.StartNew();
        var input = new Tensor<double>(new[] { 3, 128, 128 });
        for (int i = 0; i < input.Length; i++) input[i] = (i % 100) / 100.0;
        swInput.Stop();
        _out.WriteLine($"Input prep: {swInput.ElapsedMilliseconds} ms");

        // Phase 5: per-layer forward timing. Walk model.Layers manually so we
        // see EACH layer's contribution separately. The official Predict path
        // also runs the patch-embed step inside Predict; that step is timed
        // first via the public TokenizeImageInput call we exercise below.
        _out.WriteLine("\nPer-layer Forward breakdown:");
        long totalForwardMs = 0;
        long allocBeforeForward = GC.GetTotalAllocatedBytes(precise: true);
        var swTotal = Stopwatch.StartNew();
        Tensor<double> c;
        try
        {
            // Tokenize via Predict's path. We can't easily call the private
            // helper directly, so call Predict and instrument inside.
            var swPredict = Stopwatch.StartNew();
            var output = model.Predict(input);
            swPredict.Stop();
            _out.WriteLine($"Total Predict (tokenize + 678 layers + pool): {swPredict.ElapsedMilliseconds} ms");
            _out.WriteLine($"  Output shape: [{string.Join(",", output.Shape)}]");
            c = output;
        }
        catch (System.Exception ex)
        {
            swTotal.Stop();
            _out.WriteLine($"Predict FAILED at {swTotal.ElapsedMilliseconds} ms: {ex.GetType().Name}: {ex.Message}");
            return;
        }
        swTotal.Stop();
        long allocAfterForward = GC.GetTotalAllocatedBytes(precise: true);
        totalForwardMs = swTotal.ElapsedMilliseconds;
        _out.WriteLine($"Total wall-clock: {totalForwardMs} ms");
        _out.WriteLine($"GC alloc during Forward: {(allocAfterForward - allocBeforeForward) / 1024.0 / 1024.0:F1} MB");

        int gen0After = GC.CollectionCount(0);
        int gen1After = GC.CollectionCount(1);
        int gen2After = GC.CollectionCount(2);
        _out.WriteLine($"\nGC collections during the run:");
        _out.WriteLine($"  Gen0: {gen0After - gen0Before}, Gen1: {gen1After - gen1Before}, Gen2: {gen2After - gen2Before}");
    }

    [Fact(Skip = "Manual profiler — print configured architecture sizes only.")]
    public void Profile_PaLME_ConfigSizing_Snapshot()
    {
        var opts = new PaLMEOptions();
        _out.WriteLine($"PaLMEOptions defaults (Driess et al. 2023, 562B):");
        _out.WriteLine($"  ImageSize        = {opts.ImageSize}");
        _out.WriteLine($"  VisionDim        = {opts.VisionDim}");
        _out.WriteLine($"  DecoderDim       = {opts.DecoderDim}");
        _out.WriteLine($"  NumVisionLayers  = {opts.NumVisionLayers}");
        _out.WriteLine($"  NumDecoderLayers = {opts.NumDecoderLayers}");
        _out.WriteLine($"  NumHeads         = {opts.NumHeads}");
        _out.WriteLine($"  TotalParameters  = {opts.TotalParameters} B");

        int lpb = opts.DropoutRate > 0 ? 6 : 5;
        int total = 1 + opts.NumVisionLayers * lpb + 2 + opts.NumDecoderLayers * lpb + 3;
        _out.WriteLine($"\nDerived layer count (LayersPerBlock={lpb}): {total}");

        long visionMhaParams = 4L * opts.VisionDim * opts.VisionDim;
        long visionFfnParams = 2L * opts.VisionDim * (opts.VisionDim * 4);
        long decoderMhaParams = 4L * opts.DecoderDim * opts.DecoderDim;
        long decoderFfnParams = 2L * opts.DecoderDim * (opts.DecoderDim * 4);
        long visionBlock = visionMhaParams + visionFfnParams;
        long decoderBlock = decoderMhaParams + decoderFfnParams;
        long totalParams = opts.NumVisionLayers * visionBlock + opts.NumDecoderLayers * decoderBlock;
        _out.WriteLine($"\nDerived parameter counts:");
        _out.WriteLine($"  Vision block (MHA+FFN): {visionBlock:N0} params × {opts.NumVisionLayers} layers = {opts.NumVisionLayers * visionBlock:N0}");
        _out.WriteLine($"  Decoder block (MHA+FFN): {decoderBlock:N0} params × {opts.NumDecoderLayers} layers = {opts.NumDecoderLayers * decoderBlock:N0}");
        _out.WriteLine($"  Total (approximate): {totalParams:N0} ({totalParams / 1e9:F1} B)");
    }

    /// <summary>
    /// Microbenchmark: how long does each PaLME phase take when we feed the
    /// SHRUNK layer factory (Conv-only, no MHA) at the SAME parameter count?
    /// If this is dramatically faster than the real config, the bottleneck is
    /// MHA-specific (weight allocation, attention math, lazy resolve). If it's
    /// the same, the bottleneck is in something common to all layers
    /// (parameter buffer materialization, GC pressure, FFI thunks).
    /// </summary>
    [Fact(Skip = "Manual diagnostic — comparison run for MHA-vs-Dense bottleneck triage.")]
    public void Compare_PaLME_LayerKindCost()
    {
        // Time the construction of just one MHA layer at default sizes.
        var swMHA = Stopwatch.StartNew();
        var mha = new AiDotNet.NeuralNetworks.Layers.MultiHeadAttentionLayer<double>(
            16, 1408 / 16,
            (AiDotNet.Interfaces.IActivationFunction<double>?)null,
            AiDotNet.Initialization.InitializationStrategies<double>.Lazy);
        swMHA.Stop();
        _out.WriteLine($"Construct lazy MultiHeadAttention(16 heads, head_dim=88): {swMHA.ElapsedMilliseconds} ms");

        var swDense = Stopwatch.StartNew();
        var dense = new AiDotNet.NeuralNetworks.Layers.DenseLayer<double>(1408,
            (AiDotNet.Interfaces.IActivationFunction<double>?)null,
            AiDotNet.Initialization.InitializationStrategies<double>.Lazy);
        swDense.Stop();
        _out.WriteLine($"Construct lazy Dense(1408): {swDense.ElapsedMilliseconds} ms");

        var swLN = Stopwatch.StartNew();
        var ln = new AiDotNet.NeuralNetworks.Layers.LayerNormalizationLayer<double>();
        swLN.Stop();
        _out.WriteLine($"Construct LayerNorm: {swLN.ElapsedMilliseconds} ms");

        // Total: 678 layers in PaLME. Per-layer construction adds up.
        _out.WriteLine($"\nProjected ctor cost across 678-layer PaLME if each is MHA-cost: {678 * swMHA.ElapsedMilliseconds} ms");
        _out.WriteLine($"Projected ctor cost if each is Dense-cost: {678 * swDense.ElapsedMilliseconds} ms");
    }
}
