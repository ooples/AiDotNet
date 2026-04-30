using System.Diagnostics;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.VisionLanguage.Robotics;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.UnitTests.Diagnostics;

/// <summary>
/// Focused profiler for PaLME forward pass — surfaces which stage dominates the
/// 562B-default-config timeout so we can size the smoke-test default correctly.
///
/// Output prints per-stage wall-clock for: construction, layer-by-layer forward,
/// and total. Since the 562B production config is paper-faithful (Driess et al.
/// 2023) but unrunnable on CI, the goal here is to confirm where the time goes
/// and decide whether the fix is (a) a smaller research-scale default, (b) a
/// faster patch-embed, (c) a cheaper MHA path, or (d) some combination.
/// </summary>
public class PaLMEProfilerTest
{
    private readonly ITestOutputHelper _out;
    public PaLMEProfilerTest(ITestOutputHelper o) => _out = o;

    [Fact(Skip = "Manual profiler — run with --filter to enable when investigating PaLME timeouts.")]
    public void Profile_PaLME_DefaultConfig_Forward_TimeBreakdown()
    {
        var arch = new NeuralNetworkArchitecture<double>(
            inputType: AiDotNet.Enums.InputType.ThreeDimensional,
            taskType: AiDotNet.Enums.NeuralNetworkTaskType.Regression,
            inputDepth: 3, inputHeight: 128, inputWidth: 128, outputSize: 4);

        var swCtor = Stopwatch.StartNew();
        var model = new PaLME<double>(arch);
        swCtor.Stop();
        _out.WriteLine($"Construction: {swCtor.ElapsedMilliseconds} ms");
        _out.WriteLine($"  Layer count: {model.Layers.Count}");
        _out.WriteLine($"  ParameterCount: {model.ParameterCount}");

        var swInput = Stopwatch.StartNew();
        var input = new Tensor<double>(new[] { 3, 128, 128 });
        for (int i = 0; i < input.Length; i++) input[i] = (i % 100) / 100.0;
        swInput.Stop();
        _out.WriteLine($"Input prep: {swInput.ElapsedMilliseconds} ms");

        var swForward = Stopwatch.StartNew();
        try
        {
            var output = model.Predict(input);
            swForward.Stop();
            _out.WriteLine($"Total Forward: {swForward.ElapsedMilliseconds} ms (output length={output.Length})");
        }
        catch (System.Exception ex)
        {
            swForward.Stop();
            _out.WriteLine($"Forward FAILED at {swForward.ElapsedMilliseconds} ms: {ex.GetType().Name}: {ex.Message}");
        }

        // Per-layer breakdown — instrument the same iteration the model does.
        _out.WriteLine("\nPer-layer breakdown (layer, type, ms, output shape):");
        var c = (Tensor<double>)null!;
        for (int li = 0; li < model.Layers.Count; li++)
        {
            var layer = model.Layers[li];
            var sw = Stopwatch.StartNew();
            try
            {
                if (li == 0)
                {
                    // First layer needs the patched/reshaped input — skip per-layer probe
                    // for layer 0 since it sees a different shape than test.
                    sw.Stop();
                    _out.WriteLine($"  [{li:D2}] {layer.GetType().Name}: SKIPPED (input-side reshape)");
                    continue;
                }
                if (c == null)
                {
                    sw.Stop();
                    _out.WriteLine($"  [{li:D2}] {layer.GetType().Name}: SKIPPED (no chain)");
                    continue;
                }
                c = layer.Forward(c);
                sw.Stop();
                _out.WriteLine($"  [{li:D2}] {layer.GetType().Name}: {sw.ElapsedMilliseconds} ms, shape=[{string.Join(",", c.Shape)}], paramCount={layer.ParameterCount}");
            }
            catch (System.Exception ex)
            {
                sw.Stop();
                _out.WriteLine($"  [{li:D2}] {layer.GetType().Name}: FAILED at {sw.ElapsedMilliseconds} ms: {ex.GetType().Name}: {ex.Message}");
                break;
            }
        }
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

        // Per LayerHelper.CreateDefaultRoboticsActionLayers the layer count is:
        //   1 (input LN)
        // + numVisionLayers × layersPerBlock  (LN, MHA, Dense, Dense, LN, [Dropout])
        // + 2 (proj Dense, LN)
        // + numDecoderLayers × layersPerBlock
        // + 3 (action Dense, LN, action Dense)
        int lpb = opts.DropoutRate > 0 ? 6 : 5;
        int total = 1 + opts.NumVisionLayers * lpb + 2 + opts.NumDecoderLayers * lpb + 3;
        _out.WriteLine($"\nDerived layer count (LayersPerBlock={lpb}): {total}");

        // Per-MHA cost: weights are visionDim × visionDim or decoderDim × decoderDim
        // for each of Q/K/V/O = 4 projections.
        long visionMhaParams = 4L * opts.VisionDim * opts.VisionDim;
        long visionFfnParams = 2L * opts.VisionDim * (opts.VisionDim * 4); // up + down
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
}
