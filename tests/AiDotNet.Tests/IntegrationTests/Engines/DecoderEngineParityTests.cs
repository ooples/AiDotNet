using System;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.IntegrationTests.Engines;

/// <summary>
/// Asserts that a decoder-style forward gives the same answer on the CPU engine and the GPU engine.
/// </summary>
/// <remarks>
/// <para>
/// This exists because a wrong number went unexplained for far too long. A harness reported a perplexity of
/// 49,494 where the CPU path reported 15.53 on identical weights, and nothing in this repository could answer
/// whether the library's own forward had diverged across engines or whether the harness had. Answering it meant
/// hand-writing a probe, and the answer turned out to be that the library was fine.
/// </para>
/// <para>
/// A standing assertion turns "which component is wrong?" into a test result. The tolerance is relative and
/// deliberately loose: two engines accumulate in different orders, so exact equality is the wrong bar. What
/// matters is that the difference stays at float32 noise instead of growing with sequence length, which is the
/// signature of a real corruption rather than rounding.
/// </para>
/// <para>
/// The engine is process-global, so this class joins the serialized GPU collection: running it beside another
/// test that swaps the engine would make both meaningless.
/// </para>
/// </remarks>
[Collection("DiffusionGpuCuda")]
public sealed class DecoderEngineParityTests
{
    private readonly ITestOutputHelper _output;

    public DecoderEngineParityTests(ITestOutputHelper output) => _output = output;

    private const int VocabSize = 64;
    private const int ContextLength = 16;
    private const double RelativeTolerance = 1e-3;

    private static Tensor<float> Window()
    {
        // Deterministic ids: identical input on both engines, so any difference is the engine's.
        var ids = new Tensor<float>(new[] { 1, ContextLength });
        for (int i = 0; i < ContextLength; i++) ids[0, i] = ((i * 37) % (VocabSize - 1)) + 1;
        return ids;
    }

    private static double Checksum(Tensor<float> t)
    {
        // Position-weighted, so a permuted or shifted result cannot cancel to the same value.
        double sum = 0;
        var flat = t.ToArray();
        for (int i = 0; i < flat.Length; i++) sum += flat[i] * (((i % 7) + 1) * 0.125);
        return sum;
    }

    [SkippableFact]
    [Trait("category", "gpu")]
    public void A_transformer_forward_agrees_between_the_cpu_and_gpu_engines()
    {
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); }
        catch { /* no GPU backend on this host */ }

        var previous = AiDotNetEngine.Current;
        double cpu, accelerated;
        try
        {
            // Skip inside the try so the finally still disposes an engine the ctor succeeded in creating.
            Skip.If(gpu is null || !gpu.SupportsGpu || !gpu.IsGpuAvailable,
                "No GPU engine available on this host; cross-engine parity cannot be measured.");

            // ONE model, both engines. Constructing a Transformer per engine would give each run its own
            // random initialisation, and the resulting difference would measure the initialiser rather than
            // the engines — a mistake this test previously made and which produced a 66% "divergence".
            AiDotNetEngine.Current = new CpuEngine();
            var model = BuildModel();
            model.SetTrainingMode(false);
            var window = Window();

            cpu = Checksum(model.Predict(window));

            AiDotNetEngine.Current = gpu!;
            accelerated = Checksum(model.Predict(window));
        }
        finally
        {
            AiDotNetEngine.Current = previous;
            gpu?.Dispose();
        }

        double scale = Math.Max(Math.Abs(cpu), Math.Abs(accelerated));
        double relative = scale > 0 ? Math.Abs(cpu - accelerated) / scale : 0.0;

        _output.WriteLine($"cpu={cpu:G17}  gpu={accelerated:G17}  relative={relative:G6}");

        Assert.True(
            relative <= RelativeTolerance,
            $"The CPU and GPU decoder forwards diverged beyond float32 noise: cpu={cpu:G17}, gpu={accelerated:G17}, " +
            $"relative={relative:G6} (tolerance {RelativeTolerance:G3}). Accumulation order explains a difference " +
            "of this shape only when it stays small; a large or length-dependent one means a real corruption in " +
            "one engine's path.");
    }

    private static NeuralNetworkBase<float> BuildModel()
    {
        var architecture = new TransformerArchitecture<float>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.SequenceClassification,
            numEncoderLayers: 0,
            numDecoderLayers: 2,
            numHeads: 2,
            modelDimension: 32,
            feedForwardDimension: 64,
            NetworkComplexity.Simple,
            inputSize: VocabSize,
            outputSize: VocabSize);

        return new Transformer<float>(architecture);
    }
}
