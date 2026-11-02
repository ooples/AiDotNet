// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System.Threading.Tasks;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// A GPU DenseBlock inference forward frees each dense layer's intermediates when that layer is done (AiDotNet.Tensors
/// #1573, ooples/AiDotNet#2383). Held to the end of the call, a batch-16 DenseNet-121 inference peaked at 6.4-7.7 GB
/// against PyTorch's 1.3 GB. Under an enclosing <see cref="TensorScope"/>, only the running concatenation each layer
/// keeps reaches it; every BatchNorm, ReLU and convolution output went there before. Skips with no GPU backend.
/// </summary>
// Sets AiDotNetEngine.Current, a process-wide static: run apart from every other test class.
[Collection("EngineCurrentGlobalState")]
public class DenseBlockGpuInferenceScopeTests
{
    private const int NumLayers = 4, GrowthRate = 8, Batch = 2, Channels = 16, Size = 8;
    private readonly ITestOutputHelper _output;
    public DenseBlockGpuInferenceScopeTests(ITestOutputHelper output) => _output = output;

    [Fact(Timeout = 120000)]
    public async Task GpuInference_ReleasesEachLayersIntermediates_AndMatchesCpu()
    {
        await Task.Yield();
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (System.Exception ex) { _output.WriteLine($"No GPU engine: {ex.GetType().Name}: {ex.Message}"); }
        if (gpu is null || !gpu.SupportsGpu)
        {
            _output.WriteLine("No GPU backend available: skipping the DenseBlock inference scope test.");
            return;
        }

        var previous = AiDotNetEngine.Current;
        try
        {
            var block = new DenseBlock<float>(NumLayers, GrowthRate);
            var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(1573);
            var input = new Tensor<float>(new[] { Batch, Channels, Size, Size });
            for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

            block.SetTrainingMode(false);
            AiDotNetEngine.Current = previous;
            float[] cpu = block.Forward(input).ToArray();

            AiDotNetEngine.Current = gpu;
            float[] onGpu;
            int reachedEnclosingScope;
            using (var scope = TensorScope.Begin())
            {
                var output = block.Forward(input);
                reachedEnclosingScope = scope.TrackedCount;
                onGpu = scope.Keep(output).ToArray();
            }

            Assert.Equal(cpu.Length, onGpu.Length);
            for (int i = 0; i < cpu.Length; i++)
                Assert.True(System.Math.Abs(cpu[i] - onGpu[i]) <= 1e-3f,
                    $"output {i}: GPU {onGpu[i]} vs CPU {cpu[i]}");

            // One running concatenation per dense layer, each kept by that layer's scope.
            _output.WriteLine($"GPU results reaching the enclosing scope: {reachedEnclosingScope}");
            Assert.True(reachedEnclosingScope <= NumLayers,
                $"{reachedEnclosingScope} GPU results outlived their dense layer, at most {NumLayers} (the kept concatenations) should");
        }
        finally
        {
            AiDotNetEngine.Current = previous;
            gpu.Dispose();
        }
    }
}
