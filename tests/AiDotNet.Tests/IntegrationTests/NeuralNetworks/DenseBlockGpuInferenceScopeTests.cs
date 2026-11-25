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
[Trait("Category", "GPU")]
public class DenseBlockGpuInferenceScopeTests
{
    private const int NumLayers = 4, GrowthRate = 8, Batch = 2, Channels = 16, Size = 8;
    private readonly ITestOutputHelper _output;
    public DenseBlockGpuInferenceScopeTests(ITestOutputHelper output) => _output = output;

    [SkippableFact(Timeout = 120000)]
    public async Task GpuInference_ReleasesEachLayersIntermediates_AndMatchesCpu()
    {
        await Task.Yield();
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (System.Exception ex) { _output.WriteLine($"No GPU engine: {ex.GetType().Name}: {ex.Message}"); }
        var previous = AiDotNetEngine.Current;
        try
        {
            Skip.If(gpu is null || !gpu.SupportsGpu || !gpu.IsGpuAvailable,
                "No GPU backend available for the DenseBlock inference scope test.");
            AiDotNetEngine.Current = new CpuEngine();
            var block = new DenseBlock<float>(NumLayers, GrowthRate);
            var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(1573);
            var input = new Tensor<float>(new[] { Batch, Channels, Size, Size });
            for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

            block.SetTrainingMode(false);
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
            gpu?.Dispose();
        }
    }

    /// <summary>
    /// A kept concatenation goes to the enclosing scope, so before each one was released at its last use a block's
    /// concatenations all stayed on the device until the enclosing scope ended: memory quadratic in the layer count. The
    /// peak of a deep block is now a small multiple of its output, however many layers produced it.
    /// </summary>
    [SkippableFact(Timeout = 120000)]
    public async Task GpuInference_PeakDeviceMemory_DoesNotHoldEveryConcatenation()
    {
        await Task.Yield();
        const int deepLayers = 16, growth = 16, batch = 4, channels = 16, size = 32;
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (System.Exception ex) { _output.WriteLine($"No GPU engine: {ex.GetType().Name}: {ex.Message}"); }
        var previous = AiDotNetEngine.Current;
        try
        {
            Skip.If(gpu is null || !gpu.SupportsGpu || !gpu.IsGpuAvailable,
                "No GPU backend available for the DenseBlock inference peak test.");
            AiDotNetEngine.Current = gpu;
            var backend = gpu.GetBackend();
            // The allocator peak shows what the block keeps alive only where a released buffer is back in the pool
            // before a later dispatch asks for one. Vulkan queues dispatches without waiting and returns a buffer when
            // the submission that last used it completes (VulkanDevice.DeferredSubmit), so its peak is the memory of
            // the queued work whatever the block has released: 164,904,704 bytes for this block, four times all of its
            // concatenations together.
            Skip.If(backend is AiDotNet.Tensors.Engines.DirectGpu.Vulkan.VulkanBackend,
                "Vulkan's allocator peak measures queued work, not what the block keeps alive.");
            var block = new DenseBlock<float>(deepLayers, growth);
            var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(2383);
            var input = new Tensor<float>(new[] { batch, channels, size, size });
            for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            block.SetTrainingMode(false);

            for (int warm = 0; warm < 2; warm++)
            {
                using var warmScope = TensorScope.Begin();
                _ = block.Forward(input);
                backend.Synchronize();
            }
            backend.Synchronize();

            long baseline = AiDotNet.Tensors.Engines.DevicePrimitives.GpuMemoryStats.Snapshot(backend).AllocatedBytes;
            AiDotNet.Tensors.Engines.DevicePrimitives.GpuMemoryStats.ResetPeakStats(backend);
            long peak;
            using (var scope = TensorScope.Begin())
            {
                _ = block.Forward(input);
                backend.Synchronize();
                peak = AiDotNet.Tensors.Engines.DevicePrimitives.GpuMemoryStats.Snapshot(backend).PeakAllocatedBytes - baseline;
            }
            backend.Synchronize();

            long plane = (long)batch * size * size * sizeof(float);
            long outputBytes = (channels + deepLayers * growth) * plane;
            // Every running concatenation at once: sum over k = 1..L of (c + k * growth) channels.
            long everyConcatenation = (deepLayers * channels + growth * deepLayers * (deepLayers + 1) / 2) * plane;
            _output.WriteLine($"peak {peak} bytes; block output {outputBytes}; every concatenation {everyConcatenation}");
            Assert.True(peak > 0, "The device allocator must observe the block's allocations.");
            Assert.True(peak < everyConcatenation / 2,
                $"A {deepLayers}-layer block peaked at {peak} bytes: its concatenations ({everyConcatenation} bytes in all) " +
                $"are being held to the end of the call; its output is {outputBytes} bytes.");
        }
        finally
        {
            AiDotNetEngine.Current = previous;
            gpu?.Dispose();
        }
    }
}
