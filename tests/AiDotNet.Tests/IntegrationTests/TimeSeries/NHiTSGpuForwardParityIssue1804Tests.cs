// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Threading.Tasks;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TimeSeries;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.IntegrationTests.TimeSeries;

/// <summary>
/// #1804: N-HiTS sums per-stack forecasts the same way N-BEATS sums per-block ones. Each stack used to
/// return a permuted [B, H] view of a device-resident result, and TensorAdd of two such views read their
/// shared source buffers without the strides on the DirectGpu engine (AiDotNet.Tensors#1090). The batched
/// forward feeds both the GPU-resident accept/reject gate and the eager loss, so it must match the CPU
/// engine. Skips cleanly when no GPU backend is available.
/// </summary>
// Sets AiDotNetEngine.Current, a process-wide static: run apart from every other test class.
[Collection("EngineCurrentGlobalState")]
public class NHiTSGpuForwardParityIssue1804Tests
{
    private readonly ITestOutputHelper _output;
    public NHiTSGpuForwardParityIssue1804Tests(ITestOutputHelper output) => _output = output;

    // Pooling 8/4/1 (the defaults) over: a lookback every kernel divides; one that leaves a short last window
    // (50 = 6x8 + 2 = 12x4 + 2), the narrowed tail path; and one shorter than the first kernel (6 < 8), where
    // the whole lookback is that single short window.
    [SkippableTheory(Timeout = 120000)]
    [InlineData(48)]
    [InlineData(50)]
    [InlineData(6)]
    public async Task BatchedForward_OnGpu_MatchesCpu(int lookback)
    {
        await Task.Yield();
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) { _output.WriteLine($"No GPU backend: {ex.GetType().Name}: {ex.Message}"); }
        bool available = gpu is not null && gpu.SupportsGpu;
        if (!available) gpu?.Dispose();
        Skip.IfNot(available && gpu is not null, "No GPU backend available for the #1804 N-HiTS GPU parity check.");
        if (gpu is null) return;

        var previous = AiDotNetEngine.Current;
        try
        {
            const int horizon = 24, batch = 64;
            var model = new NHiTSModel<float>(new NHiTSOptions<float>
            {
                LookbackWindow = lookback, ForecastHorizon = horizon, BatchSize = batch,
            });

            var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(1804);
            var data = new float[batch * lookback];
            for (int i = 0; i < data.Length; i++) data[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

            AiDotNetEngine.Current = new CpuEngine();
            var cpuTensor = model.RunForwardBatched(new Tensor<float>(new[] { batch, lookback }, new Vector<float>((float[])data.Clone())));
            AiDotNetEngine.Current = gpu;
            var gpuTensor = model.RunForwardBatched(new Tensor<float>(new[] { batch, lookback }, new Vector<float>((float[])data.Clone())));

            Assert.NotNull(cpuTensor);
            Assert.NotNull(gpuTensor);
            float[] cpu = cpuTensor.ToArray();
            float[] gpuOut = gpuTensor.ToArray();
            Assert.Equal(batch * horizon, cpu.Length);
            Assert.Equal(cpu.Length, gpuOut.Length);
            for (int i = 0; i < cpu.Length; i++)
            {
                // GEMM summation order differs between engines; the layout defect was O(1) per element.
                Assert.True(Math.Abs(gpuOut[i] - cpu[i]) <= 1e-3f * (1f + Math.Abs(cpu[i])),
                    $"forecast[{i}]: gpu={gpuOut[i]} cpu={cpu[i]} — the GPU N-HiTS forward diverges from the CPU (#1804).");
            }
        }
        finally
        {
            AiDotNetEngine.Current = previous;
            gpu.Dispose();
        }
    }
}