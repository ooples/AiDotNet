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
/// #1804: the batched N-BEATS stack forward must give the same forecast on the DirectGpu engine as on the
/// CPU engine. Each block used to return permuted [B, L] / [B, H] views of device-resident matmul results,
/// and the stack summed those views with TensorAdd. The DirectGpu eager binary shortcut read the views'
/// shared source buffers without their strides (AiDotNet.Tensors#1090), so the GPU forecast was wrong: the
/// untrained holdout MSE was 1.12 on GPU against 1.46 on CPU. That rejected every GPU-resident training run
/// (its accept/reject gate scores through this forward) and the eager fallback then trained on wrong
/// gradients. Skips cleanly when no GPU backend is available.
/// </summary>
// Sets AiDotNetEngine.Current, a process-wide static: run apart from every other test class.
[Collection("EngineCurrentGlobalState")]
public class NBEATSGpuForwardParityIssue1804Tests
{
    private readonly ITestOutputHelper _output;
    public NBEATSGpuForwardParityIssue1804Tests(ITestOutputHelper output) => _output = output;

    [Fact(Timeout = 120000)]
    public async Task StackForward_OnGpu_MatchesCpu()
    {
        await Task.Yield();
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); } catch { /* no backend */ }
        if (gpu is null || !gpu.SupportsGpu)
        {
            _output.WriteLine("No GPU backend available — skipping #1804 GPU parity check.");
            gpu?.Dispose();
            return;
        }

        var previous = AiDotNetEngine.Current;
        try
        {
            // The #1804 workload's block shape: generic basis, 3x3 blocks, hidden 256, L=96, H=12, batch 64.
            const int lookback = 96, horizon = 12, batch = 64;
            var model = new NBEATSModel<float>(new NBEATSModelOptions<float>
            {
                NumStacks = 3, NumBlocksPerStack = 3, HiddenLayerSize = 256, NumHiddenLayers = 4,
                LookbackWindow = lookback, ForecastHorizon = horizon, BatchSize = batch,
                UseInterpretableBasis = false,
            });

            var rng = new Random(1804);
            var data = new float[batch * lookback];
            for (int i = 0; i < data.Length; i++) data[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

            AiDotNetEngine.Current = new CpuEngine();
            float[] cpu = model.RunForwardStack(new Tensor<float>(new[] { batch, lookback }, new Vector<float>((float[])data.Clone()))).ToArray();

            AiDotNetEngine.Current = gpu;
            float[] gpuOut = model.RunForwardStack(new Tensor<float>(new[] { batch, lookback }, new Vector<float>((float[])data.Clone()))).ToArray();

            Assert.Equal(batch * horizon, cpu.Length);
            Assert.Equal(cpu.Length, gpuOut.Length);
            for (int i = 0; i < cpu.Length; i++)
            {
                // GEMM summation order differs between engines, so allow float round-off relative to the
                // magnitude; the layout defect was O(1) per element.
                Assert.True(Math.Abs(gpuOut[i] - cpu[i]) <= 1e-3f * (1f + Math.Abs(cpu[i])),
                    $"forecast[{i}]: gpu={gpuOut[i]} cpu={cpu[i]} — the GPU stack forward diverges from the CPU (#1804).");
            }
        }
        finally
        {
            AiDotNetEngine.Current = previous;
            gpu.Dispose();
        }
    }
}