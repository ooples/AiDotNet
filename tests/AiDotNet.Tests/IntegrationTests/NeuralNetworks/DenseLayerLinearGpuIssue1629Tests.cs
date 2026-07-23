// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using System.Threading.Tasks;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// #1629: a DenseLayer with a LINEAR output head (<c>activationFunction: null</c> →
/// <see cref="AiDotNet.Tensors.Engines.FusedActivationType.None"/>) produced literal-zero output on the
/// GPU forward path (regression models returned all zeros on GPU-equipped boxes), while the CPU path
/// was correct. This drives the exact scenario on real GPU hardware and asserts the linear GPU forward
/// (a) is not degenerate all-zero and (b) matches the CPU forward on identical weights/input.
/// Skips cleanly when no GPU backend is available.
/// </summary>
public class DenseLayerLinearGpuIssue1629Tests
{
    private readonly ITestOutputHelper _output;
    public DenseLayerLinearGpuIssue1629Tests(ITestOutputHelper output) => _output = output;

    [Fact(Timeout = 120000)]
    public async Task LinearHead_GpuForward_IsNonZero_AndMatchesCpu()
    {
        await Task.Yield();
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); } catch { /* no backend */ }
        if (gpu is null || !gpu.SupportsGpu)
        {
            _output.WriteLine("No GPU backend available — skipping #1629 GPU repro.");
            return;
        }

        var previous = AiDotNetEngine.Current;
        try
        {
            const int inFeat = 8, outFeat = 4, batch = 5;
            // Linear output head — the exact trigger: no activation → FusedActivationType.None.
            var layer = new DenseLayer<float>(outFeat, activationFunction: null);

            var rng = new Random(1629);
            var input = new Tensor<float>(new[] { batch, inFeat });
            for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

            // CPU forward first: initializes the layer's weights against inFeat and gives the reference.
            AiDotNetEngine.Current = previous;
            var cpu = layer.Forward(input);
            float[] cpuData = cpu.ToArray();

            // GPU forward on the SAME layer (same weights/biases) + same input.
            AiDotNetEngine.Current = gpu;
            var gpuOut = layer.ForwardGpu(input);
            float[] gpuData = gpuOut.ToArray();

            // (1) Not the degenerate all-zero output the bug produced.
            bool allZero = true;
            for (int i = 0; i < gpuData.Length; i++) { if (gpuData[i] != 0f) { allZero = false; break; } }
            Assert.False(allZero, "Linear-head GPU forward returned all zeros (#1629).");

            // (2) Matches the CPU reference (GEMM+bias, no activation).
            Assert.Equal(cpuData.Length, gpuData.Length);
            for (int i = 0; i < cpuData.Length; i++)
            {
                double rel = Math.Abs(gpuData[i] - cpuData[i]) / (Math.Abs(cpuData[i]) + 1e-4);
                Assert.True(rel < 1e-3, $"GPU[{i}]={gpuData[i]} vs CPU[{i}]={cpuData[i]} (rel {rel:E3}) — linear GPU forward diverges.");
            }
        }
        finally
        {
            AiDotNetEngine.Current = previous;
            gpu.Dispose();
        }
    }
}
