// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Generic;
using System.Reflection;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DevicePrimitives;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Streaming validation reads one host scalar per batch, so a batch must leave no device memory behind. On a GPU
/// engine Predict keeps its output on the device, and the loss adds more device results. Before the fix, nothing freed
/// them until a garbage collection finalized them. For a language model each batch held [rows, vocabulary] results
/// (805 MB at 4096 x 49152): a 16-batch validation filled a 12 GB device, and the training steps after it ran 3.4x
/// slower (11.1 vs 3.3 min for 50 steps; one run stalled 10.7 h).
/// </summary>
[Collection("TrainingDiagnosticsSequential")]
public class StreamingValidationReleasesDeviceResultsTests
{
    private const int Inputs = 16, Hidden = 32, Outputs = 4096, Batch = 256, Batches = 8;

    // One batch's prediction alone: Batch x Outputs floats (4 MiB).
    private const long OutputBytes = (long)Batch * Outputs * sizeof(float);

    [SkippableFact]
    public void Validation_batches_leave_no_device_results_behind()
    {
        using var gpu = new DirectGpuTensorEngine();
        if (Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS") == "1")
            Assert.True(gpu.IsGpuAvailable, "The required GPU backend failed to initialize.");
        Skip.If(!gpu.IsGpuAvailable, "No DirectGpuTensorEngine available.");
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = gpu;
        try
        {
            var layers = new List<ILayer<float>>
            {
                new DenseLayer<float>(Hidden, (IActivationFunction<float>)new ReLUActivation<float>()),
                new DenseLayer<float>(Outputs, (IActivationFunction<float>?)null),
            };
            var architecture = new NeuralNetworkArchitecture<float>(
                inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression,
                inputSize: Inputs, outputSize: Outputs, layers: layers);
            var network = new NeuralNetwork<float>(architecture);

            var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(20261107);
            var x = new Tensor<float>(new[] { Batch, Inputs });
            var y = new Tensor<float>(new[] { Batch, Outputs });
            for (int r = 0; r < Batch; r++)
            {
                for (int i = 0; i < Inputs; i++) x[r, i] = (float)(rng.NextDouble() * 2 - 1);
                for (int o = 0; o < Outputs; o++) y[r, o] = (float)(rng.NextDouble() * 2 - 1);
            }

            // The per-batch loss of AiModelBuilder's streaming validation (EvaluateStreamingValidationAsync).
            var sampleLoss = typeof(AiModelBuilder<float, Tensor<float>, Tensor<float>>).GetMethod(
                "SampleLoss", BindingFlags.NonPublic | BindingFlags.Static)
                ?? throw new InvalidOperationException("AiModelBuilder has no SampleLoss.");
            float Loss() => (float)sampleLoss.Invoke(null, new object[] { network, network.DefaultLossFunction, x, y })!;

            float first = Loss(); // warm: weights uploaded, kernels and pools built
            long before = GpuMemoryStats.Snapshot().AllocatedBytes;
            for (int b = 0; b < Batches; b++)
                Assert.Equal(first, Loss());
            long grown = GpuMemoryStats.Snapshot().AllocatedBytes - before;

            // Each leaked batch would hold at least its prediction; all of them together at least Batches of those.
            Assert.True(grown < OutputBytes,
                $"{Batches} validation batches left {grown:N0} device bytes allocated (one prediction is {OutputBytes:N0}).");
        }
        finally { AiDotNetEngine.Current = previous; }
    }
}
