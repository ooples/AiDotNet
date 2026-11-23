// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Generic;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DevicePrimitives;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Streaming validation of a model ending in a fused readout (a dense LM head with cross-entropy) never forms the
/// <c>[rows, vocabulary]</c> logits on the device: its peak allocation stays below one such tensor (the chunked fused
/// loss holds a few row chunks), and the loss equals the loss of <c>Predict</c>. Scoring <c>Predict</c> instead held the
/// logits and the loss's intermediates over them: 7.0 GB at the first validation of a run whose training step holds 5.0.
/// </summary>
[Collection("TrainingDiagnosticsSequential")]
public class StreamingValidationFusedReadoutGpuTests
{
    private const int Inputs = 16, Hidden = 64, Vocabulary = 32768, Rows = 4096;
    private const long LogitsBytes = (long)Rows * Vocabulary * sizeof(float);   // 512 MiB

    [SkippableFact]
    public void Validation_peak_stays_below_one_logits_tensor()
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
                new DenseLayer<float>(Vocabulary, (IActivationFunction<float>?)null),
            };
            var network = new NeuralNetwork<float>(new NeuralNetworkArchitecture<float>(
                inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.MultiClassClassification,
                inputSize: Inputs, outputSize: Vocabulary, layers: layers) { RandomSeed = 7 });
            var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(20261123);
            var x = new Tensor<float>(new[] { Rows, Inputs });
            var ids = new Tensor<float>(new[] { Rows });
            for (int i = 0; i < x.Length; i++) x.SetFlat(i, (float)(rng.NextDouble() * 2 - 1));
            for (int r = 0; r < Rows; r++) ids.SetFlat(r, rng.Next(Vocabulary));
            var loss = new CrossEntropyWithLogitsLoss<float>();

            float Score() => AiModelBuilder<float, Tensor<float>, Tensor<float>>.SampleLoss(network, loss, x, ids);
            float first = Score(); // warm: layers built, weights uploaded, pools filled
            GC.Collect();
            GC.WaitForPendingFinalizers();
            long before = GpuMemoryStats.Stats()["allocated_bytes.all.current"];
            GpuMemoryStats.ResetPeakStats();
            float got = Score();
            long peak = GpuMemoryStats.Stats()["allocated_bytes.all.peak"] - before;

            Assert.True(peak < LogitsBytes,
                $"validation peaked {peak:N0} device bytes above its start; one [rows, vocabulary] tensor is {LogitsBytes:N0}.");
            Assert.Equal(first, got);
            float expected;
            using (TensorScope.Begin())
                expected = loss.ComputeTapeLoss(network.Predict(x), ids).Data.Span[0];
            Assert.True(Math.Abs(expected - got) <= 1e-5f * Math.Abs(expected),
                $"fused validation loss {got} vs the loss of Predict {expected}");
        }
        finally { AiDotNetEngine.Current = previous; }
    }

    private const int Width = 2048, Depth = 8, HeadVocabulary = 1024;
    private const long ActivationBytes = (long)Rows * Width * sizeof(float);   // 32 MiB

    /// <summary>
    /// Each layer that keeps nothing between calls has its intermediates freed as its forward returns, with no garbage
    /// collection: validation of eight layers that each make three [rows, width] intermediates peaks near the layer
    /// outputs plus one layer's work, not all of them (left to finalization they summed to about 1 GiB here, or a
    /// collection ran to reclaim them).
    /// </summary>
    [SkippableFact]
    public void Validation_frees_each_layers_intermediates_as_it_returns()
    {
        using var gpu = new DirectGpuTensorEngine();
        if (Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS") == "1")
            Assert.True(gpu.IsGpuAvailable, "The required GPU backend failed to initialize.");
        Skip.If(!gpu.IsGpuAvailable, "No DirectGpuTensorEngine available.");
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = gpu;
        try
        {
            var layers = new List<ILayer<float>>();
            for (int i = 0; i < Depth; i++) layers.Add(new ThreeIntermediatesLayer(Width));
            layers.Add(new DenseLayer<float>(HeadVocabulary, (IActivationFunction<float>?)null));
            var network = new NeuralNetwork<float>(new NeuralNetworkArchitecture<float>(
                inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.MultiClassClassification,
                inputSize: Width, outputSize: HeadVocabulary, layers: layers) { RandomSeed = 11 });
            var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(20261124);
            var x = new Tensor<float>(new[] { Rows, Width });
            var ids = new Tensor<float>(new[] { Rows });
            for (int i = 0; i < x.Length; i++) x.SetFlat(i, (float)(rng.NextDouble() * 0.2 - 0.1));
            for (int r = 0; r < Rows; r++) ids.SetFlat(r, rng.Next(HeadVocabulary));
            var loss = new CrossEntropyWithLogitsLoss<float>();

            float Score() => AiModelBuilder<float, Tensor<float>, Tensor<float>>.SampleLoss(network, loss, x, ids);
            float first = Score();
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            long before = GpuMemoryStats.Stats()["allocated_bytes.all.current"];
            GpuMemoryStats.ResetPeakStats();
            int collections = GC.CollectionCount(0);
            float got = Score();
            int collected = GC.CollectionCount(0) - collections;
            long peak = GpuMemoryStats.Stats()["allocated_bytes.all.peak"] - before;
            long after = GpuMemoryStats.Stats()["allocated_bytes.all.current"] - before;

            Assert.Equal(first, got);
            Assert.Equal(0, collected);
            // Kept: the eight outputs (8 x 32 MiB) and the input's device copy; transient: one layer's three intermediates.
            Assert.True(peak < 14 * ActivationBytes,
                $"validation peaked {peak:N0} device bytes; {Depth} layers' intermediates together are {Depth * 4 * ActivationBytes:N0}.");
            Assert.True(after < ActivationBytes, $"validation left {after:N0} device bytes allocated.");
        }
        finally { AiDotNetEngine.Current = previous; }
    }

    /// <summary>x + 0.5 (x*x + x): three intermediates and an output, and nothing kept between calls.</summary>
    [AiDotNet.Attributes.ElementWiseShape]
    private sealed class ThreeIntermediatesLayer : LayerBase<float>
    {
        internal ThreeIntermediatesLayer(int features) : base([features], [features])
        {
        }

        protected internal override bool KeepsResultsBetweenCalls => false;

        protected override Tensor<float> ForwardTraced(Tensor<float> input)
        {
            var square = Engine.TensorMultiply(input, input);
            var sum = Engine.TensorAdd(square, input);
            var half = Engine.TensorMultiplyScalar(sum, 0.5f);
            return Engine.TensorAdd(half, input);
        }

        public override bool SupportsTraining => false;

        public override void ResetState()
        {
        }
    }
}
