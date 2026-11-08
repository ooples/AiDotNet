// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Generic;
using System.Linq;
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
            var network = Network();
            Assert.True(network.ReleasesPredictResults, "Dense layers keep nothing between calls, so validation frees them.");
            var (x, y) = Batch_();

            float first = Loss(network, x, y); // warm: weights uploaded, kernels and pools built
            long before = GpuMemoryStats.Snapshot().AllocatedBytes;
            for (int b = 0; b < Batches; b++)
                Assert.Equal(first, Loss(network, x, y));
            long grown = GpuMemoryStats.Snapshot().AllocatedBytes - before;

            // Each leaked batch would hold at least its prediction; all of them together at least Batches of those.
            Assert.True(grown < OutputBytes,
                $"{Batches} validation batches left {grown:N0} device bytes allocated (one prediction is {OutputBytes:N0}).");
        }
        finally { AiDotNetEngine.Current = previous; }
    }

    /// <summary>
    /// A model that keeps a GPU result between calls (here one layer's offset, built on the device by its first forward
    /// and read by every later one, as a generation cache is) must still have it after validation. A scope around
    /// Predict would free it with the call's other results, and the next prediction would read a released buffer.
    /// </summary>
    [SkippableFact]
    public void Validation_keeps_what_a_model_holds_between_calls()
    {
        using var gpu = new DirectGpuTensorEngine();
        if (Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS") == "1")
            Assert.True(gpu.IsGpuAvailable, "The required GPU backend failed to initialize.");
        Skip.If(!gpu.IsGpuAvailable, "No DirectGpuTensorEngine available.");
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = gpu;
        try
        {
            var network = Network(new CachedOffsetLayer(Inputs));
            Assert.False(network.ReleasesPredictResults, "A layer nobody checked counts as keeping a result.");
            var (x, y) = Batch_();

            float first = Loss(network, x, y);
            for (int b = 0; b < Batches; b++)
            {
                // Other work allocates between batches, as training steps do, so a released offset buffer is reused.
                using (TensorScope.Begin())
                    gpu.TensorMultiply(x, x);
                Assert.Equal(first, Loss(network, x, y));
            }
        }
        finally { AiDotNetEngine.Current = previous; }
    }

    /// <summary>
    /// With ALiBi the attention weights are computed even in eval mode, and with the auxiliary loss on the layer keeps them
    /// for <see cref="MultiHeadAttentionLayer{T}.ComputeAuxiliaryLoss"/>, which reads them after the forward returned.
    /// Validation must leave them alive. Today that path computes the weights on the host, which no scope frees, so the
    /// layer declaring it keeps them (asserted first) is what protects a kernel that returns them on the device.
    /// </summary>
    [SkippableFact]
    public void Validation_keeps_the_attention_weights_the_auxiliary_loss_reads()
    {
        using var gpu = new DirectGpuTensorEngine();
        if (Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS") == "1")
            Assert.True(gpu.IsGpuAvailable, "The required GPU backend failed to initialize.");
        Skip.If(!gpu.IsGpuAvailable, "No DirectGpuTensorEngine available.");
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = gpu;
        try
        {
            var (network, attention) = AttentionNetwork();
            Assert.False(network.ReleasesPredictResults, "The auxiliary loss reads weights a forward kept.");
            var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(20261108);
            var x = new Tensor<float>(new[] { 2, Sequence, Heads * HeadDimension });
            var y = new Tensor<float>(new[] { 2, Sequence, Heads * HeadDimension });
            for (int i = 0; i < x.Length; i++)
            {
                x.SetFlat(i, (float)(rng.NextDouble() * 2 - 1));
                y.SetFlat(i, (float)(rng.NextDouble() * 2 - 1));
            }

            float firstLoss = Loss(network, x, y);
            float firstAuxiliary = attention.ComputeAuxiliaryLoss();
            Assert.NotEqual(0f, firstAuxiliary); // the weights were cached and read, not skipped
            for (int b = 0; b < Batches; b++)
            {
                using (TensorScope.Begin())
                    gpu.TensorMultiply(x, x);
                Assert.Equal(firstLoss, Loss(network, x, y));
                Assert.Equal(firstAuxiliary, attention.ComputeAuxiliaryLoss());
            }
        }
        finally { AiDotNetEngine.Current = previous; }
    }

    [Fact]
    public void Only_networks_whose_layers_keep_nothing_release_their_predict_results()
    {
        Assert.True(Network().ReleasesPredictResults);
        Assert.False(Network(new CachedOffsetLayer(Inputs)).ReleasesPredictResults);

        var transformer = new Transformer<float>(new TransformerArchitecture<float>(
            InputType.OneDimensional, NeuralNetworkTaskType.SequenceClassification,
            numEncoderLayers: 0, numDecoderLayers: 2, numHeads: 2, modelDimension: 32, feedForwardDimension: 64,
            NetworkComplexity.Simple, inputSize: 64, outputSize: 64));
        var keeping = string.Join(", ", transformer.Layers
            .Where(layer => layer is not LayerBase<float> { KeepsResultsBetweenCalls: false })
            .Select(layer => layer.GetType().Name));
        Assert.True(transformer.ReleasesPredictResults, $"Default transformer layers that keep a result: {keeping}.");

        // Plain attention keeps only backward caches; the auxiliary loss and ALiBi each make it keep a result.
        Assert.False(new MultiHeadAttentionLayer<float>(Heads, HeadDimension).KeepsResultsBetweenCalls);
        Assert.True(AttentionNetwork().Attention.KeepsResultsBetweenCalls);
        var alibi = new MultiHeadAttentionLayer<float>(Heads, HeadDimension);
        alibi.ConfigurePositionalEncoding(PositionalEncodingType.ALiBi);
        Assert.True(alibi.KeepsResultsBetweenCalls);
    }

    private const int Sequence = 4, Heads = 4, HeadDimension = 8;

    private static (NeuralNetwork<float> Network, MultiHeadAttentionLayer<float> Attention) AttentionNetwork()
    {
        var attention = new MultiHeadAttentionLayer<float>(Heads, HeadDimension) { UseAuxiliaryLoss = true };
        attention.ConfigurePositionalEncoding(PositionalEncodingType.ALiBi);
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.TwoDimensional, taskType: NeuralNetworkTaskType.Regression,
            inputHeight: Sequence, inputWidth: Heads * HeadDimension, outputSize: Heads * HeadDimension,
            layers: new List<ILayer<float>> { attention });
        return (new NeuralNetwork<float>(architecture), attention);
    }

    private static NeuralNetwork<float> Network(ILayer<float>? first = null)
    {
        var layers = new List<ILayer<float>>();
        if (first is not null) layers.Add(first);
        layers.Add(new DenseLayer<float>(Hidden, (IActivationFunction<float>)new ReLUActivation<float>()));
        layers.Add(new DenseLayer<float>(Outputs, (IActivationFunction<float>?)null));
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression,
            inputSize: Inputs, outputSize: Outputs, layers: layers);
        return new NeuralNetwork<float>(architecture);
    }

    private static (Tensor<float> X, Tensor<float> Y) Batch_()
    {
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(20261107);
        var x = new Tensor<float>(new[] { Batch, Inputs });
        var y = new Tensor<float>(new[] { Batch, Outputs });
        for (int r = 0; r < Batch; r++)
        {
            for (int i = 0; i < Inputs; i++) x[r, i] = (float)(rng.NextDouble() * 2 - 1);
            for (int o = 0; o < Outputs; o++) y[r, o] = (float)(rng.NextDouble() * 2 - 1);
        }

        return (x, y);
    }

    /// <summary>The per-batch loss of AiModelBuilder's streaming validation (EvaluateStreamingValidationAsync).</summary>
    private static float Loss(NeuralNetwork<float> network, Tensor<float> x, Tensor<float> y) =>
        AiModelBuilder<float, Tensor<float>, Tensor<float>>.SampleLoss(network, network.DefaultLossFunction, x, y);

    /// <summary>Adds an offset it builds on the device from its first input and keeps for every later call.</summary>
    [AiDotNet.Attributes.ElementWiseShape]
    private sealed class CachedOffsetLayer : LayerBase<float>
    {
        private Tensor<float>? _offset;

        internal CachedOffsetLayer(int features) : base([features], [features])
        {
        }

        protected override Tensor<float> ForwardTraced(Tensor<float> input)
        {
            _offset ??= Engine.TensorAdd(input, input);
            return Engine.TensorAdd(input, _offset);
        }

        public override bool SupportsTraining => false;

        public override void ResetState()
        {
        }
    }
}
