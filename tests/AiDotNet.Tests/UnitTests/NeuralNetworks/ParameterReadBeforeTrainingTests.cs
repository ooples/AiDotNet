using System;
using System.Collections.Generic;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// Reading a seeded model's parameters before it trains must not change what it trains to. Most layers draw their
/// weights on first use, so a read that comes first is the draw: it has to use the seed training would have used.
/// </summary>
public class ParameterReadBeforeTrainingTests
{
    private const int Sequence = 4;
    private const int Width = 8;

    [TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
        Direction = TensorLayoutDirection.Input, BatchOptional = true)]
    [TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
        Direction = TensorLayoutDirection.Output, BatchOptional = true)]
    private sealed class SuppliedStack : NeuralNetworkBase<float>
    {
        public SuppliedStack(NeuralNetworkArchitecture<float> architecture)
            : base(architecture, new MeanSquaredErrorLoss<float>())
        {
            InitializeLayers();
        }

        protected override void InitializeLayers() => Layers.AddRange(Architecture.Layers ?? new List<ILayer<float>>());

        public override IFullModel<float, Tensor<float>, Tensor<float>> DeepCopy() => new SuppliedStack(Architecture);

        public override ModelMetadata<float> GetModelMetadata() => new()
        {
            Name = nameof(SuppliedStack),
            Description = "Test double that runs exactly the layers its caller supplied.",
        };
    }

    private static SuppliedStack Build()
    {
        // Every model starts from the same thread state, as in a fresh process: the caller-built layers below are
        // constructed before their model exists and take their seeds from whatever construction scope is armed.
        LayerInitializationSeedScope.ResetForModelConstruction(77);
        var attention = new GroupedQueryAttentionLayer<float>(
            sequenceLength: Sequence, embeddingDimension: Width, numHeads: 4, numKVHeads: 2);
        var layers = new List<ILayer<float>>
        {
            new PreLNTransformerBlock<float>(hiddenSize: Width, ffnDim: 2 * Width, attention: attention, gated: true),
            new DenseLayer<float>(Width, (IActivationFunction<float>)new AiDotNet.ActivationFunctions.IdentityActivation<float>()),
        };
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.TwoDimensional, taskType: NeuralNetworkTaskType.SequenceToSequence,
            inputHeight: Sequence, inputWidth: Width, outputSize: Width, layers: layers)
        {
            RandomSeed = 2632,
        };
        return new SuppliedStack(architecture);
    }

    private static Tensor<float> Ramp(float scale)
    {
        var t = new Tensor<float>([1, Sequence, Width]);
        var span = t.AsWritableSpan();
        for (int i = 0; i < span.Length; i++) span[i] = scale * ((i % 7) - 3);
        return t;
    }

    private static float[] TrainedAfter(Action<SuppliedStack> before)
    {
        var model = Build();
        before(model);
        model.Train(Ramp(0.1f), Ramp(0.05f));
        return model.GetParameters().ToArray();
    }

    public static IEnumerable<object[]> Reads()
    {
        yield return new object[] { "TrainableParameterCount" };
        yield return new object[] { "GetParameters" };
        yield return new object[] { "GetParameterChunks" };
        yield return new object[] { "LayerGetTrainableParameters" };
        yield return new object[] { "Predict" };
    }

    private static void Read(SuppliedStack model, string read)
    {
        switch (read)
        {
            case "TrainableParameterCount": _ = model.TrainableParameterCount; break;
            case "GetParameters": _ = model.GetParameters(); break;
            case "GetParameterChunks": foreach (var _ in model.GetParameterChunks()) { } break;
            case "LayerGetTrainableParameters":
                // Straight to the layers, past the model, down to the sub-layers whose weights a read draws.
                foreach (var layer in model.Layers) ReadLayer((LayerBase<float>)layer);
                break;
            case "Predict": _ = model.Predict(Ramp(0.1f)); break;
            default: throw new ArgumentOutOfRangeException(nameof(read));
        }
    }

    private static void ReadLayer(LayerBase<float> layer)
    {
        _ = layer.GetTrainableParameters();
        foreach (var sub in layer.GetSubLayers())
            if (sub is LayerBase<float> child) ReadLayer(child);
    }

    [Theory]
    [MemberData(nameof(Reads))]
    public void AReadBeforeTraining_LeavesTheTrainedWeightsBitwiseEqual(string read)
    {
        var untouched = TrainedAfter(_ => { });
        Assert.NotEmpty(untouched);
        Assert.Equal(untouched, TrainedAfter(_ => { }));   // the comparison is meaningful only if training is repeatable

        Assert.Equal(untouched, TrainedAfter(m => Read(m, read)));
    }
}
