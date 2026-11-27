using System.Collections.Generic;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// Locks <c>TrainableParameterCount</c> (the PyTorch <c>requires_grad</c> numel sum) against
/// <c>ParameterCount</c> (the GetParameters/SetParameters state-vector length).
/// </summary>
public sealed class TrainableParameterCountTests
{
    private const int WeightLength = 4;
    private const int BufferLength = 3;
    private const int OuterOwnLength = 2;
    private const int InnerLength = 5;

    private static NeuralNetwork<double> NetworkOf(ILayer<double> layer)
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: WeightLength,
            outputSize: WeightLength,
            layers: new List<ILayer<double>> { layer });
        return new NeuralNetwork<double>(architecture);
    }

    [Fact]
    public void RegisteredBuffer_IsInParameterCount_ButNotTrainableParameterCount()
    {
        using var layer = new BufferedProbeLayer();

        Assert.Equal(WeightLength + BufferLength, layer.ParameterCount);
        Assert.Equal(WeightLength, layer.TrainableParameterCount);

        var model = NetworkOf(layer);
        Assert.Equal(WeightLength + BufferLength, model.ParameterCount);
        Assert.Equal(WeightLength, model.TrainableParameterCount);
        Assert.Equal(BufferLength, model.ParameterCount - model.TrainableParameterCount);
    }

    [Fact]
    public void SubLayerTensorsAlsoListedByParent_DoubleCountInParameterCount_CountOnceInTrainable()
    {
        using var layer = new SharingOuterProbeLayer();

        // ParameterCount: the outer layer's listed tensors (own 2 + inner's 5) plus the inner sub-layer's 5.
        Assert.Equal(OuterOwnLength + InnerLength + InnerLength, layer.ParameterCount);
        Assert.Equal(OuterOwnLength + InnerLength, layer.TrainableParameterCount);

        var model = NetworkOf(layer);
        Assert.Equal(OuterOwnLength + InnerLength + InnerLength, model.ParameterCount);
        Assert.Equal(OuterOwnLength + InnerLength, model.TrainableParameterCount);
    }

    [Fact]
    public void PlainModel_WithoutBuffersOrSharing_HasEqualCounts()
    {
        var architecture = new NeuralNetworkArchitecture<double>(inputFeatures: 4, outputSize: 2);
        architecture.Layers.AddRange(LayerHelper<double>.CreateDefaultLayers(
            architecture, hiddenLayerCount: 2, hiddenLayerSize: 8, outputSize: 2));
        var model = new NeuralNetwork<double>(architecture);

        // Dense 4->8, 8->8, 8->2 with biases: 40 + 72 + 18.
        const long expected = (4 * 8 + 8) + (8 * 8 + 8) + (8 * 2 + 2);
        Assert.Equal(expected, model.ParameterCount);
        Assert.Equal(expected, model.TrainableParameterCount);
    }

    [ElementWiseShape]
    private sealed class BufferedProbeLayer : LayerBase<double>
    {
        private readonly Tensor<double> _weight = new([WeightLength]);
        private readonly Tensor<double> _bias = new([BufferLength]);

        public BufferedProbeLayer() : base([WeightLength], [WeightLength])
        {
            RegisterTrainableParameter(_weight, PersistentTensorRole.Weights);
            RegisterBuffer(_bias, "router_bias");
        }

        protected override Tensor<double> ForwardTraced(Tensor<double> input) => input;

        public override bool SupportsTraining => true;

        public override void ResetState()
        {
        }
    }

    [ElementWiseShape]
    private sealed class InnerProbeLayer : LayerBase<double>
    {
        public readonly Tensor<double> Weight = new([InnerLength]);

        public InnerProbeLayer() : base([WeightLength], [WeightLength])
        {
            RegisterTrainableParameter(Weight, PersistentTensorRole.Weights);
        }

        protected override Tensor<double> ForwardTraced(Tensor<double> input) => input;

        public override bool SupportsTraining => true;

        public override void ResetState()
        {
        }
    }

    [ElementWiseShape]
    private sealed class SharingOuterProbeLayer : LayerBase<double>
    {
        private readonly Tensor<double> _own = new([OuterOwnLength]);
        private readonly InnerProbeLayer _inner = new();

        public SharingOuterProbeLayer() : base([WeightLength], [WeightLength])
        {
            RegisterTrainableParameter(_own, PersistentTensorRole.Weights);
            RegisterSubLayer(_inner);
        }

        // Lists the sub-layer's tensor as well as its own, the wrapped-tensor pattern that ParameterCount double counts.
        public override IReadOnlyList<Tensor<double>> GetTrainableParameters() => [_own, _inner.Weight];

        protected override Tensor<double> ForwardTraced(Tensor<double> input) => _inner.Forward(input);

        public override bool SupportsTraining => true;

        public override void ResetState()
        {
        }
    }
}
