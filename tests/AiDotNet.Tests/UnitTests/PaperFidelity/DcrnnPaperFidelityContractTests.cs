using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Finance.Graph;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.PaperFidelity;

public sealed class DcrnnPaperFidelityContractTests
{
    [Fact]
    public void DefaultTopologyUsesDcgruEncoderAndAutoregressiveDecoder()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 24,
            outputSize: 2);
        var options = new DCRNNOptions<double>
        {
            NumNodes = 3,
            NumFeatures = 2,
            SequenceLength = 4,
            ForecastHorizon = 2,
            HiddenDimension = 4,
            NumEncoderLayers = 2,
            NumDecoderLayers = 2,
        };

        using var model = new DCRNN<double>(architecture, options);
        var layers = ((ILayeredModel<double>)model).Layers;

        Assert.Equal(5, layers.Count);
        Assert.Equal(4, layers.OfType<DiffusionConvolutionalGRULayer<double>>().Count());
        Assert.IsType<DenseLayer<double>>(layers[^1]);
        Assert.Equal(1, model.GetLayerInfo(layers.Count - 1).OutputShape[^1]);
    }

    [Fact]
    public void DcgruGateRespondsToAnIncomingGraphNeighbor()
    {
        var identity = new double[,]
        {
            { 1.0, 0.0 },
            { 0.0, 1.0 },
        };
        var incomingFromNodeZero = new double[,]
        {
            { 1.0, 0.0 },
            { 1.0, 0.0 },
        };

        var isolated = new DiffusionConvolutionalGRULayer<double>(
            inputSize: 1,
            hiddenSize: 1,
            numNodes: 2,
            maxDiffusionStep: 1,
            forwardTransition: identity,
            backwardTransition: identity,
            useBackwardSupport: false);
        var coupled = new DiffusionConvolutionalGRULayer<double>(
            inputSize: 1,
            hiddenSize: 1,
            numNodes: 2,
            maxDiffusionStep: 1,
            forwardTransition: incomingFromNodeZero,
            backwardTransition: identity,
            useBackwardSupport: false);

        var materializationInput = new Tensor<double>([2, 1, 1]);
        isolated.Forward(materializationInput);
        coupled.Forward(materializationInput);

        Assert.True(isolated.ParameterCount > 0);
        var parameters = new Vector<double>(
            Enumerable.Repeat(0.1, checked((int)isolated.ParameterCount)).ToArray());
        isolated.SetParameters(parameters);
        coupled.SetParameters(parameters);

        var input = new Tensor<double>([2, 1, 1]);
        input[0, 0, 0] = 1.0;

        var isolatedOutput = isolated.Forward(input);
        var coupledOutput = coupled.Forward(input);

        Assert.True(
            System.Math.Abs(coupledOutput[1, 0, 0] - isolatedOutput[1, 0, 0]) > 1e-8,
            "Node 1 must respond when the diffusion support gives it an incoming edge from node 0.");
    }

    [Fact]
    public void DeserializeRebindsDiffusionSupportsToRestoredDcgruLayers()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 4,
            outputSize: 2);
        var options = new DCRNNOptions<double>
        {
            NumNodes = 2,
            NumFeatures = 1,
            SequenceLength = 2,
            ForecastHorizon = 2,
            HiddenDimension = 2,
            NumEncoderLayers = 1,
            NumDecoderLayers = 1,
            DiffusionSteps = 1,
            MaxDiffusionStep = 1,
            NumSamples = 1,
            Seed = 42,
        };
        var sourceAdjacency = new double[,]
        {
            { 1.0, 0.0 },
            { 1.0, 0.0 },
        };
        var differentConstructorAdjacency = new double[,]
        {
            { 1.0, 0.0 },
            { 0.0, 1.0 },
        };

        using var source = new DCRNN<double>(architecture, options, sourceAdjacency);
        using var restored = new DCRNN<double>(architecture, options, differentConstructorAdjacency);
        var input = new Tensor<double>([2, 2, 1]);
        input[0, 0, 0] = 1.0;
        input[0, 1, 0] = 0.5;

        var expected = source.Predict(input);
        restored.Deserialize(source.Serialize());
        var actual = restored.Predict(input);

        Assert.Equal(expected.Shape, actual.Shape);
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], actual[i], 12);
    }

    [Fact]
    public void TrainRejectsTeacherForcingTargetWithAnIncompatibleLayout()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 4,
            outputSize: 2);
        var options = new DCRNNOptions<double>
        {
            NumNodes = 2,
            NumFeatures = 1,
            SequenceLength = 2,
            ForecastHorizon = 2,
            HiddenDimension = 2,
            NumEncoderLayers = 1,
            NumDecoderLayers = 1,
            DiffusionSteps = 1,
            MaxDiffusionStep = 1,
            NumSamples = 1,
            Seed = 42,
        };

        using var model = new DCRNN<double>(architecture, options);
        var input = new Tensor<double>([2, 2, 1]);
        var malformedTarget = new Tensor<double>([2, 2, 1]);

        var exception = Assert.Throws<ArgumentException>(
            () => model.Train(input, malformedTarget));

        Assert.Equal("target", exception.ParamName);
        Assert.Contains("shape [2, >= 2]", exception.Message);
        Assert.Contains("rank 3", exception.Message);
    }
}
