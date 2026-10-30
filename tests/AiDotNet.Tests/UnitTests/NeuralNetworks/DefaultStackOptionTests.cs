using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Options;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// The sizes the default layer stacks used to hardcode are options now (#2396). Each test fails if the model
/// builds its default stack without reading the option.
/// </summary>
public class DefaultStackOptionTests
{
    private static NeuralNetworkArchitecture<double> Vector(int inputs, int outputs,
        NeuralNetworkTaskType task = NeuralNetworkTaskType.Regression) =>
        new NeuralNetworkArchitecture<double>(inputType: InputType.OneDimensional, taskType: task,
            inputSize: inputs, outputSize: outputs);

    [Fact]
    public void FeedForward_BuildsTheGivenHiddenLayers()
    {
        var model = new FeedForwardNeuralNetwork<double>(Vector(4, 2),
            options: new FeedForwardNeuralNetworkOptions { HiddenLayers = new[] { 8, 3 } });

        // 4 -> 8 -> 3 -> 2, weights plus biases.
        Assert.Equal((4 * 8 + 8) + (8 * 3 + 3) + (3 * 2 + 2), model.ParameterCount);
    }

    [Fact]
    public void FeedForward_DefaultIsTheStackItAlwaysBuilt()
    {
        Assert.Equal(new[] { 64, 64 }, new FeedForwardNeuralNetworkOptions().HiddenLayers);
    }

    [Fact]
    public void NeuralNetwork_BuildsTheGivenHiddenLayers()
    {
        var derived = new NeuralNetwork<double>(Vector(4, 2));
        var given = new NeuralNetwork<double>(Vector(4, 2),
            options: new NeuralNetworkDefaultOptions { HiddenLayers = new[] { 5 } });

        Assert.Equal((4 * 5 + 5) + (5 * 2 + 2), given.ParameterCount);
        Assert.NotEqual(derived.ParameterCount, given.ParameterCount);
    }

    [Fact]
    public void RecurrentNetwork_ReadsHiddenSizeAndLayerCount()
    {
        var arch = new NeuralNetworkArchitecture<double>(inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.Regression, inputHeight: 5, inputWidth: 3, outputSize: 2);
        int Count(int hidden, int layers) => new RecurrentNeuralNetwork<double>(arch,
            options: new RecurrentNeuralNetworkOptions { HiddenSize = hidden, NumLayers = layers }).ParameterCount;

        Assert.NotEqual(Count(7, 1), Count(9, 1));
        Assert.NotEqual(Count(7, 1), Count(7, 3));
    }

    [Fact]
    public void RadialBasisNetwork_ReadsNumCenters()
    {
        int Count(int centers) => new RadialBasisFunctionNetwork<double>(Vector(4, 2),
            options: new RadialBasisFunctionNetworkOptions { NumCenters = centers }).ParameterCount;

        Assert.NotEqual(Count(5), Count(9));
    }

    [Fact]
    public void Autoencoder_BottleneckIsTheLatentSize()
    {
        var model = new Autoencoder<double>(Vector(32, 32),
            options: new AutoencoderOptions { LatentSize = 3 });

        Assert.Equal(3, model.EncodedSize);
    }

    [Fact]
    public void DeepBeliefNetwork_StacksTheGivenRbms()
    {
        int Count(params int[] hidden) => new DeepBeliefNetwork<double>(Vector(4, 2),
            options: new DeepBeliefNetworkOptions { HiddenLayers = hidden }).ParameterCount;

        Assert.NotEqual(Count(6, 4), Count(6, 5));
        Assert.Equal(new[] { 500, 500, 2000 }, new DeepBeliefNetworkOptions().HiddenLayers);
    }

    [Fact]
    public void DeepBoltzmannMachine_StacksTheGivenLayers()
    {
        int Count(params int[] hidden) => new DeepBoltzmannMachine<double>(Vector(4, 2),
            options: new DeepBoltzmannMachineOptions { HiddenLayers = hidden }).ParameterCount;

        Assert.NotEqual(Count(5), Count(7));
        Assert.NotEqual(Count(5), Count(5, 5));
    }

    [Fact]
    public void QuantumNetwork_ReadsTheQuantumLayerCount()
    {
        int Count(int layers) => new QuantumNeuralNetwork<double>(Vector(4, 2), numQubits: 2,
            options: new QuantumNeuralNetworkOptions { NumLayers = layers }).ParameterCount;

        Assert.NotEqual(Count(1), Count(3));
    }

    [Fact]
    public void GraphAttentionNetwork_ReadsTheHiddenWidth()
    {
        GraphAttentionNetwork<double> Create(int hidden) => new GraphAttentionNetwork<double>(
            Vector(6, 3, NeuralNetworkTaskType.MultiClassClassification),
            options: new GraphAttentionNetworkOptions { HiddenSize = hidden });

        var narrow = Create(16);
        Assert.Equal(16, narrow.HiddenDim);
        Assert.NotEqual(narrow.ParameterCount, Create(32).ParameterCount);
    }

    [Fact]
    public void GraphAttentionOptions_RejectAWidthTheHeadsCannotSplit()
    {
        var options = new GraphAttentionNetworkOptions { NumHeads = 8, HiddenSize = 20 };
        Assert.Throws<System.ArgumentException>(() => options.Validate());
    }

    [Fact]
    public void SpikingNetwork_BuildsTheGivenHiddenLayers()
    {
        int Count(params int[] hidden) => new SpikingNeuralNetwork<double>(Vector(4, 2),
            (AiDotNet.Interfaces.IActivationFunction<double>?)null,
            options: new SpikingNeuralNetworkOptions { HiddenLayers = hidden }).ParameterCount;

        Assert.NotEqual(Count(8), Count(12));
    }

    [Fact]
    public void ResidualNetwork_ReadsTheBlockCount()
    {
        int Count(int blocks) => new ResidualNeuralNetwork<double>(Vector(8, 2),
            options: new ResidualNeuralNetworkOptions { NumResidualBlocks = blocks }).ParameterCount;

        Assert.NotEqual(Count(1), Count(3));
    }
}
