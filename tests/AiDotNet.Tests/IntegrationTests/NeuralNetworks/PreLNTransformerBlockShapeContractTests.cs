using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.IntegrationTests.NeuralNetworks;

public sealed class PreLNTransformerBlockShapeContractTests
{
    private const int SequenceLength = 8;
    private const int HiddenSize = 16;

    [Fact]
    public void Sequence_chain_validates_and_forwards_batched_input()
    {
        var layers = new List<ILayer<float>>
        {
            new InputLayer<float>([SequenceLength, HiddenSize]),
            new PreLNTransformerBlock<float>(
                HiddenSize,
                4 * HiddenSize,
                new MultiHeadAttentionLayer<float>(headCount: 4, headDimension: 4)),
            new SequenceTokenSliceLayer<float>(SequenceTokenSliceLayer<float>.Position.Last),
            new DenseLayer<float>(2, (IActivationFunction<float>)new IdentityActivation<float>()),
        };
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: SequenceLength,
            inputWidth: HiddenSize,
            outputSize: 2,
            layers: layers);
        var network = new FeedForwardNeuralNetwork<float>(
            architecture,
            lossFunction: new CategoricalCrossEntropyLoss<float>());

        Tensor<float> output = network.Predict(new Tensor<float>([2, SequenceLength, HiddenSize]));

        Assert.Equal([2, 2], output.Shape.ToArray());
        Assert.Equal([-1, HiddenSize], network.Layers[1].GetInputShape());
        Assert.Equal([-1, HiddenSize], network.Layers[1].GetOutputShape());
    }

    [Fact]
    public void ConstructorKnownWidths_ParameterLayoutIsStableBeforeFirstForward()
    {
        const int sequenceLength = 4;
        const int hiddenSize = 8;
        var attention = new GroupedQueryAttentionLayer<float>(
            sequenceLength: sequenceLength,
            embeddingDimension: hiddenSize,
            numHeads: 4,
            numKVHeads: 2);
        var block = new PreLNTransformerBlock<float>(
            hiddenSize: hiddenSize,
            ffnDim: 16,
            attention: attention,
            gated: true);

        var before = block.GetParameters();
        block.Forward(new Tensor<float>([1, sequenceLength, hiddenSize]));
        var after = block.GetParameters();

        Assert.True(before.Length > 0);
        Assert.Equal(before.Length, after.Length);
    }
}
