using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// Pins the classification-head contract of <see cref="VisionTransformer{T}"/>: the default loss and Predict's
/// normalization follow the head's activation, so a custom softmax head is neither trained on probabilities by a
/// logits loss nor normalized twice by Predict, and a head that is neither logits nor softmax is rejected.
/// </summary>
public class VisionTransformerHeadContractTests
{
    private const int Size = 8, Channels = 3, Patch = 4, Classes = 5;
    private static readonly VisionTransformerOptions Small = new() { HiddenDim = 16, NumLayers = 1, NumHeads = 2, MlpDim = 32 };

    private static NeuralNetworkArchitecture<float> Architecture(ILayer<float>? head = null)
    {
        List<ILayer<float>>? layers = null;
        if (head is not null)
        {
            layers = LayerHelper<float>.CreateVisionTransformerLayers(
                Size, Size, Channels, Patch, Small.HiddenDim, Small.NumLayers, Small.NumHeads, Small.MlpDim, Classes).ToList();
            layers[^1] = head;
        }
        return new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: Size, inputWidth: Size, inputDepth: Channels, outputSize: Classes, layers: layers);
    }

    private static VisionTransformer<float> Build(ILayer<float>? head = null, ILossFunction<float>? loss = null)
        => new(Architecture(head), Size, Size, Channels, Patch, Classes, loss, Small);

    private static DenseLayer<float> SoftmaxHead()
        => new(Classes, (IVectorActivationFunction<float>)new SoftmaxActivation<float>());

    [Fact]
    public void DefaultHead_TrainsOnLogits()
        => Assert.IsType<CrossEntropyWithLogitsLoss<float>>(Build().DefaultLossFunction);

    [Fact]
    public void IdentityHead_TrainsOnLogits()
        => Assert.IsType<CrossEntropyWithLogitsLoss<float>>(
            Build(new DenseLayer<float>(Classes, (IActivationFunction<float>)new IdentityActivation<float>())).DefaultLossFunction);

    [Fact]
    public void SoftmaxHead_DefaultsToCategoricalCrossEntropy()
        => Assert.IsType<CategoricalCrossEntropyLoss<float>>(Build(SoftmaxHead()).DefaultLossFunction);

    [Fact]
    public void SoftmaxHead_WithLogitsLoss_IsRejected()
        => Assert.Throws<ArgumentException>(() => Build(SoftmaxHead(), new CrossEntropyWithLogitsLoss<float>()));

    [Fact]
    public void ReluHead_IsRejected()
        => Assert.Throws<ArgumentException>(() =>
            Build(new DenseLayer<float>(Classes, (IActivationFunction<float>)new ReLUActivation<float>())));

    [Fact]
    public void LogitsHead_WithAProbabilityLoss_IsRejected()
        // CategoricalCrossEntropyLoss takes the log of its predictions: on logits it computes -log(1e-7) for five zero
        // logits instead of log 5.
        => Assert.Throws<ArgumentException>(() => Build(loss: new CategoricalCrossEntropyLoss<float>()));

    [Fact]
    public void SoftmaxHead_WithAProbabilityLoss_IsAccepted()
        => Assert.IsType<CategoricalCrossEntropyLoss<float>>(Build(SoftmaxHead(), new CategoricalCrossEntropyLoss<float>()).DefaultLossFunction);

    [Fact]
    public void SoftmaxHead_PredictIsNotNormalizedTwice()
    {
        // Zero weights and biases [10, 0, 0, 0, 0] fix the head's logits whatever reaches it, so Predict must return
        // exactly softmax(logits). A second softmax would give about [0.40, 0.15, 0.15, 0.15, 0.15].
        var head = SoftmaxHead();
        var vit = Build(head);
        var input = new Tensor<float>(new[] { 1, Channels, Size, Size });
        var random = RandomHelper.CreateSeededRandom(7);
        for (int i = 0; i < input.Length; i++)
            input[i] = (float)(random.NextDouble() * 2 - 1);
        // Size the head on its own, not through vit.Predict, so no compiled plan captures the initial weights.
        head.Forward(new Tensor<float>(new[] { 1, Small.HiddenDim }));

        var parameters = head.GetParameters();
        Assert.Equal(Small.HiddenDim * Classes + Classes, parameters.Length);
        var fixedLogits = new Vector<float>(parameters.Length);   // weights first, then the biases
        fixedLogits[parameters.Length - Classes] = 10f;
        head.SetParameters(fixedLogits);

        double denominator = Math.Exp(10) + (Classes - 1);
        var expected = new double[Classes];
        expected[0] = Math.Exp(10) / denominator;
        for (int c = 1; c < Classes; c++) expected[c] = 1 / denominator;

        var headOutput = head.Forward(new Tensor<float>(new[] { 1, Small.HiddenDim }));
        for (int c = 0; c < Classes; c++)
            Assert.Equal(expected[c], headOutput[c], 5);   // the parameter layout gives the intended logits

        var output = vit.Predict(input);
        Assert.Equal(Classes, output.Length);
        for (int c = 0; c < Classes; c++)
            Assert.Equal(expected[c], output[c], 5);
    }
}
