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
    public void SoftmaxHead_PredictIsNotNormalizedTwice()
    {
        // Softmax of a probability vector over 5 classes is confined to [1/(4+e), e/(4+e)], about [0.15, 0.40]. A head
        // normalized once can leave that band: scaling its weights up makes its softmax peaked.
        var head = SoftmaxHead();
        var vit = Build(head);
        var input = new Tensor<float>(new[] { 1, Channels, Size, Size });
        var random = RandomHelper.CreateSeededRandom(7);
        for (int i = 0; i < input.Length; i++)
            input[i] = (float)(random.NextDouble() * 2 - 1);
        // Size the head on its own, not through vit.Predict, so no compiled plan captures the unscaled weights.
        head.Forward(new Tensor<float>(new[] { 1, Small.HiddenDim }));

        var parameters = head.GetParameters();
        Assert.True(parameters.Length > 0);
        var scaled = new Vector<float>(parameters.Length);
        for (int i = 0; i < parameters.Length; i++)
            scaled[i] = parameters[i] * 50f;
        head.SetParameters(scaled);
        var output = vit.Predict(input);

        float sum = 0f, max = 0f;
        for (int i = 0; i < output.Length; i++)
        {
            sum += output[i];
            max = Math.Max(max, output[i]);
        }
        Assert.Equal(1f, sum, 3);
        Assert.True(max > 0.5f, $"The largest probability {max} is inside the band a second softmax confines it to.");
    }
}
