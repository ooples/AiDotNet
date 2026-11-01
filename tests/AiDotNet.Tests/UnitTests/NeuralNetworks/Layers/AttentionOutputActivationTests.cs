using System.Collections.Generic;
using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

/// <summary>
/// A transformer block's attention output carries no activation (Vaswani et al. 2017): softmax normalises the
/// scores, and the block's only other nonlinearity is the feed-forward's. Each layer here used to hand its
/// feed-forward activation (GELU or ReLU) to its attention sublayers as well.
/// </summary>
public class AttentionOutputActivationTests
{
    [Fact]
    public void TransformerEncoderLayer_SelfAttentionHasNoOutputActivation()
    {
        var layer = new TransformerEncoderLayer<double>(numHeads: 2, feedForwardDim: 8);
        _ = layer.Forward(Input(1, 3, 4));

        AssertNoOutputActivation(AttentionSublayers(layer), expected: 1);
    }

    [Fact]
    public void TransformerDecoderLayer_SelfAndCrossAttentionHaveNoOutputActivation()
    {
        // The FFN activation is set explicitly to show it stays on the feed-forward only.
        var layer = new TransformerDecoderLayer<double>(numHeads: 2, feedForwardDim: 8, sequenceLength: 3,
            ffnActivation: new ReLUActivation<double>());
        _ = layer.Forward(Input(1, 3, 4));

        AssertNoOutputActivation(AttentionSublayers(layer), expected: 2);
        Assert.Contains(layer.GetSubLayers().OfType<FeedForwardLayer<double>>(),
            ffn => ffn.ScalarActivation is ReLUActivation<double>);
    }

    [Fact]
    public void TimeMoEBlockLayer_SelfAttentionHasNoOutputActivation()
    {
        var layer = new TimeMoEBlockLayer<double>(hiddenDim: 4, numHeads: 2, intermediateSize: 8, numExperts: 2, topK: 1);

        AssertNoOutputActivation(AttentionSublayers(layer), expected: 1);
    }

    [Fact]
    public void UnifiedMultimodalDefaultLayers_AttentionHasNoOutputActivation()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 8,
            outputSize: 2);

        var attention = LayerHelper<double>.CreateDefaultUnifiedMultimodalLayers(architecture, embeddingDimension: 64, numTransformerLayers: 2)
            .OfType<MultiHeadAttentionLayer<double>>()
            .ToList();

        // Two unified transformer layers plus the four cross-modal attention layers.
        AssertNoOutputActivation(attention, expected: 6);
    }

    // DecoderLayer's cross-attention is an AttentionLayer, whose activation IS the score normaliser. Handing it
    // the layer's ReLU replaced softmax over the cross-attention scores with ReLU.
    [Fact]
    public void DecoderLayer_CrossAttentionNormalisesItsScoresWithSoftmax_ForBothActivationKinds()
    {
        var scalar = new DecoderLayer<double>(attentionSize: 4, feedForwardSize: 8,
            (IActivationFunction<double>)new ReLUActivation<double>());
        var vector = new DecoderLayer<double>(attentionSize: 4, feedForwardSize: 8,
            (IVectorActivationFunction<double>)new ReLUActivation<double>());

        foreach (var layer in new LayerBase<double>[] { scalar, vector })
        {
            var attention = layer.GetSubLayers().OfType<AttentionLayer<double>>().ToList();
            Assert.Equal(2, attention.Count);
            Assert.All(attention, a =>
            {
                Assert.IsType<SoftmaxActivation<double>>(a.VectorActivation);
                Assert.Null(a.ScalarActivation);
            });
        }
    }

    private static List<MultiHeadAttentionLayer<double>> AttentionSublayers(ILayer<double> layer) =>
        layer.GetSubLayers().OfType<MultiHeadAttentionLayer<double>>().ToList();

    private static void AssertNoOutputActivation(IReadOnlyCollection<MultiHeadAttentionLayer<double>> attention, int expected)
    {
        Assert.Equal(expected, attention.Count);
        Assert.All(attention, a =>
        {
            Assert.IsType<IdentityActivation<double>>(a.ScalarActivation);
            Assert.Null(a.VectorActivation);
        });
    }

    private static Tensor<double> Input(int batch, int sequence, int features)
    {
        var input = new Tensor<double>([batch, sequence, features]);
        for (int i = 0; i < input.Length; i++) input[i] = (i % 7 - 3) * 0.1;
        return input;
    }
}
