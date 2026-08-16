using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

public class TemporalFrameSplicingLayerTests
{
    [Fact]
    public void Forward_AdjacentFrames_AreConcatenatedAlongFeatureAxis()
    {
        var input = new Tensor<double>(
            new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 },
            new[] { 1, 4, 2 });
        var layer = new TemporalFrameSplicingLayer<double>(factor: 2);

        var output = layer.Forward(input);

        Assert.Equal(new[] { 1, 2, 4 }, output.Shape.ToArray());
        Assert.Equal(input.Length, output.Length);
        for (int i = 0; i < input.Length; i++)
            Assert.Equal(input[i], output[i]);
    }

    [Fact]
    public void Forward_IncompleteTrailingGroup_IsDiscardedLikeReleasedAdapter()
    {
        var input = new Tensor<double>(
            new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 },
            new[] { 1, 5, 2 });
        var layer = new TemporalFrameSplicingLayer<double>(factor: 2);

        var output = layer.Forward(input);

        Assert.Equal(new[] { 1, 2, 4 }, output.Shape.ToArray());
        Assert.Equal(8, output.Length);
        for (int i = 0; i < output.Length; i++)
            Assert.Equal(input[i], output[i]);
    }

    [Fact]
    public void Options_DefaultAndCopy_PreservePublishedFrameSplicingFactor()
    {
        var defaults = new AiDotNet.SpeechRecognition.LLMIntegrated.FireRedASRLLMOptions();
        Assert.Equal(2, defaults.AdapterFrameSplicingFactor);

        var customized = new AiDotNet.SpeechRecognition.LLMIntegrated.FireRedASRLLMOptions
        {
            AdapterFrameSplicingFactor = 4,
            Seed = 123
        };
        var copy = new AiDotNet.SpeechRecognition.LLMIntegrated.FireRedASRLLMOptions(customized);
        Assert.Equal(4, copy.AdapterFrameSplicingFactor);
        Assert.Equal(123, copy.Seed);
    }
}
