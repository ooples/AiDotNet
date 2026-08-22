#nullable disable
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.Motion;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Video;

public class RAPIDFlowReviewRegressionIntegrationTests
{
    [Fact]
    public void UpdateParameters_WithPartialVector_ThrowsBeforeMutatingLayers()
    {
        var model = CreateModel();
        var original = model.GetParameters();
        var partial = new Vector<double>(original.Length - 1);

        var ex = Assert.Throws<ArgumentException>(() => model.UpdateParameters(partial));

        Assert.Contains("Expected", ex.Message);
        var after = model.GetParameters();
        Assert.Equal(original.Length, after.Length);
        for (int i = 0; i < original.Length; i++)
        {
            Assert.Equal(original[i], after[i]);
        }
    }

    [Fact]
    public void Deserialize_WithDifferentConstructedLayerCount_RestoresSerializedTopology()
    {
        var source = CreateModel(numRefinementIterations: 1);
        var restored = CreateModel(numRefinementIterations: 2);

        restored.Deserialize(source.Serialize());

        AssertLayerGraphEqual(source, restored);
    }

    [Fact]
    public void Deserialize_WithAmbiguousConstructedLayerIdentity_ThrowsClearError()
    {
        var source = CreateModel();
        var restored = CreateModel();
        restored.Layers[0] = restored.Layers[^1];

        var ex = Assert.Throws<InvalidOperationException>(() =>
            restored.Deserialize(source.Serialize()));

        Assert.Contains("appears 2 times", ex.Message);
        Assert.Contains("ambiguous", ex.Message);
    }

    private static RAPIDFlow<double> CreateModel(int numRefinementIterations = 1)
    {
        return new RAPIDFlow<double>(
            new NeuralNetworkArchitecture<double>(
                inputType: InputType.ThreeDimensional,
                taskType: NeuralNetworkTaskType.Regression,
                inputHeight: 32,
                inputWidth: 32,
                inputDepth: 3,
                outputSize: 2),
            numRefinementIterations);
    }

    private static void AssertLayerGraphEqual(RAPIDFlow<double> expected, RAPIDFlow<double> actual)
    {
        Assert.Equal(expected.Layers.Count, actual.Layers.Count);
        for (int i = 0; i < expected.Layers.Count; i++)
        {
            Assert.Equal(expected.Layers[i].GetType(), actual.Layers[i].GetType());
        }

        var expectedParameters = expected.GetParameters();
        var actualParameters = actual.GetParameters();
        Assert.Equal(expectedParameters.Length, actualParameters.Length);
        for (int i = 0; i < expectedParameters.Length; i++)
        {
            Assert.Equal(expectedParameters[i], actualParameters[i], 10);
        }
    }
}
