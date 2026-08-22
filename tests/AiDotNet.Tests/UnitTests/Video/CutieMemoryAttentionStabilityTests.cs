using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Video.Segmentation;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Video;

public class CutieMemoryAttentionStabilityTests
{
    [Fact]
    public void MemoryAttention_FiniteInputsReturnExactScaledDotProduct()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.BinaryClassification,
            inputHeight: 4,
            inputWidth: 4,
            inputDepth: 3,
            outputSize: 1);
        using var model = new Cutie<double>(architecture, numFeatures: 2, memorySize: 1);
        using var query = new Tensor<double>([1, 2, 1, 1]);
        using var key = new Tensor<double>([1, 2, 1, 1]);
        query[0] = 2.0;
        query[1] = -3.0;
        key[0] = 4.0;
        key[1] = 5.0;

        using var score = model.ComputeFiniteMemoryAttentionScore(query, key, 0.5);

        Assert.Equal(new[] { 1, 1, 1, 1 }, score.Shape.ToArray());
        Assert.Equal(-3.5, score[0], 12);
    }

    [Fact]
    public void MemoryAttention_RejectsOverflowedScoreBeforeSoftmaxMaximum()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.BinaryClassification,
            inputHeight: 4,
            inputWidth: 4,
            inputDepth: 3,
            outputSize: 1);
        using var model = new Cutie<double>(architecture, numFeatures: 2, memorySize: 1);
        using var query = new Tensor<double>([1, 2, 1, 1]);
        using var key = new Tensor<double>([1, 2, 1, 1]);
        query.Fill(double.MaxValue);
        key.Fill(double.MaxValue);

        var error = Assert.Throws<ArithmeticException>(
            () => model.ComputeFiniteMemoryAttentionScore(query, key, 1.0));

        Assert.Contains("non-finite", error.Message, StringComparison.Ordinal);
    }
}
