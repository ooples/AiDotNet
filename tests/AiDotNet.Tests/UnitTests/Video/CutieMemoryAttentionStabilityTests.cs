using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Video.Segmentation;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Video;

public class CutieMemoryAttentionStabilityTests
{
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
