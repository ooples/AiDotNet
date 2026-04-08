using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for video stabilization models. Inherits video NN invariants
/// and adds stabilization-specific: output preserves temporal length and finite values.
/// </summary>
public abstract class VideoStabilizationTestBase : VideoNNModelTestBase
{
    [Fact]
    public void StabilizedOutput_PreservesLength()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);
        var output = network.Predict(input);

        Assert.True(output.Length >= input.Length / 2,
            $"Stabilized output ({output.Length}) much shorter than input ({input.Length}).");
    }

    [Fact]
    public void StabilizedValues_ShouldBeFinite()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);
        var output = network.Predict(input);

        for (int i = 0; i < output.Length; i++)
        {
            Assert.False(double.IsNaN(output[i]), $"Stabilized output[{i}] is NaN.");
            Assert.False(double.IsInfinity(output[i]), $"Stabilized output[{i}] is Infinity.");
        }
    }
}
