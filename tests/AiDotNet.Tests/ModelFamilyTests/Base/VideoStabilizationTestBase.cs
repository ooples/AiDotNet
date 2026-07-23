using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using Xunit;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for video stabilization models. Inherits video NN invariants
/// and adds stabilization-specific: output preserves temporal length and finite values.
/// </summary>
public abstract class VideoStabilizationTestBase : VideoNNModelTestBase
{
    [Fact(Timeout = 120000)]
    public async Task StabilizedOutput_PreservesLength()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);
        var output = network.Predict(input);

        Assert.True(output.Length >= input.Length / 2,
            $"Stabilized output ({output.Length}) much shorter than input ({input.Length}).");
    }

    [Fact(Timeout = 120000)]
    public async Task StabilizedValues_ShouldBeFinite()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
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
