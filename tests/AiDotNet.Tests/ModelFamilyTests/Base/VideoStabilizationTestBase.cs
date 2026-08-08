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
/// <remarks>
/// Generic over T so the source generator's float scaffold can emit
/// <c>VideoStabilizationTestBase&lt;float&gt;</c>. While this base was non-generic its models
/// (StabStitch, ...) were locked to &lt;double&gt; and the float-first remedy was CS0308, leaving
/// only fixture shrinks/caps. Mirrors the FinancialModelTestBase/VideoNNModelTestBase pattern.
/// </remarks>
public abstract class VideoStabilizationTestBase<T> : VideoNNModelTestBase<T>
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
            double v = ConvertToDouble(output[i]);
            Assert.False(double.IsNaN(v), $"Stabilized output[{i}] is NaN.");
            Assert.False(double.IsInfinity(v), $"Stabilized output[{i}] is Infinity.");
        }
    }
}

/// <summary>
/// Non-generic double-precision shim so existing <c>: VideoStabilizationTestBase</c> derivations
/// keep compiling (same pattern as VideoNNModelTestBase).
/// </summary>
public abstract class VideoStabilizationTestBase : VideoStabilizationTestBase<double> { }
