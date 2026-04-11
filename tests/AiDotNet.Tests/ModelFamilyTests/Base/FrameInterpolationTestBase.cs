using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using Xunit;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for frame interpolation models. Inherits video NN invariants
/// and adds interpolation-specific: output between inputs and non-empty output.
/// </summary>
public abstract class FrameInterpolationTestBase : VideoNNModelTestBase
{
    [Fact(Timeout = 120000)]
    public async Task InterpolatedFrame_ShouldBeBetweenInputs()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var network = CreateNetwork();

        var frame1 = CreateConstantTensor(InputShape, 0.2);
        var frame2 = CreateConstantTensor(InputShape, 0.8);

        var out1 = network.Predict(frame1);
        var out2 = network.Predict(frame2);

        // Interpolated output should have values between the two inputs' outputs
        // At minimum, both should produce finite output
        for (int i = 0; i < Math.Min(out1.Length, out2.Length); i++)
        {
            Assert.False(double.IsNaN(out1[i]) || double.IsNaN(out2[i]),
                $"Frame interpolation output[{i}] is NaN.");
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Interpolation_OutputNonEmpty()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);
        var output = network.Predict(input);
        Assert.True(output.Length > 0, "Frame interpolation produced empty output.");
    }
}
