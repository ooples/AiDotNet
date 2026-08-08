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
public abstract class FrameInterpolationTestBase<T> : VideoNNModelTestBase<T>
{
    /// <summary>
    /// Two independent forward passes must both produce finite, non-empty output.
    /// </summary>
    /// <remarks>
    /// RENAMED TO WHAT IT ACTUALLY CHECKS. As <c>InterpolatedFrame_ShouldBeBetweenInputs</c>
    /// this asserted nothing of the kind: two INDEPENDENT <c>Predict</c> calls cannot show
    /// that a frame lies between two others, because no interpolation is performed between
    /// them. The old body was also satisfiable three ways without the model working -- an
    /// empty output skipped the loop entirely, <c>Infinity</c> passed the NaN-only check, and
    /// a length mismatch was hidden by <c>Math.Min</c>.
    ///
    /// A genuine between-ness invariant needs an interpolation entry point taking both frames
    /// and a t; this base class has no such contract, so claiming one in the test name was the
    /// defect. Asserting the weaker property under an honest name is worth more than a strong
    /// name over a check that cannot fail.
    /// </remarks>
    [Fact(Timeout = 120000)]
    public async Task IndependentFrames_ShouldProduceFiniteOutput()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var network = CreateNetwork();

        var frame1 = CreateConstantTensor(InputShape, 0.2);
        var frame2 = CreateConstantTensor(InputShape, 0.8);

        var out1 = network.Predict(frame1);
        var out2 = network.Predict(frame2);

        // An empty output previously skipped the loop and passed.
        Assert.True(out1.Length > 0, "Frame interpolation produced empty output for frame 1.");
        Assert.True(out2.Length > 0, "Frame interpolation produced empty output for frame 2.");
        Assert.True(out1.Length == out2.Length,
            $"Two frames of identical shape produced different output lengths ({out1.Length} vs {out2.Length}).");

        for (int i = 0; i < out1.Length; i++)
        {
            double value1 = ConvertToDouble(out1[i]);
            double value2 = ConvertToDouble(out2[i]);
            // Infinity previously passed a NaN-only check.
            Assert.True(double.IsFinite(value1),
                $"Frame interpolation output[{i}] for frame 1 is {value1}.");
            Assert.True(double.IsFinite(value2),
                $"Frame interpolation output[{i}] for frame 2 is {value2}.");
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

/// <summary>Default double-precision alias for existing frame-interpolation tests.</summary>
public abstract class FrameInterpolationTestBase : FrameInterpolationTestBase<double> { }
