using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for optical flow estimation models. Inherits video NN invariants
/// and adds flow-specific: zero motion for identical frames and bounded flow vectors.
/// </summary>
public abstract class OpticalFlowTestBase : VideoNNModelTestBase
{
    [Fact]
    public void IdenticalFrames_NearZeroFlow()
    {
        var network = CreateNetwork();
        var frame = CreateConstantTensor(InputShape, 0.5);
        var output = network.Predict(frame);

        // For identical input (no motion), flow should be near zero
        double magnitude = 0;
        for (int i = 0; i < output.Length; i++)
            magnitude += output[i] * output[i];
        double rms = Math.Sqrt(magnitude / Math.Max(1, output.Length));

        Assert.True(rms < 10.0,
            $"Optical flow RMS = {rms:F4} for constant input — expected near-zero flow for no motion.");
    }

    [Fact]
    public void FlowVectors_ShouldBeBounded()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);
        var output = network.Predict(input);

        for (int i = 0; i < output.Length; i++)
        {
            Assert.False(double.IsNaN(output[i]),
                $"Flow vector[{i}] is NaN — broken motion estimation.");
            Assert.True(Math.Abs(output[i]) < 1e6,
                $"Flow vector[{i}] = {output[i]:E4} is unbounded — unrealistic motion magnitude.");
        }
    }
}
