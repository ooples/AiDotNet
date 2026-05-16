using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using Xunit;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for optical flow estimation models. Inherits video NN invariants
/// and adds flow-specific: zero motion for identical frames and bounded flow vectors.
/// </summary>
public abstract class OpticalFlowTestBase : VideoNNModelTestBase
{
    // Optical-flow input contract (OpticalFlowBase.Predict): rank-4
    // [batch, 2*channels, height, width] — two consecutive frames stacked
    // along the channel axis. The default NeuralNetworkModelTestBase shape
    // is [1, 4] and inherited bases default to odd channel counts (3 = RGB),
    // both of which fail OpticalFlowBase's rank-and-parity validation. Use
    // RAPIDFlow's paper-default evaluation crop (256x256, RGB → 2*3 = 6
    // channels stacked, 2-channel dx/dy flow output) so every optical-flow
    // model in this base sees a shape its forward pass actually accepts.
    protected override int[] InputShape => [1, 6, 256, 256];
    protected override int[] OutputShape => [1, 2, 256, 256];

    [Fact(Timeout = 120000)]
    public async Task IdenticalFrames_NearZeroFlow()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var network = CreateNetwork();
        var frame = CreateConstantTensor(InputShape, 0.5);
        var output = network.Predict(frame);

        // For identical input (no motion), flow should be near zero
        double magnitude = 0;
        for (int i = 0; i < output.Length; i++)
            magnitude += output[i] * output[i];
        double rms = Math.Sqrt(magnitude / Math.Max(1, output.Length));

        Assert.True(rms < 10.0,
            $"Optical flow RMS = {rms:F4} for constant input - expected near-zero flow for no motion.");
    }

    [Fact(Timeout = 120000)]
    public async Task FlowVectors_ShouldBeBounded()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);
        var output = network.Predict(input);

        for (int i = 0; i < output.Length; i++)
        {
            Assert.False(double.IsNaN(output[i]),
                $"Flow vector[{i}] is NaN - broken motion estimation.");
            Assert.True(Math.Abs(output[i]) < 1e6,
                $"Flow vector[{i}] = {output[i]:E4} is unbounded - unrealistic motion magnitude.");
        }
    }
}
