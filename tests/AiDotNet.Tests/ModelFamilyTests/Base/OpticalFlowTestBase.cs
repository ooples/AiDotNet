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
    // 64x64 keeps the inherited video/model-family invariants at smoke-test
    // scale. Paper-scale crops belong in model-specific tests so one model
    // does not make every optical-flow invariant prohibitively expensive.
    protected override int[] InputShape => [1, 6, 64, 64];
    protected override int[] OutputShape => [1, 2, 64, 64];

    // Optical-flow train steps run a full encoder/refinement/decoder pyramid
    // over image tensors. Keep MoreData meaningful without spending the base
    // neural-network default of 250 full image-to-image updates in one test.
    protected override int MoreDataShortIterations => 10;
    protected override int MoreDataLongIterations => 20;

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
            $"Optical flow RMS = {rms:F4} for constant input — expected near-zero flow for no motion.");
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
                $"Flow vector[{i}] is NaN — broken motion estimation.");
            Assert.True(Math.Abs(output[i]) < 1e6,
                $"Flow vector[{i}] = {output[i]:E4} is unbounded — unrealistic motion magnitude.");
        }
    }
}
