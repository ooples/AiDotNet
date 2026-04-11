using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using Xunit;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for video super-resolution models. Inherits video NN invariants
/// and adds SR-specific: output at least as large as input and finite values.
/// </summary>
public abstract class VideoSuperResolutionTestBase : VideoNNModelTestBase
{
    [Fact(Timeout = 120000)]
    public async Task Output_AtLeastAsLargeAsInput()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);
        var output = network.Predict(input);
        Assert.True(output.Length >= input.Length,
            $"Super-resolution output ({output.Length}) is smaller than input ({input.Length}). " +
            "SR models should upscale, not downscale.");
    }

    [Fact(Timeout = 120000)]
    public async Task SuperResolved_ValuesShouldBeFinite()
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
                $"SR output[{i}] is NaN — upscaling introduced numerical instability.");
        }
    }
}
