using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using Xunit;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for video inpainting models. Inherits video NN invariants
/// and adds inpainting-specific: output same size as input and bounded values.
/// </summary>
public abstract class VideoInpaintingTestBase : VideoNNModelTestBase
{
    [Fact(Timeout = 120000)]
    public async Task InpaintedOutput_SameSizeAsInput()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);
        var output = network.Predict(input);
        Assert.Equal(input.Length, output.Length);
    }

    [Fact(Timeout = 120000)]
    public async Task InpaintedValues_ShouldBeBounded()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);
        var output = network.Predict(input);

        for (int i = 0; i < output.Length; i++)
        {
            Assert.False(double.IsNaN(output[i]), $"Inpainted output[{i}] is NaN.");
            Assert.True(Math.Abs(output[i]) < 1e6,
                $"Inpainted output[{i}] = {output[i]:E4} is unbounded.");
        }
    }
}
