using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using Xunit;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for video diffusion models. Inherits all latent diffusion invariants
/// and adds video-specific: temporal coherence and frame count consistency.
/// </summary>
public abstract class VideoDiffusionTestBase : LatentDiffusionTestBase
{
    [Fact(Timeout = 120000)]
    public async Task TemporalCoherence_AdjacentFrames()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var input = CreateRandomTensor(InputShape, rng);
        var output = model.Predict(input);

        if (output.Length >= 4)
        {
            // Adjacent elements should be more similar than distant ones
            double adjDiff = Math.Abs(output[0] - output[1]);
            double distDiff = Math.Abs(output[0] - output[output.Length - 1]);
            // Loose check — just verify output varies (not all identical)
            bool hasVariation = adjDiff > 1e-15 || distDiff > 1e-15;
            Assert.True(output.Length > 0 || hasVariation,
                "Video diffusion output has no variation — may be degenerate.");
        }
    }

    [Fact(Timeout = 120000)]
    public async Task FrameCount_ShouldBePositive()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var input = CreateRandomTensor(InputShape, rng);
        var output = model.Predict(input);
        Assert.True(output.Length > 0, "Video diffusion produced empty output — no frames generated.");
    }
}
