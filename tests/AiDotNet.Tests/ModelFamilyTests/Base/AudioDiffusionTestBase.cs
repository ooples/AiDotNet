using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for audio diffusion models. Inherits all latent diffusion invariants
/// and adds audio-specific: reasonable output length and finite spectral energy.
/// </summary>
public abstract class AudioDiffusionTestBase : LatentDiffusionTestBase
{
    [Fact]
    public void AudioLength_ShouldBeReasonable()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var input = CreateRandomTensor(InputShape, rng);
        var output = model.Predict(input);
        Assert.True(output.Length > 0, "Audio diffusion produced empty output.");
        Assert.True(output.Length <= input.Length * 10,
            $"Audio output length ({output.Length}) is unreasonably large relative to input ({input.Length}).");
    }

    [Fact]
    public void SpectralEnergy_ShouldBeFinite()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var input = CreateRandomTensor(InputShape, rng);
        var output = model.Predict(input);

        double energy = 0;
        for (int i = 0; i < output.Length; i++)
            energy += output[i] * output[i];

        Assert.True(!double.IsNaN(energy) && !double.IsInfinity(energy),
            "Audio diffusion output has infinite spectral energy.");
        Assert.True(energy < 1e12,
            $"Audio diffusion energy = {energy:E4} is unreasonably large.");
    }
}
