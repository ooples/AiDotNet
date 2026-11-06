using AiDotNet.Diffusion.SuperResolution;
using AiDotNet.Models.Options;
using Xunit;

namespace AiDotNet.Tests.UnitTests.PaperFidelity;

public class UpscaleAVideoSeedTests
{
    private sealed class Probe : UpscaleAVideoModel<double>
    {
        public Probe(int? seed, DiffusionModelOptions<double>? options = null) : base(options: options, seed: seed) { }
        public double[] DrawTrainingNoise() => Enumerable.Range(0, 8).Select(_ => RandomGenerator.NextDouble()).ToArray();
    }

    [Fact]
    public void ExplicitSeedReproducesAdvancingTrainingStream()
    {
        using var a = new Probe(42);
        using var b = new Probe(42);
        using var different = new Probe(43);
        var first = a.DrawTrainingNoise();
        Assert.Equal(first, b.DrawTrainingNoise());
        Assert.False(first.SequenceEqual(different.DrawTrainingNoise()));
        var second = a.DrawTrainingNoise();
        Assert.False(first.SequenceEqual(second));
        Assert.Equal(second, b.DrawTrainingNoise());
    }

    [Fact]
    public void ExplicitSeedOverridesCopiedOptionsWithoutMutatingCaller()
    {
        var options = new DiffusionModelOptions<double> { Seed = 7 };
        using var explicitSeed = new Probe(42, options);
        using var expected = new Probe(42);
        using var optionsSeed = new Probe(null, options);
        using var expectedOptions = new Probe(7);
        Assert.Equal(expected.DrawTrainingNoise(), explicitSeed.DrawTrainingNoise());
        Assert.Equal(expectedOptions.DrawTrainingNoise(), optionsSeed.DrawTrainingNoise());
        Assert.Equal(7, options.Seed);
    }
}
