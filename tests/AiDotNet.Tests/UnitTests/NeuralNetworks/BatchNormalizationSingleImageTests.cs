using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// Batch normalization's statistics are per feature over every other axis, so a single NCHW image is a
/// batch of H*W samples per channel. Before this was fixed, any input whose leading axis was 1 took the
/// running-statistics path in training, so single-image CNN training never normalized at all.
/// </summary>
public sealed class BatchNormalizationSingleImageTests
{
    public BatchNormalizationSingleImageTests() => TestModuleInitializer.EnsureInitialized();

    [Fact]
    public void SingleImage_NormalizesEachChannelOverItsPositions_AndUpdatesRunningStatistics()
    {
        var rng = new Random(2);
        var x = new Tensor<double>(new[] { 1, 4, 8, 8 });
        for (int i = 0; i < x.Length; i++) x[i] = 100 * (rng.NextDouble() - 0.5) + 7 * (i / 64);
        var bn = new BatchNormalizationLayer<double>();
        bn.SetTrainingMode(true);
        var y = bn.Forward(x);

        for (int channel = 0; channel < 4; channel++)
        {
            double mean = 0, sq = 0;
            for (int p = 0; p < 64; p++) mean += y[channel * 64 + p];
            mean /= 64;
            for (int p = 0; p < 64; p++) sq += (y[channel * 64 + p] - mean) * (y[channel * 64 + p] - mean);
            Assert.Equal(0.0, mean, 6);
            Assert.Equal(1.0, Math.Sqrt(sq / 64), 3); // gamma = 1, biased variance, eps = 1e-5
        }

        // The running mean moved toward the channel means (all well away from 0 here).
        Assert.Contains(bn.GetRunningMean().ToArray(), m => Math.Abs(m) > 1e-3);
    }

    [Fact]
    public void OneRowOfFeatures_StillUsesRunningStatistics()
    {
        // [1, F]: each feature has exactly one sample, so batch variance is zero and the layer must not
        // normalize with it. Running statistics start at mean 0 / variance 1, so the output is ~identity.
        var x = new Tensor<double>(new[] { 1, 3 }, new Vector<double>(new[] { 5.0, -2.0, 11.0 }));
        var bn = new BatchNormalizationLayer<double>();
        bn.SetTrainingMode(true);
        var y = bn.Forward(x);
        Assert.Equal(new[] { 5.0, -2.0, 11.0 }, y.ToArray().Select(v => Math.Round(v, 3)).ToArray());
    }
}