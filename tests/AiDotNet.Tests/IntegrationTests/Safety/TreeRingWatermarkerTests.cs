using System;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using AiDotNet.Diffusion;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Safety.Watermarking;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Safety;

/// <summary>
/// Tree-Ring on a real (small, untrained) DDPM UNet: keyed rings in the initial noise's spectrum, DDIM sampling,
/// DDIM inversion, and the noncentral chi-squared test. Content is one generation: its seed and the image.
/// </summary>
public class TreeRingWatermarkerTests : WatermarkRoundTripTestBase<TreeRingWatermarkerTests.Generation>
{
    public sealed record Generation(int Seed, Tensor<double> Image);

    private static readonly int[] NoiseShape = { 1, 3, 32, 32 };
    private const int Steps = 20;

    private static readonly Lazy<DDPMModel<double>> Model = new(() => new DDPMModel<double>(
        unet: new UNetNoisePredictor<double>(inputChannels: 3, outputChannels: 3, baseChannels: 32,
            channelMultipliers: new[] { 1, 2 }, numResBlocks: 1, attentionResolutions: Array.Empty<int>(),
            contextDim: 0, numHeads: 4, inputHeight: 32, seed: 11),
        channels: 3, imageSize: 32, seed: 11));

    // Inversion does not depend on the key, so each image is inverted once however many keys test it.
    private static readonly ConditionalWeakTable<Tensor<double>, Tensor<double>> Inverted = new();

    private static TreeRingWatermarker<double> Watermarker(string key) => new(Model.Value, key, inferenceSteps: Steps);

    protected override Generation Sample(int seed) => new(seed, Watermarker(TestKey).GenerateWithoutWatermark(NoiseShape, seed));

    protected override Generation Embed(string key, Generation content) => new(content.Seed, Watermarker(key).Generate(NoiseShape, content.Seed));

    protected override WatermarkResult Detect(string key, Generation content)
    {
        var watermarker = Watermarker(key);
        var noise = Inverted.GetValue(content.Image, image => watermarker.InvertToNoise(image));
        return watermarker.DetectFromNoise(noise);
    }

    protected override Generation Perturb(Generation content, int seed)
    {
        // Additive Gaussian noise at 2% of the image's standard deviation.
        var data = content.Image.ToArray();
        double mean = 0, square = 0;
        foreach (double v in data) { mean += v; square += v * v; }
        mean /= data.Length;
        double sd = Math.Sqrt(Math.Max(0, square / data.Length - mean * mean));
        var rng = RandomHelper.CreateSeededRandom(4000 + seed);
        var noisy = new double[data.Length];
        for (int i = 0; i < data.Length; i++)
        {
            double u1 = 1.0 - rng.NextDouble(), u2 = rng.NextDouble();
            noisy[i] = data[i] + 0.02 * sd * Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
        }

        return new Generation(content.Seed, new Tensor<double>(noisy, content.Image.Shape.ToArray()));
    }

    protected override bool SameContent(Generation a, Generation b)
    {
        var x = a.Image.ToArray();
        var y = b.Image.ToArray();
        if (x.Length != y.Length) return false;
        for (int i = 0; i < x.Length; i++)
            if (x[i] != y[i]) return false;
        return true;
    }

    [Fact(Timeout = 120000)]
    public async Task WatermarkedNoise_IsRealGaussianNoise_WithTheKeyOnlyInsideTheMask()
    {
        await Task.Yield();
        var watermarker = Watermarker(TestKey);
        var marked = watermarker.WatermarkedNoise(NoiseShape, 5).ToArray();
        int plane = 32 * 32;
        double mean = 0, square = 0;
        for (int i = 0; i < plane; i++) { mean += marked[i]; square += marked[i] * marked[i]; }
        mean /= plane;
        double variance = square / plane - mean * mean;
        // The key replaces about 300 of 1024 modes with values of the same expected power, so the noise keeps unit scale.
        Assert.InRange(variance, 0.5, 1.6);

        // The other channels are the seed's untouched Gaussian noise, the same noise the unwatermarked generation uses.
        var reference = WatermarkerNoiseWithoutKey(5);
        for (int i = plane; i < marked.Length; i++) Assert.Equal(reference[i], marked[i]);

        // The keyed noise itself holds the key exactly, so the matched filter reaches its ceiling, the square root of
        // the scored degrees of freedom: about 2 * 157 + 1 for radius 10, so z near 17.7.
        var atSource = watermarker.DetectFromNoise(watermarker.WatermarkedNoise(NoiseShape, 5));
        Assert.True(atSource.ZScore > 15, $"The keyed noise itself scored z = {atSource.ZScore:F2}.");
        var foreign = Watermarker("another-key").DetectFromNoise(watermarker.WatermarkedNoise(NoiseShape, 5));
        Assert.True(foreign.ZScore < 3, $"A foreign key scored z = {foreign.ZScore:F2} on keyed noise.");
    }

    // The unwatermarked generation's starting noise: inverting it back to x_T is not needed, so read it by
    // marking with a radius-1 key on channel 0 only and comparing the untouched channels 1 and 2.
    private static double[] WatermarkerNoiseWithoutKey(int seed) =>
        new TreeRingWatermarker<double>(Model.Value, "unrelated-key", radius: 1, inferenceSteps: Steps).WatermarkedNoise(NoiseShape, seed).ToArray();

    [Fact(Timeout = 300000)]
    public async Task RingsPattern_IsDetected_ButOtherRingsKeysAreNotCalibratedAgainstIt()
    {
        await Task.Yield();
        // The paper's default pattern: detected under its own key and absent from unmarked images. With one value per
        // ring, though, another Rings key correlates with the mark by chance at the scale of a ring's mode count, so
        // its z-scores spread far wider than N(0, 1): a Rings result shows a Tree-Ring mark, not which key made it.
        TreeRingWatermarker<double> Rings(string key) => new(Model.Value, key, inferenceSteps: Steps, pattern: TreeRingKeyPattern.Rings);
        var marked = Rings(TestKey).Generate(NoiseShape, 21);
        var unmarked = Rings(TestKey).GenerateWithoutWatermark(NoiseShape, 21);
        double own = Rings(TestKey).Detect(marked).ZScore;
        double clean = Rings(TestKey).Detect(unmarked).ZScore;
        Assert.True(own >= 5.7, $"A Rings mark scored z = {own:F2} under its own key.");
        Assert.True(clean < 3, $"An unmarked image scored z = {clean:F2}.");

        var keyed = Rings(TestKey).WatermarkedNoise(NoiseShape, 21);
        var z = new double[40];
        for (int i = 0; i < z.Length; i++) z[i] = Rings($"rings-key-{i}").DetectFromNoise(keyed).ZScore;
        double mean = 0;
        foreach (double v in z) mean += v;
        mean /= z.Length;
        double sd = 0;
        foreach (double v in z) sd += (v - mean) * (v - mean);
        sd = Math.Sqrt(sd / (z.Length - 1));
        Assert.True(sd > 2, $"Other Rings keys' z-scores on Rings-keyed noise have sd {sd:F2}; the documented limit has changed.");
    }

    [Fact(Timeout = 300000)]
    public async Task MatchedFilter_IsCalibratedAgainstForeignKeys_WhereThePapersChiSquaredIsNot()
    {
        await Task.Yield();
        // On one marked image, 100 foreign Random keys: the reported z (matched filter) centres on 0 with unit spread,
        // while the paper's chi-squared z is biased upward because sigma^2 comes from the scored modes themselves.
        var marked = Embed(TestKey, Sample(31));
        var noise = Watermarker(TestKey).InvertToNoise(marked.Image);
        double sumMatched = 0, sumPaper = 0, squareMatched = 0;
        const int keys = 100;
        for (int i = 0; i < keys; i++)
        {
            var w = Watermarker($"calibration-key-{i}");
            double m = w.DetectFromNoise(noise).ZScore, c = w.PaperChiSquaredZ(noise);
            sumMatched += m;
            squareMatched += m * m;
            sumPaper += c;
        }

        double meanMatched = sumMatched / keys, meanPaper = sumPaper / keys;
        double sdMatched = Math.Sqrt(squareMatched / keys - meanMatched * meanMatched);
        Assert.True(Math.Abs(meanMatched) < 0.3 && sdMatched > 0.7 && sdMatched < 1.3,
            $"Matched-filter foreign-key z: mean {meanMatched:F3}, sd {sdMatched:F3}.");
        Assert.True(meanPaper > meanMatched,
            $"The paper's chi-squared foreign-key z averaged {meanPaper:F3}, the matched filter {meanMatched:F3}.");
    }

    [Fact]
    public void NoncentralChiSquaredCdf_MatchesTheCentralCaseAndASimulation()
    {
        // lambda = 0 is the central chi-squared: Pr(chi^2_2 <= x) = 1 - exp(-x / 2).
        double central = Math.Exp(TreeRingWatermarker<double>.NoncentralChiSquaredLogCdf(3.0, 2, 0.0));
        Assert.Equal(1 - Math.Exp(-1.5), central, 10);

        // Monte Carlo of a noncentral chi-squared(k = 6, lambda = 4) at its median region.
        var rng = RandomHelper.CreateSeededRandom(9);
        const int draws = 200000;
        int below = 0;
        for (int n = 0; n < draws; n++)
        {
            double sum = 0;
            for (int i = 0; i < 6; i++)
            {
                double u1 = 1.0 - rng.NextDouble(), u2 = rng.NextDouble();
                double g = Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2) + (i == 0 ? 2.0 : 0.0);
                sum += g * g;
            }

            if (sum <= 9.0) below++;
        }

        double cdf = Math.Exp(TreeRingWatermarker<double>.NoncentralChiSquaredLogCdf(9.0, 6, 4.0));
        Assert.InRange(cdf, (double)below / draws - 0.005, (double)below / draws + 0.005);

        // Far into the tail the z conversion stays finite and monotone.
        double z1 = TreeRingWatermarker<double>.ZFromLogUpperTail(-1000);
        double z2 = TreeRingWatermarker<double>.ZFromLogUpperTail(-2000);
        Assert.True(z1 > 40 && z2 > z1 && !double.IsInfinity(z2), $"z(-1000) = {z1}, z(-2000) = {z2}.");
        Assert.Equal(WatermarkStatistics.ZForUpperTail(1e-10), TreeRingWatermarker<double>.ZFromLogUpperTail(Math.Log(1e-10)), 9);
    }
}
