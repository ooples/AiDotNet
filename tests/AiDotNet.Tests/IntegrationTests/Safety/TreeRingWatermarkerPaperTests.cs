using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Diffusion;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Safety.Watermarking;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Safety;

/// <summary>
/// Tree-Ring (Wen et al., 2023) checked against the paper's equations with an independent naive DFT. Only the initial
/// noise and the test statistic are involved, so no sampling or inversion runs.
/// </summary>
public class TreeRingWatermarkerPaperTests
{
    private const double ApproximationTolerance = 0.01;

    private const int Size = 32;
    private const int Radius = 10;
    private static readonly int[] NoiseShape = { 1, 3, Size, Size };

    private static readonly Lazy<DDPMModel<double>> Model = new(() => new DDPMModel<double>(
        unet: new UNetNoisePredictor<double>(inputChannels: 3, outputChannels: 3, baseChannels: 32,
            channelMultipliers: new[] { 1, 2 }, numResBlocks: 1, attentionResolutions: Array.Empty<int>(),
            contextDim: 0, numHeads: 4, inputHeight: Size, seed: 11),
        channels: 3, imageSize: Size, seed: 11));

    private static TreeRingWatermarker<double> Watermarker(string key, int channel = 0,
        TreeRingKeyPattern pattern = TreeRingKeyPattern.Random) =>
        new(Model.Value, key, radius: Radius, channel: channel, inferenceSteps: 10, pattern: pattern);

    // Signed frequency of an unshifted DFT index (numpy fftfreq).
    private static int Frequency(int k) => k <= (Size - 1) / 2 ? k : k - Size;

    private static bool InMask(int ky, int kx)
    {
        int fy = Frequency(ky), fx = Frequency(kx);
        return fy * fy + fx * fx <= Radius * Radius;
    }

    // The 2-D DFT of channel 0, evaluated directly from its definition.
    private static (double Re, double Im)[,] NaiveDft(double[] sample)
    {
        var result = new (double, double)[Size, Size];
        for (int ky = 0; ky < Size; ky++)
            for (int kx = 0; kx < Size; kx++)
            {
                double re = 0, im = 0;
                for (int y = 0; y < Size; y++)
                    for (int x = 0; x < Size; x++)
                    {
                        double angle = -2 * Math.PI * ((double)ky * y + (double)kx * x) / Size;
                        double v = sample[y * Size + x];
                        re += v * Math.Cos(angle);
                        im += v * Math.Sin(angle);
                    }

                result[ky, kx] = (re, im);
            }

        return result;
    }

    // The seed's noise with nothing in channel 0: the key goes to channel 1 instead.
    private static double[] UnmarkedNoise(int seed) =>
        Watermarker("unrelated-key", channel: 1).WatermarkedNoise(NoiseShape, seed).ToArray();

    [Fact(Timeout = 120000)]
    public async System.Threading.Tasks.Task RingsKey_IsOneValuePerRing_InsideTheMask_AndTheSeedsNoiseOutsideIt()
    {
        await System.Threading.Tasks.Task.Yield();
        // Sec. 3.3, Tree-Ring_Rings: the key is constant along each ring of the Fourier-space circle; the reference
        // code gives the mode at distance d the value of ring ceil(d), the DC term ring 1. Outside the circle the
        // initial noise keeps its own spectrum. Values are real, so the keyed noise stays real.
        var marked = NaiveDft(Watermarker("paper-key", pattern: TreeRingKeyPattern.Rings).WatermarkedNoise(NoiseShape, 5).ToArray());
        var unmarked = NaiveDft(UnmarkedNoise(5));
        var ringValue = new Dictionary<int, double>();
        int inside = 0;
        for (int ky = 0; ky < Size; ky++)
            for (int kx = 0; kx < Size; kx++)
            {
                var (re, im) = marked[ky, kx];
                if (!InMask(ky, kx))
                {
                    Assert.Equal(unmarked[ky, kx].Re, re, 8);
                    Assert.Equal(unmarked[ky, kx].Im, im, 8);
                    continue;
                }

                inside++;
                int fy = Frequency(ky), fx = Frequency(kx);
                int ring = Math.Max(1, (int)Math.Ceiling(Math.Sqrt(fy * fy + fx * fx) - 1e-12));
                Assert.Equal(0.0, im, 8);
                if (ringValue.TryGetValue(ring, out double value)) Assert.Equal(value, re, 8);
                else ringValue[ring] = re;
            }

        Assert.Equal(Radius, ringValue.Count);
        // Lattice points with fy^2 + fx^2 <= 100.
        Assert.Equal(317, inside);
    }

    [Fact(Timeout = 120000)]
    public async System.Threading.Tasks.Task PaperChiSquared_MatchesEtaAndLambda_ComputedFromTheirDefinitions()
    {
        await System.Threading.Tasks.Task.Yield();
        // Sec. 3.3: eta = (1/sigma^2) sum_M |k* - y|^2 with sigma^2 = (1/|M|) sum_M |y|^2, and eta follows a
        // noncentral chi-squared with |M| degrees of freedom and lambda = (1/sigma^2) sum_M |k*|^2. Counted over the
        // independent half of the mask (one mode of each conjugate pair and DC), each real and imaginary part of an
        // off-DC mode has variance sigma^2 / 2, so it enters twice; the DC term is one real degree of freedom.
        var watermarker = Watermarker("paper-key");
        // The keyed noise holds k* exactly inside the mask.
        var key = NaiveDft(watermarker.WatermarkedNoise(NoiseShape, 5).ToArray());
        var unmarked = UnmarkedNoise(6);
        var y = NaiveDft(unmarked);

        double energy = 0, keyEnergy = 0, distance = 0;
        int pairs = 0, units = 0;
        for (int ky = 0; ky < Size; ky++)
            for (int kx = 0; kx < Size; kx++)
            {
                if (!InMask(ky, kx)) continue;
                int fy = Frequency(ky), fx = Frequency(kx);
                bool dc = fy == 0 && fx == 0;
                if (!dc && !(fy > 0 || (fy == 0 && fx > 0))) continue;
                units++;
                var (kr, ki) = key[ky, kx];
                var (yr, yi) = y[ky, kx];
                if (dc)
                {
                    energy += yr * yr;
                    keyEnergy += kr * kr;
                    distance += (kr - yr) * (kr - yr);
                }
                else
                {
                    pairs++;
                    energy += yr * yr + yi * yi;
                    keyEnergy += 2 * (kr * kr + ki * ki);
                    distance += 2 * ((kr - yr) * (kr - yr) + (ki - yi) * (ki - yi));
                }
            }

        // (317 - 1) / 2 conjugate pairs plus DC.
        Assert.Equal(159, units);
        double sigma2 = energy / units;
        double eta = distance / sigma2, lambda = keyEnergy / sigma2;
        int k = 2 * pairs + 1;
        double expected = TreeRingWatermarker<double>.ZFromLogUpperTail(
            TreeRingWatermarker<double>.NoncentralChiSquaredLogCdf(eta, k, lambda));
        double actual = watermarker.PaperChiSquaredZ(new AiDotNet.Tensors.LinearAlgebra.Tensor<double>(unmarked, NoiseShape));
        Assert.Equal(expected, actual, 6);

        // The tail itself, checked independently of the series: Patnaik's two-moment fit, noncentral chi^2_k(lambda)
        // ~ c chi^2_nu, then Wilson-Hilferty's normal approximation of that chi^2. At k = 317 the fit's z is close to
        // the exact one: measured 0.3920 against the series' 0.3940, which an independent 60-digit Poisson-mixture
        // sum also gives (0.393964). The 0.01 allowed is five times that gap.
        double scale = (k + 2 * lambda) / (k + lambda), nu = (k + lambda) * (k + lambda) / (k + 2 * lambda);
        double h = 2.0 / (9.0 * nu);
        double approximate = -(Math.Cbrt(eta / (scale * nu)) - (1 - h)) / Math.Sqrt(h);
        Assert.True(Math.Abs(expected - approximate) <= ApproximationTolerance,
            $"eta = {eta:R}, k = {k}, lambda = {lambda:R}: the series gives z = {expected:R}, the approximation {approximate:R}.");
    }
}
