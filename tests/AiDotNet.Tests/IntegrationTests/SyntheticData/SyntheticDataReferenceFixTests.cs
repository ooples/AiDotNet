using System;
using System.Collections.Generic;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.SyntheticData;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.SyntheticData;

/// <summary>
/// Reference values for the synthetic-data fixes of the element-wise sweep: TabDDPM's Gaussian and multinomial
/// diffusion (yandex-research/tab-ddpm), sdv-dev/CTGAN's sampling, activation and conditional loss, the exact normal
/// CDF and quantile (scipy.stats.norm), the NOTEARS matrix exponential (scipy.linalg.expm) and SMOTE-NC's median
/// standard deviation (Chawla et al. 2002, section 6.1).
/// </summary>
public class SyntheticDataReferenceFixTests
{
    private static TabularDataTransformer<double> TwoCategoryTransformer()
    {
        var data = new Matrix<double>(4, 1);
        data[1, 0] = 1;
        data[3, 0] = 1;
        var transformer = new TabularDataTransformer<double>(10, RandomHelper.CreateSeededRandom(1));
        transformer.Fit(data, new List<ColumnMetadata> { new("C", ColumnDataType.Categorical, new[] { "A", "B" }, 0) });
        return transformer;
    }

    private static double SampleVariance(Vector<double> v)
    {
        double mean = 0;
        for (int i = 0; i < v.Length; i++) mean += v[i];
        mean /= v.Length;
        double ss = 0;
        for (int i = 0; i < v.Length; i++) ss += (v[i] - mean) * (v[i] - mean);
        return ss / (v.Length - 1);
    }

    [Fact]
    public void GaussianDiffusion_ReverseStep_UsesBetaAsVarianceByDefault()
    {
        // TabDDPM gaussian_p_mean_variance: model_variance = betas (fixed large). With x_t = 0 and a zero noise
        // prediction the step is sigma_t z, so its sample variance estimates sigma_t^2.
        const int n = 40000;
        double beta0 = 1e-4, beta1 = 1e-4 + (0.02 - 1e-4) / 999.0;
        double posterior = beta1 * (1 - (1 - beta0)) / (1 - (1 - beta0) * (1 - beta1));

        var large = new GaussianDiffusion<double>(1000, 1e-4, 0.02, BetaSchedule.Linear,
            ReverseDiffusionVariance.FixedLarge, RandomHelper.CreateSeededRandom(3));
        Assert.Equal(ReverseDiffusionVariance.FixedLarge, large.VarianceType);
        double largeVariance = SampleVariance(large.DenoisingStep(new Vector<double>(n), new Vector<double>(n), 1));
        Assert.InRange(largeVariance / beta1, 0.97, 1.03);

        var small = new GaussianDiffusion<double>(1000, 1e-4, 0.02, BetaSchedule.Linear,
            ReverseDiffusionVariance.FixedSmall, RandomHelper.CreateSeededRandom(3));
        double smallVariance = SampleVariance(small.DenoisingStep(new Vector<double>(n), new Vector<double>(n), 1));
        Assert.InRange(smallVariance / posterior, 0.97, 1.03);

        // The string constructor keeps TabDDPM's default.
        var legacy = new GaussianDiffusion<double>(1000, 1e-4, 0.02, "linear", RandomHelper.CreateSeededRandom(3));
        Assert.Equal(ReverseDiffusionVariance.FixedLarge, legacy.VarianceType);
    }

    [Fact]
    public void MultinomialDiffusion_Loss_IsTheKlOfThePosteriorsAndTheDecoderNllAtZero()
    {
        // betas 0.1, 0.2, 0.3: alpha_1 = 0.8, alphaBar_0 = 0.9, alpha_0 = 0.9. x_t = (1, 0), x_0 = (0, 1), x0_hat = (1/2, 1/2).
        var diffusion = new MultinomialDiffusion<double>(3, 0.1, 0.3, RandomHelper.CreateSeededRandom(1));
        var xt = new Vector<double>(new[] { 1.0, 0.0 });
        var x0 = new Vector<double>(new[] { 0.0, 1.0 });
        var logits = new Vector<double>(new[] { 0.0, 0.0 });

        // q(x_0 | x_1, x_0) ~ (0.9, 0.1) * (0.05, 0.95); q(x_0 | x_1, x0_hat) ~ (0.9, 0.1) * (0.5, 0.5).
        double q0 = 0.045 / 0.14, q1 = 0.095 / 0.14, p0 = 0.9, p1 = 0.1;
        double kl = q0 * Math.Log(q0 / p0) + q1 * Math.Log(q1 / p1);
        Assert.Equal(kl, diffusion.ComputeLoss(logits, x0, xt, 1), 10);

        // t = 0: -log q(x_{-1} = x_0 | x_0, x0_hat) with q ~ (0.95, 0.05) * (0.5, 0.5).
        Assert.Equal(-Math.Log(0.05), diffusion.ComputeLoss(logits, x0, xt, 0), 10);
    }

    [Fact]
    public void MultinomialDiffusion_LossGradient_MatchesCentralDifferences()
    {
        var diffusion = new MultinomialDiffusion<double>(10, 0.02, 0.3, RandomHelper.CreateSeededRandom(1));
        var xt = new Vector<double>(new[] { 0.2, 0.5, 0.3 });
        var x0 = new Vector<double>(new[] { 0.0, 0.0, 1.0 });
        var logits = new Vector<double>(new[] { 0.3, -0.2, 0.1 });
        foreach (int t in new[] { 0, 4 })
        {
            var gradient = diffusion.ComputeLossGradient(logits, x0, xt, t);
            for (int i = 0; i < 3; i++)
            {
                const double h = 1e-6;
                var plus = new Vector<double>(new[] { logits[0], logits[1], logits[2] });
                var minus = new Vector<double>(new[] { logits[0], logits[1], logits[2] });
                plus[i] += h;
                minus[i] -= h;
                double numeric = (diffusion.ComputeLoss(plus, x0, xt, t) - diffusion.ComputeLoss(minus, x0, xt, t)) / (2 * h);
                Assert.Equal(numeric, gradient[i], 6);
            }
        }
    }

    [Fact]
    public void MultinomialDiffusion_DenoisingStep_SamplesThePosterior()
    {
        // q(x_0 | x_1 = (1, 0), x0_hat = (1/2, 1/2)) = (0.9, 0.1) for the schedule above.
        var diffusion = new MultinomialDiffusion<double>(3, 0.1, 0.3, RandomHelper.CreateSeededRandom(5));
        var xt = new Vector<double>(new[] { 1.0, 0.0 });
        var logits = new Vector<double>(new[] { 0.0, 0.0 });
        const int draws = 20000;
        int first = 0;
        for (int i = 0; i < draws; i++)
        {
            if (diffusion.DenoisingStep(xt, logits, 1)[0] > 0.5) first++;
        }
        Assert.InRange(first / (double)draws, 0.89, 0.91);
    }

    [Fact]
    public void SampleCategoricalBlocks_DrawsFromTheBlockDistribution()
    {
        // sdv-dev/CTGAN sample(): argmax of gumbel_softmax is a categorical draw, not the most likely category.
        var transformer = TwoCategoryTransformer();
        const int rows = 20000;
        var probabilities = new Tensor<double>(new[] { rows, 2 });
        for (int i = 0; i < rows; i++)
        {
            probabilities[i, 0] = 0.7;
            probabilities[i, 1] = 0.3;
        }
        var sampled = SyntheticTensorOps<double>.SampleCategoricalBlocks(probabilities, transformer, 1, RandomHelper.CreateSeededRandom(7));
        int first = 0;
        for (int i = 0; i < rows; i++)
        {
            Assert.Equal(1.0, sampled[i, 0] + sampled[i, 1]);
            if (sampled[i, 0] > 0.5) first++;
        }
        Assert.InRange(first / (double)rows, 0.68, 0.72);
    }

    [Fact]
    public void ApplyOutputActivationsBatched_IsTheGumbelSoftmax()
    {
        // sdv-dev/CTGAN _apply_activate: gumbel_softmax(logits, tau = 0.2) = softmax((logits + g) / 0.2).
        var transformer = TwoCategoryTransformer();
        var logits = new Tensor<double>(new[] { 1.0, 0.0 }, new[] { 1, 2 });
        var gumbel = new Tensor<double>(new[] { 0.3, -0.4 }, new[] { 1, 2 });
        var activated = SyntheticTensorOps<double>.ApplyOutputActivationsBatched(logits, transformer, 1, gumbel, 0.2);
        double a = 1.3 / 0.2, b = -0.4 / 0.2;
        double expected = Math.Exp(a) / (Math.Exp(a) + Math.Exp(b));
        Assert.Equal(expected, activated[0, 0], 12);
        Assert.Equal(1 - expected, activated[0, 1], 12);
        Assert.Equal(0.2, new CTGANOptions<double>().GumbelTemperature);
        Assert.Equal(0.2, new CopulaGANOptions<double>().GumbelTemperature);
        Assert.Equal(0.2, new DPCTGANOptions<double>().GumbelTemperature);
        Assert.Equal(0.2, new CTABGANPlusOptions<double>().GumbelTemperature);
    }

    [Fact]
    public void ConditionalCrossEntropy_IsTheCrossEntropyOfTheLogits()
    {
        // sdv-dev/CTGAN _cond_loss: cross_entropy(logits, argmax(c)) masked to the conditioned column, / batch.
        var logits = new Tensor<double>(new[] { 2.0, 0.5, -1.0, 0.0, 1.0, 3.0 }, new[] { 2, 3 });
        var cond = new Tensor<double>(new[] { 1.0, 0.0, 0.0, 0.0, 1.0, 0.0 }, new[] { 2, 3 });
        var mask = new Tensor<double>(new[] { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 }, new[] { 2, 3 });
        var loss = SyntheticTensorOps<double>.ConditionalCrossEntropy(logits, cond, mask, new List<(int, int)> { (0, 3) });
        double row0 = -(2.0 - Math.Log(Math.Exp(2.0) + Math.Exp(0.5) + Math.Exp(-1.0)));
        double row1 = -(1.0 - Math.Log(Math.Exp(0.0) + Math.Exp(1.0) + Math.Exp(3.0)));
        Assert.Equal((row0 + row1) / 2, loss.ToArray()[0], 12);
    }

    [Fact]
    public void StandardNormalCdfAndQuantile_MatchScipy()
    {
        // scipy.stats.norm.cdf / norm.ppf. The Abramowitz-Stegun erf they replace was off by up to 1.5e-7.
        var cdf = SyntheticTensorOps<double>.StandardNormalCdf(new Tensor<double>(new[] { 1.0, -2.5, 0.3 }, new[] { 3 }));
        Assert.Equal(0.8413447460685429, cdf[0], 12);
        Assert.Equal(0.006209665325776132, cdf[1], 12);
        Assert.Equal(0.6179114221889527, cdf[2], 12);

        var ppf = SyntheticTensorOps<double>.StandardNormalQuantile(new Tensor<double>(new[] { 0.975, 0.1, 1e-6 }, new[] { 3 }));
        Assert.Equal(1.959963984540054, ppf[0], 10);
        Assert.Equal(-1.2815515655446004, ppf[1], 10);
        Assert.Equal(-4.753424308822899, ppf[2], 8);
    }

    [Fact]
    public void MatrixExponential_MatchesClosedForms()
    {
        // NOTEARS h(W) = tr(e^(W o W)) - d uses scipy.linalg.expm.
        var nilpotent = SyntheticTensorOps<double>.MatrixExponential(new Tensor<double>(new[] { 0.0, 1.0, 0.0, 0.0 }, new[] { 2, 2 }));
        Assert.Equal(new[] { 1.0, 1.0, 0.0, 1.0 }, nilpotent.ToArray(), new DoubleTolerance(1e-12));

        var diagonal = SyntheticTensorOps<double>.MatrixExponential(new Tensor<double>(new[] { 1.0, 0.0, 0.0, 2.0 }, new[] { 2, 2 }));
        Assert.Equal(new[] { Math.E, 0.0, 0.0, Math.Exp(2.0) }, diagonal.ToArray(), new DoubleTolerance(1e-12));

        // A large norm, where a truncated series without scaling and squaring fails: e^[[0, -4], [4, 0]] is the
        // rotation by 4 radians.
        var rotation = SyntheticTensorOps<double>.MatrixExponential(new Tensor<double>(new[] { 0.0, -4.0, 4.0, 0.0 }, new[] { 2, 2 }));
        Assert.Equal(new[] { Math.Cos(4.0), -Math.Sin(4.0), Math.Sin(4.0), Math.Cos(4.0) }, rotation.ToArray(), new DoubleTolerance(1e-11));
    }

    [Fact]
    public void SmoteNc_MedianStd_AveragesTheTwoMiddleValues()
    {
        // Two continuous columns with sample standard deviations 1 and 3: the median is 2, not the upper middle 3.
        var data = new Matrix<double>(3, 3);
        for (int i = 0; i < 3; i++)
        {
            data[i, 0] = i;
            data[i, 1] = 3 * i;
            data[i, 2] = 1;
        }
        var columns = new List<ColumnMetadata>
        {
            new("A", ColumnDataType.Continuous, columnIndex: 0),
            new("B", ColumnDataType.Continuous, columnIndex: 1),
            new("Label", ColumnDataType.Categorical, new[] { "0", "1" }, 2),
        };
        var generator = new SMOTENCGenerator<double>(new SMOTENCOptions<double> { K = 1, LabelColumnIndex = 2, MinorityClassValue = 1, Seed = 1 });
        generator.Fit(data, columns);
        Assert.Equal(2.0, generator.MedianStdContinuous, 12);
    }

    private sealed class DoubleTolerance : IEqualityComparer<double>
    {
        private readonly double _tolerance;
        public DoubleTolerance(double tolerance) => _tolerance = tolerance;
        public bool Equals(double x, double y) => Math.Abs(x - y) <= _tolerance;
        public int GetHashCode(double obj) => 0;
    }
}
