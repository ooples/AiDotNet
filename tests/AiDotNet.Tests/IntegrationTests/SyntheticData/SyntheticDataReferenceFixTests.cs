using System;
using System.Collections.Generic;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
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

    [Fact]
    public void GanOptions_CopyConstructors_CopyEverySettingAndCloneTheWidths()
    {
        static void Check<TOptions>(TOptions original, Func<TOptions, TOptions> copy, Func<TOptions, int[]> generator,
            Func<TOptions, int[]> discriminator, Func<TOptions, double> temperature) where TOptions : SyntheticDataGeneratorOptions<double>
        {
            original.Seed = 11;
            original.MaxGradNorm = 2.5;
            original.NumFeatures = 3;
            original.ConfidenceLevel = 0.5;
            original.TimeHorizon = 4;
            var clone = copy(original);
            Assert.Equal((11, 2.5, 3, 0.5, 4), (clone.Seed ?? 0, clone.MaxGradNorm, clone.NumFeatures, clone.ConfidenceLevel, clone.TimeHorizon));
            Assert.Equal(0.7, temperature(clone));
            Assert.Equal(generator(original), generator(clone));
            Assert.NotSame(generator(original), generator(clone));
            Assert.NotSame(discriminator(original), discriminator(clone));
            foreach (var property in typeof(TOptions).GetProperties(
                System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.DeclaredOnly))
            {
                if (property.CanRead && property.CanWrite && property.GetIndexParameters().Length == 0
                    && !property.PropertyType.IsArray)
                {
                    Assert.Equal(property.GetValue(original), property.GetValue(clone));
                }
            }
        }

        Check(new CTGANOptions<double> { GumbelTemperature = 0.7, BatchSize = 30, PacSize = 3, GeneratorDimensions = new[] { 5, 6 } },
            o => new CTGANOptions<double>(o), o => o.GeneratorDimensions, o => o.DiscriminatorDimensions, o => o.GumbelTemperature);
        Check(new CopulaGANOptions<double> { GumbelTemperature = 0.7, Epochs = 9, VGMModes = 4 },
            o => new CopulaGANOptions<double>(o), o => o.GeneratorDimensions, o => o.DiscriminatorDimensions, o => o.GumbelTemperature);
        Check(new CTABGANPlusOptions<double> { GumbelTemperature = 0.7, ClassifierWeight = 3, TargetColumnIndex = 2 },
            o => new CTABGANPlusOptions<double>(o), o => o.GeneratorDimensions, o => o.DiscriminatorDimensions, o => o.GumbelTemperature);
        Check(new DPCTGANOptions<double> { GumbelTemperature = 0.7, Epsilon = 2, Delta = 1e-6, ClipNorm = 0.5, NoiseMultiplier = 1.5 },
            o => new DPCTGANOptions<double>(o), o => o.GeneratorDimensions, o => o.DiscriminatorDimensions, o => o.GumbelTemperature);

        var copyConstructor = typeof(CTGANOptions<double>).GetConstructor(new[] { typeof(CTGANOptions<double>) });
        Assert.NotNull(copyConstructor);
        var thrown = Assert.Throws<System.Reflection.TargetInvocationException>(() => copyConstructor?.Invoke(new object?[] { null }));
        Assert.IsType<ArgumentNullException>(thrown.InnerException);
    }

    [Fact]
    public void CopulaSynth_EmpiricalCdf_IsTheHazenPosition()
    {
        // Order statistic i of n maps to (i + 0.5) / n; a tie block maps to the Hazen position of its mid-rank.
        var sorted = new Tensor<double>(new[] { 1.0, 2.0, 3.0, 4.0 }, new[] { 4 });
        var u = CopulaSynthGenerator<double>.EmpiricalCdf(sorted, sorted);
        Assert.Equal(new[] { 0.125, 0.375, 0.625, 0.875 }, u.ToArray(), new DoubleTolerance(1e-15));

        var tied = new Tensor<double>(new[] { 1.0, 1.0, 2.0, 3.0 }, new[] { 4 });
        var tiedU = CopulaSynthGenerator<double>.EmpiricalCdf(tied, new Tensor<double>(new[] { 1.0, 2.0 }, new[] { 2 }));
        Assert.Equal(new[] { 0.25, 0.625 }, tiedU.ToArray(), new DoubleTolerance(1e-15));
    }

    [Fact]
    public void CopulaSynth_RoundTrip_LandsOnEachOrderStatistic()
    {
        // value -> u -> z -> u: on the Hazen grid the inverse position u n - 0.5 is the value's own sorted index,
        // so the quantile function returns the training value instead of blending it half a rank with its neighbour.
        var sorted = new Tensor<double>(new[] { 1.0, 3.0, 4.0, 7.0, 10.0 }, new[] { 5 });
        var z = SyntheticTensorOps<double>.StandardNormalQuantile(CopulaSynthGenerator<double>.EmpiricalCdf(sorted, sorted));
        var u = SyntheticTensorOps<double>.StandardNormalCdf(z);
        for (int i = 0; i < 5; i++)
        {
            Assert.Equal(i, u[i] * 5 - 0.5, 9);
        }
    }

    [Fact]
    public void CopulaSynth_CorrelationOfOneRow_IsTheIdentity()
    {
        var oneRow = new Tensor<double>(new[] { 0.3, -1.2, 2.0 }, new[] { 1, 3 });
        var correlation = CopulaSynthGenerator<double>.ComputeCorrelation(oneRow, 1);
        Assert.Equal(new[] { 1.0, 0, 0, 0, 1, 0, 0, 0, 1 }, correlation.ToArray());

        var data = new Matrix<double>(1, 2);
        data[0, 0] = 2.0;
        data[0, 1] = 5.0;
        var generator = new CopulaSynthGenerator<double>(new CopulaSynthOptions<double> { Seed = 1 });
        generator.Fit(data, new List<ColumnMetadata>
        {
            new("A", ColumnDataType.Continuous, columnIndex: 0),
            new("B", ColumnDataType.Continuous, columnIndex: 1),
        });
        var rows = generator.Generate(3);
        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(2.0, rows[i, 0]);
            Assert.Equal(5.0, rows[i, 1]);
        }
    }

    [Fact]
    public void GaussianDiffusion_RejectsAnUndefinedVarianceType()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new GaussianDiffusion<double>(10, 1e-4, 0.02, BetaSchedule.Linear,
            (ReverseDiffusionVariance)42, RandomHelper.CreateSeededRandom(1)));
    }

    [Fact]
    public void MultinomialDiffusion_DenoisingStep_NormalizesATinyPositivePosterior()
    {
        // beta = 1e-12, x_t = (1, 0), logits (-25, 0): the posterior ~ (1.44e-11, 5e-13) sums to under 1e-10 but is
        // positive, so category 0 is drawn with probability 0.966 (not x0_hat's 1.4e-11).
        var diffusion = new MultinomialDiffusion<double>(3, 1e-12, 1e-12, RandomHelper.CreateSeededRandom(9));
        var xt = new Vector<double>(new[] { 1.0, 0.0 });
        var logits = new Vector<double>(new[] { -25.0, 0.0 });
        const int draws = 4000;
        int first = 0;
        for (int i = 0; i < draws; i++)
        {
            if (diffusion.DenoisingStep(xt, logits, 1)[0] > 0.5) first++;
        }
        Assert.InRange(first / (double)draws, 0.95, 0.98);
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(-0.2)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NaN)]
    public void ApplyOutputActivationsBatched_RejectsAnInvalidGumbelTemperature(double temperature)
    {
        var transformer = TwoCategoryTransformer();
        var logits = new Tensor<double>(new[] { 1.0, 0.0 }, new[] { 1, 2 });
        var gumbel = new Tensor<double>(new[] { 0.3, -0.4 }, new[] { 1, 2 });
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            SyntheticTensorOps<double>.ApplyOutputActivationsBatched(logits, transformer, 1, gumbel, temperature));
    }

    private static SMOTENCGenerator<double> FittedSmoteNc(double[,] rows)
    {
        int n = rows.GetLength(0);
        var data = new Matrix<double>(n, 3);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < 3; j++) data[i, j] = rows[i, j];
        }
        var generator = new SMOTENCGenerator<double>(new SMOTENCOptions<double> { K = 1, LabelColumnIndex = 2, MinorityClassValue = 1, Seed = 1 });
        generator.Fit(data, new List<ColumnMetadata>
        {
            new("A", ColumnDataType.Continuous, columnIndex: 0),
            new("C", ColumnDataType.Categorical, new[] { "0", "1", "2", "3" }, 1),
            new("Label", ColumnDataType.Categorical, new[] { "0", "1" }, 2),
        });
        return generator;
    }

    [Fact]
    public void SmoteNc_Distance_IgnoresANaNCategoricalCell()
    {
        // A NaN category is no mismatch (|NaN - v| > 0.5 is false), as in master's per-pair distance; it must not
        // turn the continuous term into NaN.
        var generator = FittedSmoteNc(new[,] { { 0.0, double.NaN, 1 }, { 1.0, 0, 1 }, { 5.0, 0, 1 } });
        Assert.Equal(new[] { 0.0, 1.0, 5.0 }, generator.DistancesFrom(0).ToArray(), new DoubleTolerance(1e-12));
    }

    [Fact]
    public void SmoteNc_MajorityVote_CountsInLinearSpaceWithFirstMetTieBreak()
    {
        var generator = FittedSmoteNc(new[,] { { 0.0, 2, 1 }, { 1.0, 3, 1 }, { 2.0, double.NaN, 1 }, { 3.0, 3, 1 } });
        // A 2-2 tie goes to the value met first.
        Assert.Equal(2.0, generator.MajorityVoteAmongNeighbors(1, new[] { 0, 1, 3, 0 }));
        // NaN values form one group, as equal keys of master's dictionary did.
        Assert.True(double.IsNaN(generator.MajorityVoteAmongNeighbors(1, new[] { 2, 2, 1 })));
        // 60,000 neighbours: a k x k equality matrix would need 3.6e9 entries.
        var many = new int[60000];
        for (int i = 0; i < many.Length; i++) many[i] = i < 29000 ? 0 : (i % 2 == 0 ? 1 : 3);
        Assert.Equal(3.0, generator.MajorityVoteAmongNeighbors(1, many));
    }

    [Fact]
    public void ScaledNorms_DoNotOverflowFloat()
    {
        // The squares of 1e20f exceed float's range; the norm, the std and the clipped gradient do not.
        Assert.Equal(1e20f, SyntheticTabularGeneratorBase<float>.ScaledL2Norm(new Tensor<float>(new[] { 1e20f }, new[] { 1 })));

        var column = new Matrix<float>(2, 1);
        column[0, 0] = -1e20f;
        column[1, 0] = 1e20f;
        var meta = new ColumnMetadata("A", ColumnDataType.Continuous, columnIndex: 0);
        SyntheticTabularGeneratorBase<float>.ComputeColumnStatistics(column, 0, meta);
        Assert.InRange(meta.Std / (Math.Sqrt(2.0) * 1e20), 1 - 1e-6, 1 + 1e-6);

        var clipped = ClipProbe.Clip(new Tensor<float>(new[] { 1e20f }, new[] { 1 }), 1.0);
        Assert.Equal(1f, clipped[0], 5);
    }

    [Fact]
    public void Generation_WritesRowsImmediatelyAndDecodesAcrossBlocks()
    {
        // A row copied into the buffer is not changed by later writes to its source (a reused layer buffer).
        var buffer = new Tensor<double>(new[] { 2, 3 });
        var source = new Tensor<double>(new[] { 1.0, 2.0 }, new[] { 2 });
        SyntheticTensorOps<double>.WriteRow(buffer, 0, source, 3);
        source[0] = 9.0;
        SyntheticTensorOps<double>.WriteRow(buffer, 1, source, 3);
        Assert.Equal(new[] { 1.0, 2.0, 0.0, 9.0, 2.0, 0.0 }, buffer.ToArray());

        // Blocks are decoded in order and land in their own rows.
        int rows = SyntheticTensorOps<double>.GenerationBlockRows * 2 + 5;
        var generated = new Tensor<double>(new[] { rows, 2 });
        for (int i = 0; i < rows; i++)
        {
            generated[i, 0] = i;
            generated[i, 1] = -i;
        }
        int calls = 0;
        var decoded = SyntheticTensorOps<double>.DecodeInBlocks(generated, block =>
        {
            calls++;
            Assert.True(block.Shape[0] <= SyntheticTensorOps<double>.GenerationBlockRows);
            var m = block.ToMatrix();
            var result = new Matrix<double>(m.Rows, 1);
            for (int i = 0; i < m.Rows; i++) result[i, 0] = m[i, 0] - m[i, 1];
            return result;
        });
        Assert.Equal(3, calls);
        for (int i = 0; i < rows; i++) Assert.Equal(2.0 * i, decoded[i, 0]);
    }

    [Theory]
    [InlineData("CTAB")]
    [InlineData("Copula")]
    [InlineData("Causal")]
    public void Fit_ReleasesTheCachedTrainingRows(string kind)
    {
        var architecture = new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
            NeuralNetworkTaskType.Generative, inputSize: 4, outputSize: 3) { RandomSeed = 7 };
        ModelOptions options = kind switch
        {
            "CTAB" => new CTABGANPlusOptions<double>(),
            "Copula" => new CopulaGANOptions<double>(),
            _ => new CausalGANOptions<double>(),
        };
        SyntheticGanOptionFixture.PrepareOptions(options, "Epochs");
        options.Seed = 7;
        var epochs = options.GetType().GetProperty("Epochs");
        Assert.NotNull(epochs);
        epochs?.SetValue(options, 1);
        using NeuralNetworkBase<double> model = kind switch
        {
            "CTAB" => new CTABGANPlusGenerator<double>(architecture, (CTABGANPlusOptions<double>)options),
            "Copula" => new CopulaGANGenerator<double>(architecture, (CopulaGANOptions<double>)options),
            _ => new CausalGANGenerator<double>(architecture, (CausalGANOptions<double>)options),
        };
        SyntheticGanOptionFixture.Train(model);
        foreach (var name in new[] { "_realDataSource", "_realDataTensor" })
        {
            var field = model.GetType().GetField(name, System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            Assert.NotNull(field);
            Assert.Null(field?.GetValue(model));
        }
    }

    private sealed class ClipProbe : SMOTENCGenerator<float>
    {
        private ClipProbe() : base(new SMOTENCOptions<float>())
        {
        }

        public static Tensor<float> Clip(Tensor<float> gradient, double maxNorm) => ClipGradientNorm(gradient, maxNorm);
    }

    private sealed class DoubleTolerance : IEqualityComparer<double>
    {
        private readonly double _tolerance;
        public DoubleTolerance(double tolerance) => _tolerance = tolerance;
        public bool Equals(double x, double y) => Math.Abs(x - y) <= _tolerance;
        public int GetHashCode(double obj) => 0;
    }
}
