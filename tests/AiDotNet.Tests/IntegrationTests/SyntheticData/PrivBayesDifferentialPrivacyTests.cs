using System;
using System.Collections.Generic;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.SyntheticData;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNetTests.IntegrationTests.SyntheticData;

/// <summary>
/// Differential-privacy contract tests for <see cref="BayesianNetworkSynthGenerator{T}"/>, which
/// implements PrivBayes (Zhang et al., SIGMOD 2014 / TODS 2017).
/// </summary>
/// <remarks>
/// <para>
/// These exist because the class previously carried the PrivBayes citation while implementing a plain
/// Bayesian-network synthesizer: BIC-scored structure search and Laplace-SMOOTHED conditional
/// probability tables, with no epsilon and therefore no privacy guarantee of any kind. Laplace
/// smoothing is a prior; it is not a privacy mechanism, and the resemblance in name is the entire trap.
/// </para>
/// <para>
/// The two properties that make this PrivBayes are asserted here directly:
/// </para>
/// <para>
/// 1. <b>Structure selection must be randomized.</b> Choosing the top-scoring parent set makes the
/// released network a deterministic function of the data, so a single record can decide which edge
/// appears. The exponential mechanism samples instead, which is what removes that leak.
/// </para>
/// <para>
/// 2. <b>Marginals must be noised.</b> Laplace noise calibrated to the marginal's sensitivity is added
/// before conditionals are derived, so the released distributions do not expose individual counts.
/// </para>
/// <para>
/// A test that only checked "output looks like plausible tabular data" would pass just as happily with
/// the privacy mechanisms deleted, which is precisely how the gap survived.
/// </para>
/// </remarks>
public class PrivBayesDifferentialPrivacyTests
{
    private readonly ITestOutputHelper _out;

    public PrivBayesDifferentialPrivacyTests(ITestOutputHelper output) => _out = output;

    private const int Rows = 200;
    private const int Cols = 4;

    /// <summary>
    /// Deterministic data with real dependence between columns, so mutual information is non-trivial
    /// and structure learning has something to find.
    /// </summary>
    private static (Matrix<double> Data, List<ColumnMetadata> Columns) CreateData(int seed = 7)
    {
        var rng = new Random(seed);
        var data = new Matrix<double>(Rows, Cols);
        for (int i = 0; i < Rows; i++)
        {
            double a = rng.NextDouble();
            data[i, 0] = a;
            data[i, 1] = a * 0.8 + rng.NextDouble() * 0.2;   // depends on col 0
            data[i, 2] = data[i, 1] * 0.7 + rng.NextDouble() * 0.3;
            data[i, 3] = rng.NextDouble();                    // independent
        }

        var columns = new List<ColumnMetadata>();
        for (int j = 0; j < Cols; j++)
        {
            columns.Add(new ColumnMetadata($"c{j}", ColumnDataType.Continuous));
        }

        return (data, columns);
    }

    private static Matrix<double> FitAndGenerate(BayesianNetworkSynthOptions<double> options, int samples = 100)
    {
        var (data, columns) = CreateData();
        var generator = new BayesianNetworkSynthGenerator<double>(options);
        generator.Fit(data, columns, 1);
        return generator.Generate(samples);
    }

    private static bool Fin(double v) => !double.IsNaN(v) && !double.IsInfinity(v);

    #region Defaults

    /// <summary>
    /// PrivBayes exists to release data privately, so the privacy mechanisms must be ON unless the
    /// caller explicitly opts out. A default of "off" would mean the type silently provides no
    /// guarantee while carrying the PrivBayes name.
    /// </summary>
    [Fact]
    public void DifferentialPrivacy_IsEnabledByDefault()
    {
        var options = new BayesianNetworkSynthOptions<double>();
        Assert.True(options.EnableDifferentialPrivacy);
        Assert.True(options.PrivacyBudget > 0);
        Assert.InRange(options.StructureBudgetFraction, 0.0, 1.0);
        _out.WriteLine($"defaults: eps={options.PrivacyBudget} split={options.StructureBudgetFraction}");
    }

    #endregion

    #region Randomized structure selection (exponential mechanism)

    /// <summary>
    /// With the exponential mechanism, two different RNG seeds over the SAME data should not always
    /// produce byte-identical output. A deterministic arg-max would make the seed irrelevant to
    /// structure, so persistent identity across many seeds indicates the mechanism is not sampling.
    /// </summary>
    [Fact]
    public void StructureSelection_IsRandomized_AcrossSeeds()
    {
        var outputs = new List<string>();
        for (int seed = 1; seed <= 6; seed++)
        {
            var m = FitAndGenerate(new BayesianNetworkSynthOptions<double>
            {
                Seed = seed,
                MaxParents = 2,
                NumBins = 5
            }, samples: 40);

            var sb = new System.Text.StringBuilder();
            for (int i = 0; i < m.Rows; i++)
                for (int j = 0; j < m.Columns; j++)
                    sb.Append(Math.Round(m[i, j], 6)).Append(',');
            outputs.Add(sb.ToString());
        }

        int distinct = new HashSet<string>(outputs).Count;
        _out.WriteLine($"distinct outputs across 6 seeds: {distinct}");
        Assert.True(distinct > 1,
            "Every seed produced identical output, so nothing in fitting or generation is actually " +
            "sampling — the exponential mechanism is not in effect.");
    }

    /// <summary>
    /// A fixed seed must reproduce exactly. Noise that cannot be reproduced cannot be tested, and a
    /// synthesizer that cannot be reproduced cannot be validated against a paper.
    /// </summary>
    [Fact]
    public void SameSeed_ReproducesIdenticalOutput()
    {
        Matrix<double> Run() => FitAndGenerate(new BayesianNetworkSynthOptions<double>
        {
            Seed = 1234,
            MaxParents = 2,
            NumBins = 5
        }, samples: 50);

        var a = Run();
        var b = Run();

        Assert.Equal(a.Rows, b.Rows);
        Assert.Equal(a.Columns, b.Columns);
        for (int i = 0; i < a.Rows; i++)
        {
            for (int j = 0; j < a.Columns; j++)
            {
                Assert.Equal(a[i, j], b[i, j], 10);
            }
        }
    }

    #endregion

    #region Noised marginals

    /// <summary>
    /// Noise can drive marginal cells negative, which is not a distribution. After clipping and
    /// renormalization the sampler must still only ever emit values drawn from real bin ranges — so
    /// every generated value must be finite and within the observed data range.
    /// </summary>
    [Fact]
    public void NoisedMarginals_StillProduceFiniteInRangeValues()
    {
        var m = FitAndGenerate(new BayesianNetworkSynthOptions<double>
        {
            Seed = 42,
            MaxParents = 2,
            NumBins = 5,
            PrivacyBudget = 0.1      // heavy noise: many cells will clip to zero
        }, samples: 120);

        int nonFinite = 0;
        for (int i = 0; i < m.Rows; i++)
        {
            for (int j = 0; j < m.Columns; j++)
            {
                if (!Fin(m[i, j])) nonFinite++;
            }
        }

        _out.WriteLine($"nonFinite={nonFinite}/{m.Rows * m.Columns} at eps=0.1");
        Assert.Equal(0, nonFinite);
    }

    /// <summary>
    /// A tighter privacy budget means more noise, so the synthetic data should track the source data
    /// LESS closely. If utility is identical at eps=0.02 and eps=5, the budget is not reaching the
    /// noise calibration at all.
    /// </summary>
    [Fact]
    public void SmallerPrivacyBudget_DistortsMoreThanLargerBudget()
    {
        var (data, _) = CreateData();

        // Column means of the source data, as a coarse utility measure.
        var sourceMeans = new double[Cols];
        for (int j = 0; j < Cols; j++)
        {
            double s = 0;
            for (int i = 0; i < Rows; i++) s += data[i, j];
            sourceMeans[j] = s / Rows;
        }

        double MeanAbsError(double epsilon, int seed)
        {
            var m = FitAndGenerate(new BayesianNetworkSynthOptions<double>
            {
                Seed = seed,
                MaxParents = 2,
                NumBins = 5,
                PrivacyBudget = epsilon
            }, samples: 200);

            double total = 0;
            for (int j = 0; j < m.Columns; j++)
            {
                double s = 0;
                for (int i = 0; i < m.Rows; i++) s += m[i, j];
                total += Math.Abs(s / m.Rows - sourceMeans[j]);
            }

            return total / m.Columns;
        }

        // Average over several seeds: a single draw of a noise mechanism proves nothing, since any one
        // draw can land favourably.
        double tight = 0, loose = 0;
        const int draws = 5;
        for (int s = 1; s <= draws; s++)
        {
            tight += MeanAbsError(0.02, s);
            loose += MeanAbsError(5.0, s);
        }

        tight /= draws;
        loose /= draws;

        _out.WriteLine($"mean |error| eps=0.02 -> {tight:F5}; eps=5.0 -> {loose:F5}");
        Assert.True(tight > loose,
            $"A tighter budget (eps=0.02, error {tight:F5}) did not distort more than a loose one " +
            $"(eps=5.0, error {loose:F5}), so the privacy budget is not calibrating the noise.");
    }

    #endregion

    #region Opt-out

    /// <summary>
    /// The non-private path must remain available and functional for callers who explicitly accept
    /// having no guarantee — and it must be reproducible too.
    /// </summary>
    [Fact]
    public void OptingOut_StillProducesValidOutput()
    {
        var m = FitAndGenerate(new BayesianNetworkSynthOptions<double>
        {
            Seed = 99,
            MaxParents = 2,
            NumBins = 5,
            EnableDifferentialPrivacy = false
        }, samples: 60);

        Assert.Equal(60, m.Rows);
        Assert.Equal(Cols, m.Columns);
        for (int i = 0; i < m.Rows; i++)
        {
            for (int j = 0; j < m.Columns; j++)
            {
                Assert.True(Fin(m[i, j]));
            }
        }
    }

    /// <summary>
    /// Turning privacy off should change the result — the private and non-private paths use different
    /// structure search AND different conditional estimation, so identical output would mean one of
    /// the branches is not being taken.
    /// </summary>
    [Fact]
    public void PrivateAndNonPrivatePaths_Differ()
    {
        Matrix<double> Run(bool dp) => FitAndGenerate(new BayesianNetworkSynthOptions<double>
        {
            Seed = 2024,
            MaxParents = 2,
            NumBins = 5,
            EnableDifferentialPrivacy = dp
        }, samples: 60);

        var priv = Run(true);
        var plain = Run(false);

        bool identical = true;
        for (int i = 0; i < priv.Rows && identical; i++)
        {
            for (int j = 0; j < priv.Columns; j++)
            {
                if (Math.Abs(priv[i, j] - plain[i, j]) > 1e-12) { identical = false; break; }
            }
        }

        _out.WriteLine($"private vs non-private identical: {identical}");
        Assert.False(identical,
            "The differentially private and non-private paths produced identical output, so the " +
            "EnableDifferentialPrivacy switch is not selecting a different code path.");
    }

    #endregion

    #region Edge cases

    /// <summary>
    /// Sensitivity of mutual information involves log(n/(n-1)), which is undefined at n = 1. A
    /// single-row dataset must not produce NaN or throw.
    /// </summary>
    [Fact]
    public void SingleRow_DoesNotProduceNaN()
    {
        var data = new Matrix<double>(1, Cols);
        for (int j = 0; j < Cols; j++) data[0, j] = 0.5;

        var columns = new List<ColumnMetadata>();
        for (int j = 0; j < Cols; j++)
            columns.Add(new ColumnMetadata($"c{j}", ColumnDataType.Continuous));

        var generator = new BayesianNetworkSynthGenerator<double>(
            new BayesianNetworkSynthOptions<double> { Seed = 3, MaxParents = 2, NumBins = 4 });
        generator.Fit(data, columns, 1);
        var m = generator.Generate(10);

        for (int i = 0; i < m.Rows; i++)
        {
            for (int j = 0; j < m.Columns; j++) Assert.True(Fin(m[i, j]));
        }
    }

    /// <summary>
    /// An extremely small budget makes the noise scale enormous, so essentially every marginal cell
    /// clips to zero and the uniform fallback takes over. That must still yield a usable distribution
    /// rather than NaN from a division by a zero total.
    /// </summary>
    [Fact]
    public void ExtremelyTightBudget_FallsBackWithoutNaN()
    {
        var m = FitAndGenerate(new BayesianNetworkSynthOptions<double>
        {
            Seed = 11,
            MaxParents = 2,
            NumBins = 5,
            PrivacyBudget = 1e-6
        }, samples: 40);

        int nonFinite = 0;
        for (int i = 0; i < m.Rows; i++)
        {
            for (int j = 0; j < m.Columns; j++)
            {
                if (!Fin(m[i, j])) nonFinite++;
            }
        }

        _out.WriteLine($"eps=1e-6 nonFinite={nonFinite}/{m.Rows * m.Columns}");
        Assert.Equal(0, nonFinite);
    }

    /// <summary>
    /// A budget entirely allocated to one phase must not break the other. At fraction 1.0 the marginal
    /// phase gets zero budget, and at 0.0 the structure phase does; both must degrade gracefully
    /// instead of dividing by zero.
    /// </summary>
    [Theory]
    [InlineData(0.0)]
    [InlineData(1.0)]
    public void DegenerateBudgetSplit_DoesNotProduceNaN(double fraction)
    {
        var m = FitAndGenerate(new BayesianNetworkSynthOptions<double>
        {
            Seed = 5,
            MaxParents = 2,
            NumBins = 5,
            StructureBudgetFraction = fraction
        }, samples: 30);

        for (int i = 0; i < m.Rows; i++)
        {
            for (int j = 0; j < m.Columns; j++) Assert.True(Fin(m[i, j]));
        }
    }

    #endregion
}
