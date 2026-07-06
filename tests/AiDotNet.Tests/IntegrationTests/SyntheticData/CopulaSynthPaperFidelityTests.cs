using System;
using System.Collections.Generic;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.SyntheticData;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.SyntheticData;

/// <summary>
/// Paper-fidelity guards for <see cref="CopulaSynthGenerator{T}"/> (Patki et al.
/// 2016, "The Synthetic Data Vault" — Gaussian Copula). These pin the two defects
/// the Synthetic POC review measured on AiDotNet 0.213.3:
///   1. correlation preservation degraded to 0.82–0.93 (planted ~0.95), because the
///      forward empirical CDF and the inverse quantile used inconsistent grids, so
///      the probability-integral transform was not rank-preserving;
///   2. float noise leaked into categorical columns, because the column metadata
///      never reached the inverse transform and categorical values were linearly
///      interpolated.
/// Both are now fixed; these tests fail loudly if either regresses.
/// </summary>
public class CopulaSynthPaperFidelityTests
{
    private const int Seed = 42;

    /// <summary>
    /// A Gaussian copula MUST preserve the linear correlation structure of the
    /// data. We plant a strong correlation (~0.95) between two continuous columns,
    /// fit, generate 4× the rows, and require the synthetic correlation to land
    /// within 0.05 of the real one. The pre-fix code landed 0.06–0.17 LOW here
    /// (0.82–0.93 vs ~0.99); 0.05 cleanly separates "preserved" from "degraded".
    /// </summary>
    [Fact]
    public void Copula_PreservesPlantedCorrelation_WithinTolerance()
    {
        const int rows = 400;
        var rng = new Random(Seed);

        // Two continuous columns with a planted linear relationship:
        //   x ~ N(0,1);  y = rho*x + sqrt(1-rho^2)*noise  → Corr(x,y) ≈ rho.
        const double rho = 0.95;
        var data = new Matrix<double>(rows, 2);
        for (int i = 0; i < rows; i++)
        {
            double x = SampleNormal(rng);
            double e = SampleNormal(rng);
            data[i, 0] = 10.0 + 3.0 * x;                       // shift/scale: copula must be scale-invariant
            data[i, 1] = 100.0 + 5.0 * (rho * x + Math.Sqrt(1 - rho * rho) * e);
        }

        var columns = new List<ColumnMetadata>
        {
            new("X", ColumnDataType.Continuous, columnIndex: 0),
            new("Y", ColumnDataType.Continuous, columnIndex: 1),
        };

        double realCorr = Pearson(data, 0, 1, rows);

        var gen = new CopulaSynthGenerator<double>(new CopulaSynthOptions<double> { Seed = Seed });
        gen.Fit(data, columns, epochs: 1);
        var synth = gen.Generate(rows * 4);

        double synthCorr = Pearson(synth, 0, 1, synth.Rows);

        Assert.True(Math.Abs(realCorr - synthCorr) < 0.05,
            $"Copula correlation not preserved: real={realCorr:F4}, synthetic={synthCorr:F4}, " +
            $"|gap|={Math.Abs(realCorr - synthCorr):F4} exceeds 0.05 (Gaussian-copula fidelity regressed).");
    }

    /// <summary>
    /// Categorical columns must come back as values that actually occurred — never
    /// an interpolated float. We declare a 4-level categorical column encoded as
    /// {0,1,2,3} and require every generated value to be exactly one of those.
    /// </summary>
    [Fact]
    public void Copula_CategoricalColumn_NeverLeaksFloatNoise()
    {
        const int rows = 300;
        var rng = new Random(Seed);
        var data = new Matrix<double>(rows, 2);
        var valid = new HashSet<double> { 0, 1, 2, 3 };
        for (int i = 0; i < rows; i++)
        {
            data[i, 0] = 50.0 + 10.0 * SampleNormal(rng); // continuous
            data[i, 1] = rng.Next(4);                      // categorical {0,1,2,3}
        }

        var columns = new List<ColumnMetadata>
        {
            new("Income", ColumnDataType.Continuous, columnIndex: 0),
            new("Education", ColumnDataType.Categorical, new[] { "HS", "BA", "MA", "PhD" }, columnIndex: 1),
        };

        var gen = new CopulaSynthGenerator<double>(new CopulaSynthOptions<double> { Seed = Seed });
        gen.Fit(data, columns, epochs: 1);
        var synth = gen.Generate(500);

        for (int i = 0; i < synth.Rows; i++)
        {
            double v = synth[i, 1];
            Assert.True(valid.Contains(v),
                $"Categorical column leaked a non-category value at row {i}: {v} (expected one of 0,1,2,3).");
        }
    }

    /// <summary>
    /// Discrete (integer) columns must also stay integral — the same snap rule
    /// applies to <see cref="ColumnDataType.Discrete"/>, not just Categorical.
    /// </summary>
    [Fact]
    public void Copula_DiscreteColumn_StaysIntegral()
    {
        const int rows = 200;
        var rng = new Random(Seed);
        var data = new Matrix<double>(rows, 1);
        for (int i = 0; i < rows; i++) data[i, 0] = rng.Next(0, 11); // counts 0..10

        var columns = new List<ColumnMetadata>
        {
            new("Count", ColumnDataType.Discrete, columnIndex: 0),
        };

        var gen = new CopulaSynthGenerator<double>(new CopulaSynthOptions<double> { Seed = Seed });
        gen.Fit(data, columns, epochs: 1);
        var synth = gen.Generate(300);

        for (int i = 0; i < synth.Rows; i++)
        {
            double v = synth[i, 0];
            Assert.True(Math.Abs(v - Math.Round(v)) < 1e-9,
                $"Discrete column produced a non-integer value at row {i}: {v}.");
        }
    }

    private static double Pearson(Matrix<double> m, int a, int b, int n)
    {
        double ma = 0, mb = 0;
        for (int i = 0; i < n; i++) { ma += m[i, a]; mb += m[i, b]; }
        ma /= n; mb /= n;
        double cov = 0, va = 0, vb = 0;
        for (int i = 0; i < n; i++)
        {
            double da = m[i, a] - ma, db = m[i, b] - mb;
            cov += da * db; va += da * da; vb += db * db;
        }
        return cov / (Math.Sqrt(va) * Math.Sqrt(vb));
    }

    private static double SampleNormal(Random rng)
    {
        double u1 = 1.0 - rng.NextDouble();
        double u2 = 1.0 - rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }
}
