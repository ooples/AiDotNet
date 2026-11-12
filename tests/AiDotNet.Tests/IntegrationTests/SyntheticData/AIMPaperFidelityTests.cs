using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.SyntheticData;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.IntegrationTests.SyntheticData;

/// <summary>
/// Reference tests for <see cref="AIMGenerator{T}"/> against AIM (McKenna, Mullins, Sheldon, Miklau 2022)
/// and its reference implementation in the private-pgm (mbi) library.
/// </summary>
public class AIMPaperFidelityTests
{
    /// <summary>
    /// (epsilon, delta) to zCDP rho, as private-pgm's cdp_rho computes it (Canonne, Kamath, Steinke 2020).
    /// Reference values from running mbi's cdp.cdp_rho(1.0, 1e-9) and cdp_rho(5.0, 1e-6).
    /// </summary>
    [Fact]
    public void CdpRho_MatchesPrivatePgm()
    {
        Assert.Equal(0.014973057673588523, AIMGenerator<double>.CdpRho(1.0, 1e-9), 10);
        Assert.Equal(0.4630649901386301, AIMGenerator<double>.CdpRho(5.0, 1e-6), 10);
    }

    /// <summary>Algorithm 1: sigma_0 = sqrt(T / (2 alpha rho)) and epsilon_0 = sqrt(8 (1 - alpha) rho / T).</summary>
    [Fact]
    public void InitialScales_MatchAlgorithm1()
    {
        var (sigma, epsilon) = AIMGenerator<double>.InitialScales(0.5, 80, 0.9);
        Assert.Equal(9.428090415820634, sigma, 12);
        Assert.Equal(0.07071067811865475, epsilon, 12);
    }

    /// <summary>
    /// The candidate set is the downward closure of the workload with w_r = sum_s |r ∩ s|; for all pairs
    /// of three attributes every single has weight 2 and every pair weight 4. Marginals over MaxCells are
    /// dropped from the workload.
    /// </summary>
    [Fact]
    public void Workload_DownwardClosureAndWeights()
    {
        var all = AIMGenerator<double>.CompileWorkload(new[] { 3, 3, 3 }, 2, 10000)
            .ToDictionary(c => string.Join(",", c.Clique), c => c.Weight);
        Assert.Equal(6, all.Count);
        Assert.Equal(2.0, all["0"]);
        Assert.Equal(2.0, all["2"]);
        Assert.Equal(4.0, all["0,1"]);
        Assert.Equal(4.0, all["1,2"]);

        var limited = AIMGenerator<double>.CompileWorkload(new[] { 200, 200, 2 }, 2, 10000)
            .ToDictionary(c => string.Join(",", c.Clique), c => c.Weight);
        Assert.False(limited.ContainsKey("0,1"));
        Assert.Equal(1.0, limited["0"]);
        Assert.Equal(2.0, limited["2"]);
        Assert.Equal(3.0, limited["0,2"]);
    }

    /// <summary>
    /// Junction-tree size in MB (8 bytes per cell): a chain 0-1-2 over 10-value attributes has cliques
    /// {0,1},{1,2} (200 cells); a triangle needs {0,1,2} (1000 cells); three singles 30 cells.
    /// </summary>
    [Fact]
    public void ModelSize_IsJunctionTreeSize()
    {
        var cards = new[] { 10, 10, 10 };
        Assert.Equal(0.00152587890625, AIMGenerator<double>.ModelSizeMb(cards, new[] { new[] { 0, 1 }, new[] { 1, 2 } }), 15);
        Assert.Equal(0.00762939453125, AIMGenerator<double>.ModelSizeMb(cards, new[] { new[] { 0, 1 }, new[] { 1, 2 }, new[] { 0, 2 } }), 15);
        Assert.Equal(0.0002288818359375, AIMGenerator<double>.ModelSizeMb(cards, new[] { new[] { 0 }, new[] { 1 }, new[] { 2 } }), 15);
    }

    /// <summary>
    /// Quality w (||x - x_hat||_1 - sqrt(2/pi) sigma n_r): x = [5,5], x_hat = [3,7], w = 2, sigma = 1
    /// gives 2 (4 - 2 sqrt(2/pi)) = 4.808461756788539.
    /// </summary>
    [Fact]
    public void Quality_SubtractsExpectedNoiseError()
    {
        var truth = new Tensor<double>(new[] { 5.0, 5.0 }, new[] { 2 });
        var model = new Tensor<double>(new[] { 3.0, 7.0 }, new[] { 2 });
        Assert.Equal(4.808461756788539, AIMGenerator<double>.Quality(truth, model, 2.0, 1.0).GetFlat(0), 12);

        Assert.True(AIMGenerator<double>.MovedWithinNoise(truth, model, 3.0));
        Assert.False(AIMGenerator<double>.MovedWithinNoise(truth, model, 1.0));
    }

    private static (Matrix<double> Data, List<ColumnMetadata> Columns) CorrelatedCategorical(int rows)
    {
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(21);
        var data = new Matrix<double>(rows, 2);
        for (int i = 0; i < rows; i++)
        {
            int a = rng.Next(3);
            data[i, 0] = a;
            data[i, 1] = rng.NextDouble() < 0.9 ? a : (a + 1) % 3;
        }

        var columns = new List<ColumnMetadata>
        {
            new ColumnMetadata("a", ColumnDataType.Categorical),
            new ColumnMetadata("b", ColumnDataType.Categorical)
        };
        return (data, columns);
    }

    /// <summary>
    /// The one-way marginals are measured first, and the whole zCDP budget is spent by the end
    /// (the final round takes exactly what remains).
    /// </summary>
    [Fact]
    public void Fit_MeasuresOneWayFirst_AndSpendsExactlyRho()
    {
        var (data, columns) = CorrelatedCategorical(200);
        var generator = new AIMGenerator<double>(new AIMOptions<double> { Seed = 4, Epsilon = 1.0, NumIterations = 5 });
        generator.Fit(data, columns, 1);

        var log = generator.MeasurementLog;
        Assert.Equal(new[] { 0 }, log[0].Clique);
        Assert.Equal(new[] { 1 }, log[1].Clique);
        Assert.Equal(AIMGenerator<double>.CdpRho(1.0, 1e-9), generator.Rho, 15);
        Assert.Equal(generator.Rho, generator.RhoUsed, 12);
    }

    /// <summary>
    /// Private-PGM: with negligible noise the estimated graphical model reproduces the measured pairwise
    /// marginal of strongly dependent columns (L1 error under 5% of n), which an independent model of
    /// the one-way marginals cannot.
    /// </summary>
    [Fact]
    public void PrivatePgm_RecoversMeasuredJoint()
    {
        const int rows = 300;
        var (data, columns) = CorrelatedCategorical(rows);
        var generator = new AIMGenerator<double>(new AIMOptions<double>
        {
            Seed = 8,
            Epsilon = 1e6,
            NumIterations = 100
        });
        generator.Fit(data, columns, 1);

        Assert.Contains(generator.MeasurementLog, m => m.Clique.Length == 2);
        var model = generator.ModelMarginal(new[] { 0, 1 }).ToArray();
        var truth = new double[9];
        for (int i = 0; i < rows; i++) truth[(int)data[i, 0] * 3 + (int)data[i, 1]]++;

        double l1 = truth.Select((t, k) => Math.Abs(t - model[k])).Sum();
        Assert.True(l1 < 0.05 * rows, $"L1 error {l1} between the model's pair marginal and the data.");
    }

    [Fact]
    public void Generate_IsReproducible_AndUsesCategoryValues()
    {
        var (data, columns) = CorrelatedCategorical(150);
        Matrix<double> Run()
        {
            var g = new AIMGenerator<double>(new AIMOptions<double> { Seed = 12, NumIterations = 5 });
            g.Fit(data, columns, 1);
            return g.Generate(40);
        }

        var a = Run();
        var b = Run();
        for (int i = 0; i < a.Rows; i++)
        {
            for (int j = 0; j < a.Columns; j++)
            {
                Assert.Equal(a[i, j], b[i, j]);
                Assert.Contains(a[i, j], new[] { 0.0, 1.0, 2.0 });
            }
        }
    }
}
