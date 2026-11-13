using System;
using System.Threading.Tasks;
using AiDotNet.Preprocessing;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;
using AiDotNet.Preprocessing.FeatureSelection.Metaheuristic;
using AiDotNet.Preprocessing.FeatureSelection.ModelAgnostic;
using AiDotNet.Preprocessing.FeatureSelection.MultiObjective;
using AiDotNet.Preprocessing.FeatureSelection.Neural;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Preprocessing;

/// <summary>
/// Reference tests for the third review round of the 08d sparse autoencoder: the KL sparsity derivative at
/// saturated units in Half (the pairwise squared distance test moved to FeatureSelectionTensorOpsTests), and the
/// fourth round's Half guards, overflow and position fixes in the population and neural helpers.
/// </summary>
public class FeatureSelectionHelperRoundThreeTests
{
    // d/d rho^ of KL(rho || rho^) = -rho / rho^ + (1 - rho) / (1 - rho^) at rho^ clamped to [eps, 1 - eps] (Ng's
    // sparse autoencoder, with the clamp of common implementations). Units with mean activation 0 and 1 gave
    // -0.05 / 0 = -infinity and 0.95 / 0 = +infinity in Half, where the former 1e-10 guard is 0.
    [Fact(Timeout = 60000)]
    public async Task SparsityDelta_Half_StaysFiniteForSaturatedUnits()
    {
        await Task.CompletedTask;
        const double rho = 0.05;
        var mean = new Tensor<Half>(new[] { (Half)0, (Half)1, (Half)0.5 }, new[] { 3 });

        var delta = SparseAutoencoderSelector<Half>.SparsityDelta(mean, rho, 1.0);

        double eps = (double)SparseAutoencoderSelector<Half>.ActivationEpsilon();
        Assert.InRange(eps, 1e-6, 1e-3);
        // the upper clamp is the largest Half below 1, 1 - 2^-11
        double upper = 1 - Math.Pow(2, -11);
        double[] expected =
        {
            -rho / eps + (1 - rho) / (1 - eps),
            -rho / upper + (1 - rho) / (1 - upper),
            -rho / 0.5 + (1 - rho) / 0.5,
        };
        for (int i = 0; i < 3; i++)
        {
            double actual = (double)delta[i];
            Assert.False(double.IsInfinity(actual) || double.IsNaN(actual));
            Assert.InRange(actual, expected[i] - Math.Abs(expected[i]) * 0.01, expected[i] + Math.Abs(expected[i]) * 0.01);
        }
    }

    // In double the clamp margin is master's 1e-10, so the derivative is unchanged away from saturation.
    [Fact(Timeout = 60000)]
    public async Task ActivationEpsilon_Double_IsMastersGuard()
    {
        await Task.CompletedTask;
        Assert.Equal(1e-10, SparseAutoencoderSelector<double>.ActivationEpsilon());
    }

    // A constant column has population std 0. The guard std < 1e-10 is std < 0 in Half (1e-10 rounds to 0), so the
    // column kept std 0 and standardized to 0 / 0 = NaN; it must use std 1 and stay 0, as master does in double.
    [Fact(Timeout = 60000)]
    public async Task StandardizePopulation_Half_ConstantColumnStaysZero()
    {
        await Task.CompletedTask;
        var x = new Tensor<Half>(
            new[] { (Half)1, (Half)3, (Half)2, (Half)3, (Half)3, (Half)3, (Half)4, (Half)3 }, new[] { 4, 2 });

        var z = NeuralSelectorOps.StandardizePopulation(x);

        double std = Math.Sqrt(1.25);
        double[] first = { -1.5 / std, -0.5 / std, 0.5 / std, 1.5 / std };
        for (int i = 0; i < 4; i++)
        {
            Assert.Equal(first[i], (double)z[i, 0], 2);
            Assert.Equal(0.0, (double)z[i, 1]);
        }
    }

    // LIME perturbs every feature by its population std, with std 1 for a constant feature (master). In Half the
    // 1e-10 guard was 0, so a constant feature kept std 0 and was never perturbed.
    [Fact(Timeout = 60000)]
    public async Task PopulationStdOrOne_Half_GivesConstantColumnsStdOne()
    {
        await Task.CompletedTask;
        var centered = new Tensor<Half>(
            new[] { (Half)(-1), (Half)0, (Half)1, (Half)0 }, new[] { 2, 2 });

        var std = NeuralSelectorOps.PopulationStdOrOne(centered);

        Assert.Equal(1.0, (double)std[0, 0]);
        Assert.Equal(1.0, (double)std[0, 1]);
        var spread = NeuralSelectorOps.PopulationStdOrOne(new Tensor<Half>(
            new[] { (Half)(-2), (Half)0, (Half)2, (Half)0 }, new[] { 2, 2 }));
        Assert.Equal(2.0, (double)spread[0, 0]);
        Assert.Equal(1.0, (double)spread[0, 1]);
    }

    // x = 20 (-1)^i and y = (-1)^i over 300 rows have r = 1. The raw sum of squared deviations of x is
    // 300 * 400 = 120000, past Half's 65504, so sqrt(sxx) sqrt(syy) was infinite and |r| came out 0. A constant
    // column still gets the undefined value.
    [Fact(Timeout = 60000)]
    public async Task AbsTargetCorrelations_Half_StaysExactPastHalfRange()
    {
        await Task.CompletedTask;
        const int n = 300;
        var data = new Matrix<Half>(n, 2);
        var target = new Vector<Half>(n);
        for (int i = 0; i < n; i++)
        {
            double sign = i % 2 == 0 ? 1 : -1;
            data[i, 0] = (Half)(20 * sign);
            data[i, 1] = (Half)5;
            target[i] = (Half)sign;
        }

        var r = PopulationSearchOps.AbsTargetCorrelations(data, target, (Half)0.25);

        Assert.Equal(1.0, (double)r[0], 3);
        Assert.Equal(0.25, (double)r[1]);
    }

    // Kernel SHAP of f(x) = x1 + x2 + x3 at x = (1, 1, 1) against the zero baseline, from the single coalition
    // {1}: the eliminated system [[w, 0], [0, 0]] is singular and only the 1e-10 ridge makes it solvable, giving
    // phi = (1, 0, 2). In Half the ridge was 0, so the solve broke down instead of returning finite values.
    [Fact(Timeout = 60000)]
    public async Task KernelShap_Half_SolvesASingleCoalitionWithTheRidge()
    {
        await Task.CompletedTask;
        var instance = new Tensor<Half>(new[] { (Half)1, (Half)1, (Half)1 }, new[] { 3 });
        var baseline = new Tensor<Half>(new[] { 3 });
        var draws = new Tensor<Half>(new[] { (Half)0.9, (Half)0.1, (Half)0.1 }, new[] { 1, 3 });
        Func<Tensor<Half>, Tensor<Half>> predict = rows =>
        {
            int m = rows.Shape[0];
            var output = new Half[m];
            for (int i = 0; i < m; i++)
                output[i] = (Half)((double)rows[i, 0] + (double)rows[i, 1] + (double)rows[i, 2]);
            return new Tensor<Half>(output, new[] { m });
        };

        var phi = SHAPSelector<Half>.KernelShap(instance, baseline, draws, predict);

        double[] expected = { 1, 0, 2 };
        for (int j = 0; j < 3; j++)
        {
            double value = (double)phi[j];
            Assert.False(double.IsNaN(value) || double.IsInfinity(value));
            Assert.Equal(expected[j], value, 2);
        }
    }

    // MOEA/D weight vectors (i / (N - 1), 1 - i / (N - 1)). With N = 3000 in Half, i = 2999 is not exact (it
    // rounds to 3000), so the last first weight came out above 1 and its complement below 0.
    [Fact(Timeout = 60000)]
    public async Task MoeadWeightVectors_Half_StayInTheUnitIntervalFor3000Subproblems()
    {
        await Task.CompletedTask;
        const int count = 3000;

        var weights = MOEADSelector<Half>.WeightVectors(count);

        Assert.Equal(1.0, (double)weights[count - 1, 0]);
        Assert.Equal(0.0, (double)weights[count - 1, 1]);
        for (int i = 0; i < count; i++)
        {
            Assert.Equal((double)(Half)(i / (count - 1.0)), (double)weights[i, 0]);
            Assert.InRange((double)weights[i, 1], 0.0, 1.0);
        }
    }

    // SS_tot of y = (0, 2^-9) about its mean is 2^-19, below Half's smallest normal value, where Half keeps only a
    // few significant bits. The former 1e-10 guard was 0 in Half and accepted it; R^2 is reported undefined (0).
    [Fact(Timeout = 60000)]
    public async Task RSquared_Half_TreatsASubnormalTotalSumOfSquaresAsUndefined()
    {
        await Task.CompletedTask;
        var y = new Tensor<Half>(new[] { (Half)0, (Half)Math.Pow(2, -9) }, new[] { 2 });
        var prediction = new Tensor<Half>(new[] { 2 });
        var center = new Tensor<Half>(new[] { (Half)Math.Pow(2, -10) }, new[] { 1 });

        var r2 = SubsetRegressionOps.RSquared(y, prediction, center, out bool valid);

        Assert.False(valid);
        Assert.Equal(0.0, (double)r2[0]);
    }

    // One centred column x = 20 (-1)^i over 300 rows has covariance 400, so whitening maps it to +-1. Summed in
    // Half, x^T x = 120000 passed 65504 before the division by n and the whitened data came out 0 or NaN.
    [Fact(Timeout = 60000)]
    public async Task FastIcaWhiten_Half_DividesBeforeTheProduct()
    {
        await Task.CompletedTask;
        const int n = 300;
        var values = new Half[n];
        for (int i = 0; i < n; i++)
            values[i] = (Half)(i % 2 == 0 ? 20 : -20);

        var z = FastIcaOps.Whiten(new Tensor<Half>(values, new[] { n, 1 }), 1, out _);

        for (int i = 0; i < n; i++)
            Assert.Equal(1.0, Math.Abs((double)z[i, 0]), 2);
    }

    // With the default 20 ants and rho = 0.1 the pheromone settles near 200. Over 1000 features the roulette total
    // tau^alpha eta^beta was 200 * 1000 = 200000, infinite in Half, so the cumulative draw never separated the
    // features. Weights proportional to tau^alpha eta^beta are kept, and their total is finite.
    [Fact(Timeout = 60000)]
    public async Task AntColonyTransitionWeights_Half_KeepTheRouletteTotalFinite()
    {
        await Task.CompletedTask;
        const int p = 1000;
        var pheromone = new Half[p];
        var desirability = new Half[p];
        for (int j = 0; j < p; j++)
        {
            pheromone[j] = (Half)(j == 0 ? 100 : 200);
            desirability[j] = (Half)1;
        }

        var weights = AntColonyOptimization<Half>.TransitionWeights(
            new Tensor<Half>(pheromone, new[] { p }), new Tensor<Half>(desirability, new[] { p }), (Half)1);

        double total = 0;
        for (int j = 0; j < p; j++)
            total += (double)weights[j];
        Assert.Equal(p - 0.5, total, 3);
        Assert.Equal(0.5, (double)weights[0]);
        Assert.Equal(1.0, (double)weights[1]);
        var sum = AiDotNet.Tensors.Engines.AiDotNetEngine.Current.TensorSum(weights);
        Assert.False(Half.IsInfinity(sum) || Half.IsNaN(sum));
    }

    // x = y = (-1)^i over 300 rows have r = 1 and Sxx = Syy = 300. The denominator sqrt(Sxx * Syy) formed the
    // product 90000 first, infinite in Half, so the constraint-based selector scored every feature 0.
    // sqrt(Sxx) * sqrt(Syy) stays finite.
    [Fact(Timeout = 60000)]
    public async Task TargetCorrelations_Half_TakeTheRootsBeforeTheProduct()
    {
        await Task.CompletedTask;
        const int n = 300;
        var x = new Half[n];
        var y = new Half[n];
        for (int i = 0; i < n; i++)
        {
            x[i] = (Half)(i % 2 == 0 ? 1 : -1);
            y[i] = x[i];
        }

        var tx = new Tensor<Half>(x, new[] { n, 1 });
        var ty = new Tensor<Half>(y, new[] { n });

        Assert.Equal(1.0, (double)PreprocessingTensorOps.TargetCorrelations(tx, ty)[0], 3);
        Assert.Equal(1.0, (double)PreprocessingTensorOps.CorrelationsWithDenominatorGuard(tx, ty)[0], 3);
    }

    // SPEA2 measures density in the objective space (accuracy, feature count). With counts 1, 300 and 600 the
    // expansion |a|^2 + |b|^2 - 2ab squares 600 to 360000, infinite in Half, so the distances (and with them the
    // density and the archive truncation) were lost. The true distances are about 299, 300 and 599.
    [Fact(Timeout = 60000)]
    public async Task Spea2Distances_Half_StayFiniteForWideSubsets()
    {
        await Task.CompletedTask;
        var accuracy = new Tensor<Half>(new[] { (Half)0.5, (Half)0.5, (Half)0.5 }, new[] { 3 });
        var count = new Tensor<Half>(new[] { (Half)1, (Half)300, (Half)600 }, new[] { 3 });

        SPEAIISelector<Half>.ComputeFitness(accuracy, count, out var distances);

        Assert.InRange((double)distances[0, 1], 297.0, 301.0);
        Assert.InRange((double)distances[1, 2], 298.0, 302.0);
        Assert.InRange((double)distances[0, 2], 597.0, 601.0);
        Assert.Equal(0.0, (double)distances[1, 1]);
    }

    // Minority rows 300, 310 and 1000: the nearest other row of 1000 is 310. The distance expansion squared 1000
    // to 10^6, infinite in Half, so every distance was NaN and the neighbour fell back to the lowest index (300).
    // The neighbour order does not depend on the scale, so the rows are divided by their largest magnitude first.
    [Fact(Timeout = 60000)]
    public async Task SmoteMinorityNeighbors_Half_RankPastTheSquareRange()
    {
        await Task.CompletedTask;
        var minority = new Tensor<Half>(new[] { (Half)300, (Half)310, (Half)1000 }, new[] { 3, 1 });

        var neighbors = AiDotNet.Preprocessing.FeatureSelection.Imbalanced.SMOTEFeatureSelector<Half>.MinorityNeighbors(minority, 1);

        Assert.Equal(1, neighbors[0][0]);
        Assert.Equal(0, neighbors[1][0]);
        Assert.Equal(1, neighbors[2][0]);
    }
}
