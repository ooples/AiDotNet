using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Optimizers;

/// <summary>
/// Pins the incremental (ask/tell) CMA-ES API of <see cref="CMAESOptimizer{T, TInput, TOutput}"/>: Minimize is now
/// built on it and must follow exactly the trajectory the previous monolithic implementation did, and a search
/// spread over several calls (or copied between them) must be the same search.
/// </summary>
public class CMAESAskTellTests
{
    private const int Seed = 1234;

    private static CMAESOptimizer<double, Vector<double>, Vector<double>> Optimizer(int population = 8, double step = 0.4)
        => CMAESOptimizer<double, Vector<double>, Vector<double>>.CreateForFunction(
            new CMAESOptimizerOptions<double, Vector<double>, Vector<double>>
            {
                PopulationSize = population,
                InitialStepSize = step,
                Seed = Seed,
            });

    // A rotated, badly conditioned ellipsoid with its minimum at (1, -2, 0.5, 3, -1).
    private static readonly double[] Optimum = { 1.0, -2.0, 0.5, 3.0, -1.0 };

    private static double Ellipsoid(double[] x)
    {
        double total = 0.0;
        for (int i = 0; i < x.Length; i++)
        {
            double rotated = 0.0;
            for (int j = 0; j < x.Length; j++) rotated += Math.Cos(0.3 * (i + 1) * (j + 2)) * (x[j] - Optimum[j]);
            total += Math.Pow(10.0, 2.0 * i / (x.Length - 1)) * rotated * rotated;
        }
        return total;
    }

    private static double[] ToArray(Vector<double> v) => Enumerable.Range(0, v.Length).Select(i => v[i]).ToArray();

    /// <summary>
    /// The CMA-ES Minimize as it stood before the ask/tell refactor, transcribed line for line onto plain doubles
    /// (for T = double every NumOps conversion is the identity), with the same seeded generator and Box-Muller draw.
    /// It records every point it evaluates.
    /// </summary>
    private static double[] ReferenceMinimize(
        double[] initial, Func<double[], double> objective, int maxIterations, double stop,
        int populationSize, double initialStepSize, List<double[]> evaluated)
    {
        double[] bestPoint = (double[])initial.Clone();
        double bestValue = 0.0;
        int evaluations = 0;
        double Evaluate(double[] point)
        {
            evaluated.Add((double[])point.Clone());
            evaluations++;
            double value = objective(point);
            if (double.IsNaN(value)) return value;
            if (evaluations == 1 || value < bestValue)
            {
                bestValue = value;
                bestPoint = (double[])point.Clone();
            }
            return value;
        }

        Evaluate(initial);
        var random = RandomHelper.CreateSeededRandom(Seed);
        double NextGaussian()
        {
            double first = random.NextDouble();
            double second = random.NextDouble();
            return Math.Sqrt(-2.0 * Math.Log(first + 1e-12)) * Math.Cos(2.0 * Math.PI * second);
        }

        int n = initial.Length;
        int lambda = populationSize > 0 ? Math.Max(4, populationSize) : 4 + (int)(3.0 * Math.Log(n));
        int mu = Math.Max(1, lambda / 2);

        var weights = new double[mu];
        double weightSum = 0.0;
        for (int i = 0; i < mu; i++)
        {
            weights[i] = Math.Log(mu + 0.5) - Math.Log(i + 1.0);
            weightSum += weights[i];
        }

        double squaredSum = 0.0;
        for (int i = 0; i < mu; i++)
        {
            weights[i] /= weightSum;
            squaredSum += weights[i] * weights[i];
        }

        double muEffective = 1.0 / squaredSum;
        double cSigma = (muEffective + 2.0) / (n + muEffective + 5.0);
        double dSigma = 1.0 + 2.0 * Math.Max(0.0, Math.Sqrt((muEffective - 1.0) / (n + 1.0)) - 1.0) + cSigma;
        double cc = (4.0 + muEffective / n) / (n + 4.0 + 2.0 * muEffective / n);
        double c1 = 2.0 / ((n + 1.3) * (n + 1.3) + muEffective);
        double cmu = Math.Min(1.0 - c1,
            2.0 * (muEffective - 2.0 + 1.0 / muEffective) / ((n + 2.0) * (n + 2.0) + muEffective));
        double expectedNorm = Math.Sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n));

        var mean = (double[])initial.Clone();
        double sigma = initialStepSize > 0 ? initialStepSize : 0.3;
        var pathSigma = new double[n];
        var pathC = new double[n];
        var covariance = new double[n, n];
        for (int i = 0; i < n; i++) covariance[i, i] = 1.0;

        for (int generation = 0; generation < maxIterations && sigma > stop; generation++)
        {
            var factor = Cholesky(covariance, n);
            var draws = new double[lambda][];
            var values = new double[lambda];

            for (int k = 0; k < lambda; k++)
            {
                var z = new double[n];
                for (int i = 0; i < n; i++) z[i] = NextGaussian();
                var y = new double[n];
                for (int i = 0; i < n; i++)
                {
                    double total = 0.0;
                    for (int j = 0; j <= i; j++) total += factor[i, j] * z[j];
                    y[i] = total;
                }
                var point = new double[n];
                for (int i = 0; i < n; i++) point[i] = mean[i] + sigma * y[i];
                draws[k] = y;
                values[k] = Evaluate(point);
            }

            var order = Enumerable.Range(0, lambda).OrderBy(k => values[k]).ToArray();
            var weightedDraw = new double[n];
            for (int i = 0; i < n; i++)
            {
                double drawTotal = 0.0;
                for (int r = 0; r < mu; r++) drawTotal += weights[r] * draws[order[r]][i];
                weightedDraw[i] = drawTotal;
                mean[i] += sigma * drawTotal;
            }

            var conjugate = new double[n];
            for (int i = 0; i < n; i++)
            {
                double total = weightedDraw[i];
                for (int j = 0; j < i; j++) total -= factor[i, j] * conjugate[j];
                conjugate[i] = factor[i, i] == 0.0 ? 0.0 : total / factor[i, i];
            }

            double pathSigmaNorm = 0.0;
            for (int i = 0; i < n; i++)
            {
                pathSigma[i] = (1.0 - cSigma) * pathSigma[i] + Math.Sqrt(cSigma * (2.0 - cSigma) * muEffective) * conjugate[i];
                pathSigmaNorm += pathSigma[i] * pathSigma[i];
            }
            pathSigmaNorm = Math.Sqrt(pathSigmaNorm);

            double heaviside = pathSigmaNorm / Math.Sqrt(1.0 - Math.Pow(1.0 - cSigma, 2.0 * (generation + 1)))
                < (1.4 + 2.0 / (n + 1.0)) * expectedNorm ? 1.0 : 0.0;

            for (int i = 0; i < n; i++)
                pathC[i] = (1.0 - cc) * pathC[i] + heaviside * Math.Sqrt(cc * (2.0 - cc) * muEffective) * weightedDraw[i];

            double decay = 1.0 - c1 - cmu + (1.0 - heaviside) * c1 * cc * (2.0 - cc);
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    double rankMu = 0.0;
                    for (int r = 0; r < mu; r++) rankMu += weights[r] * draws[order[r]][i] * draws[order[r]][j];
                    covariance[i, j] = decay * covariance[i, j] + c1 * pathC[i] * pathC[j] + cmu * rankMu;
                }
            }

            sigma *= Math.Exp((cSigma / dSigma) * (pathSigmaNorm / expectedNorm - 1.0));
            if (double.IsNaN(sigma) || double.IsInfinity(sigma) || sigma <= 0.0) break;
        }

        return bestPoint;
    }

    private static double[,] Cholesky(double[,] matrix, int n)
    {
        for (double shift = 0.0; ; shift = shift == 0.0 ? 1e-12 : shift * 100.0)
        {
            var factor = new double[n, n];
            bool ok = true;
            for (int i = 0; i < n && ok; i++)
            {
                for (int j = 0; j <= i; j++)
                {
                    double total = matrix[i, j] + (i == j ? shift : 0.0);
                    for (int k = 0; k < j; k++) total -= factor[i, k] * factor[j, k];
                    if (i == j)
                    {
                        if (total <= 0.0 || double.IsNaN(total)) { ok = false; break; }
                        factor[i, j] = Math.Sqrt(total);
                    }
                    else
                    {
                        factor[i, j] = total / factor[j, j];
                    }
                }
            }
            if (ok) return factor;
            if (shift > 1.0)
            {
                var identity = new double[n, n];
                for (int i = 0; i < n; i++) identity[i, i] = 1.0;
                return identity;
            }
        }
    }

    [Fact]
    public void Minimize_OnAskTell_FollowsThePreRefactorTrajectoryBitForBit()
    {
        var start = new[] { 0.0, 0.0, 0.0, 0.0, 0.0 };
        var expectedPoints = new List<double[]>();
        var expectedBest = ReferenceMinimize(start, Ellipsoid, 60, 1e-12, 8, 0.4, expectedPoints);

        var actualPoints = new List<double[]>();
        var best = Optimizer().Minimize(new Vector<double>(start), v =>
        {
            var x = ToArray(v);
            actualPoints.Add(x);
            return Ellipsoid(x);
        }, 60, 1e-12);

        Assert.Equal(1 + 60 * 8, expectedPoints.Count);
        Assert.Equal(expectedPoints.Count, actualPoints.Count);
        for (int p = 0; p < expectedPoints.Count; p++)
            for (int i = 0; i < start.Length; i++)
                Assert.Equal(BitConverter.DoubleToInt64Bits(expectedPoints[p][i]), BitConverter.DoubleToInt64Bits(actualPoints[p][i]));

        var bestArray = ToArray(best);
        for (int i = 0; i < start.Length; i++)
            Assert.Equal(BitConverter.DoubleToInt64Bits(expectedBest[i]), BitConverter.DoubleToInt64Bits(bestArray[i]));
    }

    [Fact]
    public void AskTell_AdvancesOneGeneration_AndRefusesAStaleOrRepeatedTell()
    {
        var cma = Optimizer();
        var state = cma.CreateSearchState(new Vector<double>(new double[5]));
        var random = RandomHelper.CreateSeededRandom(Seed);

        var before = state.Clone();
        var generation = cma.Ask(state, random);
        Assert.Equal(8, generation.Points.Count);
        Assert.Equal(before.Mean, state.Mean);
        Assert.Equal(0, state.Generation);

        cma.Tell(state, generation, generation.Points.Select(p => Ellipsoid(ToArray(p))).ToArray());
        Assert.Equal(1, state.Generation);
        Assert.NotEqual(before.Mean, state.Mean);
        Assert.NotEqual(before.Sigma, state.Sigma);

        Assert.Throws<InvalidOperationException>(() =>
            cma.Tell(state, generation, generation.Points.Select(p => Ellipsoid(ToArray(p))).ToArray()));
        Assert.Throws<ArgumentException>(() => cma.Tell(state, cma.Ask(state, random), new double[3]));
    }

    [Fact]
    public void ASearchSplitAcrossCallsAndCopiedState_IsTheSameSearch()
    {
        var cma = Optimizer();
        double[] Run(bool copyHalfway)
        {
            var state = cma.CreateSearchState(new Vector<double>(new double[5]));
            var random = RandomHelper.CreateSeededRandom(Seed);
            for (int g = 0; g < 20; g++)
            {
                if (copyHalfway && g == 10) state = state.Clone();
                var generation = cma.Ask(state, random);
                cma.Tell(state, generation, generation.Points.Select(p => Ellipsoid(ToArray(p))).ToArray());
            }
            return state.Mean;
        }

        var straight = Run(false);
        var copied = Run(true);
        for (int i = 0; i < straight.Length; i++)
            Assert.Equal(BitConverter.DoubleToInt64Bits(straight[i]), BitConverter.DoubleToInt64Bits(copied[i]));
    }

    [Fact]
    public void Tell_MovesTheMeanTowardsLowerFitness_AndAwayWhenTheRankingIsInverted()
    {
        double DistanceAfter(Func<double, double> rank)
        {
            var cma = Optimizer();
            var state = cma.CreateSearchState(new Vector<double>(new double[5]));
            var random = RandomHelper.CreateSeededRandom(Seed);
            for (int g = 0; g < 150; g++)
            {
                var generation = cma.Ask(state, random);
                cma.Tell(state, generation, generation.Points.Select(p => rank(Ellipsoid(ToArray(p)))).ToArray());
            }
            return Math.Sqrt(state.Mean.Select((m, i) => (m - Optimum[i]) * (m - Optimum[i])).Sum());
        }

        double startDistance = Math.Sqrt(Optimum.Sum(o => o * o));
        double minimised = DistanceAfter(f => f);
        double maximised = DistanceAfter(f => -f);
        Assert.True(minimised < 0.1 * startDistance, $"CMA-ES did not approach the optimum: {minimised} from {startDistance}.");
        Assert.True(maximised > startDistance, $"Inverting the fitness should drive the mean away: {maximised}.");
    }
}
