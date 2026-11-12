using AiDotNet.Preprocessing.FeatureSelection.MultiObjective;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Preprocessing.FeatureSelection.Metaheuristic;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Preprocessing;

/// <summary>
/// Reference tests for the population-based feature selectors (swarm and evolutionary metaheuristics). Each test
/// checks one update rule against the equation in the selector's paper, computed independently here in double.
/// </summary>
public class FeatureSelectionPopulationReferenceTests
{
    private const double Tolerance = 1e-12;

    private static Tensor<double> Tensor2D(double[,] values)
    {
        int rows = values.GetLength(0);
        int cols = values.GetLength(1);
        var flat = new double[rows * cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                flat[i * cols + j] = values[i, j];
        return new Tensor<double>(flat, new[] { rows, cols });
    }

    private static Tensor<double> Tensor1D(params double[] values) =>
        new Tensor<double>((double[])values.Clone(), new[] { values.Length });

    private static double[] ToArray(Tensor<double> tensor) => tensor.ToVector().ToArray();

    private static (Matrix<double> X, Vector<double> Y) Problem(int seed, int n = 60, int p = 8)
    {
        var rng = new System.Random(seed);
        var x = new Matrix<double>(n, p);
        var y = new Vector<double>(n);
        for (int i = 0; i < n; i++)
        {
            double signal = 0;
            for (int j = 0; j < p; j++)
            {
                double v = rng.NextDouble() * 2 - 1;
                x[i, j] = v;
                if (j < 3)
                    signal += v;
            }

            y[i] = signal + 0.1 * (rng.NextDouble() - 0.5);
        }

        return (x, y);
    }

    // ---------------------------------------------------------------- Artificial Bee Colony (Karaboga 2005)

    [Fact]
    public void ArtificialBeeColony_NeighbourCandidate_ChangesOnlyOneDimension()
    {
        var foods = Tensor2D(new[,] { { 0.2, 0.6, 0.9, 0.4 }, { 0.5, 0.5, 0.5, 0.5 }, { 0.8, 0.1, 0.3, 0.7 } });
        var candidate = ToArray(ArtificialBeeColony<double>.NeighbourCandidate(foods, 0, 2, 1, Tensor1D(0.5)));

        // v_ij = x_ij + phi (x_ij - x_kj) on dimension j = 1 only
        double expected = 0.6 + 0.5 * (0.6 - 0.1);
        Assert.Equal(0.2, candidate[0], Tolerance);
        Assert.Equal(expected, candidate[1], Tolerance);
        Assert.Equal(0.9, candidate[2], Tolerance);
        Assert.Equal(0.4, candidate[3], Tolerance);

        // The moved dimension is clamped to the [0, 1] bounds
        var clamped = ToArray(ArtificialBeeColony<double>.NeighbourCandidate(foods, 0, 2, 1, Tensor1D(1.0)));
        Assert.Equal(1.0, clamped[1], Tolerance);
    }

    [Fact]
    public void ArtificialBeeColony_SelectionProbabilities_UseKarabogaFitness()
    {
        // The selector maximizes F, so the minimized cost is f = -F: fit = 1/(1+f) for f >= 0, 1+|f| for f < 0
        var probabilities = ToArray(ArtificialBeeColony<double>.SelectionProbabilities(Tensor1D(0.5, -0.25, 0.0)));
        double[] fit = { 1 + 0.5, 1 / (1 + 0.25), 1 / (1 + 0.0) };
        double total = fit[0] + fit[1] + fit[2];
        for (int i = 0; i < 3; i++)
            Assert.Equal(fit[i] / total, probabilities[i], Tolerance);
    }

    [Fact]
    public void ArtificialBeeColony_RouletteIndex_PicksFirstCumulativeAtOrAboveDraw()
    {
        var cumulative = Tensor1D(0.2, 0.5, 1.0);
        Assert.Equal(0, ArtificialBeeColony<double>.RouletteIndex(cumulative, Tensor1D(0.0)));
        Assert.Equal(0, ArtificialBeeColony<double>.RouletteIndex(cumulative, Tensor1D(0.2)));
        Assert.Equal(1, ArtificialBeeColony<double>.RouletteIndex(cumulative, Tensor1D(0.3)));
        Assert.Equal(2, ArtificialBeeColony<double>.RouletteIndex(cumulative, Tensor1D(0.99)));
    }

    [Fact]
    public void ArtificialBeeColony_RejectsColoniesWithoutTwoFoodSources()
    {
        // Two or three bees give one food source, which has no partner for the neighbourhood search
        Assert.Throws<System.ArgumentException>(() => new ArtificialBeeColony<double>(colonySize: 3));

        var (x, y) = Problem(1);
        var selector = new ArtificialBeeColony<double>(nFeaturesToSelect: 3, colonySize: 4, maxIterations: 5, randomState: 3);
        selector.Fit(x, y);
        Assert.Equal(3, selector.SelectedIndices?.Length);
    }

    // ---------------------------------------------------------------- Bat Algorithm (Yang 2010)

    private static BatAlgorithm<double>.BatSwarm TwoBats() =>
        new BatAlgorithm<double>.BatSwarm(
            Tensor2D(new[,] { { 0.2, 0.4 }, { 0.6, 0.8 } }),
            Tensor1D(1.0, 1.0),
            Tensor1D(0.5, 0.5),
            rows => AiDotNet.Tensors.Engines.AiDotNetEngine.Current.TensorNegate(
                AiDotNet.Tensors.Engines.AiDotNetEngine.Current.ReduceSum(rows, new[] { 1 }, false)));

    [Fact]
    public void BatAlgorithm_PulseRateFactors_AreNeverZero()
    {
        // r_i^t = r_i^0 (1 - exp(-gamma t)) with t = 1, 2, ...: the first accepted move must not zero the pulse rate
        var factors = ToArray(BatAlgorithm<double>.PulseRateFactors(3, 0.9));
        for (int t = 1; t <= 3; t++)
            Assert.Equal(1 - System.Math.Exp(-0.9 * t), factors[t - 1], Tolerance);
    }

    [Fact]
    public void BatAlgorithm_AcceptedMove_UpdatesLoudnessAndPulseRate()
    {
        var swarm = TwoBats();

        // f = fmin + (fmax - fmin) * 0 = -1, v = 0 + (x_1 - x*) f = -(x_1 - x*), so x_1 + v = x* = (0.2, 0.4)
        swarm.Fly(1, Tensor1D(0.0), null, Tensor1D(0.1), -1.0, 0.0, Tensor1D(0.6));

        var positions = ToArray(swarm.Positions);
        Assert.Equal(0.2, positions[2], Tolerance);
        Assert.Equal(0.4, positions[3], Tolerance);
        Assert.Equal(BatAlgorithm<double>.Alpha * 1.0, ToArray(swarm.Loudness)[1], Tolerance);
        Assert.Equal(0.5 * 0.6, ToArray(swarm.PulseRate)[1], Tolerance);
        Assert.Equal(-0.6, ToArray(swarm.Fitness)[1], Tolerance);
    }

    [Fact]
    public void BatAlgorithm_RejectedMove_LeavesBatUnchanged()
    {
        var swarm = TwoBats();

        // The candidate is better, but the acceptance draw 1.5 is not below the loudness 1.0
        swarm.Fly(1, Tensor1D(0.0), null, Tensor1D(1.5), -1.0, 0.0, Tensor1D(0.6));

        var positions = ToArray(swarm.Positions);
        Assert.Equal(0.6, positions[2], Tolerance);
        Assert.Equal(0.8, positions[3], Tolerance);
        Assert.Equal(1.0, ToArray(swarm.Loudness)[1], Tolerance);
        Assert.Equal(0.5, ToArray(swarm.PulseRate)[1], Tolerance);
        Assert.Equal(-1.4, ToArray(swarm.Fitness)[1], Tolerance);

        // The velocity is still updated: v_1 = (x_1 - x*) f = (-0.4, -0.4)
        var velocities = ToArray(swarm.Velocities);
        Assert.Equal(-0.4, velocities[2], Tolerance);
        Assert.Equal(-0.4, velocities[3], Tolerance);
    }

    [Fact]
    public void BatAlgorithm_LocalWalk_IsAroundBestScaledByMeanLoudness()
    {
        var swarm = TwoBats();

        // x_new = x* + epsilon * mean(A) with epsilon = (0.5, -0.25) and mean(A) = 1, clamped to [0, 1]
        swarm.Fly(1, Tensor1D(0.0), Tensor1D(0.5, -0.25), Tensor1D(0.1), -1.0, 0.0, Tensor1D(0.6));

        var positions = ToArray(swarm.Positions);
        Assert.Equal(0.7, positions[2], Tolerance);
        Assert.Equal(0.15, positions[3], Tolerance);
    }

    // ---------------------------------------------------------------- Grey Wolf Optimizer (Mirjalili 2014)

    [Fact]
    public void GreyWolfOptimizer_Hunt_MatchesEncirclingEquations()
    {
        var positions = Tensor2D(new[,] { { 0.5 } });
        var leaders = Tensor2D(new[,] { { 0.8 }, { 0.6 }, { 0.4 } });
        double[] r = { 0.25, 0.5, 0.75, 0.1, 0.5, 0.9 };
        var draws = new Tensor<double>((double[])r.Clone(), new[] { 1, 1, 6 });
        double a = 1.5;

        double expected = 0;
        double[] leader = { 0.8, 0.6, 0.4 };
        for (int l = 0; l < 3; l++)
        {
            double coefficientA = 2 * a * r[2 * l] - a;
            double coefficientC = 2 * r[2 * l + 1];
            double distance = System.Math.Abs(coefficientC * leader[l] - 0.5);
            expected += leader[l] - coefficientA * distance;
        }

        expected = System.Math.Max(0, System.Math.Min(1, expected / 3));
        Assert.Equal(expected, ToArray(GreyWolfOptimizer<double>.Hunt(positions, leaders, draws, a))[0], Tolerance);
    }

    [Fact]
    public void GreyWolfOptimizer_Leaders_AreTheBestFoundSoFar()
    {
        var leaders = Tensor2D(new[,] { { 0.9 }, { 0.8 }, { 0.7 } });
        var population = Tensor2D(new[,] { { 0.85 }, { 0.1 } });

        // A wolf that moved to a worse place does not replace a leader; a better one ranks among them
        var (best, fitness) = GreyWolfOptimizer<double>.UpdateLeaders(
            leaders, Tensor1D(0.9, 0.8, 0.7), population, Tensor1D(0.85, 0.1));
        Assert.Equal(new[] { 0.9, 0.85, 0.8 }, ToArray(best));
        Assert.Equal(new[] { 0.9, 0.85, 0.8 }, ToArray(fitness));
    }

    // ---------------------------------------------------------------- Whale Optimization (Mirjalili and Lewis 2016)

    [Fact]
    public void WhaleOptimization_Spiral_UsesLBetweenA2AndOne()
    {
        // l = (a2 - 1) u + 1 with a2 = -1.5 and u = 0.5 gives l = -0.25 (the old code drew l in [-1, 1])
        double a = 1.0, a2 = -1.5, l = (a2 - 1) * 0.5 + 1;
        var position = Tensor1D(0.3, 0.9);
        var leader = Tensor1D(0.5, 0.5);
        var moved = ToArray(WhaleOptimization<double>.Move(position, leader, null, Tensor1D(0.1, 0.2, 0.5, 0.7), a, a2, 1.0));
        double factor = System.Math.Exp(l) * System.Math.Cos(2 * System.Math.PI * l);
        Assert.Equal(System.Math.Max(0, System.Math.Min(1, 0.2 * factor + 0.5)), moved[0], Tolerance);
        Assert.Equal(System.Math.Max(0, System.Math.Min(1, 0.4 * factor + 0.5)), moved[1], Tolerance);
    }

    [Fact]
    public void WhaleOptimization_Encircling_And_Search_MatchPaper()
    {
        var position = Tensor1D(0.3);
        var leader = Tensor1D(0.6);

        // |A| < 1: X = X* - A |C X* - X| with A = 2 a r - a, C = 2 r2
        double a = 1.0, r = 0.75, r2 = 0.4;
        double coefficientA = 2 * a * r - a, coefficientC = 2 * r2;
        var draws = Tensor1D(r, r2, 0.5, 0.2);
        Assert.False(WhaleOptimization<double>.Explores(draws, a));
        var encircled = ToArray(WhaleOptimization<double>.Move(position, leader, null, draws, a, -1.0, 1.0));
        Assert.Equal(0.6 - coefficientA * System.Math.Abs(coefficientC * 0.6 - 0.3), encircled[0], Tolerance);

        // |A| >= 1: the same equation around a random whale
        a = 2.0;
        r = 0.0;
        coefficientA = 2 * a * r - a;
        draws = Tensor1D(r, r2, 0.5, 0.2);
        Assert.True(WhaleOptimization<double>.Explores(draws, a));
        var searched = ToArray(WhaleOptimization<double>.Move(position, leader, Tensor1D(0.4), draws, a, -1.0, 1.0));
        double expected = 0.4 - coefficientA * System.Math.Abs(coefficientC * 0.4 - 0.3);
        Assert.Equal(System.Math.Max(0, System.Math.Min(1, expected)), searched[0], Tolerance);
    }

    // ---------------------------------------------------------------- Sine Cosine Algorithm (Mirjalili 2016)

    [Fact]
    public void SineCosine_Move_MatchesSineAndCosineBranches()
    {
        var positions = Tensor2D(new[,] { { 0.3, 0.7 } });
        var destination = Tensor1D(0.6, 0.2);
        double r1 = 1.2;
        double[] u = { 0.1, 0.4, 0.2, 0.3, 0.8, 0.9 };
        var draws = new Tensor<double>((double[])u.Clone(), new[] { 1, 2, 3 });
        var moved = ToArray(SineCosine<double>.Move(positions, destination, draws, r1));

        double[] x = { 0.3, 0.7 };
        double[] best = { 0.6, 0.2 };
        for (int j = 0; j < 2; j++)
        {
            double r2 = 2 * System.Math.PI * u[3 * j], r3 = 2 * u[3 * j + 1], r4 = u[3 * j + 2];
            double wave = r4 < 0.5 ? System.Math.Sin(r2) : System.Math.Cos(r2);
            double expected = x[j] + r1 * wave * System.Math.Abs(r3 * best[j] - x[j]);
            Assert.Equal(System.Math.Max(0, System.Math.Min(1, expected)), moved[j], Tolerance);
        }
    }

    // ---------------------------------------------------------------- Salp Swarm Algorithm (Mirjalili et al. 2017)

    [Fact]
    public void SalpSwarm_MoveChain_LeadersFollowFoodAndFollowersTakeMidpoints()
    {
        var positions = Tensor2D(new[,] { { 0.1, 0.9 }, { 0.5, 0.5 }, { 0.2, 0.4 }, { 0.8, 0.6 } });
        var food = Tensor1D(0.4, 0.7);
        double c1 = 0.5;

        // Two leaders: (c2, c3) per dimension
        double[] u = { 0.2, 0.3, 0.6, 0.7, 0.4, 0.1, 0.8, 0.9 };
        var draws = new Tensor<double>((double[])u.Clone(), new[] { 2, 2, 2 });
        var moved = ToArray(SalpSwarm<double>.MoveChain(positions, food, draws, c1));

        var expected = new double[8];
        double[] f = { 0.4, 0.7 };
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
            {
                double c2 = u[(i * 2 + j) * 2], c3 = u[(i * 2 + j) * 2 + 1];
                double x = c3 < 0.5 ? f[j] + c1 * c2 : f[j] - c1 * c2;
                expected[i * 2 + j] = System.Math.Max(0, System.Math.Min(1, x));
            }

        double[] old = { 0.2, 0.4, 0.8, 0.6 };
        for (int i = 2; i < 4; i++)
            for (int j = 0; j < 2; j++)
                expected[i * 2 + j] = (old[(i - 2) * 2 + j] + expected[(i - 1) * 2 + j]) / 2;

        for (int k = 0; k < 8; k++)
            Assert.Equal(expected[k], moved[k], Tolerance);
    }

    // ---------------------------------------------------------------- Firefly Algorithm (Yang 2009)

    [Fact]
    public void FireflyAlgorithm_Attract_MatchesMovementEquation()
    {
        double alpha = 0.2, beta0 = 1.0, gamma = 2.0;
        double[] xi = { 0.2, 0.5 }, xj = { 0.6, 0.1 }, u = { 0.3, 0.8 };
        var moved = ToArray(FireflyAlgorithm<double>.Attract(Tensor1D(xi), Tensor1D(xj), Tensor1D(u), alpha, beta0, gamma));

        double r2 = (xi[0] - xj[0]) * (xi[0] - xj[0]) + (xi[1] - xj[1]) * (xi[1] - xj[1]);
        double beta = beta0 * System.Math.Exp(-gamma * r2);
        for (int k = 0; k < 2; k++)
        {
            double expected = xi[k] + beta * (xj[k] - xi[k]) + alpha * (u[k] - 0.5);
            Assert.Equal(System.Math.Max(0, System.Math.Min(1, expected)), moved[k], Tolerance);
        }
    }

    [Fact]
    public void FireflyAlgorithm_BrightestFireflyMovesRandomly()
    {
        var fireflies = Tensor2D(new[,] { { 0.6, 0.7 }, { 0.1, 0.2 } });
        System.Func<Tensor<double>, Tensor<double>> evaluate = rows =>
            AiDotNet.Tensors.Engines.AiDotNetEngine.Current.ReduceSum(rows, new[] { 1 }, false);
        var intensity = evaluate(fireflies);
        double alpha = 0.3, beta0 = 1.0, gamma = 1.0;
        FireflyAlgorithm<double>.Generation(ref fireflies, ref intensity, evaluate, new System.Random(7), alpha, beta0, gamma);

        // Same draws in the same order: firefly 0 has no brighter one and walks randomly, then firefly 1 moves
        // towards the walked firefly 0 (its intensity is updated before firefly 1 compares against it)
        var rng = new System.Random(7);
        double[] x0 = { 0.6, 0.7 }, x1 = { 0.1, 0.2 };
        for (int k = 0; k < 2; k++)
            x0[k] = System.Math.Max(0, System.Math.Min(1, x0[k] + alpha * (rng.NextDouble() - 0.5)));
        double r2 = (x0[0] - x1[0]) * (x0[0] - x1[0]) + (x0[1] - x1[1]) * (x0[1] - x1[1]);
        double beta = beta0 * System.Math.Exp(-gamma * r2);
        var moved1 = new double[2];
        for (int k = 0; k < 2; k++)
            moved1[k] = System.Math.Max(0, System.Math.Min(1, x1[k] + beta * (x0[k] - x1[k]) + alpha * (rng.NextDouble() - 0.5)));

        var result = ToArray(fireflies);
        Assert.Equal(x0[0], result[0], Tolerance);
        Assert.Equal(x0[1], result[1], Tolerance);
        Assert.Equal(moved1[0], result[2], Tolerance);
        Assert.Equal(moved1[1], result[3], Tolerance);
    }

    // ---------------------------------------------------------------- Dragonfly Algorithm (Mirjalili 2016)

    [Fact]
    public void DragonflyAlgorithm_StepCombinesSwarmBehaviours()
    {
        // t = 1 of T = 4: r = 0.25 + 2/4, w = 0.9 - 0.5/4, c' = 0.1 - 0.2/4; draws for s, a, c, f
        double[] u = { 0.5, 0.25, 0.8, 0.0 };
        var weights = DragonflyAlgorithm<double>.SwarmWeights.ForIteration(1, 4, Tensor1D(u));
        double radius = 0.75, w = 0.775, cPrime = 0.05;
        Assert.Equal(radius, weights.Radius, Tolerance);
        Assert.Equal(w, weights.Inertia, Tolerance);

        var positions = Tensor2D(new[,] { { 0.2 }, { 0.5 }, { 0.9 } });
        var steps = Tensor2D(new[,] { { 0.05 }, { -0.02 }, { 0.01 } });
        DragonflyAlgorithm<double>.MoveDragonfly(
            ref positions, ref steps, 0, Tensor1D(0.9), Tensor1D(0.2), weights, new System.Random(1));

        // Neighbours of dragonfly 0 (distance <= r and nonzero): 1 and 2
        double x = 0.2;
        double separation = -((0.5 - x) + (0.9 - x));
        double alignment = (-0.02 + 0.01) / 2;
        double cohesion = (0.5 + 0.9) / 2 - x;
        double food = 0.9 - x, enemy = 0.2 + x;
        double s = 2 * u[0] * cPrime, a = 2 * u[1] * cPrime, c = 2 * u[2] * cPrime, f = 2 * u[3], e = cPrime;
        double step = s * separation + a * alignment + c * cohesion + f * food + e * enemy + w * 0.05;
        step = System.Math.Max(-0.1, System.Math.Min(0.1, step));

        Assert.Equal(step, ToArray(steps)[0], Tolerance);
        Assert.Equal(x + step, ToArray(positions)[0], Tolerance);
    }

    [Fact]
    public void DragonflyAlgorithm_IsolatedDragonflyFliesLevy()
    {
        var weights = DragonflyAlgorithm<double>.SwarmWeights.ForIteration(0, 10, Tensor1D(0.5, 0.5, 0.5, 0.5));
        var positions = Tensor2D(new[,] { { 0.1 }, { 0.9 } });
        var steps = Tensor2D(new[,] { { 0.05 }, { 0.05 } });
        DragonflyAlgorithm<double>.MoveDragonfly(
            ref positions, ref steps, 0, Tensor1D(0.9), Tensor1D(0.9), weights, new System.Random(3));

        // Food out of reach and no neighbour: X = X + Levy .* X with uniform r1, r2 and the step reset to 0
        var rng = new System.Random(3);
        double r1 = rng.NextDouble(), r2 = rng.NextDouble();
        double sigma = System.Math.Pow(
            1.329340388179137 * System.Math.Sin(System.Math.PI * 0.75) / (0.906402477055477 * 1.5 * System.Math.Pow(2, 0.25)),
            1 / 1.5);
        double levy = 0.01 * r1 * sigma / System.Math.Pow(System.Math.Abs(r2), 1 / 1.5);
        Assert.Equal(System.Math.Min(1, 0.1 + levy * 0.1), ToArray(positions)[0], 1e-10);
        Assert.Equal(0.0, ToArray(steps)[0], Tolerance);
    }

    // ---------------------------------------------------------------- Seeded streams of the kept algorithms

    private static double[] AbsCorrelations(Matrix<double> x, Vector<double> y, double undefined)
    {
        int n = x.Rows, p = x.Columns;
        var result = new double[p];
        double yMean = 0;
        for (int i = 0; i < n; i++) yMean += y[i];
        yMean /= n;
        for (int j = 0; j < p; j++)
        {
            double xMean = 0;
            for (int i = 0; i < n; i++) xMean += x[i, j];
            xMean /= n;
            double sxy = 0, sxx = 0, syy = 0;
            for (int i = 0; i < n; i++)
            {
                double dx = x[i, j] - xMean, dy = y[i] - yMean;
                sxy += dx * dy;
                sxx += dx * dx;
                syy += dy * dy;
            }

            result[j] = sxx > 1e-10 && syy > 1e-10 ? System.Math.Abs(sxy / System.Math.Sqrt(sxx * syy)) : undefined;
        }

        return result;
    }

    private static double ThresholdFitness(double[] position, double[] scores, int targetCount)
    {
        double total = 0;
        int count = 0;
        for (int j = 0; j < position.Length; j++)
        {
            if (position[j] > 0.5)
            {
                total += scores[j];
                count++;
            }
        }

        return count == 0 ? 0 : total / count - 0.1 * System.Math.Abs(count - targetCount);
    }

    [Fact]
    public void AntColonyOptimization_MatchesAntSystemWithSameSeed()
    {
        var (x, y) = Problem(11, 50, 7);
        int k = 3, ants = 5, iterations = 6, seed = 21;
        double alpha = 1.0, beta = 2.0, rho = 0.1;
        var selector = new AntColonyOptimization<double>(k, ants, iterations, alpha, beta, rho, seed);
        selector.Fit(x, y);

        // Ant System in double: roulette over the unselected features on tau^alpha eta^beta
        var rng = new System.Random(seed);
        int p = x.Columns;
        var eta = AbsCorrelations(x, y, 0.01);
        var tau = System.Linq.Enumerable.Repeat(1.0, p).ToArray();
        int[]? best = null;
        double bestFitness = double.MinValue;
        for (int it = 0; it < iterations; it++)
        {
            var subsets = new System.Collections.Generic.List<int[]>();
            var fitness = new double[ants];
            for (int a = 0; a < ants; a++)
            {
                var available = System.Linq.Enumerable.Range(0, p).ToList();
                var chosen = new System.Collections.Generic.List<int>();
                while (chosen.Count < k)
                {
                    var w = available.Select(j => System.Math.Pow(tau[j], alpha) * System.Math.Pow(eta[j], beta)).ToArray();
                    double r = rng.NextDouble() * w.Sum(), cum = 0;
                    int pick = available.Count - 1;
                    for (int i = 0; i < w.Length; i++)
                    {
                        cum += w[i];
                        if (r <= cum)
                        {
                            pick = i;
                            break;
                        }
                    }

                    chosen.Add(available[pick]);
                    available.RemoveAt(pick);
                }

                subsets.Add(chosen.ToArray());
                fitness[a] = chosen.Average(j => eta[j]);
                if (fitness[a] > bestFitness)
                {
                    bestFitness = fitness[a];
                    best = chosen.ToArray();
                }
            }

            for (int j = 0; j < p; j++) tau[j] *= 1 - rho;
            for (int a = 0; a < ants; a++)
                foreach (int j in subsets[a]) tau[j] += fitness[a];
            for (int j = 0; j < p; j++) tau[j] = System.Math.Max(tau[j], 0.01);
        }

        Assert.NotNull(best);
        Assert.Equal((best ?? System.Array.Empty<int>()).OrderBy(j => j).ToArray(), selector.SelectedIndices);
        var pheromones = selector.Pheromones;
        Assert.NotNull(pheromones);
        for (int j = 0; j < p; j++)
            Assert.Equal(tau[j], pheromones?[j] ?? double.NaN, 9);
    }

    [Fact]
    public void HarmonySearch_MatchesGeemImprovisationWithSameSeed()
    {
        var (x, y) = Problem(12, 50, 6);
        int k = 2, memory = 6, iterations = 25, seed = 5;
        double hmcr = 0.9, par = 0.3, bw = 0.1;
        var selector = new HarmonySearch<double>(k, memory, iterations, hmcr, par, bw, seed);
        selector.Fit(x, y);

        var rng = new System.Random(seed);
        int p = x.Columns;
        var scores = AbsCorrelations(x, y, 0);
        var hm = new double[memory][];
        var fit = new double[memory];
        for (int i = 0; i < memory; i++)
        {
            hm[i] = new double[p];
            for (int j = 0; j < p; j++) hm[i][j] = rng.NextDouble();
        }

        for (int i = 0; i < memory; i++) fit[i] = ThresholdFitness(hm[i], scores, k);
        for (int it = 0; it < iterations; it++)
        {
            var h = new double[p];
            for (int j = 0; j < p; j++)
            {
                if (rng.NextDouble() < hmcr)
                {
                    h[j] = hm[rng.Next(memory)][j];
                    if (rng.NextDouble() < par)
                        h[j] = System.Math.Max(0, System.Math.Min(1, h[j] + bw * (2 * rng.NextDouble() - 1)));
                }
                else
                {
                    h[j] = rng.NextDouble();
                }
            }

            double f = ThresholdFitness(h, scores, k);
            int worst = 0;
            for (int i = 1; i < memory; i++) if (fit[i] < fit[worst]) worst = i;
            if (f > fit[worst])
            {
                hm[worst] = h;
                fit[worst] = f;
            }
        }

        int bestIdx = 0;
        for (int i = 1; i < memory; i++) if (fit[i] > fit[bestIdx]) bestIdx = i;
        var featureScores = selector.FeatureScores;
        Assert.NotNull(featureScores);
        for (int j = 0; j < p; j++)
            Assert.Equal(hm[bestIdx][j], featureScores?[j] ?? double.NaN, 12);
    }

    // ---------------------------------------------------------------- Multi-objective (NSGA-II, SPEA2, MOEA/D)

    private static bool DominatesReference(double accA, double cntA, double accB, double cntB) =>
        accA >= accB && cntA <= cntB && (accA > accB || cntA < cntB);

    [Fact]
    public void ParetoFronts_MatchBruteForceNonDominatedSorting()
    {
        double[] acc = { 0.5, 0.7, 0.7, 0.2, 0.9, 0.5, 0.3, 0.9 };
        double[] cnt = { 2, 3, 2, 1, 5, 4, 1, 4 };
        var fronts = AiDotNet.Preprocessing.FeatureSelection.Helpers.ParetoOps.Fronts(
            AiDotNet.Preprocessing.FeatureSelection.Helpers.ParetoOps.Dominance(Tensor1D(acc), Tensor1D(cnt)), out var rank);

        // Reference: peel the solutions not dominated by any remaining one
        var remaining = Enumerable.Range(0, acc.Length).ToList();
        var expectedRank = new double[acc.Length];
        int level = 0;
        var expectedFronts = new List<int[]>();
        while (remaining.Count > 0)
        {
            var front = remaining.Where(j => !remaining.Any(i => DominatesReference(acc[i], cnt[i], acc[j], cnt[j]))).ToArray();
            foreach (int j in front) expectedRank[j] = level;
            expectedFronts.Add(front);
            remaining = remaining.Except(front).ToList();
            level++;
        }

        Assert.Equal(expectedFronts.Count, fronts.Count);
        for (int f = 0; f < fronts.Count; f++)
            Assert.Equal(expectedFronts[f], fronts[f]);
        Assert.Equal(expectedRank, ToArray(rank));
    }

    [Fact]
    public void CrowdingDistance_GivesBothObjectivesInfiniteBoundaries()
    {
        // One front whose extremes differ per objective: accuracy extremes are 1 and 4, count extremes 0 and 2
        double[] acc = { 0.30, 0.10, 0.25, 0.20, 0.40, 0.35 };
        double[] cnt = { 1, 4, 6, 2, 3, 5 };
        var front = new[] { 0, 1, 2, 3, 4, 5 };
        var distance = ToArray(AiDotNet.Preprocessing.FeatureSelection.Helpers.ParetoOps.CrowdingDistance(
            new List<int[]> { front }, Tensor1D(acc), Tensor1D(cnt)));

        // Deb et al. (2002): per objective, sorted boundaries are infinite, interior adds the normalized gap
        var expected = new double[acc.Length];
        foreach (var objective in new[] { acc, cnt })
        {
            var order = front.OrderBy(i => objective[i]).ToArray();
            double range = objective[order[order.Length - 1]] - objective[order[0]];
            expected[order[0]] = double.PositiveInfinity;
            expected[order[order.Length - 1]] = double.PositiveInfinity;
            for (int r = 1; r < order.Length - 1; r++)
                expected[order[r]] += (objective[order[r + 1]] - objective[order[r - 1]]) / range;
        }

        for (int i = 0; i < acc.Length; i++)
        {
            if (double.IsPositiveInfinity(expected[i]))
                Assert.Equal(double.MaxValue, distance[i]);
            else
                Assert.Equal(expected[i], distance[i], 12);
        }

        // Solutions 2 (most features) and 0 (fewest) are boundaries of the count objective only
        Assert.Equal(double.MaxValue, distance[2]);
        Assert.Equal(double.MaxValue, distance[0]);
    }

    [Fact]
    public void NSGAII_SelectSurvivors_TakesWholeFrontsThenLeastCrowded()
    {
        var fronts = new List<int[]> { new[] { 0, 1 }, new[] { 2, 3, 4 }, new[] { 5 } };
        var crowding = Tensor1D(9, 9, 0.5, double.MaxValue, 1.0, 3);
        Assert.Equal(new[] { 0, 1, 3, 4 }, NSGAIISelector<double>.SelectSurvivors(fronts, crowding, 4));
        Assert.Equal(new[] { 0, 1, 2, 3, 4 }, NSGAIISelector<double>.SelectSurvivors(fronts, crowding, 5));
        Assert.Equal(new[] { 0, 1, 2, 3, 4, 5 }, NSGAIISelector<double>.SelectSurvivors(fronts, crowding, 6));
    }

    private static double[] Spea2FitnessReference(double[] acc, double[] cnt, out double[,] distances)
    {
        int m = acc.Length;
        var strength = new double[m];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < m; j++)
                if (DominatesReference(acc[i], cnt[i], acc[j], cnt[j])) strength[i]++;

        distances = new double[m, m];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < m; j++)
                distances[i, j] = System.Math.Sqrt((acc[i] - acc[j]) * (acc[i] - acc[j]) + (cnt[i] - cnt[j]) * (cnt[i] - cnt[j]));

        int k = (int)System.Math.Sqrt(m);
        var fitness = new double[m];
        for (int i = 0; i < m; i++)
        {
            double raw = 0;
            for (int j = 0; j < m; j++)
                if (DominatesReference(acc[j], cnt[j], acc[i], cnt[i])) raw += strength[j];

            // sigma_i^k: distance to the k-th nearest other solution
            var others = Enumerable.Range(0, m).Where(j => j != i).Select(j => distances[i, j]).OrderBy(d => d).ToArray();
            fitness[i] = raw + 1.0 / (others[k - 1] + 2);
        }

        return fitness;
    }

    [Fact]
    public void SPEAII_Fitness_IsRawStrengthPlusKthNearestDensity()
    {
        double[] acc = { 0.5, 0.7, 0.7, 0.2, 0.9, 0.5, 0.3, 0.9, 0.6 };
        double[] cnt = { 2, 3, 2, 1, 5, 4, 1, 4, 3 };
        var expected = Spea2FitnessReference(acc, cnt, out _);
        var fitness = ToArray(SPEAIISelector<double>.ComputeFitness(Tensor1D(acc), Tensor1D(cnt), out _));
        for (int i = 0; i < acc.Length; i++)
            Assert.Equal(expected[i], fitness[i], 9);
    }

    [Fact]
    public void SPEAII_EnvironmentalSelection_TruncatesNondominatedByNearestDistance()
    {
        // Five nondominated solutions on a line of accuracy versus count; 1 and 2 are the closest pair
        double[] acc = { 0.1, 0.3, 0.35, 0.6, 0.9, 0.05 };
        double[] cnt = { 1, 2, 2.1, 3, 4, 4 };
        var fitness = SPEAIISelector<double>.ComputeFitness(Tensor1D(acc), Tensor1D(cnt), out var distances);
        var fit = ToArray(fitness);
        var d = new double[acc.Length, acc.Length];
        var dist = ToArray(distances);
        for (int i = 0; i < acc.Length; i++)
            for (int j = 0; j < acc.Length; j++)
                d[i, j] = dist[i * acc.Length + j];

        // Reference truncation: drop the solution whose sorted distances are lexicographically smallest
        var members = Enumerable.Range(0, acc.Length).Where(i => fit[i] < 1).ToList();
        Assert.Equal(5, members.Count);
        while (members.Count > 3)
        {
            int remove = members[0];
            double[]? bestKey = null;
            foreach (int i in members)
            {
                var key = members.Where(j => j != i).Select(j => d[i, j]).OrderBy(v => v).ToArray();
                bool smaller = bestKey is null;
                for (int c = 0; !smaller && bestKey is not null && c < key.Length; c++)
                {
                    if (key[c] < bestKey[c]) { smaller = true; break; }
                    if (key[c] > bestKey[c]) break;
                }

                if (smaller)
                {
                    bestKey = key;
                    remove = i;
                }
            }

            members.Remove(remove);
        }

        var selected = SPEAIISelector<double>.EnvironmentalSelection(fitness, distances, 3);
        Assert.Equal(members.OrderBy(i => i).ToArray(), selected.OrderBy(i => i).ToArray());
        Assert.DoesNotContain(5, selected);

        // With room to spare the dominated solution fills the archive after the nondominated ones
        var all = SPEAIISelector<double>.EnvironmentalSelection(fitness, distances, 6);
        Assert.Equal(5, all[5]);
    }

    [Fact]
    public void MOEAD_TchebycheffAndWeights_FollowZhangAndLi()
    {
        var weights = MOEADSelector<double>.WeightVectors(5);
        Assert.Equal(new[] { 0, 1, 0.25, 0.75, 0.5, 0.5, 0.75, 0.25, 1, 0 }, ToArray(weights));

        var g = ToArray(MOEADSelector<double>.Tchebycheff(
            Tensor1D(0.4, 0.8), Tensor1D(3, 6), Tensor2D(new double[,] { { 0.25, 0.75 }, { 0.9, 0.1 } }),
            Tensor1D(0.9), Tensor1D(2)));
        Assert.Equal(System.Math.Max(0.25 * 0.5, 0.75 * 1), g[0], 12);
        Assert.Equal(System.Math.Max(0.9 * 0.1, 0.1 * 4), g[1], 12);

        Assert.Throws<System.ArgumentException>(() => new MOEADSelector<double>(populationSize: 1));
    }

    // ---------------------------------------------------------------- Simulated annealing and tabu search

    private static double MeanScore(bool[] mask, double[] scores)
    {
        double total = 0;
        int count = 0;
        for (int j = 0; j < mask.Length; j++)
        {
            if (mask[j])
            {
                total += scores[j];
                count++;
            }
        }

        return count == 0 ? 0 : total / count;
    }

    [Fact]
    public void SimulatedAnnealing_MatchesMetropolisWithSameSeed()
    {
        var (x, y) = Problem(13, 50, 9);
        int k = 3, iterations = 200, seed = 17;
        double t0 = 1.0, cooling = 0.97;
        var selector = new AiDotNet.Preprocessing.FeatureSelection.Optimization.SimulatedAnnealingSelector<double>(
            k, t0, cooling, iterations, seed);
        selector.Fit(x, y);

        var rng = new System.Random(seed);
        int p = x.Columns;
        var scores = AbsCorrelations(x, y, 0);
        var current = new bool[p];
        foreach (int j in Enumerable.Range(0, p).OrderBy(_ => rng.Next()).Take(k)) current[j] = true;
        double energy = -MeanScore(current, scores);
        var best = (bool[])current.Clone();
        double bestEnergy = energy, temperature = t0;
        for (int it = 0; it < iterations; it++)
        {
            var neighbor = (bool[])current.Clone();
            var selected = Enumerable.Range(0, p).Where(j => neighbor[j]).ToList();
            if (rng.NextDouble() < 0.5 && selected.Count > 1)
            {
                neighbor[selected[rng.Next(selected.Count)]] = false;
                var unselected = Enumerable.Range(0, p).Where(j => !neighbor[j]).ToList();
                if (unselected.Count > 0) neighbor[unselected[rng.Next(unselected.Count)]] = true;
            }
            else
            {
                int flip = rng.Next(p);
                neighbor[flip] = !neighbor[flip];
            }

            double neighborEnergy = -MeanScore(neighbor, scores);
            double delta = neighborEnergy - energy;
            if (delta < 0 || rng.NextDouble() < System.Math.Exp(-delta / temperature))
            {
                current = neighbor;
                energy = neighborEnergy;
                if (energy < bestEnergy)
                {
                    best = (bool[])current.Clone();
                    bestEnergy = energy;
                }
            }

            temperature *= cooling;
        }

        Assert.Equal(Enumerable.Range(0, p).Where(j => best[j]).ToArray(), selector.SelectedIndices);
    }

    [Fact]
    public void TabuSearch_AppliesTheEvaluatedSwapAndHonoursTabuWithAspiration()
    {
        var swap = AiDotNet.Preprocessing.FeatureSelection.Optimization.TabuSearchSelector<double>.BestSwap(
            Tensor1D(1, 1, 0, 0), Tensor1D(0.1, 0.5, 0.9, 0.3), Tensor1D(1, 0, 0, 0), Tensor1D(0.6), out var moveScore);

        // Removing 0 for 2 is best ((0.6 - 0.1 + 0.9) / 2 = 0.7) and 0 is tabu, but 0.7 beats the best 0.6
        Assert.Equal((0, 2), swap ?? (-1, -1));
        Assert.Equal(0.7, ToArray(moveScore)[0], 12);

        // Without aspiration the tabu feature 0 cannot leave: the best free swap is 1 -> 2
        swap = AiDotNet.Preprocessing.FeatureSelection.Optimization.TabuSearchSelector<double>.BestSwap(
            Tensor1D(1, 1, 0, 0), Tensor1D(0.1, 0.5, 0.9, 0.3), Tensor1D(1, 0, 0, 0), Tensor1D(0.9), out moveScore);
        Assert.Equal((1, 2), swap ?? (-1, -1));
        Assert.Equal(0.5, ToArray(moveScore)[0], 12);
    }

    [Fact]
    public void TabuSearch_MatchesGloverSwapSearchWithSameSeed()
    {
        var (x, y) = Problem(14, 50, 8);
        int k = 3, tenure = 2, iterations = 12, seed = 23;
        var selector = new AiDotNet.Preprocessing.FeatureSelection.Optimization.TabuSearchSelector<double>(
            k, tenure, iterations, seed);
        selector.Fit(x, y);

        var rng = new System.Random(seed);
        int p = x.Columns;
        var scores = AbsCorrelations(x, y, 0);
        var current = new bool[p];
        foreach (int j in Enumerable.Range(0, p).OrderBy(_ => rng.Next()).Take(k)) current[j] = true;
        double currentScore = MeanScore(current, scores);
        var best = (bool[])current.Clone();
        double bestScore = currentScore;
        var tabuUntil = new int[p];
        for (int it = 0; it < iterations; it++)
        {
            int bestR = -1, bestA = -1;
            double bestMove = double.MinValue;
            for (int r = 0; r < p; r++)
            {
                if (!current[r]) continue;
                for (int a = 0; a < p; a++)
                {
                    if (current[a]) continue;
                    var neighbor = (bool[])current.Clone();
                    neighbor[r] = false;
                    neighbor[a] = true;
                    double score = MeanScore(neighbor, scores);
                    bool admissible = (tabuUntil[r] <= it && tabuUntil[a] <= it) || score > bestScore;
                    if (admissible && score > bestMove)
                    {
                        bestMove = score;
                        bestR = r;
                        bestA = a;
                    }
                }
            }

            if (bestR >= 0)
            {
                current[bestR] = false;
                current[bestA] = true;
                tabuUntil[bestR] = it + tenure;
                tabuUntil[bestA] = it + tenure;
                currentScore = bestMove;
                if (currentScore > bestScore)
                {
                    best = (bool[])current.Clone();
                    bestScore = currentScore;
                }
            }
        }

        Assert.Equal(Enumerable.Range(0, p).Where(j => best[j]).ToArray(), selector.SelectedIndices);
    }

    // ---------------------------------------------------------------- Quantum-inspired evolutionary algorithm

    [Fact]
    public void QuantumInspired_RotationTurnsEachDifferingQubitTowardsTheBestBit()
    {
        double q = System.Math.PI / 4, a = 0.05;
        var theta = Tensor1D(q, q, q, q, 0, System.Math.PI / 2);
        var solution = Tensor1D(0, 1, 1, 0, 1, 0);
        var best = Tensor1D(1, 0, 1, 0, 0, 1);
        var rotated = ToArray(AiDotNet.Preprocessing.FeatureSelection.Quantum.QuantumInspiredSelector<double>.Rotate(
            theta, solution, best, a));

        // Han and Kim (2002): best bit 1 raises P(1) = sin^2(theta), best bit 0 lowers it; equal bits stay;
        // a qubit already at 0 (best 0) or pi / 2 (best 1) does not turn further
        Assert.Equal(q + a, rotated[0], 12);
        Assert.Equal(q - a, rotated[1], 12);
        Assert.Equal(q, rotated[2], 12);
        Assert.Equal(q, rotated[3], 12);
        Assert.Equal(0, rotated[4], 12);
        Assert.Equal(System.Math.PI / 2, rotated[5], 12);
    }

    // ---------------------------------------------------------------- Differential privacy (exponential mechanism)

    [Fact]
    public void DifferentialPrivacy_NoiseScaleIsTwoKSensitivityOverEpsilon()
    {
        // Delta = 6 B_x B_y / n = 6 * 2 * 0.5 / 100 = 0.06; scale = 2 k Delta / epsilon = 2 * 3 * 0.06 / 0.5
        Assert.Equal(0.72, AiDotNet.Preprocessing.FeatureSelection.Privacy.DifferentialPrivacySelector<double>.NoiseScaleFor(
            0.5, 3, 100, 2.0, 0.5), 12);

        var (x, y) = Problem(15, 40, 5);
        var selector = new AiDotNet.Preprocessing.FeatureSelection.Privacy.DifferentialPrivacySelector<double>(
            2, 1.0, 3, null, 1.0, 1.0);
        selector.Fit(x, y);
        Assert.Equal(2 * 2 * 6.0 / 40 / 1.0, selector.NoiseScale, 12);
        Assert.Equal(2, selector.SelectedIndices?.Length);
    }

    [Fact]
    public void DifferentialPrivacy_UtilitySensitivityIsBoundedBySixOverN()
    {
        var rng = new System.Random(31);
        int n = 20, p = 4;
        double bound = 6.0 / n, worst = 0;
        for (int trial = 0; trial < 300; trial++)
        {
            var x = new Matrix<double>(n, p);
            var y = new Vector<double>(n);
            for (int i = 0; i < n; i++)
            {
                y[i] = rng.NextDouble() < 0.5 ? -1 : 1;
                for (int j = 0; j < p; j++) x[i, j] = rng.NextDouble() < 0.5 ? -1.5 : 1.5;
            }

            var before = ToArray(AiDotNet.Preprocessing.FeatureSelection.Privacy.DifferentialPrivacySelector<double>.Utilities(x, y, 1, 1));

            // Neighbouring dataset: one row replaced by extreme values
            int row = rng.Next(n);
            y[row] = -y[row];
            for (int j = 0; j < p; j++) x[row, j] = rng.NextDouble() < 0.5 ? -3 : 3;
            var after = ToArray(AiDotNet.Preprocessing.FeatureSelection.Privacy.DifferentialPrivacySelector<double>.Utilities(x, y, 1, 1));

            for (int j = 0; j < p; j++) worst = System.Math.Max(worst, System.Math.Abs(after[j] - before[j]));
        }

        Assert.True(worst <= bound + 1e-12, $"utility changed by {worst} > {bound}");
    }

    [Fact]
    public void DifferentialPrivacy_SelectionFollowsTheExponentialMechanism()
    {
        // k = 1: P(j) = exp(u_j / scale) / sum; k = 2: sequential sampling without replacement (peeling)
        double[] u = { 0, 0.1, 0.3 };
        double scale = 0.1;
        var w = u.Select(v => System.Math.Exp(v / scale)).ToArray();
        double total = w.Sum();
        var rng = new System.Random(41);
        int trials = 20000;
        var single = new int[3];
        var pairs = new int[3]; // pair without feature i
        var utility = Tensor1D(u);
        for (int t = 0; t < trials; t++)
        {
            single[AiDotNet.Preprocessing.FeatureSelection.Privacy.DifferentialPrivacySelector<double>.ExponentialMechanismTopK(
                utility, 1, scale, rng)[0]]++;
            var two = AiDotNet.Preprocessing.FeatureSelection.Privacy.DifferentialPrivacySelector<double>.ExponentialMechanismTopK(
                utility, 2, scale, rng);
            pairs[3 - two[0] - two[1]]++;
        }

        for (int j = 0; j < 3; j++)
            Assert.True(System.Math.Abs(w[j] / total - (double)single[j] / trials) < 0.015, $"feature {j}");

        for (int missing = 0; missing < 3; missing++)
        {
            int a = (missing + 1) % 3, b = (missing + 2) % 3;
            double expected = w[a] / total * w[b] / (total - w[a]) + w[b] / total * w[a] / (total - w[b]);
            Assert.True(System.Math.Abs(expected - (double)pairs[missing] / trials) < 0.015,
                $"pair without {missing}: expected {expected}, observed {(double)pairs[missing] / trials}");
        }
    }

    // ---------------------------------------------------------------- SMOTE feature selection

    [Fact]
    public void SMOTE_MatchesChawlaOversamplingAndFisherScoresWithSameSeed()
    {
        var rng0 = new System.Random(51);
        int n = 30, p = 4, k = 3, seed = 9;
        var x = new Matrix<double>(n, p);
        var y = new Vector<double>(n);
        for (int i = 0; i < n; i++)
        {
            y[i] = i < 8 ? 1 : 0;
            for (int j = 0; j < p; j++) x[i, j] = rng0.NextDouble() + (y[i] > 0.5 ? 0.3 * j : 0);
        }

        var selector = new AiDotNet.Preprocessing.FeatureSelection.Imbalanced.SMOTEFeatureSelector<double>(2, k, 1.0, seed);
        selector.Fit(x, y);

        // Reference: synthetic = base + lambda (neighbour - base) over the k nearest minority rows
        var rng = new System.Random(seed);
        var minority = Enumerable.Range(0, n).Where(i => y[i] >= 0.5).ToArray();
        var majority = Enumerable.Range(0, n).Where(i => y[i] < 0.5).ToArray();
        var rows1 = minority.Select(i => Enumerable.Range(0, p).Select(j => x[i, j]).ToArray()).ToList();
        int nSynthetic = majority.Length - minority.Length;
        for (int s = 0; s < nSynthetic; s++)
        {
            int b = minority[rng.Next(minority.Length)];
            var neighbors = minority.Where(i => i != b)
                .OrderBy(i => System.Math.Sqrt(Enumerable.Range(0, p).Sum(j => (x[b, j] - x[i, j]) * (x[b, j] - x[i, j]))))
                .Take(k).ToArray();
            int nb = neighbors[rng.Next(neighbors.Length)];
            double lambda = rng.NextDouble();
            rows1.Add(Enumerable.Range(0, p).Select(j => x[b, j] + lambda * (x[nb, j] - x[b, j])).ToArray());
        }

        var expected = new double[p];
        for (int j = 0; j < p; j++)
        {
            double m0 = majority.Average(i => x[i, j]), m1 = rows1.Average(r => r[j]);
            double v0 = majority.Average(i => (x[i, j] - m0) * (x[i, j] - m0));
            double v1 = rows1.Average(r => (r[j] - m1) * (r[j] - m1));
            expected[j] = (m0 - m1) * (m0 - m1) / (v0 + v1);
        }

        var scores = selector.FeatureScores;
        Assert.NotNull(scores);
        for (int j = 0; j < p; j++)
            Assert.Equal(expected[j], scores?[j] ?? double.NaN, 9);
    }

    [Fact]
    public void SMOTE_SingleMinoritySampleHasNoNeighbourToInterpolate()
    {
        var x = new Matrix<double>(5, 2);
        var y = new Vector<double>(5);
        for (int i = 0; i < 5; i++)
        {
            x[i, 0] = i;
            x[i, 1] = i * i;
        }

        y[4] = 1;
        var selector = new AiDotNet.Preprocessing.FeatureSelection.Imbalanced.SMOTEFeatureSelector<double>(1, 1, 1.0, 2);
        selector.Fit(x, y);
        Assert.Equal(1, selector.SelectedIndices?.Length);
    }

    // ---------------------------------------------------------------- Stability selection

    [Fact]
    public void StabilitySelection_SelectionProbabilityIsTheFractionOfSubsamplesSelectingTheFeature()
    {
        var (x, y) = Problem(16, 40, 4);
        int seed = 4, subsamples = 20;
        var seen = new List<double[]>();
        var selector = new AiDotNet.Preprocessing.FeatureSelection.Hybrid.StabilitySelection<double>(
            2, subsamples, 0.5, 0.6,
            (m, v) =>
            {
                seen.Add(v.ToArray());
                return v[0] > 0 ? new[] { 0, 0, 1, 7 } : new[] { 2, -1 };
            },
            seed);
        selector.Fit(x, y);

        // Subsamples of floor(n / 2) rows drawn without replacement, in master's order
        var rng = new System.Random(seed);
        int positive = 0;
        for (int b = 0; b < subsamples; b++)
        {
            var rows = Enumerable.Range(0, 40).OrderBy(_ => rng.Next()).Take(20).ToArray();
            Assert.Equal(rows.Select(i => y[i]).ToArray(), seen[b]);
            if (y[rows[0]] > 0) positive++;
        }

        // A feature listed twice in one subsample still counts once; out-of-range indices are ignored
        double share = (double)positive / subsamples;
        Assert.Equal(new[] { share, share, 1 - share, 0 }, selector.SelectionProbabilities?.ToArray());
    }

    // ---------------------------------------------------------------- Hybrid selectors

    [Fact]
    public void EmbeddedWrapper_LassoCoordinateDescentDividesByTheColumnScale()
    {
        var (x, y) = Problem(17, 30, 3);
        int n = x.Rows, p = x.Columns;
        double alpha = 0.05;
        var cols = new double[p][];
        for (int j = 0; j < p; j++) cols[j] = Enumerable.Range(0, n).Select(i => x[i, j] * (j + 1)).ToArray();
        var target = Enumerable.Range(0, n).Select(i => y[i]).ToArray();

        // Friedman et al. (2010): b_j = S(rho_j, alpha) / ((1/n) sum x_ij^2), here with unscaled columns
        var beta = new double[p];
        for (int sweep = 0; sweep < 50; sweep++)
        {
            for (int j = 0; j < p; j++)
            {
                double rho = 0, scale = 0;
                for (int i = 0; i < n; i++)
                {
                    double r = target[i];
                    for (int k = 0; k < p; k++) if (k != j) r -= beta[k] * cols[k][i];
                    rho += cols[j][i] * r;
                    scale += cols[j][i] * cols[j][i];
                }

                rho /= n;
                scale /= n;
                double s = rho > alpha ? rho - alpha : rho < -alpha ? rho + alpha : 0;
                beta[j] = s / scale;
            }
        }

        var design = new double[n, p];
        for (int i = 0; i < n; i++) for (int j = 0; j < p; j++) design[i, j] = cols[j][i];
        var result = ToArray(AiDotNet.Preprocessing.FeatureSelection.Hybrid.EmbeddedWrapperHybridSelector<double>
            .LassoCoordinateDescent(Tensor2D(design), Tensor1D(target), alpha, 50));
        for (int j = 0; j < p; j++)
            Assert.Equal(beta[j], result[j], 9);
    }

    [Fact]
    public void FilterWrapper_CrossValidatedRSquaredMatchesFoldwiseLeastSquares()
    {
        var (x, y) = Problem(18, 23, 4);
        int n = x.Rows;
        var order = Enumerable.Range(0, n).Reverse().ToArray();
        var features = new List<int> { 2, 0 };
        var score = ToArray(AiDotNet.Preprocessing.FeatureSelection.Hybrid.FilterWrapperHybridSelector<double>
            .CrossValidatedRSquared(Tensor2D(ToGrid(x)), Tensor1D(Enumerable.Range(0, n).Select(i => y[i]).ToArray()), features, order))[0];

        double total = 0;
        for (int fold = 0; fold < 5; fold++)
        {
            int start = fold * n / 5, end = (fold + 1) * n / 5;
            var train = order.Where((_, i) => i < start || i >= end).ToArray();
            var test = order.Skip(start).Take(end - start).ToArray();

            // Normal equations with intercept and ridge 1e-6, solved by Cramer-free Gaussian elimination
            int k = 3;
            var a = new double[k, k + 1];
            foreach (int i in train)
            {
                var row = new[] { 1, x[i, 2], x[i, 0] };
                for (int r = 0; r < k; r++)
                {
                    for (int c = 0; c < k; c++) a[r, c] += row[r] * row[c];
                    a[r, k] += row[r] * y[i];
                }
            }

            for (int r = 0; r < k; r++) a[r, r] += 1e-6;
            for (int c = 0; c < k; c++)
                for (int r = c + 1; r < k; r++)
                {
                    double f = a[r, c] / a[c, c];
                    for (int q = c; q <= k; q++) a[r, q] -= f * a[c, q];
                }

            var b = new double[k];
            for (int r = k - 1; r >= 0; r--)
            {
                b[r] = a[r, k];
                for (int q = r + 1; q < k; q++) b[r] -= a[r, q] * b[q];
                b[r] /= a[r, r];
            }

            double mean = test.Average(i => y[i]), res = 0, tot = 0;
            foreach (int i in test)
            {
                double pred = b[0] + b[1] * x[i, 2] + b[2] * x[i, 0];
                res += (y[i] - pred) * (y[i] - pred);
                tot += (y[i] - mean) * (y[i] - mean);
            }

            total += 1 - res / tot;
        }

        Assert.Equal(total / 5, score, 8);
    }

    private static double[,] ToGrid(Matrix<double> m)
    {
        var grid = new double[m.Rows, m.Columns];
        for (int i = 0; i < m.Rows; i++) for (int j = 0; j < m.Columns; j++) grid[i, j] = m[i, j];
        return grid;
    }

    // ---------------------------------------------------------------- Model-agnostic importance

    private static double[] SolveDense(double[,] a, double[] b)
    {
        int k = b.Length;
        var m = new double[k, k + 1];
        for (int r = 0; r < k; r++)
        {
            for (int c = 0; c < k; c++) m[r, c] = a[r, c];
            m[r, k] = b[r];
        }

        for (int c = 0; c < k; c++)
        {
            int pivot = c;
            for (int r = c + 1; r < k; r++) if (System.Math.Abs(m[r, c]) > System.Math.Abs(m[pivot, c])) pivot = r;
            for (int q = 0; q <= k; q++) (m[c, q], m[pivot, q]) = (m[pivot, q], m[c, q]);
            for (int r = c + 1; r < k; r++)
            {
                double f = m[r, c] / m[c, c];
                for (int q = c; q <= k; q++) m[r, q] -= f * m[c, q];
            }
        }

        var x = new double[k];
        for (int r = k - 1; r >= 0; r--)
        {
            x[r] = m[r, k];
            for (int q = r + 1; q < k; q++) x[r] -= m[r, q] * x[q];
            x[r] /= m[r, r];
        }

        return x;
    }

    [Fact]
    public void PermutationImportance_ScoresTheModelFittedOnceOnPermutedCopies()
    {
        var (x, _) = Problem(19, 40, 3);
        var y = new Vector<double>(40);
        for (int i = 0; i < 40; i++) y[i] = 2 * x[i, 0] - x[i, 1];

        var selector = new AiDotNet.Preprocessing.FeatureSelection.ModelAgnostic.PermutationImportance<double>(
            2, 5, null, 8);
        selector.Fit(x, y);
        var importance = selector.ImportanceScores?.ToArray() ?? System.Array.Empty<double>();

        // The exact linear model has R^2 = 1 and ignores feature 2, so permuting feature 2 costs nothing
        Assert.True(System.Math.Abs(importance[2]) < 1e-6, $"importance of the unused feature: {importance[2]}");
        Assert.True(importance[0] > importance[1] && importance[1] > 0.1);

        // With a user scorer, importance is the mean drop over the seeded column shuffles
        var rng = new System.Random(8);
        var custom = new AiDotNet.Preprocessing.FeatureSelection.ModelAgnostic.PermutationImportance<double>(
            1, 3, (m, v) => m[0, 0] + 10 * m[1, 1] + 100 * m[2, 2], 8);
        custom.Fit(x, y);
        double baseline = x[0, 0] + 10 * x[1, 1] + 100 * x[2, 2];
        var expected = new double[3];
        var expectedStd = new double[3];
        for (int j = 0; j < 3; j++)
        {
            var drops = new double[3];
            for (int perm = 0; perm < 3; perm++)
            {
                var order = Enumerable.Range(0, 40).OrderBy(_ => rng.Next()).ToArray();
                double score = (j == 0 ? x[order[0], 0] : x[0, 0]) + 10 * (j == 1 ? x[order[1], 1] : x[1, 1])
                    + 100 * (j == 2 ? x[order[2], 2] : x[2, 2]);
                drops[perm] = baseline - score;
            }

            expected[j] = drops.Average();
            expectedStd[j] = System.Math.Sqrt(drops.Average(d => (d - expected[j]) * (d - expected[j])));
        }

        var scores = custom.ImportanceScores?.ToArray() ?? System.Array.Empty<double>();
        var std = custom.ImportanceStd?.ToArray() ?? System.Array.Empty<double>();
        for (int j = 0; j < 3; j++)
        {
            Assert.Equal(expected[j], scores[j], 10);
            Assert.Equal(expectedStd[j], std[j], 10);
        }
    }

    [Fact]
    public void DropColumnImportance_DefaultScorerRefitsLeastSquares()
    {
        var (x, _) = Problem(20, 40, 3);
        var y = new Vector<double>(40);
        for (int i = 0; i < 40; i++) y[i] = 3 * x[i, 0] + x[i, 1];

        var selector = new AiDotNet.Preprocessing.FeatureSelection.ModelAgnostic.DropColumnImportance<double>(2);
        selector.Fit(x, y);
        var importance = selector.ImportanceScores?.ToArray() ?? System.Array.Empty<double>();

        // Reference: R^2 of the least-squares refit without feature 0 (intercept, features 1 and 2)
        var a = new double[3, 3];
        var b = new double[3];
        for (int i = 0; i < 40; i++)
        {
            var row = new[] { 1, x[i, 1], x[i, 2] };
            for (int r = 0; r < 3; r++)
            {
                for (int c = 0; c < 3; c++) a[r, c] += row[r] * row[c];
                b[r] += row[r] * y[i];
            }
        }

        var coef = SolveDense(a, b);
        double mean = Enumerable.Range(0, 40).Average(i => y[i]), res = 0, tot = 0;
        for (int i = 0; i < 40; i++)
        {
            double pred = coef[0] + coef[1] * x[i, 1] + coef[2] * x[i, 2];
            res += (y[i] - pred) * (y[i] - pred);
            tot += (y[i] - mean) * (y[i] - mean);
        }

        Assert.Equal(res / tot, importance[0], 5);
        Assert.True(System.Math.Abs(importance[2]) < 1e-6);
        Assert.Equal(new[] { 0, 1 }, selector.SelectedIndices);
    }

    [Fact]
    public void KernelShap_RecoversExactShapleyValuesOfALinearModel()
    {
        double[] w = { 1.5, -2, 0.5, 3 };
        var instance = Tensor1D(1, 2, -1, 0.5);
        var baseline = Tensor1D(0.2, -0.4, 0.3, 1);
        System.Func<Tensor<double>, Tensor<double>> predict = rows =>
        {
            var values = ToArray(rows);
            var output = new double[rows.Shape[0]];
            for (int i = 0; i < rows.Shape[0]; i++)
                for (int j = 0; j < 4; j++) output[i] += w[j] * values[i * 4 + j];
            return Tensor1D(output);
        };

        var draws = new double[40, 4];
        var rng = new System.Random(3);
        for (int c = 0; c < 40; c++) for (int j = 0; j < 4; j++) draws[c, j] = rng.NextDouble();
        var phi = ToArray(AiDotNet.Preprocessing.FeatureSelection.ModelAgnostic.SHAPSelector<double>.KernelShap(
            instance, baseline, Tensor2D(draws), predict));

        // For f(x) = w . x with a single reference point, phi_j = w_j (x_j - b_j)
        double[] x = { 1, 2, -1, 0.5 }, b = { 0.2, -0.4, 0.3, 1 };
        for (int j = 0; j < 4; j++)
            Assert.Equal(w[j] * (x[j] - b[j]), phi[j], 6);
    }

    [Fact]
    public void LIME_MatchesWeightedSurrogateWithSameSeed()
    {
        var (x, y) = Problem(21, 12, 3);
        int seed = 6, perturbations = 30;
        double width = 0.75;
        System.Func<Matrix<double>, Vector<double>> model = m =>
        {
            var v = new Vector<double>(m.Rows);
            for (int i = 0; i < m.Rows; i++) v[i] = m[i, 0] * m[i, 0] + 0.5 * m[i, 2];
            return v;
        };

        var selector = new AiDotNet.Preprocessing.FeatureSelection.ModelAgnostic.LIMESelector<double>(
            model, 2, 2, perturbations, width, seed);
        selector.Fit(x, y);

        int n = x.Rows, p = x.Columns;
        var means = Enumerable.Range(0, p).Select(j => Enumerable.Range(0, n).Average(i => x[i, j])).ToArray();
        var stds = Enumerable.Range(0, p).Select(j => System.Math.Sqrt(Enumerable.Range(0, n).Average(i => (x[i, j] - means[j]) * (x[i, j] - means[j])))).ToArray();
        var rng = new System.Random(seed);
        var explain = Enumerable.Range(0, n).OrderBy(_ => rng.Next()).Take(2).ToArray();
        var accum = new double[p];
        foreach (int e in explain)
        {
            var a = new double[p + 1, p + 1];
            var b = new double[p + 1];
            for (int s = 0; s < perturbations; s++)
            {
                var keep = new double[p];
                var z = new double[p];
                for (int j = 0; j < p; j++)
                {
                    if (rng.NextDouble() > 0.5)
                    {
                        keep[j] = 1;
                        z[j] = x[e, j];
                    }
                    else
                    {
                        z[j] = means[j] + stds[j] * (rng.NextDouble() * 2 - 1);
                    }
                }

                double d = (p - keep.Sum()) / p;
                double weight = System.Math.Exp(-d * d / (width * width));
                double f = z[0] * z[0] + 0.5 * z[2];
                var row = new double[p + 1];
                row[0] = 1;
                for (int j = 0; j < p; j++) row[j + 1] = keep[j];
                for (int r = 0; r <= p; r++)
                {
                    for (int c = 0; c <= p; c++) a[r, c] += row[r] * weight * row[c];
                    b[r] += row[r] * weight * f;
                }
            }

            for (int r = 0; r <= p; r++) a[r, r] += 1e-6;
            var coef = SolveDense(a, b);
            for (int j = 0; j < p; j++) accum[j] += System.Math.Abs(coef[j + 1]);
        }

        var importances = selector.Importances?.ToArray() ?? System.Array.Empty<double>();
        for (int j = 0; j < p; j++)
            Assert.Equal(accum[j] / 2, importances[j], 6);
    }

    // ---------------------------------------------------------------- FastICA (projection selectors)

    [Fact]
    public void FastIca_WhitensAndSeparatesIndependentSources()
    {
        // Two independent uniform sources mixed into three features
        var rng = new System.Random(61);
        int n = 2000;
        var s = new double[n, 2];
        var x = new double[n, 3];
        for (int i = 0; i < n; i++)
        {
            s[i, 0] = rng.NextDouble() * 2 - 1;
            s[i, 1] = rng.NextDouble() < 0.5 ? -1 : 1;
            x[i, 0] = s[i, 0] + 0.5 * s[i, 1];
            x[i, 1] = 0.3 * s[i, 0] - s[i, 1];
            x[i, 2] = s[i, 0] + s[i, 1];
        }

        for (int j = 0; j < 3; j++)
        {
            double mean = 0;
            for (int i = 0; i < n; i++) mean += x[i, j];
            mean /= n;
            for (int i = 0; i < n; i++) x[i, j] -= mean;
        }

        var z = AiDotNet.Preprocessing.FeatureSelection.Helpers.FastIcaOps.Whiten(Tensor2D(x), 2, out var whitening);
        var zz = ToArray(z);
        for (int a = 0; a < 2; a++)
            for (int b = 0; b < 2; b++)
            {
                double c = 0;
                for (int i = 0; i < n; i++) c += zz[i * 2 + a] * zz[i * 2 + b];
                Assert.Equal(a == b ? 1.0 : 0.0, c / n, 6);
            }

        var w = AiDotNet.Preprocessing.FeatureSelection.Helpers.FastIcaOps.Deflation(
            z, Tensor2D(new double[,] { { 0.3, -0.2 }, { 0.1, 0.4 } }), 200, 1e-9);
        var unmixing = ToArray(w);
        for (int c = 0; c < 2; c++)
        {
            var y = new double[n];
            for (int i = 0; i < n; i++) y[i] = zz[i * 2] * unmixing[c] + zz[i * 2 + 1] * unmixing[2 + c];

            // Each recovered component matches one source up to sign and scale
            double best = 0;
            for (int src = 0; src < 2; src++)
            {
                double sy = 0, ss = 0, yy = 0;
                for (int i = 0; i < n; i++)
                {
                    sy += s[i, src] * y[i];
                    ss += s[i, src] * s[i, src];
                    yy += y[i] * y[i];
                }

                best = System.Math.Max(best, System.Math.Abs(sy / System.Math.Sqrt(ss * yy)));
            }

            Assert.True(best > 0.99, $"component {c} correlates at most {best} with a source");
        }
    }
}
