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
}
