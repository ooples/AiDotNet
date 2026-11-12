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
}
