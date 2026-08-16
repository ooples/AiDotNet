#nullable disable
using AiDotNet.Models.Options;
using AiDotNet.Solvers.QuadraticProgramming;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Solvers;

/// <summary>
/// Integration tests for the Sequential Minimal Optimization solver used to train support-vector
/// machines.
/// </summary>
/// <remarks>
/// CRITICAL: these check the KKT conditions of the SVM dual, which are necessary and sufficient for
/// optimality on this convex problem. If one fails, FIX THE SOLVER.
/// </remarks>
public class SequentialMinimalOptimizationIntegrationTests
{
    private const double Tolerance = 1e-3;

    private static Vector<double> V(params double[] values) => Vector<double>.FromArray(values);

    private static SequentialMinimalOptimizationSolver<double> Solver() =>
        new(new SequentialMinimalOptimizationOptions(), new Random(12345));

    /// <summary>
    /// Builds a linear-kernel evaluator over a small dataset.
    /// </summary>
    private static Func<int, int, double> LinearKernel(double[][] points)
    {
        return (i, j) =>
        {
            double sum = 0;
            for (int d = 0; d < points[i].Length; d++) sum += points[i][d] * points[j][d];
            return sum;
        };
    }

    /// <summary>
    /// Two linearly separable points, one per class. The dual has an exact answer: by symmetry
    /// α₁ = α₂ = 2/‖x₁ − x₂‖², which for points at ∓1 on a line is 2/4 = 0.5.
    /// </summary>
    [Fact]
    public void Solve_TwoSeparablePoints_MatchesTheAnalyticMultipliers()
    {
        var points = new[] { new[] { -1.0 }, new[] { 1.0 } };
        var labels = V(-1, 1);
        var linear = V(-1, -1);
        var upperBounds = V(1e6, 1e6);      // effectively hard margin

        var (alphas, _, _) = Solver().Solve(LinearKernel(points), labels, linear, upperBounds);

        Assert.Equal(0.5, alphas[0], 3);
        Assert.Equal(0.5, alphas[1], 3);
    }

    /// <summary>
    /// The equality constraint Σ yᵢαᵢ = 0 must hold exactly at the solution — it is the constraint
    /// that forces SMO to move two multipliers at a time, so a violation means the core invariant
    /// broke.
    /// </summary>
    [Fact]
    public void Solve_EqualityConstraint_HoldsAtTheSolution()
    {
        var points = new[]
        {
            new[] { 0.0, 0.0 }, new[] { 1.0, 0.5 }, new[] { 0.2, 1.1 },
            new[] { 2.0, 2.0 }, new[] { 2.5, 1.8 }, new[] { 1.9, 2.6 },
        };
        var labels = V(-1, -1, -1, 1, 1, 1);
        var linear = V(-1, -1, -1, -1, -1, -1);
        var upperBounds = V(1, 1, 1, 1, 1, 1);

        var (alphas, _, _) = Solver().Solve(LinearKernel(points), labels, linear, upperBounds);

        double sum = 0;
        for (int i = 0; i < alphas.Length; i++) sum += labels[i] * alphas[i];

        Assert.True(Math.Abs(sum) < 1e-6, $"Sum of y·alpha was {sum}, expected 0.");
    }

    /// <summary>
    /// Every multiplier must land inside its box. A value outside [0, C] is not a feasible point of
    /// the dual at all.
    /// </summary>
    [Fact]
    public void Solve_AllMultipliers_StayWithinTheirBounds()
    {
        var points = new[]
        {
            new[] { 0.0 }, new[] { 0.4 }, new[] { 1.6 }, new[] { 2.0 },
        };
        var labels = V(-1, -1, 1, 1);
        var linear = V(-1, -1, -1, -1);
        double c = 0.75;
        var upperBounds = V(c, c, c, c);

        var (alphas, _, _) = Solver().Solve(LinearKernel(points), labels, linear, upperBounds);

        for (int i = 0; i < alphas.Length; i++)
        {
            Assert.True(alphas[i] >= -1e-9, $"alpha[{i}] = {alphas[i]} is below 0.");
            Assert.True(alphas[i] <= c + 1e-9, $"alpha[{i}] = {alphas[i]} exceeds C = {c}.");
        }
    }

    /// <summary>
    /// The KKT conditions of the SVM dual, which are necessary and sufficient here. For each point,
    /// with the decision value f(xᵢ) and margin rᵢ = yᵢ·f(xᵢ):
    ///   α = 0  requires r ≥ 1 (outside the margin),
    ///   0 &lt; α &lt; C requires r = 1 (on the margin),
    ///   α = C  requires r ≤ 1 (inside the margin or misclassified).
    /// </summary>
    [Fact]
    public void Solve_Solution_SatisfiesTheDualKktConditions()
    {
        var points = new[]
        {
            new[] { 0.0, 0.0 }, new[] { 0.6, 0.3 }, new[] { 0.1, 0.8 },
            new[] { 2.2, 2.1 }, new[] { 2.6, 1.7 }, new[] { 1.8, 2.9 },
            new[] { 1.2, 1.3 },
        };
        var labels = V(-1, -1, -1, 1, 1, 1, 1);
        var linear = V(-1, -1, -1, -1, -1, -1, -1);
        double c = 5.0;
        var upperBounds = V(c, c, c, c, c, c, c);

        var kernel = LinearKernel(points);
        var (alphas, bias, _) = Solver().Solve(kernel, labels, linear, upperBounds);

        for (int i = 0; i < alphas.Length; i++)
        {
            double decision = bias;
            for (int j = 0; j < alphas.Length; j++)
            {
                decision += alphas[j] * labels[j] * kernel(j, i);
            }

            double margin = labels[i] * decision;

            if (alphas[i] <= Tolerance)
            {
                Assert.True(margin >= 1.0 - 1e-2, $"alpha[{i}]=0 but margin {margin} < 1.");
            }
            else if (alphas[i] >= c - Tolerance)
            {
                Assert.True(margin <= 1.0 + 1e-2, $"alpha[{i}]=C but margin {margin} > 1.");
            }
            else
            {
                Assert.True(
                    Math.Abs(margin - 1.0) < 1e-2,
                    $"alpha[{i}] is free but its margin {margin} is not 1.");
            }
        }
    }

    /// <summary>
    /// Non-separable data must still produce a feasible solution rather than diverging: the
    /// multipliers of the offending points saturate at C, which is what the soft margin is for.
    /// </summary>
    [Fact]
    public void Solve_NonSeparableData_SaturatesAtTheUpperBoundWithoutDiverging()
    {
        // Two identical points with opposite labels cannot be separated by anything.
        var points = new[] { new[] { 1.0 }, new[] { 1.0 } };
        var labels = V(-1, 1);
        var linear = V(-1, -1);
        double c = 0.4;
        var upperBounds = V(c, c);

        var (alphas, _, _) = Solver().Solve(LinearKernel(points), labels, linear, upperBounds);

        Assert.Equal(c, alphas[0], 3);
        Assert.Equal(c, alphas[1], 3);
    }

    /// <summary>
    /// Determinism: a seeded solver must give the same answer twice, so training is reproducible.
    /// </summary>
    [Fact]
    public void Solve_WithSeededRandom_IsReproducible()
    {
        var points = new[]
        {
            new[] { 0.0, 1.0 }, new[] { 1.0, 0.0 }, new[] { 2.0, 2.5 }, new[] { 3.0, 2.0 },
        };
        var labels = V(-1, -1, 1, 1);
        var linear = V(-1, -1, -1, -1);
        var upperBounds = V(1, 1, 1, 1);

        var first = new SequentialMinimalOptimizationSolver<double>(
            new SequentialMinimalOptimizationOptions(), new Random(7))
            .Solve(LinearKernel(points), labels, linear, upperBounds);

        var second = new SequentialMinimalOptimizationSolver<double>(
            new SequentialMinimalOptimizationOptions(), new Random(7))
            .Solve(LinearKernel(points), labels, linear, upperBounds);

        for (int i = 0; i < first.Alphas.Length; i++)
        {
            Assert.Equal(first.Alphas[i], second.Alphas[i], 10);
        }

        Assert.Equal(first.Bias, second.Bias, 10);
    }

    #region Validation

    [Fact]
    public void Solve_NullKernel_Throws()
    {
        Assert.Throws<ArgumentNullException>(() =>
            Solver().Solve(null, V(1, -1), V(-1, -1), V(1, 1)));
    }

    [Fact]
    public void Solve_MismatchedLengths_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            Solver().Solve((i, j) => 1.0, V(1, -1), V(-1), V(1, 1)));
    }

    #endregion
}
