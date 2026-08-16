#nullable disable
using AiDotNet.Models.Options;
using AiDotNet.Solvers;
using AiDotNet.Solvers.LinearProgramming;
using AiDotNet.Solvers.QuadraticProgramming;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Solvers;

/// <summary>
/// Integration tests for the active-set quadratic-programming solver.
/// </summary>
/// <remarks>
/// CRITICAL: expected values are derived analytically (by solving the KKT system by hand) or
/// verified with an independent KKT residual check, not recorded from the solver's output. If a
/// test fails, FIX THE SOLVER.
/// </remarks>
public class QuadraticProgramSolverIntegrationTests
{
    private static Vector<double> V(params double[] values) => Vector<double>.FromArray(values);

    private static Matrix<double> M(double[,] values)
    {
        var matrix = new Matrix<double>(values.GetLength(0), values.GetLength(1));
        for (int r = 0; r < values.GetLength(0); r++)
        {
            for (int c = 0; c < values.GetLength(1); c++) matrix[r, c] = values[r, c];
        }

        return matrix;
    }

    private static ActiveSetQuadraticProgramSolver<double> Solver() => new();

    /// <summary>
    /// Unconstrained: minimizing ½xᵀQx + cᵀx puts the answer at x = −Q⁻¹c.
    /// With Q = I and c = (−3, 5) that is exactly (3, −5), objective −17.
    /// </summary>
    [Fact]
    public void Solve_Unconstrained_FindsClosedFormMinimum()
    {
        var program = new QuadraticProgram<double>(
            quadratic: M(new[,] { { 1.0, 0.0 }, { 0.0, 1.0 } }),
            linear: V(-3, 5));

        var solution = Solver().Solve(program);

        Assert.Equal(LinearProgramStatus.Optimal, solution.Status);
        Assert.Equal(3.0, solution.Solution[0], 6);
        Assert.Equal(-5.0, solution.Solution[1], 6);
        Assert.Equal(-17.0, solution.ObjectiveValue, 6);
    }

    /// <summary>
    /// Equality-constrained: minimize ½(x² + y²) subject to x + y = 2.
    /// Symmetry (and the KKT system) give x = y = 1, objective 1, multiplier −1.
    /// </summary>
    [Fact]
    public void Solve_EqualityConstrained_FindsAnalyticSolution()
    {
        var program = new QuadraticProgram<double>(
            quadratic: M(new[,] { { 1.0, 0.0 }, { 0.0, 1.0 } }),
            linear: V(0, 0),
            equalityMatrix: M(new[,] { { 1.0, 1.0 } }),
            equalityBounds: V(2));

        var solution = Solver().Solve(program);

        Assert.Equal(LinearProgramStatus.Optimal, solution.Status);
        Assert.Equal(1.0, solution.Solution[0], 6);
        Assert.Equal(1.0, solution.Solution[1], 6);
        Assert.Equal(1.0, solution.ObjectiveValue, 6);
    }

    /// <summary>
    /// An inactive constraint must leave the answer at the unconstrained minimum. Here the
    /// unconstrained optimum (1, 1) satisfies x + y ≤ 10 comfortably, so the constraint does
    /// nothing and its multiplier must be zero.
    /// </summary>
    [Fact]
    public void Solve_NonBindingConstraint_LeavesUnconstrainedOptimum()
    {
        var program = new QuadraticProgram<double>(
            quadratic: M(new[,] { { 1.0, 0.0 }, { 0.0, 1.0 } }),
            linear: V(-1, -1),
            inequalityMatrix: M(new[,] { { 1.0, 1.0 } }),
            inequalityBounds: V(10));

        var solution = Solver().Solve(program);

        Assert.Equal(LinearProgramStatus.Optimal, solution.Status);
        Assert.Equal(1.0, solution.Solution[0], 6);
        Assert.Equal(1.0, solution.Solution[1], 6);
        Assert.Equal(0.0, solution.InequalityMultipliers[0], 6);
    }

    /// <summary>
    /// A binding constraint pulls the answer off the unconstrained optimum. The unconstrained
    /// minimum of ½(x² + y²) − 4x − 4y is (4, 4), but x + y ≤ 2 forces the closest feasible point,
    /// which by symmetry is (1, 1).
    /// </summary>
    [Fact]
    public void Solve_BindingConstraint_MovesToConstrainedOptimum()
    {
        var program = new QuadraticProgram<double>(
            quadratic: M(new[,] { { 1.0, 0.0 }, { 0.0, 1.0 } }),
            linear: V(-4, -4),
            inequalityMatrix: M(new[,] { { 1.0, 1.0 } }),
            inequalityBounds: V(2));

        var solution = Solver().Solve(program);

        Assert.Equal(LinearProgramStatus.Optimal, solution.Status);
        Assert.Equal(1.0, solution.Solution[0], 6);
        Assert.Equal(1.0, solution.Solution[1], 6);
        Assert.True(
            solution.InequalityMultipliers[0] > 1e-6,
            "A binding constraint must carry a positive multiplier.");
    }

    /// <summary>
    /// Non-negative least squares, the shape SuperLearner's meta-learner and NMF both need. With
    /// X = (1, −1), the unconstrained optimum is (y₀ − y₁) / 2. Thus y = (−1, 1)
    /// gives −1 and is clipped to 0, while y = (2, −2) gives 2 and remains unchanged.
    /// </summary>
    [Theory]
    [InlineData(-1.0, 1.0, 0.0)]     // unconstrained optimum is negative -> clipped to the bound
    [InlineData(2.0, -2.0, 2.0)]     // unconstrained optimum is positive -> unchanged
    public async Task Solve_NonNegativeLeastSquares_RespectsBound(double y0, double y1, double expected)
    {
        await Task.Yield();
        // Q = XᵀX, c = -Xᵀy for X = [[1], [-1]].
        double xtx = 2.0;
        double xty = y0 - y1;

        var program = new QuadraticProgram<double>(
            quadratic: M(new[,] { { xtx } }),
            linear: V(-xty),
            lowerBounds: V(0));

        var solution = Solver().Solve(program);

        Assert.Equal(LinearProgramStatus.Optimal, solution.Status);
        Assert.Equal(expected, solution.Solution[0], 6);
        Assert.True(solution.Solution[0] >= -1e-9, "The non-negativity bound was violated.");
    }

    /// <summary>
    /// The shape of the gradient-episodic-memory projection: minimize ½λᵀGλ − aᵀλ subject to
    /// λ ≥ 0. With G = I and a = (−1, 2) the unconstrained optimum is (−1, 2); the first component
    /// is clipped to 0 and the second stays at 2.
    /// </summary>
    [Fact]
    public void Solve_BoundConstrainedDualQp_ClipsOnlyTheViolatingComponent()
    {
        var program = new QuadraticProgram<double>(
            quadratic: M(new[,] { { 1.0, 0.0 }, { 0.0, 1.0 } }),
            linear: V(1, -2),
            lowerBounds: V(0, 0));

        var solution = Solver().Solve(program);

        Assert.Equal(LinearProgramStatus.Optimal, solution.Status);
        Assert.Equal(0.0, solution.Solution[0], 6);
        Assert.Equal(2.0, solution.Solution[1], 6);
    }

    /// <summary>
    /// Portfolio shape: minimum-variance weights that must sum to one and stay non-negative.
    /// With a diagonal covariance the analytic minimum-variance weights are proportional to the
    /// inverse variances: for variances (1, 4) that is (4/5, 1/5).
    /// </summary>
    [Fact]
    public void Solve_MinimumVariancePortfolio_MatchesAnalyticWeights()
    {
        var program = new QuadraticProgram<double>(
            quadratic: M(new[,] { { 1.0, 0.0 }, { 0.0, 4.0 } }),
            linear: V(0, 0),
            equalityMatrix: M(new[,] { { 1.0, 1.0 } }),
            equalityBounds: V(1),
            lowerBounds: V(0, 0));

        var solution = Solver().Solve(program);

        Assert.Equal(LinearProgramStatus.Optimal, solution.Status);
        Assert.Equal(0.8, solution.Solution[0], 6);
        Assert.Equal(0.2, solution.Solution[1], 6);
    }

    /// <summary>
    /// Contradictory constraints have no feasible point, which the feasibility phase must detect.
    /// </summary>
    [Fact]
    public void Solve_ContradictoryConstraints_ReportsInfeasible()
    {
        var program = new QuadraticProgram<double>(
            quadratic: M(new[,] { { 1.0 } }),
            linear: V(0),
            inequalityMatrix: M(new[,] { { 1.0 }, { -1.0 } }),
            inequalityBounds: V(1, -5));                       // x <= 1 and x >= 5

        var solution = Solver().Solve(program);

        Assert.Equal(LinearProgramStatus.Infeasible, solution.Status);
        Assert.Null(solution.Solution);
    }

    #region KKT certification

    /// <summary>
    /// The strongest available check: the returned point and multipliers are fed to an independent
    /// KKT residual evaluator. For a convex problem all four residuals being zero is necessary AND
    /// sufficient for optimality, so this certifies the answer without trusting the solver.
    /// </summary>
    [Fact]
    public void Solve_ReturnedPointAndMultipliers_SatisfyKktConditions()
    {
        var quadratic = M(new[,] { { 2.0, 0.5 }, { 0.5, 1.0 } });
        var linear = V(-4, -4);
        var inequalityMatrix = M(new[,] { { 1.0, 1.0 }, { 1.0, 0.0 } });
        var inequalityBounds = V(2, 1.5);

        var program = new QuadraticProgram<double>(
            quadratic, linear, inequalityMatrix, inequalityBounds);

        var solution = Solver().Solve(program);

        Assert.Equal(LinearProgramStatus.Optimal, solution.Status);

        var residual = KktConditions.Evaluate(
            quadratic, linear, solution.Solution,
            inequalityMatrix, inequalityBounds, solution.InequalityMultipliers);

        Assert.True(residual.Stationarity < 1e-6, $"Stationarity residual {residual.Stationarity}");
        Assert.True(residual.PrimalFeasibility < 1e-6, $"Primal residual {residual.PrimalFeasibility}");
        Assert.True(residual.DualFeasibility < 1e-6, $"Dual residual {residual.DualFeasibility}");
        Assert.True(
            residual.ComplementarySlackness < 1e-6,
            $"Complementary slackness residual {residual.ComplementarySlackness}");
    }

    /// <summary>
    /// The KKT evaluator must actually reject a wrong answer, or the test above proves nothing.
    /// </summary>
    [Fact]
    public void KktConditions_RejectAPointThatIsNotOptimal()
    {
        var quadratic = M(new[,] { { 1.0, 0.0 }, { 0.0, 1.0 } });
        var linear = V(-4, -4);

        // The true unconstrained optimum is (4, 4); (0, 0) is not stationary.
        var residual = KktConditions.Evaluate(quadratic, linear, V(0, 0));

        Assert.True(residual.Stationarity > 1.0, "A clearly non-optimal point was certified.");
    }

    /// <summary>
    /// A point that violates a constraint must be caught by the primal-feasibility residual, even
    /// if it happens to look stationary.
    /// </summary>
    [Fact]
    public void KktConditions_DetectAnInfeasiblePoint()
    {
        var quadratic = M(new[,] { { 1.0 } });
        var linear = V(0);
        var inequalityMatrix = M(new[,] { { 1.0 } });
        var inequalityBounds = V(1);

        var residual = KktConditions.Evaluate(
            quadratic, linear, V(5), inequalityMatrix, inequalityBounds, V(0));

        Assert.True(residual.PrimalFeasibility > 3.0, "A constraint violation went undetected.");
    }

    #endregion

    #region Validation

    [Fact]
    public void Solve_NullProgram_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => Solver().Solve(null));
    }

    [Fact]
    public void Construct_NonSquareQuadratic_Throws()
    {
        Assert.Throws<ArgumentException>(() => new QuadraticProgram<double>(
            M(new[,] { { 1.0, 0.0 } }), V(1, 1)));
    }

    [Fact]
    public void Construct_QuadraticSizeMismatch_Throws()
    {
        Assert.Throws<ArgumentException>(() => new QuadraticProgram<double>(
            M(new[,] { { 1.0 } }), V(1, 1)));
    }

    [Fact]
    public void Construct_EmptyObjective_Throws()
    {
        Assert.Throws<ArgumentException>(() => new QuadraticProgram<double>(
            new Matrix<double>(0, 0), new Vector<double>(0)));
    }

    #endregion
}
