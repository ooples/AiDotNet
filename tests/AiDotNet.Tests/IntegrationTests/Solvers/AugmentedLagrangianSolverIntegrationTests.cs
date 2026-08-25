#nullable disable
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Solvers.Constrained;
using AiDotNet.Solvers.LinearProgramming;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Solvers;

/// <summary>
/// Integration tests for the augmented Lagrangian (method of multipliers) solver.
/// </summary>
/// <remarks>
/// CRITICAL: Every expected point and multiplier here is derived by hand from the KKT conditions,
/// which for these small problems reduce to a linear system with a unique solution. The multipliers
/// are checked as well as the points — a solver that lands on the right answer with the wrong prices
/// has not actually solved the dual, and the multipliers are what an augmented Lagrangian method
/// exists to produce. If a test fails, FIX THE SOLVER — do not relax the assertion.
/// </remarks>
public class AugmentedLagrangianSolverIntegrationTests
{
    private const double Tolerance = 1e-5;

    private static Vector<double> V(params double[] values) => Vector<double>.FromArray(values);

    private static Matrix<double> Row(params double[] values)
    {
        var matrix = new Matrix<double>(1, values.Length);
        for (int c = 0; c < values.Length; c++) matrix[0, c] = values[c];
        return matrix;
    }

    private static AugmentedLagrangianSolver<double> Solver() => new();

    /// <summary>
    /// minimize x² + y² subject to x + y = 1.
    /// The KKT system is 2x + λ = 0, 2y + λ = 0, x + y = 1, so x = y = 1/2 and λ = −1.
    /// </summary>
    [Fact]
    public void Solve_SumEqualityOnCircle_FindsTheClosestPointAndItsPrice()
    {
        var problem = new ConstrainedProblem<double>(
            objective: p => (p[0] * p[0] + p[1] * p[1], V(2 * p[0], 2 * p[1])),
            equalityConstraints: p => (V(p[0] + p[1] - 1.0), Row(1.0, 1.0)));

        var solution = Solver().Solve(problem, V(0.0, 0.0));

        Assert.Equal(LinearProgramStatus.Optimal, solution.Status);
        Assert.Equal(0.5, solution.Solution[0], 5);
        Assert.Equal(0.5, solution.Solution[1], 5);
        Assert.Equal(0.5, solution.ObjectiveValue, 5);
        Assert.Equal(-1.0, solution.EqualityMultipliers[0], 4);
        Assert.True(solution.ConstraintViolation < Tolerance);
    }

    /// <summary>
    /// The same objective with a starting point far from feasible. An augmented Lagrangian method
    /// does not require a feasible start, which is most of why it is used.
    /// </summary>
    [Fact]
    public void Solve_FromAnInfeasibleStart_StillConverges()
    {
        var problem = new ConstrainedProblem<double>(
            objective: p => (p[0] * p[0] + p[1] * p[1], V(2 * p[0], 2 * p[1])),
            equalityConstraints: p => (V(p[0] + p[1] - 1.0), Row(1.0, 1.0)));

        var solution = Solver().Solve(problem, V(-40.0, 75.0));

        Assert.Equal(LinearProgramStatus.Optimal, solution.Status);
        Assert.Equal(0.5, solution.Solution[0], 5);
        Assert.Equal(0.5, solution.Solution[1], 5);
    }

    /// <summary>
    /// minimize (x − 3)² + (y − 2)² subject to x + y ≤ 1.
    /// The unconstrained minimum (3, 2) violates the limit, so the constraint binds and the answer
    /// is the projection of (3, 2) onto the line x + y = 1: (1, 0). The KKT system
    /// 2(x−3) + μ = 0, 2(y−2) + μ = 0, x + y = 1 gives μ = 4.
    /// </summary>
    [Fact]
    public void Solve_BindingInequality_ProjectsOntoTheBoundary()
    {
        var problem = new ConstrainedProblem<double>(
            objective: p => (
                (p[0] - 3) * (p[0] - 3) + (p[1] - 2) * (p[1] - 2),
                V(2 * (p[0] - 3), 2 * (p[1] - 2))),
            inequalityConstraints: p => (V(p[0] + p[1] - 1.0), Row(1.0, 1.0)));

        var solution = Solver().Solve(problem, V(0.0, 0.0));

        Assert.Equal(LinearProgramStatus.Optimal, solution.Status);
        Assert.Equal(1.0, solution.Solution[0], 4);
        Assert.Equal(0.0, solution.Solution[1], 4);
        Assert.Equal(4.0, solution.InequalityMultipliers[0], 4);
    }

    /// <summary>
    /// The same problem with a limit the unconstrained minimum already satisfies. The constraint
    /// must then not bend the answer at all, and complementary slackness requires its multiplier to
    /// be zero — the property Rockafellar's max(0, ·) form exists to deliver.
    /// </summary>
    [Fact]
    public void Solve_SlackInequality_LeavesTheAnswerAndPriceUntouched()
    {
        var problem = new ConstrainedProblem<double>(
            objective: p => (
                (p[0] - 3) * (p[0] - 3) + (p[1] - 2) * (p[1] - 2),
                V(2 * (p[0] - 3), 2 * (p[1] - 2))),
            inequalityConstraints: p => (V(p[0] + p[1] - 100.0), Row(1.0, 1.0)));

        var solution = Solver().Solve(problem, V(0.0, 0.0));

        Assert.Equal(LinearProgramStatus.Optimal, solution.Status);
        Assert.Equal(3.0, solution.Solution[0], 4);
        Assert.Equal(2.0, solution.Solution[1], 4);
        Assert.Equal(0.0, solution.InequalityMultipliers[0], 6);
    }

    /// <summary>
    /// Equality and inequality together: minimize x² + y² subject to x + y = 2 and x ≤ 0.5.
    /// The equality alone would give (1, 1), which violates x ≤ 0.5, so both constraints bind and
    /// the answer is (0.5, 1.5).
    /// </summary>
    [Fact]
    public void Solve_BothConstraintKinds_SatisfiesBoth()
    {
        var problem = new ConstrainedProblem<double>(
            objective: p => (p[0] * p[0] + p[1] * p[1], V(2 * p[0], 2 * p[1])),
            equalityConstraints: p => (V(p[0] + p[1] - 2.0), Row(1.0, 1.0)),
            inequalityConstraints: p => (V(p[0] - 0.5), Row(1.0, 0.0)));

        var solution = Solver().Solve(problem, V(0.0, 0.0));

        Assert.Equal(LinearProgramStatus.Optimal, solution.Status);
        Assert.Equal(0.5, solution.Solution[0], 4);
        Assert.Equal(1.5, solution.Solution[1], 4);
        Assert.True(
            solution.InequalityMultipliers[0] > 0.0,
            "A binding inequality must carry a strictly positive multiplier.");
    }

    /// <summary>
    /// Two equality constraints that between them pin the answer completely:
    ///   minimize x² + y² + z² subject to x + y + z = 3 and x − y = 1.
    /// The KKT conditions give 2x + λ₁ + λ₂ = 0, 2y + λ₁ − λ₂ = 0, 2z + λ₁ = 0, which with the two
    /// constraints solve to λ₂ = −1, λ₁ = −2 and hence (3/2, 1/2, 1) with an objective of 7/2.
    /// Note that many points satisfy both constraints — (4/3, 1/3, 4/3) does, with an objective of
    /// 11/3 — so satisfying them is not on its own evidence of optimality.
    /// </summary>
    [Fact]
    public void Solve_TwoEqualities_PinsTheAnswer()
    {
        var problem = new ConstrainedProblem<double>(
            objective: p => (
                p[0] * p[0] + p[1] * p[1] + p[2] * p[2],
                V(2 * p[0], 2 * p[1], 2 * p[2])),
            equalityConstraints: p =>
            {
                var jacobian = new Matrix<double>(2, 3);
                jacobian[0, 0] = 1.0; jacobian[0, 1] = 1.0; jacobian[0, 2] = 1.0;
                jacobian[1, 0] = 1.0; jacobian[1, 1] = -1.0; jacobian[1, 2] = 0.0;
                return (V(p[0] + p[1] + p[2] - 3.0, p[0] - p[1] - 1.0), jacobian);
            });

        var solution = Solver().Solve(problem, V(0.0, 0.0, 0.0));

        Assert.Equal(LinearProgramStatus.Optimal, solution.Status);
        Assert.Equal(1.5, solution.Solution[0], 4);
        Assert.Equal(0.5, solution.Solution[1], 4);
        Assert.Equal(1.0, solution.Solution[2], 4);
        Assert.Equal(3.5, solution.ObjectiveValue, 4);
    }

    /// <summary>
    /// A nonlinear constraint, where the linear-programming machinery does not apply at all:
    ///   minimize x + y subject to x² + y² = 2. By symmetry the answer is (−1, −1).
    /// </summary>
    [Fact]
    public void Solve_NonlinearConstraint_FindsTheOptimumOnTheCircle()
    {
        var problem = new ConstrainedProblem<double>(
            objective: p => (p[0] + p[1], V(1.0, 1.0)),
            equalityConstraints: p => (
                V(p[0] * p[0] + p[1] * p[1] - 2.0),
                Row(2 * p[0], 2 * p[1])));

        var solution = Solver().Solve(problem, V(-0.5, -1.2));

        Assert.Equal(LinearProgramStatus.Optimal, solution.Status);
        Assert.Equal(-1.0, solution.Solution[0], 4);
        Assert.Equal(-1.0, solution.Solution[1], 4);
        Assert.Equal(-2.0, solution.ObjectiveValue, 4);
    }

    /// <summary>
    /// Constrained Rosenbrock: minimize the Rosenbrock function subject to x + y = 1.
    /// Substituting y = 1 − x leaves g(x) = (1 − x)² + 100(1 − x − x²)², whose stationary point is
    /// x* = 0.6187956190750254 (verified independently: g'(x*) = 4·10⁻¹⁴).
    ///
    /// The tempting answer is the golden-ratio conjugate (√5 − 1)/2 ≈ 0.6180340, where the line
    /// crosses the valley floor y = x² and the second term vanishes. It is not the optimum: g there
    /// is 0.1458980 against 0.1456070 at x*. Zeroing the valley term costs more in the (1 − x)² term
    /// than it saves, so the true optimum sits slightly off the valley floor — which is exactly the
    /// kind of trade-off a constrained solver has to get right and an eyeballed one gets wrong.
    /// </summary>
    [Fact]
    public void Solve_ConstrainedRosenbrock_TradesValleyDepthAgainstDistance()
    {
        var problem = new ConstrainedProblem<double>(
            objective: p =>
            {
                double a = 1 - p[0];
                double b = p[1] - p[0] * p[0];
                double value = a * a + 100 * b * b;
                var gradient = V(-2 * a - 400 * p[0] * b, 200 * b);
                return (value, gradient);
            },
            equalityConstraints: p => (V(p[0] + p[1] - 1.0), Row(1.0, 1.0)));

        var solution = Solver().Solve(problem, V(0.0, 0.0));

        Assert.Equal(LinearProgramStatus.Optimal, solution.Status);

        const double ExpectedX = 0.6187956190750254;
        Assert.Equal(ExpectedX, solution.Solution[0], 6);
        Assert.Equal(1.0 - ExpectedX, solution.Solution[1], 6);
        Assert.Equal(0.1456070180282598, solution.ObjectiveValue, 6);
        Assert.True(solution.ConstraintViolation < Tolerance);
    }

    /// <summary>
    /// A problem with no constraints must reduce to a plain unconstrained minimization rather than
    /// be rejected — the degenerate case is a legitimate input.
    /// </summary>
    [Fact]
    public void Solve_NoConstraints_ReducesToUnconstrainedMinimization()
    {
        var problem = new ConstrainedProblem<double>(
            objective: p => (
                (p[0] - 2) * (p[0] - 2) + (p[1] + 3) * (p[1] + 3),
                V(2 * (p[0] - 2), 2 * (p[1] + 3))));

        var solution = Solver().Solve(problem, V(0.0, 0.0));

        Assert.Equal(LinearProgramStatus.Optimal, solution.Status);
        Assert.Equal(2.0, solution.Solution[0], 4);
        Assert.Equal(-3.0, solution.Solution[1], 4);
        Assert.Null(solution.EqualityMultipliers);
    }

    /// <summary>
    /// The multipliers must satisfy the stationarity condition ∇f + Σλ∇h = 0 at the returned point,
    /// which is the definition of a KKT point and a check independent of the hand-derived answer.
    /// </summary>
    [Fact]
    public void Solve_ReturnedMultipliers_SatisfyStationarity()
    {
        var problem = new ConstrainedProblem<double>(
            objective: p => (
                3 * p[0] * p[0] + 2 * p[1] * p[1],
                V(6 * p[0], 4 * p[1])),
            equalityConstraints: p => (V(2 * p[0] + p[1] - 4.0), Row(2.0, 1.0)));

        var solution = Solver().Solve(problem, V(0.0, 0.0));

        Assert.Equal(LinearProgramStatus.Optimal, solution.Status);

        double lambda = solution.EqualityMultipliers[0];
        double stationarityX = 6 * solution.Solution[0] + lambda * 2.0;
        double stationarityY = 4 * solution.Solution[1] + lambda * 1.0;

        Assert.Equal(0.0, stationarityX, 4);
        Assert.Equal(0.0, stationarityY, 4);
    }

    /// <summary>
    /// A caller may substitute any unconstrained optimizer for the subproblems, and a different
    /// inner solver must reach the same constrained optimum — the outer method is what determines
    /// the answer.
    /// </summary>
    [Fact]
    public void Solve_WithASubstitutedInnerOptimizer_ReachesTheSameOptimum()
    {
        var problem = new ConstrainedProblem<double>(
            objective: p => (p[0] * p[0] + p[1] * p[1], V(2 * p[0], 2 * p[1])),
            equalityConstraints: p => (V(p[0] + p[1] - 1.0), Row(1.0, 1.0)));

        var solver = new AugmentedLagrangianSolver<double>(
            new AugmentedLagrangianSolverOptions(),
            new AdamOptimizer<double, Tensor<double>, Tensor<double>>(null));

        var solution = solver.Solve(problem, V(0.0, 0.0));

        Assert.Equal(LinearProgramStatus.Optimal, solution.Status);
        Assert.Equal(0.5, solution.Solution[0], 4);
        Assert.Equal(0.5, solution.Solution[1], 4);
    }

    #region Configuration and validation

    [Fact]
    public void Constructor_NullOptions_Throws()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new AugmentedLagrangianSolver<double>(null));
    }

    [Fact]
    public void Constructor_NullInnerOptimizer_Throws()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new AugmentedLagrangianSolver<double>(new AugmentedLagrangianSolverOptions(), null));
    }

    [Fact]
    public void Constructor_PenaltyGrowthFactorOfOne_Throws()
    {
        // A factor of one never tightens the penalty, so an infeasible iterate would never be
        // driven back — silently accepting it would mean never converging.
        Assert.Throws<ArgumentException>(() =>
            new AugmentedLagrangianSolver<double>(
                new AugmentedLagrangianSolverOptions { PenaltyGrowthFactor = 1.0 }));
    }

    [Fact]
    public void Constructor_MaximumPenaltyBelowInitial_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            new AugmentedLagrangianSolver<double>(
                new AugmentedLagrangianSolverOptions
                {
                    InitialPenalty = 100.0,
                    MaximumPenalty = 1.0,
                }));
    }

    [Fact]
    public void Solve_NullProblem_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => Solver().Solve(null, V(0.0)));
    }

    [Fact]
    public void Solve_EmptyStartingPoint_Throws()
    {
        var problem = new ConstrainedProblem<double>(p => (0.0, new Vector<double>(0)));
        Assert.Throws<ArgumentException>(() => Solver().Solve(problem, new Vector<double>(0)));
    }

    /// <summary>
    /// A problem whose constraints cannot be satisfied must report that it ran out of iterations
    /// with a violation still on the books, rather than claim an optimum it never reached.
    /// </summary>
    [Fact]
    public void Solve_ContradictoryConstraints_ReportsTheRemainingViolation()
    {
        var problem = new ConstrainedProblem<double>(
            objective: p => (p[0] * p[0], V(2 * p[0])),
            equalityConstraints: p =>
            {
                var jacobian = new Matrix<double>(2, 1);
                jacobian[0, 0] = 1.0;
                jacobian[1, 0] = 1.0;
                return (V(p[0] - 1.0, p[0] - 5.0), jacobian);
            });

        var solution = new AugmentedLagrangianSolver<double>(
            new AugmentedLagrangianSolverOptions { MaxOuterIterations = 20 })
            .Solve(problem, V(0.0));

        Assert.Equal(LinearProgramStatus.IterationLimit, solution.Status);
        Assert.True(
            solution.ConstraintViolation > 1.0,
            $"Contradictory constraints cannot both be met, so a substantial violation must be " +
            $"reported; got {solution.ConstraintViolation}.");
    }

    #endregion
}
