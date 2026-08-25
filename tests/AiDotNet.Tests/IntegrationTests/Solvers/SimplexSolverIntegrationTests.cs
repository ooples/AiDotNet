#nullable disable
using AiDotNet.Models.Options;
using AiDotNet.Solvers.LinearProgramming;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Solvers;

/// <summary>
/// Integration tests for the two-phase simplex linear-programming solver.
/// </summary>
/// <remarks>
/// CRITICAL: Every expected value here is either hand-solvable or fixed by linear-programming
/// theory (complementary slackness, strong duality). If a test fails, FIX THE SOLVER — do not
/// relax the assertion to match the output.
/// </remarks>
public class SimplexSolverIntegrationTests
{
    private const double Tolerance = 1e-6;

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

    private static SimplexSolver<double> Solver() => new();

    #region Known optima

    /// <summary>
    /// The textbook furniture problem:
    ///   maximize 30·t + 20·c  subject to  4t + 3c ≤ 240,  2t + c ≤ 100,  t, c ≥ 0.
    /// Solving the two binding constraints simultaneously gives t = 30, c = 40, profit 1700.
    /// Every other vertex — (0,0), (50,0), (0,80) — gives 0, 1500 and 1600 respectively.
    /// </summary>
    [Fact]
    public void Solve_FurnitureProblem_FindsKnownVertex()
    {
        var program = new LinearProgram<double>(
            objective: V(-30, -20),                       // maximize by minimizing the negation
            inequalityMatrix: M(new[,] { { 4.0, 3.0 }, { 2.0, 1.0 } }),
            inequalityBounds: V(240, 100));

        var solution = Solver().Solve(program);

        Assert.Equal(LinearProgramStatus.Optimal, solution.Status);
        Assert.Equal(30.0, solution.Solution[0], 6);
        Assert.Equal(40.0, solution.Solution[1], 6);
        Assert.Equal(-1700.0, solution.ObjectiveValue, 6);
    }

    /// <summary>
    /// Minimizing with a binding equality:
    ///   minimize 2x + 3y  subject to  x + y = 10,  x ≤ 4,  x, y ≥ 0.
    /// y is the more expensive variable, so push x to its ceiling: x = 4, y = 6, cost 26.
    /// </summary>
    [Fact]
    public void Solve_EqualityConstraint_FindsKnownOptimum()
    {
        var program = new LinearProgram<double>(
            objective: V(2, 3),
            inequalityMatrix: M(new[,] { { 1.0, 0.0 } }),
            inequalityBounds: V(4),
            equalityMatrix: M(new[,] { { 1.0, 1.0 } }),
            equalityBounds: V(10));

        var solution = Solver().Solve(program);

        Assert.Equal(LinearProgramStatus.Optimal, solution.Status);
        Assert.Equal(4.0, solution.Solution[0], 6);
        Assert.Equal(6.0, solution.Solution[1], 6);
        Assert.Equal(26.0, solution.ObjectiveValue, 6);
    }

    /// <summary>
    /// A greater-than constraint forces phase one to do real work: the all-slack basis is
    /// infeasible, so artificial variables are needed to find a starting vertex.
    ///   minimize x + y  subject to  x + 2y ≥ 6,  x, y ≥ 0.
    /// y is twice as effective per unit cost, so put everything into y: x = 0, y = 3, cost 3.
    /// </summary>
    [Fact]
    public void Solve_GreaterThanConstraint_RunsPhaseOneAndFindsOptimum()
    {
        var program = new LinearProgram<double>(
            objective: V(1, 1),
            inequalityMatrix: M(new[,] { { -1.0, -2.0 } }),   // -x - 2y <= -6  ==  x + 2y >= 6
            inequalityBounds: V(-6));

        var solution = Solver().Solve(program);

        Assert.Equal(LinearProgramStatus.Optimal, solution.Status);
        Assert.Equal(0.0, solution.Solution[0], 6);
        Assert.Equal(3.0, solution.Solution[1], 6);
        Assert.Equal(3.0, solution.ObjectiveValue, 6);
    }

    /// <summary>
    /// Explicit variable bounds must be honoured without the caller adding constraint rows.
    ///   minimize -x - y  subject to  1 ≤ x ≤ 3,  2 ≤ y ≤ 5.
    /// The objective pushes both to their ceilings: x = 3, y = 5, value -8.
    /// </summary>
    [Fact]
    public void Solve_WithVariableBounds_RespectsBoxAndFindsCorner()
    {
        var program = new LinearProgram<double>(
            objective: V(-1, -1),
            lowerBounds: V(1, 2),
            upperBounds: V(3, 5));

        var solution = Solver().Solve(program);

        Assert.Equal(LinearProgramStatus.Optimal, solution.Status);
        Assert.Equal(3.0, solution.Solution[0], 6);
        Assert.Equal(5.0, solution.Solution[1], 6);
        Assert.Equal(-8.0, solution.ObjectiveValue, 6);
    }

    /// <summary>
    /// A free variable (unbounded in both directions) has to be split into positive and negative
    /// parts internally. The answer here is negative, which a solver that silently assumed
    /// non-negativity could never produce.
    ///   minimize x  subject to  x ≥ -7, x free.
    /// </summary>
    [Fact]
    public void Solve_FreeVariable_CanReturnNegativeValue()
    {
        var program = new LinearProgram<double>(
            objective: V(1),
            inequalityMatrix: M(new[,] { { -1.0 } }),        // -x <= 7  ==  x >= -7
            inequalityBounds: V(7),
            lowerBounds: V(double.NegativeInfinity),
            upperBounds: V(double.PositiveInfinity));

        var solution = Solver().Solve(program);

        Assert.Equal(LinearProgramStatus.Optimal, solution.Status);
        Assert.Equal(-7.0, solution.Solution[0], 6);
        Assert.Equal(-7.0, solution.ObjectiveValue, 6);
    }

    #endregion

    #region Statuses

    /// <summary>
    /// Contradictory constraints (x ≥ 5 and x ≤ 2) have no feasible point at all. Phase one must
    /// detect this rather than returning a wrong answer.
    /// </summary>
    [Fact]
    public void Solve_ContradictoryConstraints_ReportsInfeasible()
    {
        var program = new LinearProgram<double>(
            objective: V(1),
            inequalityMatrix: M(new[,] { { -1.0 }, { 1.0 } }),
            inequalityBounds: V(-5, 2));                      // x >= 5 and x <= 2

        var solution = Solver().Solve(program);

        Assert.Equal(LinearProgramStatus.Infeasible, solution.Status);
        Assert.Null(solution.Solution);
    }

    /// <summary>
    /// Maximizing an unconstrained-above variable can improve forever, which is a modelling error
    /// the solver has to name rather than silently truncate.
    /// </summary>
    [Fact]
    public void Solve_ObjectiveImprovesForever_ReportsUnbounded()
    {
        var program = new LinearProgram<double>(
            objective: V(-1),                                 // maximize x
            inequalityMatrix: M(new[,] { { -1.0 } }),         // only a LOWER bound: x >= -3
            inequalityBounds: V(3));

        var solution = Solver().Solve(program);

        Assert.Equal(LinearProgramStatus.Unbounded, solution.Status);
        Assert.Null(solution.Solution);
    }

    /// <summary>
    /// A degenerate problem — more constraints meeting at the optimal vertex than the dimension
    /// requires — is where Dantzig's rule can cycle. Bland's rule must take over and terminate.
    ///   minimize -x - y  subject to  x + y ≤ 4,  x + y ≤ 4 (repeated),  x ≤ 4,  y ≤ 4.
    /// Optimal value -4 along the whole edge x + y = 4.
    /// </summary>
    [Fact]
    public void Solve_DegenerateProblem_TerminatesWithCorrectObjective()
    {
        var program = new LinearProgram<double>(
            objective: V(-1, -1),
            inequalityMatrix: M(new[,]
            {
                { 1.0, 1.0 },
                { 1.0, 1.0 },
                { 1.0, 0.0 },
                { 0.0, 1.0 },
            }),
            inequalityBounds: V(4, 4, 4, 4));

        var solution = Solver().Solve(program);

        Assert.Equal(LinearProgramStatus.Optimal, solution.Status);
        Assert.Equal(-4.0, solution.ObjectiveValue, 6);
        Assert.Equal(4.0, solution.Solution[0] + solution.Solution[1], 6);
    }

    #endregion

    #region Duality

    /// <summary>
    /// Strong duality: at the optimum the primal objective equals the dual objective bᵀy. For the
    /// furniture problem the duals are the marginal value of an hour of each resource.
    /// </summary>
    [Fact]
    public void Solve_DualValues_SatisfyStrongDuality()
    {
        var program = new LinearProgram<double>(
            objective: V(-30, -20),
            inequalityMatrix: M(new[,] { { 4.0, 3.0 }, { 2.0, 1.0 } }),
            inequalityBounds: V(240, 100));

        var solution = Solver().Solve(program);

        Assert.Equal(LinearProgramStatus.Optimal, solution.Status);
        Assert.NotNull(solution.InequalityDualValues);

        double dualObjective =
            240 * solution.InequalityDualValues[0] + 100 * solution.InequalityDualValues[1];

        Assert.Equal(solution.ObjectiveValue, dualObjective, 6);
    }

    /// <summary>
    /// Complementary slackness: a constraint that is not tight at the optimum has a dual value of
    /// zero — extra capacity you are not using is worth nothing.
    ///   minimize -x  subject to  x ≤ 2 (binding),  x ≤ 100 (slack).
    /// </summary>
    [Fact]
    public void Solve_SlackConstraint_HasZeroDualValue()
    {
        var program = new LinearProgram<double>(
            objective: V(-1),
            inequalityMatrix: M(new[,] { { 1.0 }, { 1.0 } }),
            inequalityBounds: V(2, 100));

        var solution = Solver().Solve(program);

        Assert.Equal(LinearProgramStatus.Optimal, solution.Status);
        Assert.Equal(2.0, solution.Solution[0], 6);
        Assert.Equal(0.0, solution.InequalityDualValues[1], 6);
        Assert.True(
            Math.Abs(solution.InequalityDualValues[0]) > Tolerance,
            "The binding constraint should carry a non-zero shadow price.");
    }

    #endregion

    #region Configuration and validation

    /// <summary>
    /// A deliberately tiny iteration budget must surface as IterationLimit, never as a false
    /// "optimal" claim.
    /// </summary>
    [Fact]
    public void Solve_WithTinyIterationBudget_ReportsIterationLimit()
    {
        var program = new LinearProgram<double>(
            objective: V(-30, -20),
            inequalityMatrix: M(new[,] { { 4.0, 3.0 }, { 2.0, 1.0 } }),
            inequalityBounds: V(240, 100));

        var solver = new SimplexSolver<double>(new SimplexSolverOptions { MaxIterations = 1 });
        var solution = solver.Solve(program);

        Assert.Equal(LinearProgramStatus.IterationLimit, solution.Status);
    }

    [Fact]
    public void Solve_NullProgram_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => Solver().Solve(null));
    }

    [Fact]
    public void Construct_EmptyObjective_Throws()
    {
        Assert.Throws<ArgumentException>(() => new LinearProgram<double>(new Vector<double>(0)));
    }

    [Fact]
    public void Construct_ConstraintMatrixWithoutBounds_Throws()
    {
        Assert.Throws<ArgumentException>(() => new LinearProgram<double>(
            objective: V(1, 1),
            inequalityMatrix: M(new[,] { { 1.0, 1.0 } })));
    }

    [Fact]
    public void Construct_ConstraintColumnMismatch_Throws()
    {
        Assert.Throws<ArgumentException>(() => new LinearProgram<double>(
            objective: V(1, 1),
            inequalityMatrix: M(new[,] { { 1.0, 1.0, 1.0 } }),
            inequalityBounds: V(1)));
    }

    [Fact]
    public void Construct_BoundsLengthMismatch_Throws()
    {
        Assert.Throws<ArgumentException>(() => new LinearProgram<double>(
            objective: V(1, 1),
            lowerBounds: V(0)));
    }

    #endregion
}
