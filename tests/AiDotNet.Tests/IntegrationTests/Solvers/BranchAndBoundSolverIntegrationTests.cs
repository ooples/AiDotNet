#nullable disable
using AiDotNet.Models.Options;
using AiDotNet.Solvers.LinearProgramming;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Solvers;

/// <summary>
/// Integration tests for the branch-and-bound integer-programming solver.
/// </summary>
/// <remarks>
/// CRITICAL: expected values are enumerated by hand from small problems, so they are the true
/// optima and not a record of current behaviour. If one fails, FIX THE SOLVER.
/// </remarks>
public class BranchAndBoundSolverIntegrationTests
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

    private static BranchAndBoundSolver<double> Solver() => new();

    /// <summary>
    /// The classic case where rounding the relaxation gives the wrong answer.
    ///   maximize x + y  subject to  2x + 2y ≤ 3,  x, y ∈ {0, 1, 2, ...}
    /// The relaxation reaches 1.5 (any x + y = 1.5), but no whole-number point exceeds x + y = 1.
    /// Rounding 1.5 up to 2 would be infeasible; rounding pieces down can miss the optimum.
    /// </summary>
    [Fact]
    public void Solve_RelaxationIsFractional_FindsTrueIntegerOptimum()
    {
        var relaxation = new LinearProgram<double>(
            objective: V(-1, -1),
            inequalityMatrix: M(new[,] { { 2.0, 2.0 } }),
            inequalityBounds: V(3));

        var solution = Solver().Solve(new IntegerProgram<double>(relaxation));

        Assert.Equal(LinearProgramStatus.Optimal, solution.Status);
        Assert.Equal(-1.0, solution.ObjectiveValue, 6);
        Assert.Equal(1.0, solution.Solution[0] + solution.Solution[1], 6);
    }

    /// <summary>
    /// A 0/1 knapsack solved by exhaustive reasoning.
    ///   values  = (10, 10, 12, 18), weights = (2, 4, 6, 9), capacity = 15.
    /// The feasible subsets and their values: {0,1,2} weighs 12 → 32; {0,3} weighs 11 → 28;
    /// {1,3} weighs 13 → 28; {0,1,3} weighs 15 → 38; {2,3} weighs 15 → 30.
    /// The best is {0,1,3} with value 38 at exactly the capacity.
    /// </summary>
    [Fact]
    public void Solve_ZeroOneKnapsack_FindsBestSubset()
    {
        var relaxation = new LinearProgram<double>(
            objective: V(-10, -10, -12, -18),
            inequalityMatrix: M(new[,] { { 2.0, 4.0, 6.0, 9.0 } }),
            inequalityBounds: V(15),
            lowerBounds: V(0, 0, 0, 0),
            upperBounds: V(1, 1, 1, 1));

        var solution = Solver().Solve(new IntegerProgram<double>(relaxation));

        Assert.Equal(LinearProgramStatus.Optimal, solution.Status);
        Assert.Equal(-38.0, solution.ObjectiveValue, 6);
        Assert.Equal(1.0, solution.Solution[0], 6);
        Assert.Equal(1.0, solution.Solution[1], 6);
        Assert.Equal(0.0, solution.Solution[2], 6);
        Assert.Equal(1.0, solution.Solution[3], 6);
    }

    /// <summary>
    /// Mixed-integer: one variable must be whole, the other may be fractional.
    ///   minimize -x - y  subject to  x + y ≤ 3.5,  0 ≤ x ≤ 10 (integral),  0 ≤ y ≤ 0.5.
    /// y is capped at 0.5, so x takes the remaining 3.0 exactly; value -3.5.
    /// </summary>
    [Fact]
    public void Solve_MixedInteger_LeavesContinuousVariableFractional()
    {
        var relaxation = new LinearProgram<double>(
            objective: V(-1, -1),
            inequalityMatrix: M(new[,] { { 1.0, 1.0 } }),
            inequalityBounds: V(3.5),
            lowerBounds: V(0, 0),
            upperBounds: V(10, 0.5));

        var solution = Solver().Solve(
            new IntegerProgram<double>(relaxation, new[] { true, false }));

        Assert.Equal(LinearProgramStatus.Optimal, solution.Status);
        Assert.Equal(3.0, solution.Solution[0], 6);
        Assert.Equal(0.5, solution.Solution[1], 6);
        Assert.Equal(-3.5, solution.ObjectiveValue, 6);
    }

    /// <summary>
    /// Integer values must come back exactly whole, not as 2.9999999997 left over from the
    /// floating-point relaxation.
    /// </summary>
    [Fact]
    public void Solve_IntegerComponents_AreExactlyWhole()
    {
        var relaxation = new LinearProgram<double>(
            objective: V(-1),
            inequalityMatrix: M(new[,] { { 3.0 } }),
            inequalityBounds: V(10));                       // 3x <= 10  =>  x <= 3.33...

        var solution = Solver().Solve(new IntegerProgram<double>(relaxation));

        Assert.Equal(LinearProgramStatus.Optimal, solution.Status);
        Assert.Equal(3.0, solution.Solution[0]);            // exact equality, no tolerance
    }

    /// <summary>
    /// A problem whose relaxation is feasible but which admits no whole-number point at all:
    ///   2x = 1 with x integral.
    /// </summary>
    [Fact]
    public void Solve_NoIntegerPointExists_ReportsInfeasible()
    {
        var relaxation = new LinearProgram<double>(
            objective: V(1),
            equalityMatrix: M(new[,] { { 2.0 } }),
            equalityBounds: V(1),
            lowerBounds: V(0),
            upperBounds: V(10));

        var solution = Solver().Solve(new IntegerProgram<double>(relaxation));

        Assert.Equal(LinearProgramStatus.Infeasible, solution.Status);
        Assert.Null(solution.Solution);
    }

    /// <summary>
    /// An unbounded relaxation makes the integer problem unbounded too.
    /// </summary>
    [Fact]
    public void Solve_UnboundedRelaxation_ReportsUnbounded()
    {
        var relaxation = new LinearProgram<double>(objective: V(-1));

        var solution = Solver().Solve(new IntegerProgram<double>(relaxation));

        Assert.Equal(LinearProgramStatus.Unbounded, solution.Status);
    }

    /// <summary>
    /// A search cut short by the node budget must never claim optimality, and must never claim
    /// infeasibility either: with branches left unexplored it simply does not know, and
    /// "infeasible" is the much stronger assertion that no whole-number point exists anywhere.
    /// </summary>
    /// <remarks>
    /// The relaxation here is genuinely fractional (2x + 2y ≤ 3 gives x + y = 1.5), so one node is
    /// not enough to finish — unlike the knapsack above, whose relaxation is integral at the root
    /// and for which a single node really is a complete, optimal search.
    /// </remarks>
    [Fact]
    public void Solve_NodeBudgetExhaustedBeforeAnyCandidate_ReportsIterationLimitNotInfeasible()
    {
        var relaxation = new LinearProgram<double>(
            objective: V(-1, -1),
            inequalityMatrix: M(new[,] { { 2.0, 2.0 } }),
            inequalityBounds: V(3));

        var solver = new BranchAndBoundSolver<double>(
            new BranchAndBoundSolverOptions { MaxNodes = 1 });
        var solution = solver.Solve(new IntegerProgram<double>(relaxation));

        Assert.Equal(LinearProgramStatus.IterationLimit, solution.Status);
    }

    /// <summary>
    /// When the root relaxation is already integral, a single node is a complete search and
    /// claiming optimality is correct — the knapsack's ratio ordering happens to give exactly this.
    /// </summary>
    [Fact]
    public void Solve_RootRelaxationAlreadyIntegral_IsOptimalInOneNode()
    {
        var relaxation = new LinearProgram<double>(
            objective: V(-10, -10, -12, -18),
            inequalityMatrix: M(new[,] { { 2.0, 4.0, 6.0, 9.0 } }),
            inequalityBounds: V(15),
            lowerBounds: V(0, 0, 0, 0),
            upperBounds: V(1, 1, 1, 1));

        var solver = new BranchAndBoundSolver<double>(
            new BranchAndBoundSolverOptions { MaxNodes = 1 });
        var solution = solver.Solve(new IntegerProgram<double>(relaxation));

        Assert.Equal(LinearProgramStatus.Optimal, solution.Status);
        Assert.Equal(-38.0, solution.ObjectiveValue, 6);
    }

    /// <summary>
    /// Bounding must actually prune: solving the knapsack should cost far fewer nodes than the
    /// 2^4 = 16 subsets a brute-force enumeration would examine.
    /// </summary>
    [Fact]
    public void Solve_BoundingPrunesTheSearchTree()
    {
        var relaxation = new LinearProgram<double>(
            objective: V(-10, -10, -12, -18),
            inequalityMatrix: M(new[,] { { 2.0, 4.0, 6.0, 9.0 } }),
            inequalityBounds: V(15),
            lowerBounds: V(0, 0, 0, 0),
            upperBounds: V(1, 1, 1, 1));

        var solution = Solver().Solve(new IntegerProgram<double>(relaxation));

        Assert.Equal(LinearProgramStatus.Optimal, solution.Status);
        Assert.True(
            solution.Iterations < 16,
            $"Explored {solution.Iterations} nodes; bounding should beat enumerating all 16 subsets.");
    }

    #region Validation

    [Fact]
    public void Solve_NullProgram_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => Solver().Solve(null));
    }

    [Fact]
    public void Construct_NullRelaxation_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => new IntegerProgram<double>(null));
    }

    [Fact]
    public void Construct_MaskLengthMismatch_Throws()
    {
        var relaxation = new LinearProgram<double>(V(1, 1));
        Assert.Throws<ArgumentException>(() =>
            new IntegerProgram<double>(relaxation, new[] { true }));
    }

    [Fact]
    public void Construct_MaskWithNoIntegralVariables_Throws()
    {
        var relaxation = new LinearProgram<double>(V(1, 1));
        Assert.Throws<ArgumentException>(() =>
            new IntegerProgram<double>(relaxation, new[] { false, false }));
    }

    #endregion
}
