#nullable disable
using AiDotNet.Solvers.Assignment;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Solvers;

/// <summary>
/// Integration tests for the Hungarian linear-assignment solver.
/// </summary>
/// <remarks>
/// CRITICAL: expected costs are verified by exhaustive enumeration of all permutations, so they are
/// the true optima. If a test fails, FIX THE SOLVER.
/// </remarks>
public class LinearAssignmentSolverIntegrationTests
{
    private static Matrix<double> M(double[,] values)
    {
        var matrix = new Matrix<double>(values.GetLength(0), values.GetLength(1));
        for (int r = 0; r < values.GetLength(0); r++)
        {
            for (int c = 0; c < values.GetLength(1); c++) matrix[r, c] = values[r, c];
        }

        return matrix;
    }

    private static LinearAssignmentSolver<double> Solver() => new();

    /// <summary>
    /// Brute-force optimum over every maximum-cardinality assignment of distinct columns to rows,
    /// used as the independent oracle.
    /// </summary>
    /// <remarks>
    /// A rectangular assignment matches exactly <c>min(rows, columns)</c> pairs: when rows
    /// outnumber columns some rows must go unmatched, so the search is allowed to skip a row while
    /// still being required to reach that cardinality. Demanding that EVERY row be matched would
    /// make the rows-exceed-columns case vacuously infeasible.
    /// </remarks>
    private static double BruteForceOptimum(double[,] cost)
    {
        foreach (double value in cost)
        {
            if (value < 0)
            {
                throw new ArgumentException(
                    "The branch-and-bound brute-force oracle requires non-negative costs.",
                    nameof(cost));
            }
        }

        int rows = cost.GetLength(0);
        int columns = cost.GetLength(1);
        int required = Math.Min(rows, columns);
        double best = double.PositiveInfinity;

        var used = new bool[columns];

        void Recurse(int row, int matched, double runningCost)
        {
            if (runningCost >= best) return;

            if (matched == required)
            {
                best = runningCost;
                return;
            }

            // Not enough rows left to reach the required number of matches.
            if (rows - row < required - matched) return;

            for (int c = 0; c < columns; c++)
            {
                if (used[c]) continue;
                used[c] = true;
                Recurse(row + 1, matched + 1, runningCost + cost[row, c]);
                used[c] = false;
            }

            // Leaving this row unmatched is only allowed when the remaining rows can still supply
            // the required number of matches.
            Recurse(row + 1, matched, runningCost);
        }

        Recurse(0, 0, 0.0);
        return best;
    }

    /// <summary>
    /// The canonical case where greedy fails. Greedy grabs the cheapest cell (0,0) = 1, which forces
    /// (1,1) = 100 for a total of 101. The optimal pairing is (0,1) + (1,0) = 2 + 3 = 5.
    /// </summary>
    [Fact]
    public void Solve_WhereGreedyIsSuboptimal_FindsTheOptimalPairing()
    {
        var cost = new[,] { { 1.0, 2.0 }, { 3.0, 100.0 } };

        var assignment = Solver().Solve(M(cost));
        double total = Solver().TotalCost(M(cost), assignment);

        Assert.Equal(5.0, total, 6);
        Assert.Equal(1, assignment[0]);
        Assert.Equal(0, assignment[1]);
    }

    /// <summary>
    /// A 4x4 instance checked against exhaustive enumeration of all 24 permutations.
    /// </summary>
    [Fact]
    public void Solve_SquareMatrix_MatchesBruteForceOptimum()
    {
        var cost = new[,]
        {
            { 82.0, 83.0, 69.0, 92.0 },
            { 77.0, 37.0, 49.0, 92.0 },
            { 11.0, 69.0,  5.0, 86.0 },
            {  8.0,  9.0, 98.0, 23.0 },
        };

        var assignment = Solver().Solve(M(cost));
        double total = Solver().TotalCost(M(cost), assignment);

        Assert.Equal(BruteForceOptimum(cost), total, 6);
    }

    /// <summary>
    /// Every row must receive a distinct column — an assignment that reuses a column is not an
    /// assignment at all.
    /// </summary>
    [Fact]
    public void Solve_AssignsEachColumnAtMostOnce()
    {
        var cost = new[,]
        {
            { 4.0, 1.0, 3.0 },
            { 2.0, 0.0, 5.0 },
            { 3.0, 2.0, 2.0 },
        };

        var assignment = Solver().Solve(M(cost));

        var seen = new HashSet<int>();
        for (int i = 0; i < assignment.Length; i++)
        {
            Assert.True(assignment[i] >= 0, $"Row {i} was left unmatched in a square problem.");
            Assert.True(seen.Add(assignment[i]), $"Column {assignment[i]} was assigned twice.");
        }

        Assert.Equal(BruteForceOptimum(cost), Solver().TotalCost(M(cost), assignment), 6);
    }

    /// <summary>
    /// More columns than rows: every row is matched, spare columns go unused. This is the DETR case
    /// — more object queries than ground-truth boxes.
    /// </summary>
    [Fact]
    public void Solve_MoreColumnsThanRows_MatchesEveryRow()
    {
        var cost = new[,]
        {
            { 9.0, 1.0, 8.0, 7.0 },
            { 6.0, 5.0, 2.0, 9.0 },
        };

        var assignment = Solver().Solve(M(cost));
        double total = Solver().TotalCost(M(cost), assignment);

        Assert.True(assignment[0] >= 0 && assignment[1] >= 0, "Every row should be matched.");
        Assert.NotEqual(assignment[0], assignment[1]);
        Assert.Equal(BruteForceOptimum(cost), total, 6);
    }

    /// <summary>
    /// More rows than columns: only as many rows as there are columns can be matched, and the rest
    /// must be reported as unmatched rather than silently sharing a column.
    /// </summary>
    [Fact]
    public void Solve_MoreRowsThanColumns_LeavesSurplusRowsUnmatched()
    {
        var cost = new[,]
        {
            { 4.0, 1.0 },
            { 2.0, 9.0 },
            { 7.0, 3.0 },
        };

        var assignment = Solver().Solve(M(cost));

        int matched = 0;
        var seen = new HashSet<int>();
        for (int i = 0; i < assignment.Length; i++)
        {
            if (assignment[i] < 0) continue;
            matched++;
            Assert.True(seen.Add(assignment[i]), $"Column {assignment[i]} was assigned twice.");
        }

        Assert.Equal(2, matched);
        Assert.Equal(BruteForceOptimum(cost), Solver().TotalCost(M(cost), assignment), 6);
    }

    /// <summary>
    /// Negative costs (rewards) must work: the algorithm minimizes, so a cost matrix of negated
    /// scores maximizes the score.
    /// </summary>
    [Fact]
    public void Solve_NegativeCosts_StillMinimizes()
    {
        var cost = new[,]
        {
            { -5.0, -1.0 },
            { -2.0, -9.0 },
        };

        var assignment = Solver().Solve(M(cost));
        double total = Solver().TotalCost(M(cost), assignment);

        Assert.Equal(-14.0, total, 6);
        Assert.Equal(0, assignment[0]);
        Assert.Equal(1, assignment[1]);
    }

    /// <summary>A single cell is a degenerate but legal problem.</summary>
    [Fact]
    public void Solve_SingleElement_MatchesIt()
    {
        var assignment = Solver().Solve(M(new[,] { { 42.0 } }));
        Assert.Equal(0, assignment[0]);
    }

    /// <summary>
    /// Ties everywhere must still produce a valid permutation rather than collapsing onto one
    /// column.
    /// </summary>
    [Fact]
    public void Solve_AllCostsEqual_StillProducesAPermutation()
    {
        var cost = new[,]
        {
            { 1.0, 1.0, 1.0 },
            { 1.0, 1.0, 1.0 },
            { 1.0, 1.0, 1.0 },
        };

        var assignment = Solver().Solve(M(cost));

        var seen = new HashSet<int>();
        for (int i = 0; i < assignment.Length; i++)
        {
            Assert.True(assignment[i] >= 0);
            Assert.True(seen.Add(assignment[i]));
        }
    }

    [Fact]
    public void Solve_NullCost_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => Solver().Solve(null));
    }

    [Fact]
    public void Solve_EmptyCost_Throws()
    {
        Assert.Throws<ArgumentException>(() => Solver().Solve(new Matrix<double>(0, 0)));
    }
}
