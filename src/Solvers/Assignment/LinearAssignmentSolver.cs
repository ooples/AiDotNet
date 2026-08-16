using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Solvers.Assignment;

/// <summary>
/// Solves the linear assignment problem: pair every row with a distinct column so that the total
/// cost is as small as possible.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Implements the Hungarian algorithm (Kuhn, 1955; Munkres, 1957) in its O(n³) shortest-augmenting-
/// path form, generalized to rectangular cost matrices. The result is the <b>globally optimal</b>
/// assignment, not a good one.
/// </para>
/// <para>
/// The assignment problem is a linear program whose constraint matrix is totally unimodular, which
/// is why its linear relaxation always has an integral optimum and why a specialized combinatorial
/// algorithm can solve it exactly in polynomial time. The greedy alternative — repeatedly taking
/// the cheapest remaining pair — is not optimal, and the gap is not a rounding detail: committing
/// early to a locally cheap pair can force an arbitrarily expensive one later.
/// </para>
/// <para><b>For Beginners:</b> You have workers and jobs, and a table of what each worker costs on
/// each job. Every worker takes exactly one job and every job goes to exactly one worker. Picking
/// the cheapest pair, then the cheapest of what remains, and so on feels sensible but is provably
/// not the best plan — the cheap pair you grabbed first may have been the only affordable option
/// for a job that now costs a fortune. This finds the genuinely cheapest overall plan.
/// </para>
/// <example>
/// <code>
/// var solver = new LinearAssignmentSolver&lt;double&gt;();
/// var assignment = solver.Solve(costMatrix);
/// // assignment[i] is the column matched to row i, or -1 when the row is unmatched
/// // (possible only when there are more rows than columns).
/// </code>
/// </example>
/// </remarks>
public sealed class LinearAssignmentSolver<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Finds the minimum-cost assignment of rows to columns.
    /// </summary>
    /// <param name="cost">
    /// The cost matrix; <c>cost[i, j]</c> is the cost of pairing row <c>i</c> with column <c>j</c>.
    /// The matrix need not be square.
    /// </param>
    /// <returns>
    /// An array with one entry per row giving the column assigned to it, or <c>-1</c> when that row
    /// is left unmatched (which happens only when there are more rows than columns).
    /// </returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="cost"/> is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the cost matrix has no rows or no columns.</exception>
    public Vector<int> Solve(Matrix<T> cost)
    {
        if (cost is null) throw new ArgumentNullException(nameof(cost));
        if (cost.Rows == 0 || cost.Columns == 0)
        {
            throw new ArgumentException(
                "The cost matrix must have at least one row and one column.", nameof(cost));
        }

        int rowCount = cost.Rows;
        int columnCount = cost.Columns;

        // The algorithm below assigns every ROW, so it needs at least as many columns as rows. When
        // rows outnumber columns the problem is transposed and the answer mapped back, which leaves
        // the surplus rows unmatched — the correct outcome, since a column cannot take two rows.
        bool transposed = rowCount > columnCount;
        int n = transposed ? columnCount : rowCount;   // rows in the working orientation
        int m = transposed ? rowCount : columnCount;   // columns in the working orientation

        T Cost(int i, int j) => transposed ? cost[j, i] : cost[i, j];

        // Potentials (dual variables) for rows and columns, and the row currently matched to each
        // column. Index 0 of each array is a sentinel used by the augmenting-path search.
        var rowPotential = new T[n + 1];
        var columnPotential = new T[m + 1];
        var matchedRowOfColumn = new int[m + 1];
        var previousColumn = new int[m + 1];

        for (int i = 0; i <= n; i++) rowPotential[i] = NumOps.Zero;
        for (int j = 0; j <= m; j++)
        {
            columnPotential[j] = NumOps.Zero;
            matchedRowOfColumn[j] = 0;
            previousColumn[j] = -1;
        }

        var infinity = NumOps.FromDouble(double.PositiveInfinity);

        for (int row = 1; row <= n; row++)
        {
            // Grow a shortest augmenting path from this row until it reaches a free column.
            matchedRowOfColumn[0] = row;
            int currentColumn = 0;

            var minimumSlack = new T[m + 1];
            var visited = new bool[m + 1];
            for (int j = 0; j <= m; j++)
            {
                minimumSlack[j] = infinity;
                visited[j] = false;
                previousColumn[j] = -1;
            }

            do
            {
                visited[currentColumn] = true;
                int currentRow = matchedRowOfColumn[currentColumn];
                T delta = infinity;
                int nextColumn = -1;

                for (int j = 1; j <= m; j++)
                {
                    if (visited[j]) continue;

                    // Reduced cost of pairing currentRow with column j under the current potentials.
                    T reducedCost = NumOps.Subtract(
                        NumOps.Subtract(Cost(currentRow - 1, j - 1), rowPotential[currentRow]),
                        columnPotential[j]);

                    if (NumOps.LessThan(reducedCost, minimumSlack[j]))
                    {
                        minimumSlack[j] = reducedCost;
                        previousColumn[j] = currentColumn;
                    }

                    if (NumOps.LessThan(minimumSlack[j], delta))
                    {
                        delta = minimumSlack[j];
                        nextColumn = j;
                    }
                }

                if (nextColumn < 0)
                {
                    // No unvisited column remains reachable. With m >= n this cannot happen for a
                    // finite cost matrix; bail out rather than loop forever on malformed input.
                    break;
                }

                // Shift the potentials so the chosen edge becomes tight, keeping every previously
                // tight edge tight. This is the step that makes the algorithm run in O(n³) instead
                // of re-deriving the shortest paths from scratch.
                for (int j = 0; j <= m; j++)
                {
                    if (visited[j])
                    {
                        rowPotential[matchedRowOfColumn[j]] =
                            NumOps.Add(rowPotential[matchedRowOfColumn[j]], delta);
                        columnPotential[j] = NumOps.Subtract(columnPotential[j], delta);
                    }
                    else
                    {
                        minimumSlack[j] = NumOps.Subtract(minimumSlack[j], delta);
                    }
                }

                currentColumn = nextColumn;
            }
            while (matchedRowOfColumn[currentColumn] != 0);

            // Walk the augmenting path back to the start, flipping matched and unmatched edges.
            while (currentColumn != 0)
            {
                int predecessor = previousColumn[currentColumn];
                if (predecessor < 0) break;
                matchedRowOfColumn[currentColumn] = matchedRowOfColumn[predecessor];
                currentColumn = predecessor;
            }
        }

        var assignment = new Vector<int>(rowCount);
        for (int i = 0; i < rowCount; i++) assignment[i] = -1;

        for (int j = 1; j <= m; j++)
        {
            int matchedRow = matchedRowOfColumn[j];
            if (matchedRow <= 0 || matchedRow > n) continue;

            if (transposed)
            {
                // Working orientation had rows and columns swapped: working row (matchedRow-1) is an
                // original COLUMN, and working column (j-1) is an original ROW.
                assignment[j - 1] = matchedRow - 1;
            }
            else
            {
                assignment[matchedRow - 1] = j - 1;
            }
        }

        return assignment;
    }

    /// <summary>
    /// Returns the total cost of an assignment produced by <see cref="Solve"/>.
    /// </summary>
    /// <param name="cost">The cost matrix the assignment was computed from.</param>
    /// <param name="assignment">The column assigned to each row, with -1 for unmatched rows.</param>
    /// <returns>The sum of the costs of the matched pairs.</returns>
    /// <exception cref="ArgumentNullException">
    /// Thrown when <paramref name="cost"/> or <paramref name="assignment"/> is null.
    /// </exception>
    public T TotalCost(Matrix<T> cost, Vector<int> assignment)
    {
        if (cost is null) throw new ArgumentNullException(nameof(cost));
        if (assignment is null) throw new ArgumentNullException(nameof(assignment));

        T total = NumOps.Zero;
        for (int i = 0; i < assignment.Length; i++)
        {
            int column = assignment[i];
            if (column >= 0) total = NumOps.Add(total, cost[i, column]);
        }

        return total;
    }
}
