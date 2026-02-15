using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.ScoreBased;

/// <summary>
/// Hill Climbing — greedy score-based DAG structure learning.
/// </summary>
/// <remarks>
/// <para>
/// Hill climbing searches over individual DAGs (not equivalence classes like GES) by
/// greedily applying the single-edge operation (add, remove, or reverse) that most
/// improves the BIC score. It terminates when no operation improves the score.
/// </para>
/// <para>
/// <b>Operations per step:</b>
/// <list type="bullet">
/// <item>Add an edge i → j (if no cycle is created)</item>
/// <item>Remove an existing edge i → j</item>
/// <item>Reverse an existing edge i → j to j → i (if no cycle is created)</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> Hill climbing is the simplest score-based approach. At each step,
/// it tries adding, removing, or flipping every possible edge and picks the change that
/// improves the score the most. It's like climbing a hill in fog — you always step in
/// the steepest upward direction, but you might get stuck on a local peak.
/// </para>
/// <para>
/// Reference: Heckerman et al. (1995), "Learning Bayesian Networks: The Combination
/// of Knowledge and Statistical Data".
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class HillClimbingAlgorithm<T> : ScoreBasedBase<T>
{
    /// <inheritdoc/>
    public override string Name => "Hill Climbing";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    /// <summary>
    /// Initializes Hill Climbing with optional configuration.
    /// </summary>
    public HillClimbingAlgorithm(CausalDiscoveryOptions? options = null)
    {
        ApplyScoreOptions(options);
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;
        var X = new double[n, d];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < d; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);

        var parentSets = new HashSet<int>[d];
        for (int i = 0; i < d; i++) parentSets[i] = [];

        var scores = new double[d];
        for (int i = 0; i < d; i++)
            scores[i] = ComputeBIC(X, n, i, parentSets[i]);

        for (int iteration = 0; iteration < MaxIterations; iteration++)
        {
            double bestImprovement = 0;
            int bestType = -1; // 0=add, 1=remove, 2=reverse
            int bestFrom = -1, bestTo = -1;

            // Try all add operations
            for (int to = 0; to < d; to++)
            {
                if (parentSets[to].Count >= MaxParents) continue;
                for (int from = 0; from < d; from++)
                {
                    if (from == to || parentSets[to].Contains(from)) continue;
                    if (WouldCreateCycle(parentSets, from, to, d)) continue;

                    var testParents = new HashSet<int>(parentSets[to]) { from };
                    double imp = ComputeBIC(X, n, to, testParents) - scores[to];
                    if (imp > bestImprovement)
                    {
                        bestImprovement = imp;
                        bestType = 0;
                        bestFrom = from;
                        bestTo = to;
                    }
                }
            }

            // Try all remove operations
            for (int to = 0; to < d; to++)
            {
                foreach (int from in parentSets[to])
                {
                    var testParents = new HashSet<int>(parentSets[to]);
                    testParents.Remove(from);
                    double imp = ComputeBIC(X, n, to, testParents) - scores[to];
                    if (imp > bestImprovement)
                    {
                        bestImprovement = imp;
                        bestType = 1;
                        bestFrom = from;
                        bestTo = to;
                    }
                }
            }

            // Try all reverse operations
            for (int to = 0; to < d; to++)
            {
                foreach (int from in parentSets[to])
                {
                    // Remove from → to, add to → from
                    var testParentsTo = new HashSet<int>(parentSets[to]);
                    testParentsTo.Remove(from);

                    if (parentSets[from].Count >= MaxParents) continue;

                    var testParentsFrom = new HashSet<int>(parentSets[from]) { to };

                    // Check cycle with reversed edge
                    var tempParents = (HashSet<int>[])parentSets.Clone();
                    tempParents[to] = testParentsTo;
                    tempParents[from] = testParentsFrom;
                    if (WouldCreateCycle(tempParents, to, from, d)) continue;

                    double impTo = ComputeBIC(X, n, to, testParentsTo) - scores[to];
                    double impFrom = ComputeBIC(X, n, from, testParentsFrom) - scores[from];
                    double totalImp = impTo + impFrom;

                    if (totalImp > bestImprovement)
                    {
                        bestImprovement = totalImp;
                        bestType = 2;
                        bestFrom = from;
                        bestTo = to;
                    }
                }
            }

            if (bestType < 0) break; // No improvement found

            // Apply best operation
            switch (bestType)
            {
                case 0: // Add
                    parentSets[bestTo].Add(bestFrom);
                    scores[bestTo] = ComputeBIC(X, n, bestTo, parentSets[bestTo]);
                    break;
                case 1: // Remove
                    parentSets[bestTo].Remove(bestFrom);
                    scores[bestTo] = ComputeBIC(X, n, bestTo, parentSets[bestTo]);
                    break;
                case 2: // Reverse
                    parentSets[bestTo].Remove(bestFrom);
                    parentSets[bestFrom].Add(bestTo);
                    scores[bestTo] = ComputeBIC(X, n, bestTo, parentSets[bestTo]);
                    scores[bestFrom] = ComputeBIC(X, n, bestFrom, parentSets[bestFrom]);
                    break;
            }
        }

        // Build adjacency
        var W = new double[d, d];
        for (int to = 0; to < d; to++)
            foreach (int from in parentSets[to])
                W[from, to] = Math.Max(0.01, ComputeAbsCorr(X, n, from, to));

        return DoubleArrayToMatrix(W);
    }

    private static double ComputeAbsCorr(double[,] X, int n, int i, int j)
    {
        double mi = 0, mj = 0;
        for (int k = 0; k < n; k++) { mi += X[k, i]; mj += X[k, j]; }
        mi /= n; mj /= n;

        double sij = 0, sii = 0, sjj = 0;
        for (int k = 0; k < n; k++)
        {
            double di = X[k, i] - mi, dj = X[k, j] - mj;
            sij += di * dj; sii += di * di; sjj += dj * dj;
        }

        return (sii > 1e-10 && sjj > 1e-10) ? Math.Abs(sij / Math.Sqrt(sii * sjj)) : 0;
    }
}
