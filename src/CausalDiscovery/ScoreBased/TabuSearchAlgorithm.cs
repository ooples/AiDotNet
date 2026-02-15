using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.ScoreBased;

/// <summary>
/// Tabu Search — score-based DAG learning with memory to escape local optima.
/// </summary>
/// <remarks>
/// <para>
/// Tabu search extends hill climbing by maintaining a "tabu list" of recently visited
/// states that cannot be revisited. This prevents cycling and helps escape local optima
/// that trap simple hill climbing.
/// </para>
/// <para>
/// <b>Algorithm:</b>
/// <list type="number">
/// <item>Start with empty graph or initial structure</item>
/// <item>At each step, apply the best non-tabu operation (add/remove edge)</item>
/// <item>Add the reverse of the operation to the tabu list</item>
/// <item>Allow tabu moves if they improve upon the best-known solution (aspiration)</item>
/// <item>Continue for a fixed number of iterations or until convergence</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> Tabu search is like hill climbing but with a memory. If you just
/// climbed from point A to point B, you're not allowed to go back to A for a while.
/// This forces the algorithm to explore new areas and often finds better solutions
/// than simple hill climbing, which can get stuck.
/// </para>
/// <para>
/// Reference: Glover (1989, 1990), "Tabu Search" — applied to Bayesian network
/// structure learning.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class TabuSearchAlgorithm<T> : ScoreBasedBase<T>
{
    private const int DEFAULT_TABU_SIZE = 100;

    /// <inheritdoc/>
    public override string Name => "Tabu Search";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    /// <summary>
    /// Initializes Tabu Search with optional configuration.
    /// </summary>
    public TabuSearchAlgorithm(CausalDiscoveryOptions? options = null)
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

        double bestTotalScore = scores.Sum();
        var bestParentSets = CloneParentSets(parentSets, d);

        // Tabu list: stores (operation_type, from, to) tuples
        var tabuList = new Queue<(int type, int from, int to)>();

        double currentTotal = scores.Sum();

        for (int iteration = 0; iteration < MaxIterations; iteration++)
        {
            double bestImprovement = double.NegativeInfinity;
            int bestType = -1, bestFrom = -1, bestTo = -1;

            // Evaluate all operations
            for (int to = 0; to < d; to++)
            {
                // Add operations
                if (parentSets[to].Count < MaxParents)
                {
                    for (int from = 0; from < d; from++)
                    {
                        if (from == to || parentSets[to].Contains(from)) continue;
                        if (WouldCreateCycle(parentSets, from, to, d)) continue;

                        bool isTabu = tabuList.Any(t => t.type == 1 && t.from == from && t.to == to);

                        var testParents = new HashSet<int>(parentSets[to]) { from };
                        double imp = ComputeBIC(X, n, to, testParents) - scores[to];

                        // Aspiration criterion: allow tabu move if it improves global best
                        double newTotal = currentTotal + imp;
                        if (isTabu && newTotal <= bestTotalScore) continue;

                        if (imp > bestImprovement)
                        {
                            bestImprovement = imp;
                            bestType = 0;
                            bestFrom = from;
                            bestTo = to;
                        }
                    }
                }

                // Remove operations
                foreach (int from in parentSets[to])
                {
                    bool isTabu = tabuList.Any(t => t.type == 0 && t.from == from && t.to == to);

                    var testParents = new HashSet<int>(parentSets[to]);
                    testParents.Remove(from);
                    double imp = ComputeBIC(X, n, to, testParents) - scores[to];

                    double newTotal = currentTotal + imp;
                    if (isTabu && newTotal <= bestTotalScore) continue;

                    if (imp > bestImprovement)
                    {
                        bestImprovement = imp;
                        bestType = 1;
                        bestFrom = from;
                        bestTo = to;
                    }
                }
            }

            if (bestType < 0) break;

            // Apply operation
            switch (bestType)
            {
                case 0: // Add
                    parentSets[bestTo].Add(bestFrom);
                    scores[bestTo] = ComputeBIC(X, n, bestTo, parentSets[bestTo]);
                    tabuList.Enqueue((1, bestFrom, bestTo)); // Tabu the reverse (remove)
                    break;
                case 1: // Remove
                    parentSets[bestTo].Remove(bestFrom);
                    scores[bestTo] = ComputeBIC(X, n, bestTo, parentSets[bestTo]);
                    tabuList.Enqueue((0, bestFrom, bestTo)); // Tabu the reverse (add)
                    break;
            }

            // Maintain tabu list size
            while (tabuList.Count > DEFAULT_TABU_SIZE)
                tabuList.Dequeue();

            // Track best solution
            currentTotal = scores.Sum();
            if (currentTotal > bestTotalScore)
            {
                bestTotalScore = currentTotal;
                bestParentSets = CloneParentSets(parentSets, d);
            }
        }

        // Build adjacency from best parent sets
        var W = new double[d, d];
        for (int to = 0; to < d; to++)
            foreach (int from in bestParentSets[to])
                W[from, to] = Math.Max(0.01, ComputeAbsCorr(X, n, from, to));

        return DoubleArrayToMatrix(W);
    }

    private static HashSet<int>[] CloneParentSets(HashSet<int>[] parentSets, int d)
    {
        var clone = new HashSet<int>[d];
        for (int i = 0; i < d; i++)
            clone[i] = new HashSet<int>(parentSets[i]);
        return clone;
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
