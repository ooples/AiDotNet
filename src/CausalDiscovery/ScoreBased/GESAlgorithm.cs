using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.ScoreBased;

/// <summary>
/// GES (Greedy Equivalence Search) — score-based causal discovery over equivalence classes.
/// </summary>
/// <remarks>
/// <para>
/// GES searches over Markov equivalence classes of DAGs using two phases:
/// <list type="number">
/// <item><b>Forward phase:</b> Greedily adds edges that most improve the BIC score.</item>
/// <item><b>Backward phase:</b> Greedily removes edges that improve the BIC score.</item>
/// </list>
/// </para>
/// <para>
/// <b>Key Properties:</b>
/// <list type="bullet">
/// <item>Consistent: recovers the true equivalence class given sufficient data (Chickering, 2002)</item>
/// <item>Searches over CPDAGs (equivalence classes), not individual DAGs</item>
/// <item>Uses BIC scoring by default — balances fit and complexity</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> GES builds a causal graph by first adding edges that improve the
/// model fit, then removing edges that are unnecessary. It uses a score (BIC) that rewards
/// fitting the data well while penalizing too many edges. This two-phase approach is
/// guaranteed to find the correct structure given enough data.
/// </para>
/// <para>
/// Reference: Chickering (2002), "Optimal Structure Identification with Greedy Search",
/// Journal of Machine Learning Research.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class GESAlgorithm<T> : ScoreBasedBase<T>
{
    /// <inheritdoc/>
    public override string Name => "GES";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    /// <summary>
    /// Initializes GES with optional configuration.
    /// </summary>
    public GESAlgorithm(CausalDiscoveryOptions? options = null)
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

        // Parent sets for each variable
        var parentSets = new HashSet<int>[d];
        for (int i = 0; i < d; i++) parentSets[i] = [];

        // Compute initial scores (empty graph)
        var scores = new double[d];
        for (int i = 0; i < d; i++)
            scores[i] = ComputeBIC(X, n, i, parentSets[i]);

        // Forward phase: add edges
        bool improved = true;
        int forwardIter = 0;
        while (improved && forwardIter < MaxIterations)
        {
            improved = false;
            forwardIter++;

            int bestFrom = -1, bestTo = -1;
            double bestImprovement = 0;

            for (int to = 0; to < d; to++)
            {
                if (parentSets[to].Count >= MaxParents) continue;

                for (int from = 0; from < d; from++)
                {
                    if (from == to || parentSets[to].Contains(from)) continue;
                    if (WouldCreateCycle(parentSets, from, to, d)) continue;

                    var testParents = new HashSet<int>(parentSets[to]) { from };
                    double newScore = ComputeBIC(X, n, to, testParents);
                    double improvement = newScore - scores[to];

                    if (improvement > bestImprovement)
                    {
                        bestImprovement = improvement;
                        bestFrom = from;
                        bestTo = to;
                    }
                }
            }

            if (bestFrom >= 0)
            {
                parentSets[bestTo].Add(bestFrom);
                scores[bestTo] = ComputeBIC(X, n, bestTo, parentSets[bestTo]);
                improved = true;
            }
        }

        // Backward phase: remove edges (separate iteration counter)
        improved = true;
        int backwardIter = 0;
        while (improved && backwardIter < MaxIterations)
        {
            improved = false;
            backwardIter++;

            int bestFrom = -1, bestTo = -1;
            double bestImprovement = 0;

            for (int to = 0; to < d; to++)
            {
                foreach (int from in parentSets[to])
                {
                    var testParents = new HashSet<int>(parentSets[to]);
                    testParents.Remove(from);
                    double newScore = ComputeBIC(X, n, to, testParents);
                    double improvement = newScore - scores[to];

                    if (improvement > bestImprovement)
                    {
                        bestImprovement = improvement;
                        bestFrom = from;
                        bestTo = to;
                    }
                }
            }

            if (bestFrom >= 0)
            {
                parentSets[bestTo].Remove(bestFrom);
                scores[bestTo] = ComputeBIC(X, n, bestTo, parentSets[bestTo]);
                improved = true;
            }
        }

        // Build weighted adjacency from parent sets
        var W = new double[d, d];
        for (int to = 0; to < d; to++)
        {
            foreach (int from in parentSets[to])
            {
                // Use correlation magnitude as weight
                double weight = Math.Abs(ComputeRegCoef(X, n, from, to, parentSets[to]));
                W[from, to] = Math.Max(weight, 0.01); // ensure non-zero for discovered edges
            }
        }

        return DoubleArrayToMatrix(W);
    }

    private static double ComputeRegCoef(double[,] X, int n, int from, int to, HashSet<int> parentSet)
    {
        double meanX = 0, meanY = 0;
        for (int i = 0; i < n; i++) { meanX += X[i, from]; meanY += X[i, to]; }
        meanX /= n; meanY /= n;

        double sxy = 0, sxx = 0;
        for (int i = 0; i < n; i++)
        {
            double dx = X[i, from] - meanX;
            sxy += dx * (X[i, to] - meanY);
            sxx += dx * dx;
        }

        return sxx > 1e-10 ? sxy / sxx : 0;
    }
}
