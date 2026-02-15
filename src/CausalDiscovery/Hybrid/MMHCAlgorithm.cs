using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.Hybrid;

/// <summary>
/// MMHC — Max-Min Hill-Climbing, a hybrid constraint-based + score-based algorithm.
/// </summary>
/// <remarks>
/// <para>
/// MMHC combines two phases:
/// <list type="number">
/// <item><b>MMPC phase (constraint-based):</b> For each variable, identify candidate parents/children
/// using the Max-Min Parents and Children heuristic. This restricts the search space.</item>
/// <item><b>HC phase (score-based):</b> Run greedy Hill Climbing search within the restricted
/// space defined by the MMPC skeleton, optimizing BIC score.</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> MMHC first quickly identifies which variables MIGHT be related
/// (using statistical tests), then carefully determines the exact direction and strength
/// of those relationships (using a scoring approach). This makes it both fast and accurate.
/// </para>
/// <para>
/// Reference: Tsamardinos et al. (2006), "The Max-Min Hill-Climbing Bayesian Network
/// Structure Learning Algorithm", Machine Learning.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class MMHCAlgorithm<T> : HybridBase<T>
{
    /// <inheritdoc/>
    public override string Name => "MMHC";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    public MMHCAlgorithm(CausalDiscoveryOptions? options = null)
    {
        ApplyHybridOptions(options);
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

        // Phase 1: MMPC — find candidate parents/children for each variable
        var skeleton = MMPCPhase(X, n, d);

        // Phase 2: Hill Climbing within restricted space
        var W = HillClimbPhase(X, n, d, skeleton);

        return DoubleArrayToMatrix(W);
    }

    /// <summary>
    /// MMPC phase: for each variable, identify candidate neighbors using Max-Min heuristic.
    /// </summary>
    private bool[,] MMPCPhase(double[,] X, int n, int d)
    {
        var candidates = new bool[d, d];

        for (int target = 0; target < d; target++)
        {
            var cpc = new List<int>();

            // Forward phase: greedily add variables with max-min association
            bool added = true;
            while (added && cpc.Count < MaxParents)
            {
                added = false;
                double bestMinAssoc = -1;
                int bestVar = -1;

                for (int candidate = 0; candidate < d; candidate++)
                {
                    if (candidate == target || cpc.Contains(candidate)) continue;

                    // Min association over conditioning sets
                    double minAssoc = ComputeMinAssociation(X, n, target, candidate, cpc);
                    if (minAssoc > bestMinAssoc)
                    {
                        bestMinAssoc = minAssoc;
                        bestVar = candidate;
                    }
                }

                if (bestVar >= 0 && bestMinAssoc > Alpha)
                {
                    cpc.Add(bestVar);
                    added = true;
                }
            }

            // Backward phase: remove false positives
            var toRemove = new List<int>();
            foreach (int var in cpc)
            {
                var subset = cpc.Where(v => v != var).ToList();
                double minAssoc = ComputeMinAssociation(X, n, target, var, subset);
                if (minAssoc <= Alpha)
                    toRemove.Add(var);
            }

            foreach (int v in toRemove) cpc.Remove(v);

            // Set symmetric skeleton
            foreach (int v in cpc)
            {
                candidates[target, v] = true;
                candidates[v, target] = true;
            }
        }

        return candidates;
    }

    /// <summary>
    /// Computes minimum association (absolute correlation) over conditioning subsets.
    /// </summary>
    private double ComputeMinAssociation(double[,] X, int n, int target, int candidate, List<int> condSet)
    {
        if (condSet.Count == 0)
            return Math.Abs(ComputeCorrelation(X, n, target, candidate));

        double minAssoc = double.MaxValue;

        // Test with empty conditioning
        double baseCorr = Math.Abs(ComputeCorrelation(X, n, target, candidate));
        minAssoc = Math.Min(minAssoc, baseCorr);

        // Test conditioning on each individual variable
        foreach (int cond in condSet)
        {
            double partialCorr = ComputePartialCorrelation(X, n, target, candidate, cond);
            minAssoc = Math.Min(minAssoc, Math.Abs(partialCorr));
        }

        return minAssoc;
    }

    /// <summary>
    /// Computes partial correlation conditioned on a single variable.
    /// </summary>
    private static double ComputePartialCorrelation(double[,] X, int n, int i, int j, int cond)
    {
        double rij = ComputeCorrelation(X, n, i, j);
        double ric = ComputeCorrelation(X, n, i, cond);
        double rjc = ComputeCorrelation(X, n, j, cond);

        double denom = Math.Sqrt((1 - ric * ric) * (1 - rjc * rjc));
        return denom > 1e-10 ? (rij - ric * rjc) / denom : 0;
    }

    /// <summary>
    /// Hill Climbing phase within the skeleton-restricted space.
    /// </summary>
    private double[,] HillClimbPhase(double[,] X, int n, int d, bool[,] skeleton)
    {
        var parents = new List<int>[d];
        for (int i = 0; i < d; i++) parents[i] = [];

        double totalBIC = 0;
        for (int i = 0; i < d; i++)
            totalBIC += ComputeBIC(X, n, d, i, parents[i]);

        // Greedy hill climbing with add/remove operations
        bool improved = true;
        int maxIter = 100;
        while (improved && maxIter-- > 0)
        {
            improved = false;
            double bestDelta = 0;
            int bestOp = -1; // 0=add, 1=remove
            int bestFrom = -1, bestTo = -1;

            for (int i = 0; i < d; i++)
            {
                for (int j = 0; j < d; j++)
                {
                    if (i == j || !skeleton[i, j]) continue;

                    if (!parents[j].Contains(i))
                    {
                        // Try adding edge i→j
                        if (parents[j].Count >= MaxParents) continue;
                        var newParents = new List<int>(parents[j]) { i };
                        if (WouldCreateCycle(parents, d, i, j)) continue;

                        double oldBIC = ComputeBIC(X, n, d, j, parents[j]);
                        double newBIC = ComputeBIC(X, n, d, j, newParents);
                        double delta = oldBIC - newBIC; // positive = improvement

                        if (delta > bestDelta)
                        {
                            bestDelta = delta;
                            bestOp = 0;
                            bestFrom = i;
                            bestTo = j;
                        }
                    }
                    else
                    {
                        // Try removing edge i→j
                        var newParents = parents[j].Where(p => p != i).ToList();
                        double oldBIC = ComputeBIC(X, n, d, j, parents[j]);
                        double newBIC = ComputeBIC(X, n, d, j, newParents);
                        double delta = oldBIC - newBIC;

                        if (delta > bestDelta)
                        {
                            bestDelta = delta;
                            bestOp = 1;
                            bestFrom = i;
                            bestTo = j;
                        }
                    }
                }
            }

            if (bestDelta > 0)
            {
                if (bestOp == 0) parents[bestTo].Add(bestFrom);
                else parents[bestTo].Remove(bestFrom);
                improved = true;
            }
        }

        // Convert parent sets to adjacency matrix with BIC-improvement weights
        var W = new double[d, d];
        for (int j = 0; j < d; j++)
        {
            foreach (int i in parents[j])
            {
                double corr = Math.Abs(ComputeCorrelation(X, n, i, j));
                W[i, j] = Math.Max(corr, 0.1);
            }
        }

        return W;
    }

    /// <summary>
    /// Checks if adding edge from → to would create a cycle.
    /// </summary>
    private static bool WouldCreateCycle(List<int>[] parents, int d, int from, int to)
    {
        var visited = new bool[d];
        var stack = new Stack<int>();
        stack.Push(from);

        while (stack.Count > 0)
        {
            int node = stack.Pop();
            if (node == to) continue; // skip the edge we're checking
            if (visited[node]) continue;
            visited[node] = true;

            foreach (int parent in parents[node])
            {
                if (parent == to) return true; // cycle found
                stack.Push(parent);
            }
        }

        return false;
    }
}
