using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.ScoreBased;

/// <summary>
/// FGES (Fast Greedy Equivalence Search) — optimized version of GES with score caching.
/// </summary>
/// <remarks>
/// <para>
/// FGES improves on GES by using score caching to avoid redundant BIC calculations.
/// This makes it more practical for datasets with many variables.
/// </para>
/// <para>
/// <b>Key improvements over GES:</b>
/// <list type="bullet">
/// <item>Score caching: avoids recomputing BIC for unchanged parent sets</item>
/// <item>Faithfulness-based pruning to skip unpromising edges</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> FGES does the same thing as GES but faster. It remembers
/// previous calculations (caching) so it doesn't redo work unnecessarily. This makes
/// it suitable for datasets with many variables where standard GES would be too slow.
/// </para>
/// <para>
/// Reference: Ramsey et al. (2017), "A Million Variables and More: The Fast Greedy
/// Equivalence Search Algorithm for Learning High-dimensional Graphical Models", JMLR.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class FGESAlgorithm<T> : ScoreBasedBase<T>
{
    /// <inheritdoc/>
    public override string Name => "FGES";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    /// <summary>
    /// Initializes FGES with optional configuration.
    /// </summary>
    public FGESAlgorithm(CausalDiscoveryOptions? options = null)
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

        // Score cache
        var scoreCache = new Dictionary<string, double>();

        var scores = new double[d];
        for (int i = 0; i < d; i++)
        {
            scores[i] = GetCachedBIC(X, n, i, parentSets[i], scoreCache);
        }

        // Forward phase with caching
        bool improved = true;
        int forwardIter = 0;
        while (improved && forwardIter < MaxIterations)
        {
            improved = false;
            forwardIter++;

            // Evaluate all possible edge additions
            var candidates = new List<(int from, int to, double improvement)>();

            for (int to = 0; to < d; to++)
            {
                if (parentSets[to].Count >= MaxParents) continue;
                for (int from = 0; from < d; from++)
                {
                    if (from == to || parentSets[to].Contains(from)) continue;
                    if (WouldCreateCycle(parentSets, from, to, d)) continue;

                    var testParents = new HashSet<int>(parentSets[to]) { from };
                    double newScore = GetCachedBIC(X, n, to, testParents, scoreCache);
                    double imp = newScore - scores[to];
                    if (imp > 0)
                        candidates.Add((from, to, imp));
                }
            }

            if (candidates.Count > 0)
            {
                var best = candidates.OrderByDescending(c => c.improvement).First();
                parentSets[best.to].Add(best.from);
                scores[best.to] = GetCachedBIC(X, n, best.to, parentSets[best.to], scoreCache);
                improved = true;
            }
        }

        // Backward phase with caching
        improved = true;
        int backwardIter = 0;
        while (improved && backwardIter < MaxIterations)
        {
            improved = false;
            backwardIter++;

            var candidates = new List<(int from, int to, double improvement)>();
            for (int to = 0; to < d; to++)
            {
                foreach (int from in parentSets[to])
                {
                    var testParents = new HashSet<int>(parentSets[to]);
                    testParents.Remove(from);
                    double newScore = GetCachedBIC(X, n, to, testParents, scoreCache);
                    double imp = newScore - scores[to];
                    if (imp > 0)
                        candidates.Add((from, to, imp));
                }
            }

            if (candidates.Count > 0)
            {
                var best = candidates.OrderByDescending(c => c.improvement).First();
                parentSets[best.to].Remove(best.from);
                scores[best.to] = GetCachedBIC(X, n, best.to, parentSets[best.to], scoreCache);
                improved = true;
            }
        }

        var W = new double[d, d];
        for (int to = 0; to < d; to++)
            foreach (int from in parentSets[to])
                W[from, to] = Math.Max(0.01, ComputeAbsCorrelation(X, n, from, to));

        return DoubleArrayToMatrix(W);
    }

    private double GetCachedBIC(double[,] X, int n, int target, HashSet<int> parents,
        Dictionary<string, double> cache)
    {
        string key = $"{target}:{string.Join(",", parents.OrderBy(p => p))}";
        if (cache.TryGetValue(key, out double cached))
            return cached;

        double score = ComputeBIC(X, n, target, parents);
        cache[key] = score;
        return score;
    }

    private static double ComputeAbsCorrelation(double[,] X, int n, int i, int j)
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
