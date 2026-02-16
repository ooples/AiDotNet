using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.ScoreBased;

/// <summary>
/// FGES (Fast Greedy Equivalence Search) — greedy DAG search with BIC score caching.
/// </summary>
/// <remarks>
/// <para>
/// This implementation performs greedy forward (edge addition) and backward (edge removal)
/// search over DAG structures with BIC score caching for efficiency. It uses the same
/// forward-backward structure as GES but with cached score evaluations to avoid redundant
/// BIC computations.
/// </para>
/// <para>
/// <b>For Beginners:</b> FGES searches for the best graph by first greedily adding edges
/// that improve the score, then removing edges that improve the score. Caching remembered
/// scores avoids recalculating the same parent-set configurations.
/// </para>
/// <para>
/// Reference: Ramsey et al. (2017), "A Million Variables and More", JMLR.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class FGESAlgorithm<T> : ScoreBasedBase<T>
{
    /// <inheritdoc/>
    public override string Name => "FGES";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    public FGESAlgorithm(CausalDiscoveryOptions? options = null)
    {
        ApplyScoreOptions(options);
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int d = data.Columns;

        var parentSets = new HashSet<int>[d];
        for (int i = 0; i < d; i++) parentSets[i] = [];

        var scoreCache = new Dictionary<string, double>();
        var scores = new double[d];
        for (int i = 0; i < d; i++)
            scores[i] = GetCachedBIC(data, i, parentSets[i], scoreCache);

        // Forward phase with caching
        bool improved = true;
        int forwardIter = 0;
        while (improved && forwardIter < MaxIterations)
        {
            improved = false;
            forwardIter++;

            var candidates = new List<(int from, int to, double improvement)>();

            for (int to = 0; to < d; to++)
            {
                if (parentSets[to].Count >= MaxParents) continue;
                for (int from = 0; from < d; from++)
                {
                    if (from == to || parentSets[to].Contains(from)) continue;
                    if (WouldCreateCycle(parentSets, from, to)) continue;

                    var testParents = new HashSet<int>(parentSets[to]) { from };
                    double newScore = GetCachedBIC(data, to, testParents, scoreCache);
                    double imp = newScore - scores[to];
                    if (imp > 0)
                        candidates.Add((from, to, imp));
                }
            }

            if (candidates.Count > 0)
            {
                var best = candidates.OrderByDescending(c => c.improvement).First();
                parentSets[best.to].Add(best.from);
                scores[best.to] = GetCachedBIC(data, best.to, parentSets[best.to], scoreCache);
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
                    double newScore = GetCachedBIC(data, to, testParents, scoreCache);
                    double imp = newScore - scores[to];
                    if (imp > 0)
                        candidates.Add((from, to, imp));
                }
            }

            if (candidates.Count > 0)
            {
                var best = candidates.OrderByDescending(c => c.improvement).First();
                parentSets[best.to].Remove(best.from);
                scores[best.to] = GetCachedBIC(data, best.to, parentSets[best.to], scoreCache);
                improved = true;
            }
        }

        var W = new Matrix<T>(d, d);
        for (int to = 0; to < d; to++)
            foreach (int from in parentSets[to])
                W[from, to] = NumOps.FromDouble(Math.Max(1e-10, ComputeAbsCorrelation(data, from, to)));

        return W;
    }

    private double GetCachedBIC(Matrix<T> data, int target, HashSet<int> parents,
        Dictionary<string, double> cache)
    {
        string key = $"{target}:{string.Join(",", parents.OrderBy(p => p))}";
        if (cache.TryGetValue(key, out double cached))
            return cached;

        double score = ComputeBIC(data, target, parents);
        cache[key] = score;
        return score;
    }
}
