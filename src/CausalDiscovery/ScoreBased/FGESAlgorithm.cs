using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.ScoreBased;

/// <summary>
/// FGES (Fast Greedy Equivalence Search) — optimized version of GES with score caching.
/// </summary>
/// <remarks>
/// <para>
/// FGES improves on GES by using score caching to avoid redundant BIC calculations.
/// </para>
/// <para>
/// <b>For Beginners:</b> FGES does the same thing as GES but faster. It remembers
/// previous calculations (caching) so it doesn't redo work unnecessarily.
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
                    if (WouldCreateCycle(parentSets, from, to, d)) continue;

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
                W[from, to] = NumOps.FromDouble(Math.Max(0.01, ComputeAbsCorrelation(data, from, to)));

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
