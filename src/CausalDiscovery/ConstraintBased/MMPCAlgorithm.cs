using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.ConstraintBased;

/// <summary>
/// MMPC (Max-Min Parents and Children) — identifies the parents and children of each variable.
/// </summary>
/// <remarks>
/// <para>
/// MMPC identifies the parents and children (direct causes and effects) of each variable
/// using a forward-backward selection procedure based on conditional independence tests.
/// </para>
/// <para>
/// <b>For Beginners:</b> MMPC finds the "direct neighbors" of each variable in the
/// causal graph. It's faster than PC for large graphs because it works locally
/// (one variable at a time) rather than globally.
/// </para>
/// <para>
/// Reference: Tsamardinos et al. (2003), "Algorithms for Large Scale Markov Blanket Discovery".
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class MMPCAlgorithm<T> : ConstraintBasedBase<T>
{
    private double _minAssocThreshold = 0.05;

    /// <inheritdoc/>
    public override string Name => "MMPC";

    /// <inheritdoc/>
    public override bool SupportsLatentConfounders => false;

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    public MMPCAlgorithm(CausalDiscoveryOptions? options = null)
    {
        ApplyConstraintOptions(options);
        if (options?.SignificanceLevel.HasValue == true)
            _minAssocThreshold = options.SignificanceLevel.Value;
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int d = data.Columns;

        // Find parents/children of each variable
        var pcSets = new HashSet<int>[d];
        for (int target = 0; target < d; target++)
        {
            pcSets[target] = FindParentsChildren(data, target);
        }

        // Build symmetric adjacency and orient using constraint tests
        var W = new Matrix<T>(d, d);
        for (int i = 0; i < d; i++)
        {
            for (int j = i + 1; j < d; j++)
            {
                if (pcSets[i].Contains(j) || pcSets[j].Contains(i))
                {
                    double weight = Math.Abs(ComputeCorrelation(data, i, j));
                    T weightT = NumOps.FromDouble(weight);
                    W[i, j] = weightT;
                    W[j, i] = weightT;
                }
            }
        }

        return W;
    }

    private HashSet<int> FindParentsChildren(Matrix<T> data, int target)
    {
        int d = data.Columns;
        var candidates = new HashSet<int>();

        // Forward phase: max-min criterion
        while (candidates.Count < d - 1)
        {
            int bestFeature = -1;
            double bestMinAssoc = double.MinValue;

            for (int j = 0; j < d; j++)
            {
                if (j == target || candidates.Contains(j)) continue;

                double minAssoc = Math.Abs(ComputeCorrelation(data, j, target));

                foreach (int s in candidates)
                {
                    double condAssoc = Math.Abs(ComputePartialCorr(data, j, target, [s]));
                    minAssoc = Math.Min(minAssoc, condAssoc);
                }

                if (minAssoc > bestMinAssoc)
                {
                    bestMinAssoc = minAssoc;
                    bestFeature = j;
                }
            }

            if (bestFeature < 0 || bestMinAssoc < _minAssocThreshold)
                break;

            candidates.Add(bestFeature);
        }

        // Backward phase: remove false positives
        var toRemove = new List<int>();
        foreach (int j in candidates)
        {
            var others = candidates.Where(k => k != j).ToList();
            if (others.Count == 0) continue;

            bool isIndependent = true;
            foreach (int s in others)
            {
                double condAssoc = Math.Abs(ComputePartialCorr(data, j, target, [s]));
                if (condAssoc >= _minAssocThreshold)
                {
                    isIndependent = false;
                    break;
                }
            }

            if (isIndependent)
                toRemove.Add(j);
        }

        foreach (int j in toRemove)
            candidates.Remove(j);

        return candidates;
    }

}
