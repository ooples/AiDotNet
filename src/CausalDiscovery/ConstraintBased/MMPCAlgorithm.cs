using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.ConstraintBased;

/// <summary>
/// MMPC (Max-Min Parents and Children) — identifies the parents and children of each variable.
/// </summary>
/// <remarks>
/// <para>
/// MMPC identifies the parents and children (direct causes and effects) of each variable
/// using a forward-backward selection procedure based on conditional independence tests.
/// Unlike the PC algorithm which learns the full graph, MMPC focuses on finding the
/// local neighborhood of each variable, then assembles a global graph.
/// </para>
/// <para>
/// <b>Algorithm:</b>
/// <list type="number">
/// <item>Forward phase: Greedily add the feature most associated with the target,
/// conditioned on the current set (max-min criterion).</item>
/// <item>Backward phase: Remove features that become conditionally independent
/// given the others.</item>
/// <item>Repeat for all variables to build the full graph.</item>
/// </list>
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

    /// <summary>
    /// Initializes MMPC with optional configuration.
    /// </summary>
    public MMPCAlgorithm(CausalDiscoveryOptions? options = null)
    {
        ApplyConstraintOptions(options);
        if (options?.SignificanceLevel.HasValue == true)
            _minAssocThreshold = options.SignificanceLevel.Value;
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

        // Find parents/children of each variable
        var pcSets = new HashSet<int>[d];
        for (int target = 0; target < d; target++)
        {
            pcSets[target] = FindParentsChildren(X, n, d, target);
        }

        // Build symmetric adjacency and orient using constraint tests
        var W = new double[d, d];
        for (int i = 0; i < d; i++)
        {
            for (int j = i + 1; j < d; j++)
            {
                if (pcSets[i].Contains(j) || pcSets[j].Contains(i))
                {
                    double weight = Math.Abs(ComputeCorrelation(X, n, i, j));
                    W[i, j] = weight;
                    W[j, i] = weight;
                }
            }
        }

        return DoubleArrayToMatrix(W);
    }

    private HashSet<int> FindParentsChildren(double[,] X, int n, int d, int target)
    {
        var candidates = new HashSet<int>();

        // Forward phase: max-min criterion
        while (candidates.Count < d - 1)
        {
            int bestFeature = -1;
            double bestMinAssoc = double.MinValue;

            for (int j = 0; j < d; j++)
            {
                if (j == target || candidates.Contains(j)) continue;

                double minAssoc = Math.Abs(ComputeCorrelation(X, n, j, target));

                foreach (int s in candidates)
                {
                    double condAssoc = Math.Abs(ComputePartialCorr(X, n, j, target, [s]));
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
                double condAssoc = Math.Abs(ComputePartialCorr(X, n, j, target, [s]));
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

    private Matrix<T> DoubleArrayToMatrix(double[,] data)
    {
        int rows = data.GetLength(0), cols = data.GetLength(1);
        var result = new Matrix<T>(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result[i, j] = NumOps.FromDouble(data[i, j]);
        return result;
    }
}
