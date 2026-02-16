using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.ConstraintBased;

/// <summary>
/// Markov Blanket (Grow-Shrink) Algorithm — discovers the Markov blanket of each variable.
/// </summary>
/// <remarks>
/// <para>
/// The Grow-Shrink algorithm identifies the Markov blanket of each variable through two phases:
/// <list type="number">
/// <item><b>Growing:</b> Add variables that increase conditional mutual information with the target.</item>
/// <item><b>Shrinking:</b> Remove variables that become conditionally independent given the rest.</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of a variable's Markov blanket as a "shield" — if you know
/// all variables in the blanket, no other variable can provide additional information about
/// the target. This algorithm finds that shield by adding helpful variables (growing) and
/// then removing redundant ones (shrinking).
/// </para>
/// <para>
/// Reference: Margaritis and Thrun (1999), "Bayesian Network Induction via Local Neighborhoods".
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
internal class MarkovBlanketAlgorithm<T> : ConstraintBasedBase<T>
{
    private const int DefaultDiscretizationBins = 10;
    private const double DefaultMIThreshold = 0.01;
    private const double MinDiscretizationRange = 1e-10;
    private const int MinGroupSizeForMI = 5;

    private readonly int _nBins = DefaultDiscretizationBins;
    private readonly double _miThreshold = DefaultMIThreshold;

    /// <inheritdoc/>
    public override string Name => "Markov Blanket (Grow-Shrink)";

    /// <inheritdoc/>
    public override bool SupportsLatentConfounders => false;

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    public MarkovBlanketAlgorithm(CausalDiscoveryOptions? options = null)
    {
        ApplyConstraintOptions(options);
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;

        if (n == 0)
            return new Matrix<T>(d, d);

        // Discretize data for MI computation (uses int[,] which is legitimate for binned indices)
        var X = DiscretizeData(data, n, d);

        // Find Markov blanket for each variable
        var blankets = new HashSet<int>[d];
        for (int target = 0; target < d; target++)
        {
            blankets[target] = FindMarkovBlanket(X, n, d, target);
        }

        // Build adjacency using Matrix<T>
        var W = new Matrix<T>(d, d);
        for (int i = 0; i < d; i++)
        {
            for (int j = i + 1; j < d; j++)
            {
                if (blankets[i].Contains(j) || blankets[j].Contains(i))
                {
                    double mi = ComputeMI(X, i, j, n);
                    T miT = NumOps.FromDouble(mi);
                    W[i, j] = miT;
                    W[j, i] = miT;
                }
            }
        }

        return W;
    }

    private HashSet<int> FindMarkovBlanket(int[,] X, int n, int d, int target)
    {
        var scores = new double[d];
        for (int j = 0; j < d; j++)
        {
            if (j == target) continue;
            scores[j] = ComputeMI(X, j, target, n);
        }

        var candidates = Enumerable.Range(0, d)
            .Where(j => j != target)
            .OrderByDescending(j => scores[j])
            .ToList();

        // Growing phase
        var blanket = new HashSet<int>();
        foreach (int candidate in candidates)
        {
            double condMI = ComputeConditionalMI(X, candidate, target, blanket, n);
            if (condMI > _miThreshold)
                blanket.Add(candidate);
        }

        // Shrinking phase
        var toRemove = new List<int>();
        foreach (int feature in blanket)
        {
            var others = blanket.Where(f => f != feature).ToHashSet();
            double condMI = ComputeConditionalMI(X, feature, target, others, n);
            if (condMI <= _miThreshold)
                toRemove.Add(feature);
        }

        foreach (int feature in toRemove)
            blanket.Remove(feature);

        return blanket;
    }

    private int[,] DiscretizeData(Matrix<T> data, int n, int d)
    {
        var result = new int[n, d];
        for (int j = 0; j < d; j++)
        {
            double min = double.MaxValue, max = double.MinValue;
            for (int i = 0; i < n; i++)
            {
                double val = NumOps.ToDouble(data[i, j]);
                min = Math.Min(min, val);
                max = Math.Max(max, val);
            }

            double range = max - min;
            for (int i = 0; i < n; i++)
            {
                double val = NumOps.ToDouble(data[i, j]);
                result[i, j] = range > MinDiscretizationRange
                    ? Math.Min((int)((val - min) / range * (_nBins - 1)), _nBins - 1)
                    : 0;
            }
        }

        return result;
    }

    private double ComputeMI(int[,] X, int col1, int col2, int n)
    {
        var jointCounts = new Dictionary<(int, int), int>();
        var counts1 = new int[_nBins];
        var counts2 = new int[_nBins];

        for (int i = 0; i < n; i++)
        {
            int v1 = X[i, col1], v2 = X[i, col2];
            counts1[v1]++;
            counts2[v2]++;
            var key = (v1, v2);
            jointCounts[key] = jointCounts.GetValueOrDefault(key) + 1;
        }

        double mi = 0;
        foreach (var kvp in jointCounts)
        {
            double pxy = (double)kvp.Value / n;
            double px = (double)counts1[kvp.Key.Item1] / n;
            double py = (double)counts2[kvp.Key.Item2] / n;
            if (pxy > 0 && px > 0 && py > 0)
                mi += pxy * Math.Log(pxy / (px * py));
        }

        return mi;
    }

    private double ComputeConditionalMI(int[,] X, int feature, int target, HashSet<int> condSet, int n)
    {
        if (condSet.Count == 0)
            return ComputeMI(X, feature, target, n);

        var condVars = condSet.ToArray();
        var groups = new Dictionary<string, List<int>>();
        for (int i = 0; i < n; i++)
        {
            var key = string.Join(",", condVars.Select(c => X[i, c].ToString()));
            if (!groups.TryGetValue(key, out var list))
            {
                list = [];
                groups[key] = list;
            }
            list.Add(i);
        }

        double condMI = 0;
        foreach (var group in groups.Values)
        {
            if (group.Count <= MinGroupSizeForMI) continue;
            int groupN = group.Count;

            var jointCounts = new Dictionary<(int, int), int>();
            var xCounts = new int[_nBins];
            var yCounts = new int[_nBins];

            foreach (int i in group)
            {
                int xVal = X[i, feature], yVal = X[i, target];
                xCounts[xVal]++;
                yCounts[yVal]++;
                var key = (xVal, yVal);
                jointCounts[key] = jointCounts.GetValueOrDefault(key) + 1;
            }

            double groupMI = 0;
            foreach (var kvp in jointCounts)
            {
                double pxy = (double)kvp.Value / groupN;
                double px = (double)xCounts[kvp.Key.Item1] / groupN;
                double py = (double)yCounts[kvp.Key.Item2] / groupN;
                if (pxy > 0 && px > 0 && py > 0)
                    groupMI += pxy * Math.Log(pxy / (px * py));
            }

            condMI += groupMI * ((double)groupN / n);
        }

        return condMI;
    }

}
