using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Clustering.Evaluation;

/// <summary>
/// Adjusted Rand Index for comparing cluster assignments.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The Adjusted Rand Index (ARI) measures agreement between two clusterings,
/// adjusted for chance. Values range from -1 to 1:
/// - 1: Perfect agreement
/// - 0: Random labeling (expected by chance)
/// - Negative: Less agreement than random
/// </para>
/// <para>
/// ARI = (RI - Expected_RI) / (max(RI) - Expected_RI)
/// Where RI is the Rand Index (fraction of pairs that agree).
/// </para>
/// <para><b>For Beginners:</b> ARI measures how well clusterings agree.
///
/// Consider all pairs of points:
/// - True Positive: Same cluster in both
/// - True Negative: Different clusters in both
/// - Pairs that disagree: One says same, other says different
///
/// Rand Index = (TP + TN) / (all pairs)
/// But random clusterings can have high RI by chance!
///
/// Adjusted Rand Index corrects for this:
/// - ARI = 1: Perfect match
/// - ARI = 0: No better than random
/// - ARI < 0: Worse than random (rare)
/// </para>
/// </remarks>
public class AdjustedRandIndex<T> : IExternalClusterMetric<T>
{
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new AdjustedRandIndex instance.
    /// </summary>
    public AdjustedRandIndex()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <inheritdoc />
    public double Compute(Vector<T> trueLabels, Vector<T> predictedLabels)
    {
        int n = trueLabels.Length;

        if (n != predictedLabels.Length)
        {
            throw new ArgumentException("Label vectors must have the same length.");
        }

        if (n < 2)
        {
            return 0;
        }

        // Get unique labels
        var trueUnique = new HashSet<int>();
        var predUnique = new HashSet<int>();

        for (int i = 0; i < n; i++)
        {
            trueUnique.Add((int)_numOps.ToDouble(trueLabels[i]));
            predUnique.Add((int)_numOps.ToDouble(predictedLabels[i]));
        }

        var trueList = trueUnique.ToList();
        var predList = predUnique.ToList();

        // Build contingency table
        var contingency = new int[trueList.Count, predList.Count];
        var trueMap = trueList.Select((v, i) => (v, i)).ToDictionary(x => x.v, x => x.i);
        var predMap = predList.Select((v, i) => (v, i)).ToDictionary(x => x.v, x => x.i);

        for (int i = 0; i < n; i++)
        {
            int trueIdx = trueMap[(int)_numOps.ToDouble(trueLabels[i])];
            int predIdx = predMap[(int)_numOps.ToDouble(predictedLabels[i])];
            contingency[trueIdx, predIdx]++;
        }

        // Compute sums
        var rowSums = new long[trueList.Count];
        var colSums = new long[predList.Count];

        for (int i = 0; i < trueList.Count; i++)
        {
            for (int j = 0; j < predList.Count; j++)
            {
                rowSums[i] += contingency[i, j];
                colSums[j] += contingency[i, j];
            }
        }

        // Compute combinatorial terms
        // Sum of C(n_ij, 2) for all cells
        long sumNij = 0;
        for (int i = 0; i < trueList.Count; i++)
        {
            for (int j = 0; j < predList.Count; j++)
            {
                int nij = contingency[i, j];
                sumNij += Choose2(nij);
            }
        }

        // Sum of C(a_i, 2) for row sums
        long sumA = 0;
        foreach (long a in rowSums)
        {
            sumA += Choose2((int)a);
        }

        // Sum of C(b_j, 2) for column sums
        long sumB = 0;
        foreach (long b in colSums)
        {
            sumB += Choose2((int)b);
        }

        // Total pairs
        long totalPairs = Choose2(n);

        if (totalPairs == 0)
        {
            return 0;
        }

        // Expected index
        double expectedIndex = (double)(sumA * sumB) / totalPairs;

        // Max index
        double maxIndex = 0.5 * (sumA + sumB);

        // Adjusted Rand Index
        double numerator = sumNij - expectedIndex;
        double denominator = maxIndex - expectedIndex;

        if (Math.Abs(denominator) < 1e-10)
        {
            return 0;
        }

        return numerator / denominator;
    }

    private static long Choose2(int n)
    {
        if (n < 2) return 0;
        return (long)n * (n - 1) / 2;
    }
}
