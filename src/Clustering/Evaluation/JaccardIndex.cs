using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Clustering.Evaluation;

/// <summary>
/// Jaccard Index for comparing clustering results against ground truth.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The Jaccard Index measures the similarity between two clusterings by computing
/// the ratio of pairs that are in the same cluster in both clusterings to pairs
/// that are in the same cluster in at least one clustering.
/// </para>
/// <para>
/// Jaccard = a / (a + b + c)
/// Where:
/// - a = pairs in same cluster in both clusterings
/// - b = pairs in same cluster only in true labels
/// - c = pairs in same cluster only in predicted labels
/// </para>
/// <para><b>For Beginners:</b> Jaccard Index asks "How similar are the groupings?"
///
/// It compares every pair of points:
/// - "Are these two together in the true groups?"
/// - "Are these two together in the predicted groups?"
///
/// Then it calculates:
/// - Agreement = Both say together OR both say apart
/// - Jaccard = Pairs together in both / Pairs together in at least one
///
/// Values range from 0 (completely different) to 1 (identical clusterings).
/// Higher is better!
/// </para>
/// </remarks>
public class JaccardIndex<T> : IExternalClusterMetric<T>
{
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new JaccardIndex instance.
    /// </summary>
    public JaccardIndex()
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
            return 1.0; // No pairs to compare
        }

        long a = 0; // Same in both
        long b = 0; // Same in true only
        long c = 0; // Same in predicted only

        for (int i = 0; i < n; i++)
        {
            int trueI = (int)_numOps.ToDouble(trueLabels[i]);
            int predI = (int)_numOps.ToDouble(predictedLabels[i]);

            for (int j = i + 1; j < n; j++)
            {
                int trueJ = (int)_numOps.ToDouble(trueLabels[j]);
                int predJ = (int)_numOps.ToDouble(predictedLabels[j]);

                bool sameTrue = trueI == trueJ;
                bool samePred = predI == predJ;

                if (sameTrue && samePred)
                {
                    a++; // Same in both
                }
                else if (sameTrue && !samePred)
                {
                    b++; // Same in true only
                }
                else if (!sameTrue && samePred)
                {
                    c++; // Same in predicted only
                }
                // d = different in both (not used in Jaccard)
            }
        }

        long denominator = a + b + c;
        return denominator > 0 ? (double)a / denominator : 1.0;
    }

    /// <summary>
    /// Computes the pair confusion matrix components.
    /// </summary>
    /// <param name="trueLabels">The ground truth labels.</param>
    /// <param name="predictedLabels">The clustering assignments.</param>
    /// <returns>Tuple of (a, b, c, d) where a=same in both, b=same in true only, c=same in pred only, d=different in both.</returns>
    public (long A, long B, long C, long D) ComputePairConfusionMatrix(Vector<T> trueLabels, Vector<T> predictedLabels)
    {
        int n = trueLabels.Length;

        long a = 0, b = 0, c = 0, d = 0;

        for (int i = 0; i < n; i++)
        {
            int trueI = (int)_numOps.ToDouble(trueLabels[i]);
            int predI = (int)_numOps.ToDouble(predictedLabels[i]);

            for (int j = i + 1; j < n; j++)
            {
                int trueJ = (int)_numOps.ToDouble(trueLabels[j]);
                int predJ = (int)_numOps.ToDouble(predictedLabels[j]);

                bool sameTrue = trueI == trueJ;
                bool samePred = predI == predJ;

                if (sameTrue && samePred) a++;
                else if (sameTrue && !samePred) b++;
                else if (!sameTrue && samePred) c++;
                else d++;
            }
        }

        return (a, b, c, d);
    }
}

/// <summary>
/// Rand Index and Adjusted Rand Index for comparing clustering results.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The Rand Index measures the percentage of pair decisions that agree between
/// two clusterings. The Adjusted Rand Index corrects for chance agreement.
/// </para>
/// <para><b>For Beginners:</b>
/// Rand Index asks "What fraction of pairs do we agree on?"
/// - If true labels say "together", do we also say "together"?
/// - If true labels say "apart", do we also say "apart"?
///
/// Adjusted Rand Index goes further by asking "How much better than random?"
/// - ARI = 0 means no better than random
/// - ARI = 1 means perfect agreement
/// - ARI can be negative if worse than random
/// </para>
/// </remarks>
public class RandIndex<T> : IExternalClusterMetric<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly JaccardIndex<T> _jaccard;
    private readonly bool _adjusted;

    /// <summary>
    /// Initializes a new RandIndex instance.
    /// </summary>
    /// <param name="adjusted">If true, compute Adjusted Rand Index. Default is false.</param>
    public RandIndex(bool adjusted = false)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _jaccard = new JaccardIndex<T>();
        _adjusted = adjusted;
    }

    /// <inheritdoc />
    public double Compute(Vector<T> trueLabels, Vector<T> predictedLabels)
    {
        int n = trueLabels.Length;

        if (n < 2)
        {
            return 1.0;
        }

        var (a, b, c, d) = _jaccard.ComputePairConfusionMatrix(trueLabels, predictedLabels);

        if (_adjusted)
        {
            // For adjusted, we need to compute expected value
            return ComputeAdjustedRandIndex(trueLabels, predictedLabels);
        }

        // Simple Rand Index
        long totalPairs = a + b + c + d;
        return (double)(a + d) / totalPairs;
    }

    private double ComputeAdjustedRandIndex(Vector<T> trueLabels, Vector<T> predictedLabels)
    {
        int n = trueLabels.Length;

        // Build contingency table
        var contingency = new Dictionary<(int True, int Pred), int>();
        var trueCounts = new Dictionary<int, int>();
        var predCounts = new Dictionary<int, int>();

        for (int i = 0; i < n; i++)
        {
            int trueLabel = (int)_numOps.ToDouble(trueLabels[i]);
            int predLabel = (int)_numOps.ToDouble(predictedLabels[i]);

            var key = (trueLabel, predLabel);
            if (!contingency.ContainsKey(key))
                contingency[key] = 0;
            contingency[key]++;

            if (!trueCounts.ContainsKey(trueLabel))
                trueCounts[trueLabel] = 0;
            trueCounts[trueLabel]++;

            if (!predCounts.ContainsKey(predLabel))
                predCounts[predLabel] = 0;
            predCounts[predLabel]++;
        }

        // Calculate sums using the contingency table
        double sumNij2 = 0;
        foreach (var count in contingency.Values)
        {
            sumNij2 += Binomial(count, 2);
        }

        double sumA2 = 0;
        foreach (var count in trueCounts.Values)
        {
            sumA2 += Binomial(count, 2);
        }

        double sumB2 = 0;
        foreach (var count in predCounts.Values)
        {
            sumB2 += Binomial(count, 2);
        }

        double totalPairs = Binomial(n, 2);
        double expectedIndex = (sumA2 * sumB2) / totalPairs;
        double maxIndex = 0.5 * (sumA2 + sumB2);

        if (maxIndex == expectedIndex)
        {
            return 1.0; // Perfect agreement when all same
        }

        return (sumNij2 - expectedIndex) / (maxIndex - expectedIndex);
    }

    private static double Binomial(int n, int k)
    {
        if (k > n || k < 0) return 0;
        if (k == 0 || k == n) return 1;
        if (k == 1) return n;
        if (k == 2) return (double)n * (n - 1) / 2;

        // General case
        double result = 1;
        for (int i = 0; i < k; i++)
        {
            result = result * (n - i) / (i + 1);
        }
        return result;
    }
}
