using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Clustering.Evaluation;

/// <summary>
/// Variation of Information (VI) for comparing clustering results.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Variation of Information is an information-theoretic measure of the distance
/// between two clusterings. It measures the amount of information lost and gained
/// when going from one clustering to another.
/// </para>
/// <para>
/// VI(C, K) = H(C) + H(K) - 2*I(C, K)
///          = H(C|K) + H(K|C)
/// Where:
/// - H(C) = entropy of true clustering
/// - H(K) = entropy of predicted clustering
/// - I(C, K) = mutual information
/// - H(C|K) = conditional entropy of C given K
/// </para>
/// <para><b>For Beginners:</b> Variation of Information asks "How different are these groupings?"
///
/// It measures the information distance between two clusterings:
/// - VI = 0: Identical clusterings (no information lost or gained)
/// - VI &gt; 0: Different clusterings (some information differs)
///
/// Unlike other metrics, LOWER is better for VI (it's a distance measure).
///
/// Think of it as: "How much would I need to change one grouping to get the other?"
/// </para>
/// </remarks>
public class VariationOfInformation<T> : IExternalClusterMetric<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly bool _normalized;

    /// <summary>
    /// Initializes a new VariationOfInformation instance.
    /// </summary>
    /// <param name="normalized">If true, normalize VI to [0, 1] range. Default is false.</param>
    public VariationOfInformation(bool normalized = false)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _normalized = normalized;
    }

    /// <inheritdoc />
    public double Compute(Vector<T> trueLabels, Vector<T> predictedLabels)
    {
        int n = trueLabels.Length;

        if (n != predictedLabels.Length)
        {
            throw new ArgumentException("Label vectors must have the same length.");
        }

        if (n == 0)
        {
            return 0;
        }

        // Compute entropies
        double hC = ComputeEntropy(trueLabels, n);
        double hK = ComputeEntropy(predictedLabels, n);
        double mi = ComputeMutualInformation(trueLabels, predictedLabels, n);

        double vi = hC + hK - 2 * mi;

        if (_normalized)
        {
            double maxEntropy = Math.Max(hC, hK);
            if (maxEntropy > 0)
            {
                // Normalized VI = VI / (H(C) + H(K))
                double jointEntropy = hC + hK;
                if (jointEntropy > 0)
                {
                    return vi / jointEntropy;
                }
            }
            return 0;
        }

        return vi;
    }

    /// <summary>
    /// Computes the Normalized Mutual Information (NMI).
    /// </summary>
    /// <param name="trueLabels">The ground truth labels.</param>
    /// <param name="predictedLabels">The clustering assignments.</param>
    /// <returns>NMI value in [0, 1] where 1 indicates identical clusterings.</returns>
    /// <remarks>
    /// <para>
    /// NMI = 2 * I(C, K) / (H(C) + H(K))
    /// This is the complement of normalized VI: NMI = 1 - NVI
    /// </para>
    /// </remarks>
    public double ComputeNMI(Vector<T> trueLabels, Vector<T> predictedLabels)
    {
        int n = trueLabels.Length;

        if (n == 0)
        {
            return 1.0;
        }

        double hC = ComputeEntropy(trueLabels, n);
        double hK = ComputeEntropy(predictedLabels, n);
        double mi = ComputeMutualInformation(trueLabels, predictedLabels, n);

        double denominator = hC + hK;
        if (denominator == 0)
        {
            return 1.0; // Both have zero entropy (single cluster each)
        }

        return 2 * mi / denominator;
    }

    /// <summary>
    /// Computes the Adjusted Mutual Information (AMI).
    /// </summary>
    /// <param name="trueLabels">The ground truth labels.</param>
    /// <param name="predictedLabels">The clustering assignments.</param>
    /// <returns>AMI value, corrected for chance. 1.0 indicates perfect agreement.</returns>
    /// <remarks>
    /// <para>
    /// AMI adjusts NMI for chance agreement, similar to how ARI adjusts Rand Index.
    /// AMI = (MI - E[MI]) / (max(H(C), H(K)) - E[MI])
    /// </para>
    /// </remarks>
    public double ComputeAMI(Vector<T> trueLabels, Vector<T> predictedLabels)
    {
        int n = trueLabels.Length;

        if (n == 0)
        {
            return 1.0;
        }

        // Build contingency table
        var contingency = new Dictionary<(int True, int Pred), int>();
        var trueCounts = new Dictionary<int, int>();
        var predCounts = new Dictionary<int, int>();

        for (int i = 0; i < n; i++)
        {
            int trueLabel = (int)_numOps.ToDouble(trueLabels[i]);
            int predLabel = (int)_numOps.ToDouble(predictedLabels[i]);

            var key = (trueLabel, predLabel);
            contingency.TryAdd(key, 0);
            contingency[key]++;

            trueCounts.TryAdd(trueLabel, 0);
            trueCounts[trueLabel]++;

            predCounts.TryAdd(predLabel, 0);
            predCounts[predLabel]++;
        }

        double hC = ComputeEntropy(trueLabels, n);
        double hK = ComputeEntropy(predictedLabels, n);
        double mi = ComputeMutualInformation(trueLabels, predictedLabels, n);

        // Compute expected MI under hypergeometric model
        double emi = ComputeExpectedMI(trueCounts, predCounts, n);

        double maxH = Math.Max(hC, hK);
        double denominator = maxH - emi;

        if (Math.Abs(denominator) < 1e-10)
        {
            return 1.0;
        }

        return (mi - emi) / denominator;
    }

    private double ComputeEntropy(Vector<T> labels, int n)
    {
        var counts = new Dictionary<int, int>();

        for (int i = 0; i < n; i++)
        {
            int label = (int)_numOps.ToDouble(labels[i]);
            counts.TryAdd(label, 0);
            counts[label]++;
        }

        double entropy = 0;
        foreach (int count in counts.Values)
        {
            if (count > 0)
            {
                double p = (double)count / n;
                entropy -= p * Math.Log2(p);
            }
        }

        return entropy;
    }

    private double ComputeMutualInformation(Vector<T> trueLabels, Vector<T> predictedLabels, int n)
    {
        // Build contingency table
        var contingency = new Dictionary<(int True, int Pred), int>();
        var trueCounts = new Dictionary<int, int>();
        var predCounts = new Dictionary<int, int>();

        for (int i = 0; i < n; i++)
        {
            int trueLabel = (int)_numOps.ToDouble(trueLabels[i]);
            int predLabel = (int)_numOps.ToDouble(predictedLabels[i]);

            var key = (trueLabel, predLabel);
            contingency.TryAdd(key, 0);
            contingency[key]++;

            trueCounts.TryAdd(trueLabel, 0);
            trueCounts[trueLabel]++;

            predCounts.TryAdd(predLabel, 0);
            predCounts[predLabel]++;
        }

        // Compute MI
        double mi = 0;
        foreach (var kvp in contingency)
        {
            int nij = kvp.Value;
            int ni = trueCounts[kvp.Key.True];
            int nj = predCounts[kvp.Key.Pred];

            if (nij > 0)
            {
                double pij = (double)nij / n;
                double pi = (double)ni / n;
                double pj = (double)nj / n;
                mi += pij * Math.Log2(pij / (pi * pj));
            }
        }

        return mi;
    }

    private double ComputeExpectedMI(Dictionary<int, int> trueCounts, Dictionary<int, int> predCounts, int n)
    {
        // Approximate expected MI using the formula:
        // E[MI] ≈ sum over all (i,j) of expected contribution
        // This is a simplified approximation

        double emi = 0;
        var trueList = trueCounts.Values.ToList();
        var predList = predCounts.Values.ToList();

        foreach (int ai in trueList)
        {
            foreach (int bj in predList)
            {
                // Expected overlap under hypergeometric distribution
                double expectedNij = (double)ai * bj / n;
                if (expectedNij > 0)
                {
                    double pij = expectedNij / n;
                    double pi = (double)ai / n;
                    double pj = (double)bj / n;
                    if (pij > 0 && pi > 0 && pj > 0)
                    {
                        emi += pij * Math.Log2(pij / (pi * pj));
                    }
                }
            }
        }

        return Math.Max(0, emi);
    }
}

/// <summary>
/// Adjusted Mutual Information (AMI) for comparing clustering results.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// AMI adjusts NMI for chance agreement. Random clusterings will have
/// AMI close to 0, while perfect agreement gives AMI = 1.
/// </para>
/// <para><b>For Beginners:</b> AMI is like NMI but corrected for luck.
///
/// Regular NMI can be high just by chance if you have many clusters.
/// AMI asks "How much better than random?"
///
/// - AMI = 0: No better than random
/// - AMI = 1: Perfect agreement
/// - AMI can be negative if worse than random
/// </para>
/// </remarks>
public class AdjustedMutualInformation<T> : IExternalClusterMetric<T>
{
    private readonly VariationOfInformation<T> _vi;

    /// <summary>
    /// Initializes a new AdjustedMutualInformation instance.
    /// </summary>
    public AdjustedMutualInformation()
    {
        _vi = new VariationOfInformation<T>();
    }

    /// <inheritdoc />
    public double Compute(Vector<T> trueLabels, Vector<T> predictedLabels)
    {
        return _vi.ComputeAMI(trueLabels, predictedLabels);
    }
}
