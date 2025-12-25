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
    private static readonly double Log2 = Math.Log(2);

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

        // Build contingency table once
        var (contingency, trueCounts, predCounts) = BuildContingencyTable(trueLabels, predictedLabels);

        // Compute entropies
        double hC = ComputeEntropyFromCounts(trueCounts, n);
        double hK = ComputeEntropyFromCounts(predCounts, n);
        double mi = ComputeMutualInformationFromContingency(contingency, trueCounts, predCounts, n);

        double vi = hC + hK - 2 * mi;

        if (_normalized)
        {
            double jointEntropy = hC + hK;
            if (jointEntropy > 0)
            {
                return vi / jointEntropy;
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
    /// This is the arithmetic mean normalization variant.
    /// </para>
    /// </remarks>
    public double ComputeNMI(Vector<T> trueLabels, Vector<T> predictedLabels)
    {
        int n = trueLabels.Length;

        if (n == 0)
        {
            return 1.0;
        }

        var (contingency, trueCounts, predCounts) = BuildContingencyTable(trueLabels, predictedLabels);

        double hC = ComputeEntropyFromCounts(trueCounts, n);
        double hK = ComputeEntropyFromCounts(predCounts, n);
        double mi = ComputeMutualInformationFromContingency(contingency, trueCounts, predCounts, n);

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
    /// AMI adjusts NMI for chance agreement using the hypergeometric model.
    /// AMI = (MI - E[MI]) / (mean(H(C), H(K)) - E[MI])
    ///
    /// This implementation uses the exact hypergeometric expectation formula
    /// from Vinh, Epps, and Bailey (2010).
    /// </para>
    /// </remarks>
    public double ComputeAMI(Vector<T> trueLabels, Vector<T> predictedLabels)
    {
        int n = trueLabels.Length;

        if (n == 0)
        {
            return 1.0;
        }

        var (contingency, trueCounts, predCounts) = BuildContingencyTable(trueLabels, predictedLabels);

        double hC = ComputeEntropyFromCounts(trueCounts, n);
        double hK = ComputeEntropyFromCounts(predCounts, n);
        double mi = ComputeMutualInformationFromContingency(contingency, trueCounts, predCounts, n);

        // Compute expected MI under hypergeometric null model
        double emi = ComputeExpectedMI(trueCounts, predCounts, n);

        // Use arithmetic mean of entropies as normalizer (most common variant)
        double meanH = (hC + hK) / 2.0;
        double denominator = meanH - emi;

        if (Math.Abs(denominator) < 1e-15)
        {
            // When denominator is zero, AMI is defined as 1 if MI == EMI, else 0
            return Math.Abs(mi - emi) < 1e-15 ? 1.0 : 0.0;
        }

        return (mi - emi) / denominator;
    }

    private (Dictionary<(int, int), int>, Dictionary<int, int>, Dictionary<int, int>) BuildContingencyTable(
        Vector<T> trueLabels, Vector<T> predictedLabels)
    {
        int n = trueLabels.Length;
        var contingency = new Dictionary<(int, int), int>();
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

        return (contingency, trueCounts, predCounts);
    }

    private double ComputeEntropyFromCounts(Dictionary<int, int> counts, int n)
    {
        double entropy = 0;
        foreach (int count in counts.Values)
        {
            if (count > 0)
            {
                double p = (double)count / n;
                entropy -= p * Math.Log(p) / Log2;
            }
        }
        return entropy;
    }

    private double ComputeMutualInformationFromContingency(
        Dictionary<(int, int), int> contingency,
        Dictionary<int, int> trueCounts,
        Dictionary<int, int> predCounts,
        int n)
    {
        double mi = 0;
        foreach (var kvp in contingency)
        {
            int nij = kvp.Value;
            int ni = trueCounts[kvp.Key.Item1];
            int nj = predCounts[kvp.Key.Item2];

            if (nij > 0)
            {
                double pij = (double)nij / n;
                double pi = (double)ni / n;
                double pj = (double)nj / n;
                mi += pij * Math.Log(pij / (pi * pj)) / Log2;
            }
        }
        return mi;
    }

    /// <summary>
    /// Computes the expected mutual information under the hypergeometric null model.
    /// Uses the exact formula from Vinh, Epps, and Bailey (2010):
    /// "Information Theoretic Measures for Clusterings Comparison"
    /// </summary>
    private double ComputeExpectedMI(Dictionary<int, int> trueCounts, Dictionary<int, int> predCounts, int n)
    {
        double emi = 0;
        var trueList = trueCounts.Values.ToArray();
        var predList = predCounts.Values.ToArray();

        // Precompute log factorials for efficiency
        var logFact = new double[n + 1];
        logFact[0] = 0;
        for (int i = 1; i <= n; i++)
        {
            logFact[i] = logFact[i - 1] + Math.Log(i);
        }

        foreach (int ai in trueList)
        {
            foreach (int bj in predList)
            {
                // Sum over all possible values of nij
                int nijMin = Math.Max(1, ai + bj - n);
                int nijMax = Math.Min(ai, bj);

                for (int nij = nijMin; nij <= nijMax; nij++)
                {
                    // Compute log of hypergeometric probability
                    // P(nij) = C(ai, nij) * C(n-ai, bj-nij) / C(n, bj)
                    double logProb = LogBinomial(ai, nij, logFact)
                                   + LogBinomial(n - ai, bj - nij, logFact)
                                   - LogBinomial(n, bj, logFact);

                    if (double.IsNegativeInfinity(logProb))
                        continue;

                    double prob = Math.Exp(logProb);

                    // Contribution to EMI
                    double logTerm = Math.Log((double)nij * n / ((double)ai * bj));
                    if (!double.IsNaN(logTerm) && !double.IsInfinity(logTerm))
                    {
                        emi += (double)nij / n * logTerm / Log2 * prob;
                    }
                }
            }
        }

        return Math.Max(0, emi);
    }

    private static double LogBinomial(int n, int k, double[] logFact)
    {
        if (k < 0 || k > n)
            return double.NegativeInfinity;
        if (k == 0 || k == n)
            return 0;
        return logFact[n] - logFact[k] - logFact[n - k];
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
