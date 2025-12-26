using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Clustering.Evaluation;

/// <summary>
/// Normalized Mutual Information for comparing cluster assignments.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Normalized Mutual Information (NMI) measures the mutual information
/// between two clusterings, normalized to range [0, 1]:
/// - 1: Perfect agreement
/// - 0: No mutual information (independent)
/// </para>
/// <para>
/// NMI = 2 * I(U, V) / (H(U) + H(V))
/// Where:
/// - I(U, V) = Mutual information between clusterings U and V
/// - H(U), H(V) = Entropy of each clustering
/// </para>
/// <para><b>For Beginners:</b> NMI measures shared information between clusterings.
///
/// Mutual Information asks:
/// "How much does knowing clustering U tell me about clustering V?"
///
/// Entropy H(U) measures uncertainty:
/// - Low entropy: Few clusters, predictable
/// - High entropy: Many equal-sized clusters, less predictable
///
/// NMI normalizes by average entropy so:
/// - NMI = 1: Knowing U completely determines V
/// - NMI = 0: U tells nothing about V (independent)
///
/// Unlike ARI, NMI is always non-negative.
/// </para>
/// </remarks>
public class NormalizedMutualInformation<T> : IExternalClusterMetric<T>
{
    private readonly NMINormalization _normalization;
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new NormalizedMutualInformation instance.
    /// </summary>
    /// <param name="normalization">Normalization method. Default is Arithmetic.</param>
    public NormalizedMutualInformation(NMINormalization normalization = NMINormalization.Arithmetic)
    {
        _normalization = normalization;
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

        if (n == 0)
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

        // Compute marginal probabilities
        var rowSums = new int[trueList.Count];
        var colSums = new int[predList.Count];

        for (int i = 0; i < trueList.Count; i++)
        {
            for (int j = 0; j < predList.Count; j++)
            {
                rowSums[i] += contingency[i, j];
                colSums[j] += contingency[i, j];
            }
        }

        // Compute entropy of true labels H(U)
        double hU = 0;
        foreach (int sum in rowSums)
        {
            if (sum > 0)
            {
                double p = (double)sum / n;
                hU -= p * Math.Log(p);
            }
        }

        // Compute entropy of predicted labels H(V)
        double hV = 0;
        foreach (int sum in colSums)
        {
            if (sum > 0)
            {
                double p = (double)sum / n;
                hV -= p * Math.Log(p);
            }
        }

        // Compute mutual information I(U, V)
        double mi = 0;
        for (int i = 0; i < trueList.Count; i++)
        {
            for (int j = 0; j < predList.Count; j++)
            {
                if (contingency[i, j] > 0 && rowSums[i] > 0 && colSums[j] > 0)
                {
                    double pij = (double)contingency[i, j] / n;
                    double pi = (double)rowSums[i] / n;
                    double pj = (double)colSums[j] / n;
                    mi += pij * Math.Log(pij / (pi * pj));
                }
            }
        }

        // Normalize
        double normalizer = _normalization switch
        {
            NMINormalization.Arithmetic => (hU + hV) / 2,
            NMINormalization.Geometric => Math.Sqrt(hU * hV),
            NMINormalization.Min => Math.Min(hU, hV),
            NMINormalization.Max => Math.Max(hU, hV),
            _ => (hU + hV) / 2
        };

        if (normalizer < 1e-10)
        {
            return 0;
        }

        return mi / normalizer;
    }
}

/// <summary>
/// Normalization methods for NMI.
/// </summary>
public enum NMINormalization
{
    /// <summary>
    /// Arithmetic mean of entropies: (H(U) + H(V)) / 2
    /// </summary>
    Arithmetic,

    /// <summary>
    /// Geometric mean of entropies: sqrt(H(U) * H(V))
    /// </summary>
    Geometric,

    /// <summary>
    /// Minimum of entropies: min(H(U), H(V))
    /// </summary>
    Min,

    /// <summary>
    /// Maximum of entropies: max(H(U), H(V))
    /// </summary>
    Max
}
