using AiDotNet.Clustering.Base;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Clustering.Evaluation;

/// <summary>
/// Base class for external cluster evaluation metrics that compare against ground truth.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// External cluster metrics compare clustering results against known ground truth labels.
/// This base class provides common functionality shared by all external metrics including
/// contingency table construction, entropy calculations, and pair counting.
/// </para>
/// <para><b>For Beginners:</b> This base class handles the common math that all
/// external metrics need:
/// - Building contingency tables (cross-tabulation of true vs predicted)
/// - Computing entropy (how "mixed" or "uncertain" clusters are)
/// - Counting pairs (how many point-pairs agree/disagree)
///
/// Individual metrics focus on their specific formulas while this class handles the plumbing.
/// </para>
/// </remarks>
public abstract class ExternalClusterMetricBase<T> : IExternalClusterMetric<T>
{
    /// <summary>
    /// The numeric operations instance for type T.
    /// </summary>
    protected readonly INumericOperations<T> NumOps;

    /// <summary>
    /// Precomputed log(2) for entropy calculations.
    /// </summary>
    protected static readonly double Log2 = Math.Log(2);

    /// <summary>
    /// Initializes a new instance of the ExternalClusterMetricBase class.
    /// </summary>
    protected ExternalClusterMetricBase()
    {
        NumOps = MathHelper.GetNumericOperations<T>();
    }

    /// <inheritdoc />
    public abstract double Compute(Vector<T> trueLabels, Vector<T> predictedLabels);

    /// <summary>
    /// Builds a contingency table from true and predicted labels.
    /// </summary>
    /// <param name="trueLabels">The ground truth labels.</param>
    /// <param name="predictedLabels">The predicted cluster assignments.</param>
    /// <returns>A tuple containing the contingency table, true label counts, and predicted label counts.</returns>
    /// <remarks>
    /// <para>
    /// The contingency table is a cross-tabulation showing how many points
    /// with each true label ended up in each predicted cluster.
    /// </para>
    /// </remarks>
    protected (Dictionary<(int True, int Pred), int> Contingency,
               Dictionary<int, int> TrueCounts,
               Dictionary<int, int> PredCounts)
        BuildContingencyTable(Vector<T> trueLabels, Vector<T> predictedLabels)
    {
        int n = trueLabels.Length;
        var contingency = new Dictionary<(int True, int Pred), int>();
        var trueCounts = new Dictionary<int, int>();
        var predCounts = new Dictionary<int, int>();

        var trueIds = ClusteringTensorOps.ToIntArray(trueLabels);
        var predIds = ClusteringTensorOps.ToIntArray(predictedLabels);
        for (int i = 0; i < n; i++)
        {
            int trueLabel = trueIds[i];
            int predLabel = predIds[i];

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

    /// <summary>
    /// Computes the entropy of a label distribution.
    /// </summary>
    /// <param name="counts">Dictionary mapping labels to their counts.</param>
    /// <param name="total">Total number of samples.</param>
    /// <returns>The entropy in bits.</returns>
    protected double ComputeEntropyFromCounts(Dictionary<int, int> counts, int total)
    {
        if (total == 0) return 0;

        double entropy = 0;
        foreach (int count in counts.Values)
        {
            if (count > 0)
            {
                double p = (double)count / total;
                entropy -= p * Math.Log(p) / Log2;
            }
        }
        return entropy;
    }

    /// <summary>
    /// Computes the entropy of a label vector.
    /// </summary>
    /// <param name="labels">The label vector.</param>
    /// <returns>The entropy in bits.</returns>
    protected double ComputeEntropy(Vector<T> labels)
    {
        int n = labels.Length;
        if (n == 0) return 0;

        var counts = new Dictionary<int, int>();
        foreach (int label in ClusteringTensorOps.ToIntArray(labels))
        {
            if (!counts.ContainsKey(label))
                counts[label] = 0;
            counts[label]++;
        }

        return ComputeEntropyFromCounts(counts, n);
    }

    /// <summary>
    /// Computes mutual information from a contingency table.
    /// </summary>
    /// <param name="contingency">The contingency table.</param>
    /// <param name="trueCounts">Counts per true label.</param>
    /// <param name="predCounts">Counts per predicted label.</param>
    /// <param name="total">Total number of samples.</param>
    /// <returns>The mutual information in bits.</returns>
    protected double ComputeMutualInformation(
        Dictionary<(int True, int Pred), int> contingency,
        Dictionary<int, int> trueCounts,
        Dictionary<int, int> predCounts,
        int total)
    {
        if (total == 0) return 0;

        double mi = 0;
        foreach (var kvp in contingency)
        {
            int nij = kvp.Value;
            int ni = trueCounts[kvp.Key.True];
            int nj = predCounts[kvp.Key.Pred];

            if (nij > 0)
            {
                double pij = (double)nij / total;
                double pi = (double)ni / total;
                double pj = (double)nj / total;
                mi += pij * Math.Log(pij / (pi * pj)) / Log2;
            }
        }
        return mi;
    }

    /// <summary>
    /// Computes the pair confusion matrix components.
    /// </summary>
    /// <param name="trueLabels">The ground truth labels.</param>
    /// <param name="predictedLabels">The predicted cluster assignments.</param>
    /// <returns>
    /// A tuple of (A, B, C, D) where:
    /// A = pairs in same cluster in both clusterings
    /// B = pairs in same cluster only in true labels
    /// C = pairs in same cluster only in predicted labels
    /// D = pairs in different clusters in both
    /// </returns>
    protected (long A, long B, long C, long D) ComputePairConfusionMatrix(
        Vector<T> trueLabels, Vector<T> predictedLabels)
    {
        long n = trueLabels.Length;
        var (contingency, trueCounts, predCounts) = BuildContingencyTable(trueLabels, predictedLabels);

        // Pairs in the same true and same predicted cluster, same true only, same predicted only, neither.
        long a = contingency.Values.Sum(count => PairCount(count));
        long sameTrue = trueCounts.Values.Sum(count => PairCount(count));
        long samePred = predCounts.Values.Sum(count => PairCount(count));
        long b = sameTrue - a;
        long c = samePred - a;
        long d = n * (n - 1) / 2 - a - b - c;

        return (a, b, c, d);
    }

    /// <summary>Number of unordered pairs among <paramref name="count"/> items.</summary>
    private static long PairCount(long count) => count * (count - 1) / 2;

    /// <summary>
    /// Computes the binomial coefficient C(n, k).
    /// </summary>
    /// <param name="n">Total number of items.</param>
    /// <param name="k">Number to choose.</param>
    /// <returns>The binomial coefficient.</returns>
    protected static double Binomial(int n, int k)
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

    /// <summary>
    /// Validates that the input label vectors have the same length.
    /// </summary>
    /// <param name="trueLabels">The ground truth labels.</param>
    /// <param name="predictedLabels">The predicted cluster assignments.</param>
    /// <exception cref="ArgumentException">Thrown when vectors have different lengths.</exception>
    protected static void ValidateLabelVectors(Vector<T> trueLabels, Vector<T> predictedLabels)
    {
        if (trueLabels.Length != predictedLabels.Length)
        {
            throw new ArgumentException("Label vectors must have the same length.");
        }
    }
}
