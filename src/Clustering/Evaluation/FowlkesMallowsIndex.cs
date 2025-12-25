using AiDotNet.Clustering.Interfaces;

namespace AiDotNet.Clustering.Evaluation;

/// <summary>
/// Computes the Fowlkes-Mallows Index for cluster-label agreement.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The Fowlkes-Mallows Index (FMI) is the geometric mean of precision and recall
/// for pairs of points. It measures similarity between two clusterings and requires
/// ground truth labels.
/// </para>
/// <para>
/// Formula: FM = sqrt(TP / (TP + FP) * TP / (TP + FN))
/// where:
/// - TP = pairs correctly put in same cluster
/// - FP = pairs incorrectly put in same cluster
/// - FN = pairs incorrectly put in different clusters
/// </para>
/// <para><b>For Beginners:</b> The FM Index measures agreement between clusterings.
///
/// For every pair of points, ask two questions:
/// 1. Are they in the same true class?
/// 2. Are they in the same predicted cluster?
///
/// Possible outcomes:
/// - TP: Both same class AND same cluster (correct!)
/// - TN: Both different class AND different cluster (correct!)
/// - FP: Different class but same cluster (wrong!)
/// - FN: Same class but different cluster (wrong!)
///
/// FM Index = sqrt(Precision * Recall)
/// - Precision: Of pairs we grouped together, how many should be?
/// - Recall: Of pairs that should be together, how many did we group?
///
/// Range: 0 (no agreement) to 1 (perfect agreement)
/// Random clustering: FM ≈ sqrt(1/K) where K is number of clusters
/// </para>
/// </remarks>
public class FowlkesMallowsIndex<T> : IClusterMetric<T>, IExternalClusterMetric<T>
{
    /// <inheritdoc />
    public string Name => "Fowlkes-Mallows Index";

    /// <inheritdoc />
    public bool HigherIsBetter => true;

    /// <summary>
    /// Computes FM Index comparing predicted labels to true labels.
    /// </summary>
    /// <param name="data">The data matrix (not used, can be null).</param>
    /// <param name="predictedLabels">The predicted cluster assignments.</param>
    /// <param name="trueLabels">The ground truth class labels.</param>
    /// <returns>FM Index between 0 and 1.</returns>
    public double ComputeWithTrueLabels(Matrix<T>? data, Vector<T> predictedLabels, Vector<T> trueLabels)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int n = predictedLabels.Length;

        if (n != trueLabels.Length)
        {
            throw new ArgumentException("Predicted and true labels must have the same length.");
        }

        // Build contingency table
        var trueClasses = new Dictionary<int, int>();
        var predClusters = new Dictionary<int, int>();

        for (int i = 0; i < n; i++)
        {
            int trueLabel = (int)numOps.ToDouble(trueLabels[i]);
            int predLabel = (int)numOps.ToDouble(predictedLabels[i]);

            if (!trueClasses.ContainsKey(trueLabel))
                trueClasses[trueLabel] = trueClasses.Count;
            if (!predClusters.ContainsKey(predLabel))
                predClusters[predLabel] = predClusters.Count;
        }

        int numClasses = trueClasses.Count;
        int numClusters = predClusters.Count;

        // Contingency matrix
        var contingency = new long[numClusters, numClasses];

        for (int i = 0; i < n; i++)
        {
            int trueIdx = trueClasses[(int)numOps.ToDouble(trueLabels[i])];
            int predIdx = predClusters[(int)numOps.ToDouble(predictedLabels[i])];
            contingency[predIdx, trueIdx]++;
        }

        // Compute TP, FP, FN using the contingency table
        // TP = sum(n_ij choose 2) for all i,j
        // Sum of same cluster = sum(a_i choose 2) where a_i = row sum
        // Sum of same class = sum(b_j choose 2) where b_j = col sum

        long sumNijC2 = 0; // sum of (n_ij choose 2)
        var rowSums = new long[numClusters];
        var colSums = new long[numClasses];

        for (int k = 0; k < numClusters; k++)
        {
            for (int c = 0; c < numClasses; c++)
            {
                long nij = contingency[k, c];
                sumNijC2 += nij * (nij - 1) / 2;
                rowSums[k] += nij;
                colSums[c] += nij;
            }
        }

        long sumAiC2 = 0;
        for (int k = 0; k < numClusters; k++)
        {
            sumAiC2 += rowSums[k] * (rowSums[k] - 1) / 2;
        }

        long sumBjC2 = 0;
        for (int c = 0; c < numClasses; c++)
        {
            sumBjC2 += colSums[c] * (colSums[c] - 1) / 2;
        }

        // TP = sumNijC2
        // TP + FP = sumAiC2 (pairs in same predicted cluster)
        // TP + FN = sumBjC2 (pairs in same true class)

        long tp = sumNijC2;
        long tpPlusFp = sumAiC2;
        long tpPlusFn = sumBjC2;

        if (tpPlusFp == 0 || tpPlusFn == 0)
        {
            return 0;
        }

        double precision = (double)tp / tpPlusFp;
        double recall = (double)tp / tpPlusFn;

        return Math.Sqrt(precision * recall);
    }

    /// <inheritdoc />
    public double Compute(Matrix<T> data, Vector<T> labels)
    {
        // FM Index requires true labels; without them, return 0
        return 0;
    }

    /// <inheritdoc />
    double IExternalClusterMetric<T>.Compute(Vector<T> trueLabels, Vector<T> predictedLabels)
    {
        return ComputeWithTrueLabels(null, predictedLabels, trueLabels);
    }
}
