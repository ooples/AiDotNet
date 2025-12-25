using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Clustering.Evaluation;

/// <summary>
/// F-Measure (F-Score) for comparing clustering results against ground truth.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The F-Measure combines precision and recall for clustering evaluation.
/// For each true class, it finds the best matching cluster, then averages
/// the F-scores weighted by class sizes.
/// </para>
/// <para>
/// F(C_i, K_j) = (2 * Precision * Recall) / (Precision + Recall)
/// Where:
/// - Precision = |C_i ∩ K_j| / |K_j|
/// - Recall = |C_i ∩ K_j| / |C_i|
/// </para>
/// <para><b>For Beginners:</b> F-Measure asks "How well do clusters match classes?"
///
/// For each true class:
/// - Find which cluster matches it best
/// - Calculate how well that cluster captures the class
///
/// Precision: "Of the cluster, how many belong to this class?"
/// Recall: "Of this class, how many are in this cluster?"
/// F-Measure: Balances both (harmonic mean)
///
/// Higher is better! 1.0 = perfect, 0 = no agreement.
/// </para>
/// </remarks>
public class FMeasure<T> : IExternalClusterMetric<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly double _beta;

    /// <summary>
    /// Initializes a new FMeasure instance.
    /// </summary>
    /// <param name="beta">Beta parameter for F-beta score. Default is 1.0 (F1 score).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Beta controls the precision-recall tradeoff:
    /// - beta = 1: F1 score, equal weight to precision and recall
    /// - beta &lt; 1: Emphasizes precision (fewer false positives)
    /// - beta &gt; 1: Emphasizes recall (fewer false negatives)
    /// </para>
    /// </remarks>
    public FMeasure(double beta = 1.0)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _beta = beta;
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

        // Build contingency table
        var contingency = new Dictionary<(int True, int Pred), int>();
        var trueCounts = new Dictionary<int, int>();
        var predCounts = new Dictionary<int, int>();
        var trueClasses = new HashSet<int>();
        var predClusters = new HashSet<int>();

        for (int i = 0; i < n; i++)
        {
            int trueLabel = (int)_numOps.ToDouble(trueLabels[i]);
            int predLabel = (int)_numOps.ToDouble(predictedLabels[i]);

            trueClasses.Add(trueLabel);
            predClusters.Add(predLabel);

            var key = (trueLabel, predLabel);
            contingency.TryAdd(key, 0);
            contingency[key]++;

            trueCounts.TryAdd(trueLabel, 0);
            trueCounts[trueLabel]++;

            predCounts.TryAdd(predLabel, 0);
            predCounts[predLabel]++;
        }

        // For each true class, find the best F-score with any cluster
        double totalFScore = 0;
        double betaSquared = _beta * _beta;

        foreach (int trueClass in trueClasses)
        {
            double bestFScore = 0;
            int trueClassSize = trueCounts[trueClass];

            foreach (int predCluster in predClusters)
            {
                int predClusterSize = predCounts[predCluster];
                int intersection = 0;

                if (contingency.TryGetValue((trueClass, predCluster), out int count))
                {
                    intersection = count;
                }

                if (intersection == 0) continue;

                double precision = (double)intersection / predClusterSize;
                double recall = (double)intersection / trueClassSize;

                double fScore = (1 + betaSquared) * precision * recall / (betaSquared * precision + recall);

                bestFScore = Math.Max(bestFScore, fScore);
            }

            totalFScore += bestFScore * trueClassSize;
        }

        return totalFScore / n;
    }

    /// <summary>
    /// Computes the F-Measure matrix between all classes and clusters.
    /// </summary>
    /// <param name="trueLabels">The ground truth labels.</param>
    /// <param name="predictedLabels">The clustering assignments.</param>
    /// <returns>Dictionary mapping (class, cluster) pairs to their F-scores.</returns>
    public Dictionary<(int Class, int Cluster), double> ComputeMatrix(Vector<T> trueLabels, Vector<T> predictedLabels)
    {
        int n = trueLabels.Length;
        var result = new Dictionary<(int, int), double>();

        // Build contingency table
        var contingency = new Dictionary<(int True, int Pred), int>();
        var trueCounts = new Dictionary<int, int>();
        var predCounts = new Dictionary<int, int>();
        var trueClasses = new HashSet<int>();
        var predClusters = new HashSet<int>();

        for (int i = 0; i < n; i++)
        {
            int trueLabel = (int)_numOps.ToDouble(trueLabels[i]);
            int predLabel = (int)_numOps.ToDouble(predictedLabels[i]);

            trueClasses.Add(trueLabel);
            predClusters.Add(predLabel);

            var key = (trueLabel, predLabel);
            contingency.TryAdd(key, 0);
            contingency[key]++;

            trueCounts.TryAdd(trueLabel, 0);
            trueCounts[trueLabel]++;

            predCounts.TryAdd(predLabel, 0);
            predCounts[predLabel]++;
        }

        double betaSquared = _beta * _beta;

        foreach (int trueClass in trueClasses)
        {
            int trueClassSize = trueCounts[trueClass];

            foreach (int predCluster in predClusters)
            {
                int predClusterSize = predCounts[predCluster];
                int intersection = 0;

                if (contingency.TryGetValue((trueClass, predCluster), out int count))
                {
                    intersection = count;
                }

                if (intersection == 0)
                {
                    result[(trueClass, predCluster)] = 0;
                    continue;
                }

                double precision = (double)intersection / predClusterSize;
                double recall = (double)intersection / trueClassSize;

                double fScore = (1 + betaSquared) * precision * recall / (betaSquared * precision + recall);

                result[(trueClass, predCluster)] = fScore;
            }
        }

        return result;
    }

    /// <summary>
    /// Computes pair-counting based F-Measure (BCubed F-Measure).
    /// </summary>
    /// <param name="trueLabels">The ground truth labels.</param>
    /// <param name="predictedLabels">The clustering assignments.</param>
    /// <returns>The BCubed F-Measure.</returns>
    /// <remarks>
    /// <para>
    /// BCubed evaluates each point individually based on how well its cluster
    /// matches its class, then averages. This handles multi-class clusters better.
    /// </para>
    /// </remarks>
    public double ComputeBCubed(Vector<T> trueLabels, Vector<T> predictedLabels)
    {
        int n = trueLabels.Length;

        if (n == 0)
        {
            return 0;
        }

        // Group by predicted cluster
        var clusterToPoints = new Dictionary<int, List<int>>();
        // Group by true class
        var classToPoints = new Dictionary<int, List<int>>();

        for (int i = 0; i < n; i++)
        {
            int trueLabel = (int)_numOps.ToDouble(trueLabels[i]);
            int predLabel = (int)_numOps.ToDouble(predictedLabels[i]);

            if (!clusterToPoints.ContainsKey(predLabel))
            {
                clusterToPoints[predLabel] = new List<int>();
            }
            clusterToPoints[predLabel].Add(i);

            if (!classToPoints.ContainsKey(trueLabel))
            {
                classToPoints[trueLabel] = new List<int>();
            }
            classToPoints[trueLabel].Add(i);
        }

        double totalPrecision = 0;
        double totalRecall = 0;

        for (int i = 0; i < n; i++)
        {
            int trueLabel = (int)_numOps.ToDouble(trueLabels[i]);
            int predLabel = (int)_numOps.ToDouble(predictedLabels[i]);

            var clusterPoints = clusterToPoints[predLabel];
            var classPoints = classToPoints[trueLabel];

            // Count points in same cluster with same class
            int correctInCluster = 0;
            foreach (int j in clusterPoints)
            {
                if ((int)_numOps.ToDouble(trueLabels[j]) == trueLabel)
                {
                    correctInCluster++;
                }
            }

            // Count points in same class in same cluster
            int correctInClass = 0;
            foreach (int j in classPoints)
            {
                if ((int)_numOps.ToDouble(predictedLabels[j]) == predLabel)
                {
                    correctInClass++;
                }
            }

            // Precision for this point: fraction of cluster mates with same class
            totalPrecision += (double)correctInCluster / clusterPoints.Count;

            // Recall for this point: fraction of class mates in same cluster
            totalRecall += (double)correctInClass / classPoints.Count;
        }

        double avgPrecision = totalPrecision / n;
        double avgRecall = totalRecall / n;

        if (avgPrecision + avgRecall == 0)
        {
            return 0;
        }

        double betaSquared = _beta * _beta;
        return (1 + betaSquared) * avgPrecision * avgRecall / (betaSquared * avgPrecision + avgRecall);
    }
}
