using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Clustering.Evaluation;

/// <summary>
/// Purity metric for evaluating clustering against ground truth labels.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Purity measures the fraction of correctly assigned points, where "correct"
/// means each cluster is assigned to its majority class. Values range from
/// 1/k (random) to 1 (perfect).
/// </para>
/// <para>
/// Purity = (1/n) * sum over all clusters k of max_j |c_k ∩ t_j|
/// Where c_k is cluster k and t_j is true class j.
/// </para>
/// <para><b>For Beginners:</b> Purity asks "How pure are the clusters?"
///
/// For each cluster:
/// - Find the most common true class
/// - Count how many points belong to that class
///
/// Purity = Total correctly assigned / Total points
///
/// Example:
/// - Cluster 1: 30 cats, 10 dogs → 30 correct (majority is cats)
/// - Cluster 2: 20 cats, 40 dogs → 40 correct (majority is dogs)
/// - Purity = (30 + 40) / 100 = 0.70
///
/// Higher purity = Better clustering!
/// But beware: Purity increases as k increases (k=n gives purity=1).
/// </para>
/// </remarks>
public class Purity<T> : IExternalClusterMetric<T>
{
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new Purity instance.
    /// </summary>
    public Purity()
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

        if (n == 0)
        {
            return 0;
        }

        // Build contingency table
        var contingency = new Dictionary<(int Cluster, int Class), int>();
        var clusterLabels = new HashSet<int>();
        var trueClasses = new HashSet<int>();

        for (int i = 0; i < n; i++)
        {
            int cluster = (int)_numOps.ToDouble(predictedLabels[i]);
            int trueClass = (int)_numOps.ToDouble(trueLabels[i]);

            clusterLabels.Add(cluster);
            trueClasses.Add(trueClass);

            var key = (cluster, trueClass);
            contingency.TryAdd(key, 0);
            contingency[key]++;
        }

        // For each cluster, find the majority class count
        int correctlyAssigned = 0;

        foreach (int cluster in clusterLabels)
        {
            int maxCount = 0;

            foreach (int trueClass in trueClasses)
            {
                if (contingency.TryGetValue((cluster, trueClass), out int count))
                {
                    maxCount = Math.Max(maxCount, count);
                }
            }

            correctlyAssigned += maxCount;
        }

        return (double)correctlyAssigned / n;
    }

    /// <summary>
    /// Computes purity per cluster.
    /// </summary>
    /// <param name="trueLabels">The ground truth labels.</param>
    /// <param name="predictedLabels">The clustering assignments.</param>
    /// <returns>Dictionary mapping cluster labels to their purity values.</returns>
    public Dictionary<int, double> ComputePerCluster(Vector<T> trueLabels, Vector<T> predictedLabels)
    {
        int n = trueLabels.Length;

        var contingency = new Dictionary<(int Cluster, int Class), int>();
        var clusterCounts = new Dictionary<int, int>();
        var clusterLabels = new HashSet<int>();
        var trueClasses = new HashSet<int>();

        for (int i = 0; i < n; i++)
        {
            int cluster = (int)_numOps.ToDouble(predictedLabels[i]);
            int trueClass = (int)_numOps.ToDouble(trueLabels[i]);

            clusterLabels.Add(cluster);
            trueClasses.Add(trueClass);

            var key = (cluster, trueClass);
            contingency.TryAdd(key, 0);
            contingency[key]++;

            clusterCounts.TryAdd(cluster, 0);
            clusterCounts[cluster]++;
        }

        var result = new Dictionary<int, double>();

        foreach (int cluster in clusterLabels)
        {
            int maxCount = 0;

            foreach (int trueClass in trueClasses)
            {
                if (contingency.TryGetValue((cluster, trueClass), out int count))
                {
                    maxCount = Math.Max(maxCount, count);
                }
            }

            result[cluster] = clusterCounts[cluster] > 0 ? (double)maxCount / clusterCounts[cluster] : 0;
        }

        return result;
    }
}
