using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Clustering.Evaluation;

/// <summary>
/// Entropy-based metrics for evaluating clustering against ground truth labels.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Entropy measures the "disorder" or "uncertainty" in cluster assignments.
/// Lower entropy indicates that clusters are more homogeneous with respect
/// to the true classes.
/// </para>
/// <para>
/// Entropy of cluster k = -sum over all classes j of p(j|k) * log(p(j|k))
/// Overall entropy = sum over all clusters k of (n_k/n) * Entropy(k)
/// </para>
/// <para><b>For Beginners:</b> Entropy measures "How mixed are the clusters?"
///
/// Think of it like sorting socks:
/// - Low entropy: Each drawer has one color (easy to find socks!)
/// - High entropy: Each drawer is a random mix (chaos!)
///
/// For clustering:
/// - Entropy = 0: Perfect! Each cluster has only one class
/// - Entropy = log(C): Worst! Each cluster has equal parts of all classes
///
/// We want LOW entropy (pure clusters).
/// </para>
/// </remarks>
public class ClusteringEntropy<T> : IExternalClusterMetric<T>
{
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new ClusteringEntropy instance.
    /// </summary>
    public ClusteringEntropy()
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

        // Build contingency table and counts
        var contingency = new Dictionary<(int Cluster, int Class), int>();
        var clusterCounts = new Dictionary<int, int>();
        var trueClasses = new HashSet<int>();

        for (int i = 0; i < n; i++)
        {
            int cluster = (int)_numOps.ToDouble(predictedLabels[i]);
            int trueClass = (int)_numOps.ToDouble(trueLabels[i]);

            trueClasses.Add(trueClass);

            var key = (cluster, trueClass);
            contingency.TryAdd(key, 0);
            contingency[key]++;

            clusterCounts.TryAdd(cluster, 0);
            clusterCounts[cluster]++;
        }

        // Compute weighted average entropy
        double totalEntropy = 0;

        foreach (var clusterKvp in clusterCounts)
        {
            int cluster = clusterKvp.Key;
            int clusterSize = clusterKvp.Value;

            if (clusterSize == 0) continue;

            double clusterEntropy = 0;

            foreach (int trueClass in trueClasses)
            {
                if (contingency.TryGetValue((cluster, trueClass), out int count) && count > 0)
                {
                    double p = (double)count / clusterSize;
                    clusterEntropy -= p * Math.Log2(p);
                }
            }

            totalEntropy += ((double)clusterSize / n) * clusterEntropy;
        }

        return totalEntropy;
    }

    /// <summary>
    /// Computes entropy for each cluster.
    /// </summary>
    /// <param name="trueLabels">The ground truth labels.</param>
    /// <param name="predictedLabels">The clustering assignments.</param>
    /// <returns>Dictionary mapping cluster labels to their entropy values.</returns>
    public Dictionary<int, double> ComputePerCluster(Vector<T> trueLabels, Vector<T> predictedLabels)
    {
        int n = trueLabels.Length;

        var contingency = new Dictionary<(int Cluster, int Class), int>();
        var clusterCounts = new Dictionary<int, int>();
        var trueClasses = new HashSet<int>();

        for (int i = 0; i < n; i++)
        {
            int cluster = (int)_numOps.ToDouble(predictedLabels[i]);
            int trueClass = (int)_numOps.ToDouble(trueLabels[i]);

            trueClasses.Add(trueClass);

            var key = (cluster, trueClass);
            contingency.TryAdd(key, 0);
            contingency[key]++;

            clusterCounts.TryAdd(cluster, 0);
            clusterCounts[cluster]++;
        }

        var result = new Dictionary<int, double>();

        foreach (var clusterKvp in clusterCounts)
        {
            int cluster = clusterKvp.Key;
            int clusterSize = clusterKvp.Value;

            if (clusterSize == 0)
            {
                result[cluster] = 0;
                continue;
            }

            double clusterEntropy = 0;

            foreach (int trueClass in trueClasses)
            {
                if (contingency.TryGetValue((cluster, trueClass), out int count) && count > 0)
                {
                    double p = (double)count / clusterSize;
                    clusterEntropy -= p * Math.Log2(p);
                }
            }

            result[cluster] = clusterEntropy;
        }

        return result;
    }

    /// <summary>
    /// Computes normalized entropy (0 = perfect, 1 = worst).
    /// </summary>
    /// <param name="trueLabels">The ground truth labels.</param>
    /// <param name="predictedLabels">The clustering assignments.</param>
    /// <returns>Normalized entropy value.</returns>
    public double ComputeNormalized(Vector<T> trueLabels, Vector<T> predictedLabels)
    {
        int n = trueLabels.Length;

        // Count number of classes
        var trueClasses = new HashSet<int>();
        for (int i = 0; i < n; i++)
        {
            trueClasses.Add((int)_numOps.ToDouble(trueLabels[i]));
        }

        int numClasses = trueClasses.Count;
        if (numClasses <= 1)
        {
            return 0; // No entropy possible with one class
        }

        double maxEntropy = Math.Log2(numClasses);
        double entropy = Compute(trueLabels, predictedLabels);

        return maxEntropy > 0 ? entropy / maxEntropy : 0;
    }
}

/// <summary>
/// Conditional Entropy H(C|K) - entropy of true classes given clusters.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class ConditionalEntropy<T> : IExternalClusterMetric<T>
{
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new ConditionalEntropy instance.
    /// </summary>
    public ConditionalEntropy()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <inheritdoc />
    public double Compute(Vector<T> trueLabels, Vector<T> predictedLabels)
    {
        // Conditional entropy H(C|K) is the same as clustering entropy
        var clusteringEntropy = new ClusteringEntropy<T>();
        return clusteringEntropy.Compute(trueLabels, predictedLabels);
    }
}

/// <summary>
/// Homogeneity score - measures if clusters contain only members of a single class.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class Homogeneity<T> : IExternalClusterMetric<T>
{
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new Homogeneity instance.
    /// </summary>
    public Homogeneity()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <inheritdoc />
    public double Compute(Vector<T> trueLabels, Vector<T> predictedLabels)
    {
        int n = trueLabels.Length;

        if (n == 0) return 1;

        // Compute H(C|K) - conditional entropy of classes given clusters
        double hCK = ComputeConditionalEntropy(trueLabels, predictedLabels);

        // Compute H(C) - entropy of true classes
        double hC = ComputeEntropy(trueLabels);

        if (hC == 0)
        {
            return 1; // Perfect homogeneity if only one class
        }

        return 1 - hCK / hC;
    }

    private double ComputeConditionalEntropy(Vector<T> trueLabels, Vector<T> predictedLabels)
    {
        var entropy = new ClusteringEntropy<T>();
        return entropy.Compute(trueLabels, predictedLabels);
    }

    private double ComputeEntropy(Vector<T> labels)
    {
        int n = labels.Length;
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
}

/// <summary>
/// Completeness score - measures if all members of a given class are assigned to the same cluster.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class Completeness<T> : IExternalClusterMetric<T>
{
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new Completeness instance.
    /// </summary>
    public Completeness()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <inheritdoc />
    public double Compute(Vector<T> trueLabels, Vector<T> predictedLabels)
    {
        int n = trueLabels.Length;

        if (n == 0) return 1;

        // Compute H(K|C) - conditional entropy of clusters given classes
        double hKC = ComputeConditionalEntropy(predictedLabels, trueLabels);

        // Compute H(K) - entropy of clusters
        double hK = ComputeEntropy(predictedLabels);

        if (hK == 0)
        {
            return 1; // Perfect completeness if only one cluster
        }

        return 1 - hKC / hK;
    }

    private double ComputeConditionalEntropy(Vector<T> labels, Vector<T> givenLabels)
    {
        // This computes H(labels | givenLabels)
        var entropy = new ClusteringEntropy<T>();
        return entropy.Compute(labels, givenLabels);
    }

    private double ComputeEntropy(Vector<T> labels)
    {
        int n = labels.Length;
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
}
