using AiDotNet.Clustering.Interfaces;

namespace AiDotNet.Clustering.Evaluation;

/// <summary>
/// Computes the Dunn Index for cluster validity assessment.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The Dunn Index is the ratio of the minimum inter-cluster distance to the
/// maximum intra-cluster distance. Higher values indicate better clustering.
/// </para>
/// <para>
/// Formula: D = min(d(C_i, C_j)) / max(diam(C_k))
/// where:
/// - d(C_i, C_j) = minimum distance between points in different clusters
/// - diam(C_k) = maximum distance between points within a cluster
/// </para>
/// <para><b>For Beginners:</b> The Dunn Index asks two questions:
///
/// 1. How far apart are different clusters? (larger = better)
/// 2. How spread out is each cluster? (smaller = better)
///
/// A good clustering has:
/// - Large gaps between clusters
/// - Tight, compact clusters
///
/// The Dunn Index is the ratio: (smallest gap) / (largest spread)
/// - Higher values = better clustering
/// - Maximum when clusters are tight and well-separated
///
/// Limitations:
/// - Sensitive to outliers (a single far point increases diameter)
/// - Computationally expensive for large datasets (O(n²))
/// </para>
/// </remarks>
public class DunnIndex<T> : IClusterMetric<T>
{
    /// <inheritdoc />
    public string Name => "Dunn Index";

    /// <inheritdoc />
    public bool HigherIsBetter => true;

    /// <inheritdoc />
    public double Compute(Matrix<T> data, Vector<T> labels)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int n = data.Rows;
        int d = data.Columns;

        // Get unique cluster labels
        var clusterLabels = new HashSet<int>();
        for (int i = 0; i < n; i++)
        {
            int label = (int)numOps.ToDouble(labels[i]);
            if (label >= 0)
            {
                clusterLabels.Add(label);
            }
        }

        var clusters = clusterLabels.ToArray();
        if (clusters.Length < 2)
        {
            return 0;
        }

        // Group points by cluster
        var clusterPoints = new Dictionary<int, List<int>>();
        foreach (int c in clusters)
        {
            clusterPoints[c] = new List<int>();
        }

        for (int i = 0; i < n; i++)
        {
            int label = (int)numOps.ToDouble(labels[i]);
            if (label >= 0 && clusterPoints.ContainsKey(label))
            {
                clusterPoints[label].Add(i);
            }
        }

        // Compute minimum inter-cluster distance
        double minInterCluster = double.MaxValue;
        for (int ci = 0; ci < clusters.Length; ci++)
        {
            for (int cj = ci + 1; cj < clusters.Length; cj++)
            {
                int c1 = clusters[ci];
                int c2 = clusters[cj];

                foreach (int i in clusterPoints[c1])
                {
                    foreach (int j in clusterPoints[c2])
                    {
                        double dist = ComputeDistance(data, i, j, d, numOps);
                        minInterCluster = Math.Min(minInterCluster, dist);
                    }
                }
            }
        }

        // Compute maximum intra-cluster diameter
        double maxIntraCluster = 0;
        foreach (int c in clusters)
        {
            var points = clusterPoints[c];
            for (int i = 0; i < points.Count; i++)
            {
                for (int j = i + 1; j < points.Count; j++)
                {
                    double dist = ComputeDistance(data, points[i], points[j], d, numOps);
                    maxIntraCluster = Math.Max(maxIntraCluster, dist);
                }
            }
        }

        if (maxIntraCluster == 0)
        {
            return 0;
        }

        return minInterCluster / maxIntraCluster;
    }

    private double ComputeDistance(Matrix<T> data, int i, int j, int d, INumericOperations<T> numOps)
    {
        double sum = 0;
        for (int k = 0; k < d; k++)
        {
            double diff = numOps.ToDouble(data[i, k]) - numOps.ToDouble(data[j, k]);
            sum += diff * diff;
        }
        return Math.Sqrt(sum);
    }
}
