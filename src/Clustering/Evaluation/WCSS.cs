using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Clustering.Evaluation;

/// <summary>
/// Within-Cluster Sum of Squares (WCSS) metric for evaluating cluster compactness.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// WCSS measures the total squared distance of each point to its cluster centroid.
/// Lower values indicate more compact clusters.
/// </para>
/// <para>
/// WCSS = sum over all clusters k of sum over all points i in k of ||x_i - c_k||^2
/// </para>
/// <para><b>For Beginners:</b> WCSS measures how "tight" your clusters are.
///
/// For each point:
/// - Find the center of its cluster
/// - Measure the distance squared
/// - Add up all these distances
///
/// Lower WCSS = Better clustering (tighter clusters)
///
/// Also called "inertia" in scikit-learn.
/// </para>
/// </remarks>
public class WCSS<T> : IClusterMetric<T>
{
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new WCSS instance.
    /// </summary>
    public WCSS()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <inheritdoc />
    public string Name => "Within-Cluster Sum of Squares";

    /// <inheritdoc />
    public double Compute(Matrix<T> data, Vector<T> labels)
    {
        int n = data.Rows;
        int d = data.Columns;

        // Get unique labels (excluding -1 for noise)
        var uniqueLabels = new HashSet<int>();
        for (int i = 0; i < n; i++)
        {
            int label = (int)_numOps.ToDouble(labels[i]);
            if (label >= 0)
            {
                uniqueLabels.Add(label);
            }
        }

        if (uniqueLabels.Count == 0)
        {
            return 0;
        }

        // Compute centroids
        var centroids = new Dictionary<int, double[]>();
        var counts = new Dictionary<int, int>();

        foreach (int label in uniqueLabels)
        {
            centroids[label] = new double[d];
            counts[label] = 0;
        }

        for (int i = 0; i < n; i++)
        {
            int label = (int)_numOps.ToDouble(labels[i]);
            if (label < 0) continue;

            counts[label]++;
            for (int j = 0; j < d; j++)
            {
                centroids[label][j] += _numOps.ToDouble(data[i, j]);
            }
        }

        foreach (int label in uniqueLabels)
        {
            if (counts[label] > 0)
            {
                for (int j = 0; j < d; j++)
                {
                    centroids[label][j] /= counts[label];
                }
            }
        }

        // Compute WCSS
        double wcss = 0;
        for (int i = 0; i < n; i++)
        {
            int label = (int)_numOps.ToDouble(labels[i]);
            if (label < 0) continue;

            double distSq = 0;
            for (int j = 0; j < d; j++)
            {
                double diff = _numOps.ToDouble(data[i, j]) - centroids[label][j];
                distSq += diff * diff;
            }
            wcss += distSq;
        }

        return wcss;
    }

    /// <summary>
    /// Computes WCSS per cluster.
    /// </summary>
    /// <param name="data">The data matrix.</param>
    /// <param name="labels">The cluster assignments.</param>
    /// <returns>Dictionary mapping cluster labels to their WCSS values.</returns>
    public Dictionary<int, double> ComputePerCluster(Matrix<T> data, Vector<T> labels)
    {
        int n = data.Rows;
        int d = data.Columns;

        var uniqueLabels = new HashSet<int>();
        for (int i = 0; i < n; i++)
        {
            int label = (int)_numOps.ToDouble(labels[i]);
            if (label >= 0)
            {
                uniqueLabels.Add(label);
            }
        }

        // Compute centroids
        var centroids = new Dictionary<int, double[]>();
        var counts = new Dictionary<int, int>();

        foreach (int label in uniqueLabels)
        {
            centroids[label] = new double[d];
            counts[label] = 0;
        }

        for (int i = 0; i < n; i++)
        {
            int label = (int)_numOps.ToDouble(labels[i]);
            if (label < 0) continue;

            counts[label]++;
            for (int j = 0; j < d; j++)
            {
                centroids[label][j] += _numOps.ToDouble(data[i, j]);
            }
        }

        foreach (int label in uniqueLabels)
        {
            if (counts[label] > 0)
            {
                for (int j = 0; j < d; j++)
                {
                    centroids[label][j] /= counts[label];
                }
            }
        }

        // Compute WCSS per cluster
        var result = new Dictionary<int, double>();
        foreach (int label in uniqueLabels)
        {
            result[label] = 0;
        }

        for (int i = 0; i < n; i++)
        {
            int label = (int)_numOps.ToDouble(labels[i]);
            if (label < 0) continue;

            double distSq = 0;
            for (int j = 0; j < d; j++)
            {
                double diff = _numOps.ToDouble(data[i, j]) - centroids[label][j];
                distSq += diff * diff;
            }
            result[label] += distSq;
        }

        return result;
    }
}

/// <summary>
/// Between-Cluster Sum of Squares (BCSS) metric for evaluating cluster separation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// BCSS measures the weighted sum of squared distances between cluster centroids
/// and the overall data centroid. Higher values indicate better separated clusters.
/// </para>
/// <para>
/// BCSS = sum over all clusters k of n_k * ||c_k - c_global||^2
/// </para>
/// <para><b>For Beginners:</b> BCSS measures how "spread apart" your cluster centers are.
///
/// - Find the overall center of all data
/// - For each cluster, measure distance from cluster center to overall center
/// - Weight by cluster size
///
/// Higher BCSS = Better clustering (clusters are more separated)
///
/// Total variance = WCSS + BCSS
/// Good clustering has low WCSS (tight) and high BCSS (separated).
/// </para>
/// </remarks>
public class BCSS<T> : IClusterMetric<T>
{
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new BCSS instance.
    /// </summary>
    public BCSS()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <inheritdoc />
    public string Name => "Between-Cluster Sum of Squares";

    /// <inheritdoc />
    public double Compute(Matrix<T> data, Vector<T> labels)
    {
        int n = data.Rows;
        int d = data.Columns;

        // Get unique labels (excluding -1 for noise)
        var uniqueLabels = new HashSet<int>();
        for (int i = 0; i < n; i++)
        {
            int label = (int)_numOps.ToDouble(labels[i]);
            if (label >= 0)
            {
                uniqueLabels.Add(label);
            }
        }

        if (uniqueLabels.Count < 2)
        {
            return 0; // Need at least 2 clusters
        }

        // Compute overall centroid
        var globalCentroid = new double[d];
        int validPoints = 0;

        for (int i = 0; i < n; i++)
        {
            int label = (int)_numOps.ToDouble(labels[i]);
            if (label < 0) continue;

            validPoints++;
            for (int j = 0; j < d; j++)
            {
                globalCentroid[j] += _numOps.ToDouble(data[i, j]);
            }
        }

        if (validPoints == 0) return 0;

        for (int j = 0; j < d; j++)
        {
            globalCentroid[j] /= validPoints;
        }

        // Compute cluster centroids
        var centroids = new Dictionary<int, double[]>();
        var counts = new Dictionary<int, int>();

        foreach (int label in uniqueLabels)
        {
            centroids[label] = new double[d];
            counts[label] = 0;
        }

        for (int i = 0; i < n; i++)
        {
            int label = (int)_numOps.ToDouble(labels[i]);
            if (label < 0) continue;

            counts[label]++;
            for (int j = 0; j < d; j++)
            {
                centroids[label][j] += _numOps.ToDouble(data[i, j]);
            }
        }

        foreach (int label in uniqueLabels)
        {
            if (counts[label] > 0)
            {
                for (int j = 0; j < d; j++)
                {
                    centroids[label][j] /= counts[label];
                }
            }
        }

        // Compute BCSS
        double bcss = 0;
        foreach (int label in uniqueLabels)
        {
            double distSq = 0;
            for (int j = 0; j < d; j++)
            {
                double diff = centroids[label][j] - globalCentroid[j];
                distSq += diff * diff;
            }
            bcss += counts[label] * distSq;
        }

        return bcss;
    }
}
