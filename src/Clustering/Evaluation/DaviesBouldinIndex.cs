using AiDotNet.Clustering.DistanceMetrics;
using AiDotNet.Clustering.Interfaces;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Clustering.Evaluation;

/// <summary>
/// Davies-Bouldin Index for evaluating cluster quality.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The Davies-Bouldin Index measures the average similarity between
/// each cluster and its most similar cluster. Lower values indicate
/// better clustering (more compact and well-separated clusters).
/// </para>
/// <para>
/// For each cluster i:
/// - S(i) = average distance from points to cluster centroid
/// - d(i,j) = distance between cluster centroids
/// - R(i,j) = (S(i) + S(j)) / d(i,j)
/// - DB = (1/k) * sum(max_j(R(i,j)))
/// </para>
/// <para><b>For Beginners:</b> Davies-Bouldin measures cluster separation.
///
/// The idea:
/// - Good clusters are compact (small S)
/// - Good clusters are well-separated (large d)
/// - For each cluster, find the worst overlap with another
/// - Average these worst cases
///
/// Lower score = Better clustering!
/// (Unlike Silhouette where higher is better)
///
/// A score of 0 would be perfect separation.
/// </para>
/// </remarks>
public class DaviesBouldinIndex<T> : IClusterMetric<T>
{
    private readonly IDistanceMetric<T>? _distanceMetric;
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new DaviesBouldinIndex instance.
    /// </summary>
    /// <param name="distanceMetric">Distance metric to use, or null for Euclidean.</param>
    public DaviesBouldinIndex(IDistanceMetric<T>? distanceMetric = null)
    {
        _distanceMetric = distanceMetric;
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <inheritdoc />
    public string Name => "Davies-Bouldin Index";

    /// <inheritdoc />
    public double Compute(Matrix<T> data, Vector<T> labels)
    {
        int n = data.Rows;
        int d = data.Columns;
        var metric = _distanceMetric ?? new EuclideanDistance<T>();

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

        int k = uniqueLabels.Count;
        if (k < 2)
        {
            return 0; // Need at least 2 clusters
        }

        var labelList = uniqueLabels.ToList();

        // Compute centroids
        var centroids = new Dictionary<int, double[]>();
        var clusterCounts = new Dictionary<int, int>();

        foreach (int label in labelList)
        {
            centroids[label] = new double[d];
            clusterCounts[label] = 0;
        }

        for (int i = 0; i < n; i++)
        {
            int label = (int)_numOps.ToDouble(labels[i]);
            if (label < 0) continue;

            for (int j = 0; j < d; j++)
            {
                centroids[label][j] += _numOps.ToDouble(data[i, j]);
            }
            clusterCounts[label]++;
        }

        foreach (int label in labelList)
        {
            if (clusterCounts[label] > 0)
            {
                for (int j = 0; j < d; j++)
                {
                    centroids[label][j] /= clusterCounts[label];
                }
            }
        }

        // Compute scatter (average distance to centroid) for each cluster
        var scatter = new Dictionary<int, double>();
        foreach (int label in labelList)
        {
            scatter[label] = 0;
        }

        // Precompute centroid vectors for distance metric
        var centroidVectors = new Dictionary<int, Vector<T>>();
        foreach (int label in labelList)
        {
            var vec = new Vector<T>(d);
            for (int j = 0; j < d; j++)
            {
                vec[j] = _numOps.FromDouble(centroids[label][j]);
            }
            centroidVectors[label] = vec;
        }

        for (int i = 0; i < n; i++)
        {
            int label = (int)_numOps.ToDouble(labels[i]);
            if (label < 0) continue;

            // Use the configured distance metric
            var pointVec = data.GetRow(i);
            double dist = _numOps.ToDouble(metric.Compute(pointVec, centroidVectors[label]));
            scatter[label] += dist;
        }

        foreach (int label in labelList)
        {
            if (clusterCounts[label] > 0)
            {
                scatter[label] /= clusterCounts[label];
            }
        }

        // Compute Davies-Bouldin index
        double dbSum = 0;

        for (int ii = 0; ii < k; ii++)
        {
            int labelI = labelList[ii];
            double maxR = 0;

            for (int jj = 0; jj < k; jj++)
            {
                if (ii == jj) continue;

                int labelJ = labelList[jj];

                // Distance between centroids using configured metric
                double centroidDist = _numOps.ToDouble(metric.Compute(centroidVectors[labelI], centroidVectors[labelJ]));

                if (centroidDist > 0)
                {
                    double r = (scatter[labelI] + scatter[labelJ]) / centroidDist;
                    maxR = Math.Max(maxR, r);
                }
            }

            dbSum += maxR;
        }

        return dbSum / k;
    }
}
