using AiDotNet.Clustering.DistanceMetrics;
using AiDotNet.Clustering.Interfaces;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Clustering.Evaluation;

/// <summary>
/// Silhouette Score for evaluating cluster quality.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The Silhouette Score measures how similar a point is to its own cluster
/// compared to other clusters. Values range from -1 to 1:
/// - Near +1: Points are well-matched to their cluster
/// - Near 0: Points are on cluster boundaries
/// - Near -1: Points may be assigned to wrong clusters
/// </para>
/// <para>
/// For each point i:
/// - a(i) = mean distance to other points in same cluster
/// - b(i) = mean distance to points in nearest other cluster
/// - s(i) = (b(i) - a(i)) / max(a(i), b(i))
/// </para>
/// <para><b>For Beginners:</b> Silhouette Score asks "Is each point in the right cluster?"
///
/// For each point, we measure:
/// 1. How close it is to its cluster-mates (a)
/// 2. How far it is from the nearest other cluster (b)
///
/// If b >> a: Point is clearly in the right place (+1)
/// If b ≈ a: Point is between clusters (0)
/// If b << a: Point might be in wrong cluster (-1)
///
/// The overall score is the average of all points.
/// Higher is better!
/// </para>
/// </remarks>
public class SilhouetteScore<T> : IClusterMetric<T>
{
    private readonly IDistanceMetric<T>? _distanceMetric;
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new SilhouetteScore instance.
    /// </summary>
    /// <param name="distanceMetric">Distance metric to use, or null for Euclidean.</param>
    public SilhouetteScore(IDistanceMetric<T>? distanceMetric = null)
    {
        _distanceMetric = distanceMetric;
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <inheritdoc />
    public string Name => "Silhouette Score";

    /// <inheritdoc />
    public double Compute(Matrix<T> data, Vector<T> labels)
    {
        int n = data.Rows;

        if (n < 2)
        {
            return 0;
        }

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

        if (uniqueLabels.Count < 2)
        {
            return 0; // Need at least 2 clusters
        }

        // Compute per-sample silhouette scores
        var silhouetteScores = new double[n];

        for (int i = 0; i < n; i++)
        {
            int labelI = (int)_numOps.ToDouble(labels[i]);

            if (labelI < 0)
            {
                silhouetteScores[i] = 0; // Noise points
                continue;
            }

            var pointI = GetRow(data, i);

            // Compute distances to all other points, grouped by cluster
            var clusterDistances = new Dictionary<int, List<double>>();
            foreach (int label in uniqueLabels)
            {
                clusterDistances[label] = new List<double>();
            }

            for (int j = 0; j < n; j++)
            {
                if (i == j) continue;

                int labelJ = (int)_numOps.ToDouble(labels[j]);
                if (labelJ < 0) continue;

                var pointJ = GetRow(data, j);
                double dist = _numOps.ToDouble(metric.Compute(pointI, pointJ));
                clusterDistances[labelJ].Add(dist);
            }

            // a(i) = mean distance to same cluster
            double a = clusterDistances[labelI].Count > 0
                ? clusterDistances[labelI].Average()
                : 0;

            // b(i) = min mean distance to other clusters
            double b = double.MaxValue;
            foreach (var kvp in clusterDistances)
            {
                if (kvp.Key != labelI && kvp.Value.Count > 0)
                {
                    double meanDist = kvp.Value.Average();
                    b = Math.Min(b, meanDist);
                }
            }

            if (b == double.MaxValue) b = 0;

            // Silhouette score for point i
            double maxAB = Math.Max(a, b);
            silhouetteScores[i] = maxAB > 0 ? (b - a) / maxAB : 0;
        }

        // Count non-noise points
        int validPoints = 0;
        double sum = 0;
        for (int i = 0; i < n; i++)
        {
            int label = (int)_numOps.ToDouble(labels[i]);
            if (label >= 0)
            {
                sum += silhouetteScores[i];
                validPoints++;
            }
        }

        return validPoints > 0 ? sum / validPoints : 0;
    }

    /// <summary>
    /// Computes per-sample silhouette scores.
    /// </summary>
    /// <param name="data">The data matrix.</param>
    /// <param name="labels">The cluster assignments.</param>
    /// <returns>Array of silhouette scores for each sample.</returns>
    public double[] ComputeSampleScores(Matrix<T> data, Vector<T> labels)
    {
        int n = data.Rows;
        var metric = _distanceMetric ?? new EuclideanDistance<T>();

        var uniqueLabels = new HashSet<int>();
        for (int i = 0; i < n; i++)
        {
            int label = (int)_numOps.ToDouble(labels[i]);
            if (label >= 0)
            {
                uniqueLabels.Add(label);
            }
        }

        var scores = new double[n];

        for (int i = 0; i < n; i++)
        {
            int labelI = (int)_numOps.ToDouble(labels[i]);

            if (labelI < 0 || uniqueLabels.Count < 2)
            {
                scores[i] = 0;
                continue;
            }

            var pointI = GetRow(data, i);

            var clusterDistances = new Dictionary<int, List<double>>();
            foreach (int label in uniqueLabels)
            {
                clusterDistances[label] = new List<double>();
            }

            for (int j = 0; j < n; j++)
            {
                if (i == j) continue;

                int labelJ = (int)_numOps.ToDouble(labels[j]);
                if (labelJ < 0) continue;

                var pointJ = GetRow(data, j);
                double dist = _numOps.ToDouble(metric.Compute(pointI, pointJ));
                clusterDistances[labelJ].Add(dist);
            }

            double a = clusterDistances[labelI].Count > 0
                ? clusterDistances[labelI].Average()
                : 0;

            double b = double.MaxValue;
            foreach (var kvp in clusterDistances)
            {
                if (kvp.Key != labelI && kvp.Value.Count > 0)
                {
                    double meanDist = kvp.Value.Average();
                    b = Math.Min(b, meanDist);
                }
            }

            if (b == double.MaxValue) b = 0;

            double maxAB = Math.Max(a, b);
            scores[i] = maxAB > 0 ? (b - a) / maxAB : 0;
        }

        return scores;
    }

    private Vector<T> GetRow(Matrix<T> matrix, int row)
    {
        var result = new Vector<T>(matrix.Columns);
        for (int j = 0; j < matrix.Columns; j++)
        {
            result[j] = matrix[row, j];
        }
        return result;
    }
}
