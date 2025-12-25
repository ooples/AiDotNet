using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Clustering.Evaluation;

/// <summary>
/// Calinski-Harabasz Index (Variance Ratio Criterion) for evaluating cluster quality.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The Calinski-Harabasz Index is the ratio of between-cluster variance
/// to within-cluster variance. Higher values indicate better clustering.
/// </para>
/// <para>
/// CH = (BGS / (k-1)) / (WGS / (n-k))
/// Where:
/// - BGS = Between-Group Sum of Squares (cluster separation)
/// - WGS = Within-Group Sum of Squares (cluster compactness)
/// - k = number of clusters
/// - n = number of points
/// </para>
/// <para><b>For Beginners:</b> Calinski-Harabasz measures variance ratio.
///
/// Good clustering has:
/// - High variance BETWEEN clusters (clusters are different)
/// - Low variance WITHIN clusters (clusters are tight)
///
/// The index is the ratio of these variances.
/// Higher score = Better clustering!
///
/// Think of it like:
/// - Numerator: How spread apart are the cluster centers?
/// - Denominator: How tight are points around their centers?
/// - We want big numerator, small denominator.
/// </para>
/// </remarks>
public class CalinskiHarabaszIndex<T> : IClusterMetric<T>
{
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new CalinskiHarabaszIndex instance.
    /// </summary>
    public CalinskiHarabaszIndex()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <inheritdoc />
    public string Name => "Calinski-Harabasz Index";

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

        int k = uniqueLabels.Count;
        if (k < 2)
        {
            return 0; // Need at least 2 clusters
        }

        var labelList = uniqueLabels.ToList();

        // Compute overall centroid
        var overallCentroid = new double[d];
        int validPoints = 0;

        for (int i = 0; i < n; i++)
        {
            int label = (int)_numOps.ToDouble(labels[i]);
            if (label < 0) continue;

            for (int j = 0; j < d; j++)
            {
                overallCentroid[j] += _numOps.ToDouble(data[i, j]);
            }
            validPoints++;
        }

        if (validPoints == 0) return 0;

        for (int j = 0; j < d; j++)
        {
            overallCentroid[j] /= validPoints;
        }

        // Compute cluster centroids
        var clusterCentroids = new Dictionary<int, double[]>();
        var clusterCounts = new Dictionary<int, int>();

        foreach (int label in labelList)
        {
            clusterCentroids[label] = new double[d];
            clusterCounts[label] = 0;
        }

        for (int i = 0; i < n; i++)
        {
            int label = (int)_numOps.ToDouble(labels[i]);
            if (label < 0) continue;

            for (int j = 0; j < d; j++)
            {
                clusterCentroids[label][j] += _numOps.ToDouble(data[i, j]);
            }
            clusterCounts[label]++;
        }

        foreach (int label in labelList)
        {
            if (clusterCounts[label] > 0)
            {
                for (int j = 0; j < d; j++)
                {
                    clusterCentroids[label][j] /= clusterCounts[label];
                }
            }
        }

        // Compute Between-Group Sum of Squares (BGS)
        double bgs = 0;
        foreach (int label in labelList)
        {
            double distSq = 0;
            for (int j = 0; j < d; j++)
            {
                double diff = clusterCentroids[label][j] - overallCentroid[j];
                distSq += diff * diff;
            }
            bgs += clusterCounts[label] * distSq;
        }

        // Compute Within-Group Sum of Squares (WGS)
        double wgs = 0;
        for (int i = 0; i < n; i++)
        {
            int label = (int)_numOps.ToDouble(labels[i]);
            if (label < 0) continue;

            double distSq = 0;
            for (int j = 0; j < d; j++)
            {
                double diff = _numOps.ToDouble(data[i, j]) - clusterCentroids[label][j];
                distSq += diff * diff;
            }
            wgs += distSq;
        }

        // Compute Calinski-Harabasz index
        if (wgs == 0 || validPoints <= k)
        {
            return 0;
        }

        return (bgs / (k - 1)) / (wgs / (validPoints - k));
    }
}
