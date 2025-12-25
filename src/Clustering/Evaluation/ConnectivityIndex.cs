using AiDotNet.Clustering.DistanceMetrics;
using AiDotNet.Clustering.Interfaces;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Clustering.Evaluation;

/// <summary>
/// Computes the Connectivity Index for cluster validity assessment.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The Connectivity Index measures the degree to which points are connected
/// to their neighbors within the same cluster. Lower values indicate better
/// clustering where nearby points are grouped together.
/// </para>
/// <para>
/// Formula: Connectivity = sum over all points i, over L nearest neighbors:
///   (1/j) if the j-th nearest neighbor is in a different cluster
/// </para>
/// <para><b>For Beginners:</b> Connectivity checks if neighbors are together.
///
/// For each point, look at its closest neighbors:
/// - 1st nearest neighbor: If in different cluster, add 1/1 = 1.0
/// - 2nd nearest neighbor: If in different cluster, add 1/2 = 0.5
/// - 3rd nearest neighbor: If in different cluster, add 1/3 = 0.33
/// - And so on...
///
/// The penalty is higher for closer neighbors being separated.
///
/// Interpretation:
/// - 0 = Perfect (all nearest neighbors in same cluster)
/// - Higher values = Worse (clusters split natural groupings)
///
/// Unlike other metrics:
/// - Lower is better (0 is ideal)
/// - No upper bound
/// - Intuitive: "Are nearby points kept together?"
/// </para>
/// </remarks>
public class ConnectivityIndex<T> : IClusterMetric<T>
{
    private readonly int _numNeighbors;
    private readonly IDistanceMetric<T>? _distanceMetric;
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new ConnectivityIndex instance.
    /// </summary>
    /// <param name="numNeighbors">Number of nearest neighbors to consider. Default is 10.</param>
    /// <param name="distanceMetric">Distance metric to use, or null for Euclidean (default).</param>
    public ConnectivityIndex(int numNeighbors = 10, IDistanceMetric<T>? distanceMetric = null)
    {
        if (numNeighbors <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numNeighbors),
                "Number of neighbors must be greater than 0.");
        }

        _numNeighbors = numNeighbors;
        _distanceMetric = distanceMetric;
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <inheritdoc />
    public string Name => "Connectivity Index";

    /// <inheritdoc />
    public bool HigherIsBetter => false;

    /// <inheritdoc />
    public double Compute(Matrix<T> data, Vector<T> labels)
    {
        int n = data.Rows;

        int L = Math.Min(_numNeighbors, n - 1);
        double connectivity = 0;

        // Use configured distance metric or default to Euclidean
        var metric = _distanceMetric ?? new EuclideanDistance<T>();

        for (int i = 0; i < n; i++)
        {
            int iLabel = (int)_numOps.ToDouble(labels[i]);

            // Find L nearest neighbors
            var neighbors = FindNearestNeighbors(data, i, L, n, metric);

            // Add penalty for neighbors in different clusters
            for (int j = 0; j < neighbors.Length; j++)
            {
                int neighborIdx = neighbors[j];
                int neighborLabel = (int)_numOps.ToDouble(labels[neighborIdx]);

                if (neighborLabel != iLabel)
                {
                    connectivity += 1.0 / (j + 1);
                }
            }
        }

        return connectivity;
    }

    private int[] FindNearestNeighbors(Matrix<T> data, int pointIdx, int L, int n, IDistanceMetric<T> metric)
    {
        var distances = new List<(int idx, double dist)>();
        var pointI = data.GetRow(pointIdx);

        for (int j = 0; j < n; j++)
        {
            if (j == pointIdx) continue;

            var pointJ = data.GetRow(j);
            double dist = _numOps.ToDouble(metric.Compute(pointI, pointJ));

            distances.Add((j, dist));
        }

        return distances.OrderBy(x => x.dist).Take(L).Select(x => x.idx).ToArray();
    }
}
