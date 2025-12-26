using AiDotNet.Clustering.Interfaces;
using AiDotNet.Clustering.DistanceMetrics;

namespace AiDotNet.Clustering.SpatialIndex;

/// <summary>
/// Ball Tree for efficient nearest neighbor queries with arbitrary distance metrics.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// A Ball Tree partitions space using hyperspheres (balls) rather than hyperplanes.
/// This makes it more effective than KD-Trees for high-dimensional data and
/// for non-Euclidean distance metrics.
/// </para>
/// <para><b>For Beginners:</b> A Ball Tree groups nearby points into "balls" (spheres).
/// Each ball contains a center point and a radius that encloses all points in that ball.
///
/// When searching for neighbors:
/// - If the query point is far from a ball's center (farther than the ball's radius
///   plus your search radius), you can skip the entire ball.
/// - This saves a lot of computation for large datasets.
///
/// Ball Trees work better than KD-Trees when:
/// - Your data has many dimensions (>20)
/// - You're using non-Euclidean distance metrics (like cosine distance)
/// </para>
/// </remarks>
public class BallTree<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly IDistanceMetric<T> _distanceMetric;
    private readonly int _leafSize;
    private BallNode? _root;
    private Matrix<T>? _data;
    private int _dimensions;

    /// <summary>
    /// Initializes a new Ball Tree instance.
    /// </summary>
    /// <param name="distanceMetric">The distance metric to use (defaults to Euclidean).</param>
    /// <param name="leafSize">Maximum points in a leaf node.</param>
    public BallTree(IDistanceMetric<T>? distanceMetric = null, int leafSize = 40)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _distanceMetric = distanceMetric ?? new EuclideanDistance<T>();
        _leafSize = leafSize;
    }

    /// <summary>
    /// Gets the number of points in the tree.
    /// </summary>
    public int Count => _data?.Rows ?? 0;

    /// <summary>
    /// Gets the number of dimensions.
    /// </summary>
    public int Dimensions => _dimensions;

    /// <summary>
    /// Builds the Ball Tree from the given data.
    /// </summary>
    /// <param name="data">Matrix where each row is a point.</param>
    public void Build(Matrix<T> data)
    {
        if (data.Rows == 0)
        {
            throw new ArgumentException("Data cannot be empty.", nameof(data));
        }

        _data = data;
        _dimensions = data.Columns;

        var indices = new int[data.Rows];
        for (int i = 0; i < data.Rows; i++)
        {
            indices[i] = i;
        }

        _root = BuildRecursive(indices, 0, indices.Length);
    }

    /// <summary>
    /// Finds the k nearest neighbors to the query point.
    /// </summary>
    /// <param name="query">The query point.</param>
    /// <param name="k">Number of neighbors to find.</param>
    /// <returns>Array of (index, distance) pairs sorted by distance.</returns>
    public (int Index, T Distance)[] QueryKNearest(Vector<T> query, int k)
    {
        if (_root is null || _data is null)
        {
            throw new InvalidOperationException("Tree must be built before querying.");
        }

        if (query.Length != _dimensions)
        {
            throw new ArgumentException(
                $"Query dimension ({query.Length}) must match tree dimension ({_dimensions}).");
        }

        k = Math.Min(k, _data.Rows);
        var results = new List<(int Index, T Distance)>();

        QueryKNearestRecursive(_root, query, k, results);

        results.Sort((a, b) => _numOps.ToDouble(a.Distance).CompareTo(_numOps.ToDouble(b.Distance)));
        return results.Take(k).ToArray();
    }

    /// <summary>
    /// Finds all points within the given radius of the query point.
    /// </summary>
    /// <param name="query">The query point.</param>
    /// <param name="radius">The search radius.</param>
    /// <returns>Array of (index, distance) pairs for points within radius.</returns>
    public (int Index, T Distance)[] QueryRadius(Vector<T> query, T radius)
    {
        if (_root is null || _data is null)
        {
            throw new InvalidOperationException("Tree must be built before querying.");
        }

        if (query.Length != _dimensions)
        {
            throw new ArgumentException(
                $"Query dimension ({query.Length}) must match tree dimension ({_dimensions}).");
        }

        var results = new List<(int Index, T Distance)>();
        QueryRadiusRecursive(_root, query, radius, results);

        results.Sort((a, b) => _numOps.ToDouble(a.Distance).CompareTo(_numOps.ToDouble(b.Distance)));
        return results.ToArray();
    }

    private BallNode BuildRecursive(int[] indices, int start, int end)
    {
        int count = end - start;

        // Compute centroid of all points
        var centroid = ComputeCentroid(indices, start, end);

        if (count <= _leafSize)
        {
            // Compute radius
            T radius = ComputeRadius(centroid, indices, start, end);

            var leafIndices = new int[count];
            Array.Copy(indices, start, leafIndices, 0, count);

            return new BallNode
            {
                IsLeaf = true,
                Centroid = centroid,
                Radius = radius,
                Indices = leafIndices
            };
        }

        // Find the dimension with maximum spread
        int splitDim = FindMaxSpreadDimension(indices, start, end);

        // Sort indices by split dimension
        Array.Sort(indices, start, count, Comparer<int>.Create((a, b) =>
            _numOps.ToDouble(_data![a, splitDim]).CompareTo(_numOps.ToDouble(_data[b, splitDim]))));

        int mid = start + count / 2;

        // Compute radius for this node
        T nodeRadius = ComputeRadius(centroid, indices, start, end);

        var node = new BallNode
        {
            IsLeaf = false,
            Centroid = centroid,
            Radius = nodeRadius
        };

        node.Left = BuildRecursive(indices, start, mid);
        node.Right = BuildRecursive(indices, mid, end);

        return node;
    }

    private Vector<T> ComputeCentroid(int[] indices, int start, int end)
    {
        var centroid = new Vector<T>(_dimensions);
        int count = end - start;

        for (int j = 0; j < _dimensions; j++)
        {
            T sum = _numOps.Zero;
            for (int i = start; i < end; i++)
            {
                sum = _numOps.Add(sum, _data![indices[i], j]);
            }
            centroid[j] = _numOps.Divide(sum, _numOps.FromDouble(count));
        }

        return centroid;
    }

    private T ComputeRadius(Vector<T> centroid, int[] indices, int start, int end)
    {
        T maxDist = _numOps.Zero;

        for (int i = start; i < end; i++)
        {
            var point = GetRow(_data!, indices[i]);
            T dist = _distanceMetric.Compute(centroid, point);
            if (_numOps.ToDouble(dist) > _numOps.ToDouble(maxDist))
            {
                maxDist = dist;
            }
        }

        return maxDist;
    }

    private int FindMaxSpreadDimension(int[] indices, int start, int end)
    {
        int maxDim = 0;
        double maxSpread = 0;

        for (int d = 0; d < _dimensions; d++)
        {
            double min = double.MaxValue;
            double max = double.MinValue;

            for (int i = start; i < end; i++)
            {
                double val = _numOps.ToDouble(_data![indices[i], d]);
                min = Math.Min(min, val);
                max = Math.Max(max, val);
            }

            double spread = max - min;
            if (spread > maxSpread)
            {
                maxSpread = spread;
                maxDim = d;
            }
        }

        return maxDim;
    }

    private void QueryKNearestRecursive(BallNode node, Vector<T> query, int k, List<(int Index, T Distance)> results)
    {
        // Distance from query to ball center
        T distToCenter = _distanceMetric.Compute(query, node.Centroid);

        // If this ball is too far away and we have k results, skip it
        if (results.Count >= k)
        {
            T kthDist = results[results.Count - 1].Distance;
            double minPossibleDist = _numOps.ToDouble(distToCenter) - _numOps.ToDouble(node.Radius);
            if (minPossibleDist > _numOps.ToDouble(kthDist))
            {
                return;
            }
        }

        if (node.IsLeaf)
        {
            foreach (int idx in node.Indices!)
            {
                var point = GetRow(_data!, idx);
                T dist = _distanceMetric.Compute(query, point);
                results.Add((idx, dist));
            }

            // Keep only k best
            if (results.Count > k)
            {
                results.Sort((a, b) => _numOps.ToDouble(a.Distance).CompareTo(_numOps.ToDouble(b.Distance)));
                results.RemoveRange(k, results.Count - k);
            }
            return;
        }

        // Visit children (closer one first)
        T distToLeft = _distanceMetric.Compute(query, node.Left!.Centroid);
        T distToRight = _distanceMetric.Compute(query, node.Right!.Centroid);

        if (_numOps.ToDouble(distToLeft) <= _numOps.ToDouble(distToRight))
        {
            QueryKNearestRecursive(node.Left, query, k, results);
            QueryKNearestRecursive(node.Right, query, k, results);
        }
        else
        {
            QueryKNearestRecursive(node.Right, query, k, results);
            QueryKNearestRecursive(node.Left, query, k, results);
        }
    }

    private void QueryRadiusRecursive(BallNode node, Vector<T> query, T radius, List<(int Index, T Distance)> results)
    {
        // Distance from query to ball center
        T distToCenter = _distanceMetric.Compute(query, node.Centroid);

        // If this ball is too far away, skip it entirely
        double minPossibleDist = _numOps.ToDouble(distToCenter) - _numOps.ToDouble(node.Radius);
        if (minPossibleDist > _numOps.ToDouble(radius))
        {
            return;
        }

        if (node.IsLeaf)
        {
            foreach (int idx in node.Indices!)
            {
                var point = GetRow(_data!, idx);
                T dist = _distanceMetric.Compute(query, point);
                if (_numOps.ToDouble(dist) <= _numOps.ToDouble(radius))
                {
                    results.Add((idx, dist));
                }
            }
            return;
        }

        QueryRadiusRecursive(node.Left!, query, radius, results);
        QueryRadiusRecursive(node.Right!, query, radius, results);
    }

    private Vector<T> GetRow(Matrix<T> matrix, int rowIndex)
    {
        var row = new Vector<T>(matrix.Columns);
        for (int j = 0; j < matrix.Columns; j++)
        {
            row[j] = matrix[rowIndex, j];
        }
        return row;
    }

    /// <summary>
    /// Internal node structure for the Ball Tree.
    /// </summary>
    private class BallNode
    {
        public bool IsLeaf { get; set; }
        public Vector<T> Centroid { get; set; } = null!;
        public T Radius { get; set; } = default!;
        public int[]? Indices { get; set; }
        public BallNode? Left { get; set; }
        public BallNode? Right { get; set; }
    }
}
