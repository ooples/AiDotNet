using AiDotNet.Tensors.Engines;
using AiDotNet.Clustering.Base;
using AiDotNet.Attributes;
using AiDotNet.Clustering.DistanceMetrics;
using AiDotNet.Clustering.Interfaces;
using AiDotNet.Enums;

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
[ComponentType(ComponentType.Evaluator)]
[PipelineStage(PipelineStage.Evaluation)]
public class BallTree<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly IDistanceMetric<T> _distanceMetric;
    private readonly int _leafSize;
    private BallNode? _root;
    private Matrix<T>? _data;
    private int _dimensions;
    private int[][]? _ranks;

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

        // Dense per-dimension ranks: integer comparisons of ranks order points exactly like their values
        _ranks = new int[_dimensions][];
        for (int j = 0; j < _dimensions; j++)
        {
            _ranks[j] = ClusteringTensorOps.DenseRanks(data.GetColumn(j));
        }

        var indices = Enumerable.Range(0, data.Rows).ToArray();

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

        results.Sort(CompareByDistance);
        return results.Take(k).ToArray();
    }

    /// <summary>Orders (index, distance) pairs by ascending distance.</summary>
    private int CompareByDistance((int Index, T Distance) a, (int Index, T Distance) b)
        => _numOps.Compare(a.Distance, b.Distance);

    /// <summary>True when <paramref name="distance"/> is at most <paramref name="limit"/>.</summary>
    private bool IsWithin(T distance, T limit) => _numOps.LessThanOrEquals(distance, limit);

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

        results.Sort(CompareByDistance);
        return results.ToArray();
    }

    private BallNode BuildRecursive(int[] indices, int start, int end)
    {
        int count = end - start;
        var data = _data ?? throw new InvalidOperationException("BallTree: Data not initialized.");
        var members = ClusteringTensorOps.SelectRows(data, new ArraySegment<int>(indices, start, count).ToArray());

        // Compute centroid of all points
        var centroid = ClusteringTensorOps.ColumnMeans(members);

        // Radius: largest distance from the centroid to a member
        T radius = ClusteringTensorOps.MaxOrZero(ClusteringTensorOps.DistancesToPoint(_distanceMetric, members, centroid));

        if (count <= _leafSize)
        {
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
        int splitDim = FindMaxSpreadDimension(members);

        // Sort indices by split dimension
        var ranks = (_ranks ?? throw new InvalidOperationException("BallTree: Data not initialized."))[splitDim];
        Array.Sort(indices, start, count, Comparer<int>.Create((a, b) => ranks[a].CompareTo(ranks[b])));

        int mid = start + count / 2;

        var node = new BallNode
        {
            IsLeaf = false,
            Centroid = centroid,
            Radius = radius
        };

        node.Left = BuildRecursive(indices, start, mid);
        node.Right = BuildRecursive(indices, mid, end);

        return node;
    }

    private int FindMaxSpreadDimension(Matrix<T> members)
    {
        // First dimension with the largest (max - min) spread; 0 when every dimension is constant
        var rows = Tensor<T>.FromMatrix(members);
        var spread = Engine.TensorSubtract(
            Engine.ReduceMax(rows, new[] { 0 }, false), Engine.TensorAmin(rows, new[] { 0 }, false));
        return ClusteringTensorOps.ArgMax(spread.ToVector());
    }

    private static IEngine Engine => AiDotNetEngine.Current;

    private void QueryKNearestRecursive(BallNode node, Vector<T> query, int k, List<(int Index, T Distance)> results)
    {
        // Distance from query to ball center
        T distToCenter = _distanceMetric.Compute(query, node.Centroid);

        // If this ball is too far away and we have k results, skip it
        if (results.Count >= k)
        {
            T kthDist = results[results.Count - 1].Distance;
            if (_numOps.GreaterThan(_numOps.Subtract(distToCenter, node.Radius), kthDist))
            {
                return;
            }
        }

        if (node.IsLeaf)
        {
            var data = _data ?? throw new InvalidOperationException("BallTree: Data not initialized.");
            foreach (int idx in (node.Indices ?? throw new InvalidOperationException("BallTree: Leaf node has null indices.")))
            {
                T dist = _distanceMetric.Compute(query, data.GetRow(idx));
                results.Add((idx, dist));
            }

            // Keep only k best
            if (results.Count > k)
            {
                results.Sort(CompareByDistance);
                results.RemoveRange(k, results.Count - k);
            }
            return;
        }

        // Visit children (closer one first)
        T distToLeft = _distanceMetric.Compute(query, (node.Left ?? throw new InvalidOperationException("BallTree: Non-leaf node has null children.")).Centroid);
        T distToRight = _distanceMetric.Compute(query, (node.Right ?? throw new InvalidOperationException("BallTree: Non-leaf node has null children.")).Centroid);

        if (_numOps.LessThanOrEquals(distToLeft, distToRight))
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
        if (_numOps.GreaterThan(_numOps.Subtract(distToCenter, node.Radius), radius))
        {
            return;
        }

        if (node.IsLeaf)
        {
            var data = _data ?? throw new InvalidOperationException("BallTree: Data not initialized.");
            foreach (int idx in (node.Indices ?? throw new InvalidOperationException("BallTree: Leaf node has null indices.")))
            {
                T dist = _distanceMetric.Compute(query, data.GetRow(idx));
                if (IsWithin(dist, radius))
                {
                    results.Add((idx, dist));
                }
            }
            return;
        }

        QueryRadiusRecursive((node.Left ?? throw new InvalidOperationException("BallTree: Non-leaf node has null children.")), query, radius, results);
        QueryRadiusRecursive((node.Right ?? throw new InvalidOperationException("BallTree: Non-leaf node has null children.")), query, radius, results);
    }

    /// <summary>
    /// Internal node structure for the Ball Tree.
    /// </summary>
    private class BallNode
    {
        public bool IsLeaf { get; set; }
        public required Vector<T> Centroid { get; set; }
        public required T Radius { get; set; }
        public int[]? Indices { get; set; }
        public BallNode? Left { get; set; }
        public BallNode? Right { get; set; }
    }
}
