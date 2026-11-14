using AiDotNet.Clustering.Base;
using AiDotNet.Attributes;
using AiDotNet.Clustering.DistanceMetrics;
using AiDotNet.Clustering.Interfaces;
using AiDotNet.Enums;

namespace AiDotNet.Clustering.SpatialIndex;

#pragma warning disable CS8601, CS8618 // Generic T defaults use default(T) - always used with value types

/// <summary>
/// K-dimensional tree for efficient nearest neighbor queries.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// KD-Tree is a space-partitioning data structure for organizing points in k-dimensional
/// space. It enables efficient nearest neighbor searches in O(log n) average case,
/// compared to O(n) for brute-force search.
/// </para>
/// <para><b>For Beginners:</b> A KD-Tree is like a special way of organizing data points
/// so you can quickly find nearby points.
///
/// Imagine organizing a phone book not just by last name, but alternating between
/// first name and last name at each level. This lets you narrow down your search
/// very quickly.
///
/// KD-Trees are used in:
/// - DBSCAN clustering (to find points within epsilon radius)
/// - K-Nearest Neighbors (KNN) classification
/// - Range searches (find all points in a region)
/// </para>
/// </remarks>
[ComponentType(ComponentType.Evaluator)]
[PipelineStage(PipelineStage.Evaluation)]
public class KDTree<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly IDistanceMetric<T> _distanceMetric;
    private readonly int _leafSize;
    private KDNode? _root;
    private Matrix<T>? _data;
    private int _dimensions;
    private int[][]? _ranks;

    /// <summary>
    /// Initializes a new KD-Tree instance.
    /// </summary>
    /// <param name="distanceMetric">The distance metric to use (defaults to Euclidean).</param>
    /// <param name="leafSize">Maximum points in a leaf node before splitting.</param>
    public KDTree(IDistanceMetric<T>? distanceMetric = null, int leafSize = 10)
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
    /// Builds the KD-Tree from the given data.
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

        // Create indices array for all points
        var indices = Enumerable.Range(0, data.Rows).ToArray();

        _root = BuildRecursive(indices, 0, indices.Length, 0);
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
        var heap = new BoundedMaxHeap<T>(k, _numOps);

        QueryKNearestRecursive(_root, query, heap, 0);

        return heap.ToSortedArray();
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
        QueryRadiusRecursive(_root, query, radius, results, 0);

        // Sort by distance
        results.Sort(CompareByDistance);

        return results.ToArray();
    }

    /// <summary>Orders (index, distance) pairs by ascending distance.</summary>
    private int CompareByDistance((int Index, T Distance) a, (int Index, T Distance) b)
        => _numOps.Compare(a.Distance, b.Distance);

    /// <summary>True when <paramref name="distance"/> is at most <paramref name="limit"/>.</summary>
    private bool IsWithin(T distance, T limit) => _numOps.LessThanOrEquals(distance, limit);

    /// <summary>
    /// Finds all points within the given squared radius of the query point.
    /// More efficient when you can work with squared distances.
    /// </summary>
    /// <param name="query">The query point.</param>
    /// <param name="squaredRadius">The squared search radius.</param>
    /// <returns>Array of point indices within the squared radius.</returns>
    public int[] QueryRadiusIndices(Vector<T> query, T squaredRadius)
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

        var results = new List<int>();
        QueryRadiusIndicesRecursive(_root, query, squaredRadius, results, 0);

        return results.ToArray();
    }

    private KDNode BuildRecursive(int[] indices, int start, int end, int depth)
    {
        int count = end - start;

        if (count <= _leafSize)
        {
            // Create leaf node
            var leafIndices = new int[count];
            Array.Copy(indices, start, leafIndices, 0, count);
            return new KDNode { Indices = leafIndices, IsLeaf = true };
        }

        // Choose split dimension (cycle through dimensions)
        int splitDim = depth % _dimensions;

        // Find median using QuickSelect
        int medianIdx = start + count / 2;
        QuickSelect(indices, start, end - 1, medianIdx, splitDim);

        int splitPointIndex = indices[medianIdx];
        T splitValue = (_data ?? throw new InvalidOperationException("KDTree: Data not initialized."))[splitPointIndex, splitDim];

        var node = new KDNode
        {
            SplitDimension = splitDim,
            SplitValue = splitValue,
            PointIndex = splitPointIndex,
            IsLeaf = false
        };

        // Build subtrees
        if (medianIdx > start)
        {
            node.Left = BuildRecursive(indices, start, medianIdx, depth + 1);
        }

        if (medianIdx + 1 < end)
        {
            node.Right = BuildRecursive(indices, medianIdx + 1, end, depth + 1);
        }

        return node;
    }

    private void QuickSelect(int[] indices, int left, int right, int k, int dimension)
    {
        while (left < right)
        {
            int pivotIndex = Partition(indices, left, right, dimension);

            if (pivotIndex == k)
            {
                return;
            }
            else if (pivotIndex < k)
            {
                left = pivotIndex + 1;
            }
            else
            {
                right = pivotIndex - 1;
            }
        }
    }

    private int Partition(int[] indices, int left, int right, int dimension)
    {
        // Use median of three for pivot selection
        int mid = (left + right) / 2;
        if (Compare(indices[mid], indices[left], dimension) < 0)
        {
            Swap(indices, left, mid);
        }
        if (Compare(indices[right], indices[left], dimension) < 0)
        {
            Swap(indices, left, right);
        }
        if (Compare(indices[mid], indices[right], dimension) < 0)
        {
            Swap(indices, mid, right);
        }

        var ranks = Ranks(dimension);
        int pivotRank = ranks[indices[right]];
        int storeIndex = left;

        for (int i = left; i < right; i++)
        {
            if (ranks[indices[i]] < pivotRank)
            {
                Swap(indices, i, storeIndex);
                storeIndex++;
            }
        }

        Swap(indices, storeIndex, right);
        return storeIndex;
    }

    private int Compare(int idx1, int idx2, int dimension)
    {
        var ranks = Ranks(dimension);
        return ranks[idx1].CompareTo(ranks[idx2]);
    }

    /// <summary>Dense value ranks of every point along one dimension.</summary>
    private int[] Ranks(int dimension)
        => (_ranks ?? throw new InvalidOperationException("KDTree: Data not initialized."))[dimension];

    private static void Swap(int[] array, int i, int j)
    {
        int temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }

    private void QueryKNearestRecursive(KDNode node, Vector<T> query, BoundedMaxHeap<T> heap, int depth)
    {
        if (node.IsLeaf)
        {
            // Check all points in leaf
            foreach (int idx in (node.Indices ?? throw new InvalidOperationException("KDTree: Leaf node has null indices.")))
            {
                T dist = ComputeDistance(query, idx);
                heap.TryAdd(idx, dist);
            }
            return;
        }

        // Check current node's point
        if (node.PointIndex >= 0)
        {
            T dist = ComputeDistance(query, node.PointIndex);
            heap.TryAdd(node.PointIndex, dist);
        }

        T diff = _numOps.Subtract(query[node.SplitDimension], node.SplitValue);

        // Determine which subtree to search first
        bool goLeft = !_numOps.GreaterThan(diff, _numOps.Zero);
        KDNode? nearNode = goLeft ? node.Left : node.Right;
        KDNode? farNode = goLeft ? node.Right : node.Left;

        // Search near subtree first
        if (nearNode is not null)
        {
            QueryKNearestRecursive(nearNode, query, heap, depth + 1);
        }

        // Check if we need to search far subtree
        T axisDist = _numOps.Abs(diff);
        if (farNode is not null && (heap.Count < heap.Capacity || _numOps.LessThan(axisDist, heap.MaxDistance)))
        {
            QueryKNearestRecursive(farNode, query, heap, depth + 1);
        }
    }

    private void QueryRadiusRecursive(KDNode node, Vector<T> query, T radius, List<(int, T)> results, int depth)
    {
        if (node.IsLeaf)
        {
            foreach (int idx in (node.Indices ?? throw new InvalidOperationException("KDTree: Leaf node has null indices.")))
            {
                T dist = ComputeDistance(query, idx);
                if (IsWithin(dist, radius))
                {
                    results.Add((idx, dist));
                }
            }
            return;
        }

        // Check current node's point
        if (node.PointIndex >= 0)
        {
            T dist = ComputeDistance(query, node.PointIndex);
            if (IsWithin(dist, radius))
            {
                results.Add((node.PointIndex, dist));
            }
        }

        T queryValue = query[node.SplitDimension];

        // Search left subtree if it might contain points within radius
        if (node.Left is not null && _numOps.LessThanOrEquals(_numOps.Subtract(queryValue, radius), node.SplitValue))
        {
            QueryRadiusRecursive(node.Left, query, radius, results, depth + 1);
        }

        // Search right subtree if it might contain points within radius
        if (node.Right is not null && _numOps.GreaterThanOrEquals(_numOps.Add(queryValue, radius), node.SplitValue))
        {
            QueryRadiusRecursive(node.Right, query, radius, results, depth + 1);
        }
    }

    private void QueryRadiusIndicesRecursive(KDNode node, Vector<T> query, T squaredRadius, List<int> results, int depth)
    {
        if (node.IsLeaf)
        {
            foreach (int idx in (node.Indices ?? throw new InvalidOperationException("KDTree: Leaf node has null indices.")))
            {
                T squaredDist = ComputeSquaredDistance(query, idx);
                if (IsWithin(squaredDist, squaredRadius))
                {
                    results.Add(idx);
                }
            }
            return;
        }

        // Check current node's point
        if (node.PointIndex >= 0)
        {
            T squaredDist = ComputeSquaredDistance(query, node.PointIndex);
            if (IsWithin(squaredDist, squaredRadius))
            {
                results.Add(node.PointIndex);
            }
        }

        T queryValue = query[node.SplitDimension];
        T radius = _numOps.Sqrt(squaredRadius);

        // Search left subtree if it might contain points within radius
        if (node.Left is not null && _numOps.LessThanOrEquals(_numOps.Subtract(queryValue, radius), node.SplitValue))
        {
            QueryRadiusIndicesRecursive(node.Left, query, squaredRadius, results, depth + 1);
        }

        // Search right subtree if it might contain points within radius
        if (node.Right is not null && _numOps.GreaterThanOrEquals(_numOps.Add(queryValue, radius), node.SplitValue))
        {
            QueryRadiusIndicesRecursive(node.Right, query, squaredRadius, results, depth + 1);
        }
    }

    private T ComputeDistance(Vector<T> query, int dataIndex)
    {
        var point = (_data ?? throw new InvalidOperationException("KDTree: Data not initialized.")).GetRow(dataIndex);
        return _distanceMetric.Compute(query, point);
    }

    private T ComputeSquaredDistance(Vector<T> query, int dataIndex)
    {
        // Sum of squared differences as whole-vector tensor ops (no per-element numeric calls)
        var data = _data ?? throw new InvalidOperationException("KDTree: Data not initialized.");
        return ClusteringTensorOps.SquaredDistancesToPoint(query, data.GetSubMatrix(dataIndex, 0, 1, data.Columns))[0];
    }

    /// <summary>
    /// Internal node structure for the KD-Tree.
    /// </summary>
    private class KDNode
    {
        public bool IsLeaf { get; set; }
        public int[]? Indices { get; set; }  // For leaf nodes
        public int SplitDimension { get; set; }
        public T SplitValue { get; set; } = default;
        public int PointIndex { get; set; } = -1;
        public KDNode? Left { get; set; }
        public KDNode? Right { get; set; }
    }

    /// <summary>
    /// Bounded max-heap for k-nearest neighbor queries.
    /// </summary>
    private class BoundedMaxHeap<TNum>
    {
        private readonly List<(int Index, TNum Distance)> _heap;
        private readonly int _capacity;
        private readonly INumericOperations<TNum> _ops;

        public BoundedMaxHeap(int capacity, INumericOperations<TNum> ops)
        {
            _capacity = capacity;
            _ops = ops;
            _heap = new List<(int, TNum)>(capacity + 1);
        }

        public int Count => _heap.Count;
        public int Capacity => _capacity;

        public TNum MaxDistance => _heap.Count > 0 ? _heap[0].Distance : _ops.FromDouble(double.MaxValue);

        public void TryAdd(int index, TNum distance)
        {
            if (_heap.Count < _capacity)
            {
                _heap.Add((index, distance));
                BubbleUp(_heap.Count - 1);
            }
            else if (_ops.LessThan(distance, _heap[0].Distance))
            {
                _heap[0] = (index, distance);
                BubbleDown(0);
            }
        }

        public (int Index, TNum Distance)[] ToSortedArray()
        {
            var result = _heap.ToArray();
            Array.Sort(result, CompareByDistance);
            return result;
        }

        private int CompareByDistance((int Index, TNum Distance) a, (int Index, TNum Distance) b)
            => _ops.Compare(a.Distance, b.Distance);

        /// <summary>True when the entry at <paramref name="i"/> is farther than the entry at <paramref name="j"/>.</summary>
        private bool IsFarther(int i, int j) => _ops.GreaterThan(_heap[i].Distance, _heap[j].Distance);

        private void BubbleUp(int index)
        {
            while (index > 0)
            {
                int parent = (index - 1) / 2;
                if (!IsFarther(index, parent))
                {
                    break;
                }
                SwapItems(index, parent);
                index = parent;
            }
        }

        private void BubbleDown(int index)
        {
            while (true)
            {
                int largest = index;
                int left = 2 * index + 1;
                int right = 2 * index + 2;

                if (left < _heap.Count && IsFarther(left, largest))
                {
                    largest = left;
                }

                if (right < _heap.Count && IsFarther(right, largest))
                {
                    largest = right;
                }

                if (largest == index)
                {
                    break;
                }

                SwapItems(index, largest);
                index = largest;
            }
        }

        private void SwapItems(int i, int j)
        {
            var temp = _heap[i];
            _heap[i] = _heap[j];
            _heap[j] = temp;
        }
    }
}
