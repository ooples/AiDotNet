using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.DimensionalityReduction;

/// <summary>
/// Isomap (Isometric Mapping) for nonlinear dimensionality reduction.
/// </summary>
/// <remarks>
/// <para>
/// Isomap extends classical MDS by estimating geodesic distances along the data
/// manifold instead of using Euclidean distances. It builds a neighborhood graph
/// and computes shortest paths to estimate geodesic distances.
/// </para>
/// <para>
/// The algorithm:
/// 1. Build a k-nearest neighbors graph
/// 2. Compute shortest path distances (Floyd-Warshall or Dijkstra)
/// 3. Apply MDS to the geodesic distance matrix
/// </para>
/// <para><b>For Beginners:</b> Isomap "unrolls" curved data:
/// - Imagine data lying on a Swiss roll (curved surface)
/// - Regular PCA can't flatten it properly
/// - Isomap finds distances along the surface, not through the air
/// - Result: Points that are nearby on the surface stay nearby
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.DimensionReducer)]
[PipelineStage(PipelineStage.Preprocessing)]
public class Isomap<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nComponents;
    private readonly int _nNeighbors;
    private readonly IsomapNeighborAlgorithm _neighborAlgorithm;
    private readonly IsomapPathAlgorithm _pathAlgorithm;

    // Fitted parameters
    private Matrix<T>? _embedding;
    private Matrix<T>? _geodesicDistances;
    private int _nSamples;
    private int _nFeaturesIn;

    /// <summary>
    /// Gets the number of components.
    /// </summary>
    public int NComponents => _nComponents;

    /// <summary>
    /// Gets the number of neighbors for graph construction.
    /// </summary>
    public int NNeighbors => _nNeighbors;

    /// <summary>
    /// Gets the geodesic distance matrix.
    /// </summary>
    public Matrix<T>? GeodesicDistances => _geodesicDistances;

    /// <summary>
    /// Gets the embedding result.
    /// </summary>
    public Matrix<T>? Embedding => _embedding;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="Isomap{T}"/>.
    /// </summary>
    /// <param name="nComponents">Target dimensionality. Defaults to 2.</param>
    /// <param name="nNeighbors">Number of neighbors for graph construction. Defaults to 5.</param>
    /// <param name="neighborAlgorithm">Algorithm for neighbor search. Defaults to Auto.</param>
    /// <param name="pathAlgorithm">Algorithm for shortest paths. Defaults to Dijkstra.</param>
    /// <param name="columnIndices">The column indices to use, or null for all columns.</param>
    public Isomap(
        int nComponents = 2,
        int nNeighbors = 5,
        IsomapNeighborAlgorithm neighborAlgorithm = IsomapNeighborAlgorithm.Auto,
        IsomapPathAlgorithm pathAlgorithm = IsomapPathAlgorithm.Dijkstra,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nComponents < 1)
        {
            throw new ArgumentException("Number of components must be at least 1.", nameof(nComponents));
        }

        if (nNeighbors < 1)
        {
            throw new ArgumentException("Number of neighbors must be at least 1.", nameof(nNeighbors));
        }

        _nComponents = nComponents;
        _nNeighbors = nNeighbors;
        _neighborAlgorithm = neighborAlgorithm;
        _pathAlgorithm = pathAlgorithm;
    }

    /// <summary>
    /// Fits Isomap by computing geodesic distances and embedding.
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        _nSamples = data.Rows;
        _nFeaturesIn = data.Columns;

        var x = Tensor<T>.FromMatrix(data);

        // Step 1: Compute pairwise Euclidean distances
        var distances = DimensionalityReductionOps.EuclideanDistances(x, x);

        // Step 2: Build k-nearest neighbors graph
        var graph = BuildNeighborGraph(distances);

        // Step 3: Compute shortest path distances (geodesic)
        var geodesic = _pathAlgorithm == IsomapPathAlgorithm.FloydWarshall
            ? FloydWarshall(graph)
            : Dijkstra(graph);

        // Check for disconnected graph
        if (NumOps.IsInfinity(Engine.TensorMaxValue(geodesic)))
        {
            throw new InvalidOperationException(
                "The neighborhood graph is disconnected. Try increasing n_neighbors.");
        }

        _geodesicDistances = geodesic.ToMatrix();

        // Step 4: Apply classical MDS to geodesic distance matrix
        _embedding = DimensionalityReductionOps.ClassicalMds(geodesic, _nComponents).ToMatrix();
    }

    /// <summary>
    /// Symmetric k-nearest-neighbour graph: edge weight = distance, 0 on the diagonal, +infinity where there is no edge.
    /// </summary>
    private Tensor<T> BuildNeighborGraph(Tensor<T> distances)
    {
        var engine = Engine;
        int n = distances.Shape[0];
        var infinity = NumOps.FromDouble(double.PositiveInfinity);

        var neighbors = DimensionalityReductionOps.NearestNeighbors(distances, _nNeighbors);
        var directed = DimensionalityReductionOps.ScatterRows(
            neighbors, DimensionalityReductionOps.TakeRows(distances, neighbors), n, infinity);

        // An edge found from either end is kept in both directions
        var graph = engine.TensorMin(directed, engine.TensorTranspose(directed));
        engine.TensorFillDiagonalInPlace(graph, NumOps.Zero);
        return graph;
    }

    /// <summary>
    /// All-pairs shortest paths by Floyd-Warshall: dist = min(dist, dist[:, k] + dist[k, :]) for every k.
    /// </summary>
    private Tensor<T> FloydWarshall(Tensor<T> graph)
    {
        var engine = Engine;
        int n = graph.Shape[0];
        var dist = graph;
        for (int k = 0; k < n; k++)
        {
            var throughK = engine.TensorAdd(engine.TensorNarrow(dist, 1, k, 1), engine.TensorNarrow(dist, 0, k, 1));
            dist = engine.TensorMin(dist, throughK);
        }

        return dist;
    }

    /// <summary>
    /// All-pairs shortest paths by Dijkstra, run for every source at once: each step settles, per source row,
    /// the closest unsettled vertex u and relaxes d[source, v] = min(d[source, v], d[source, u] + w(u, v)).
    /// </summary>
    private Tensor<T> Dijkstra(Tensor<T> graph)
    {
        var engine = Engine;
        int n = graph.Shape[0];
        var infinity = NumOps.FromDouble(double.PositiveInfinity);
        var infinite = DimensionalityReductionOps.Full(new[] { n, n }, infinity);

        var dist = infinite.Clone();
        engine.TensorFillDiagonalInPlace(dist, NumOps.Zero);
        var settled = new Tensor<T>(new[] { n, n });
        var settleMark = DimensionalityReductionOps.Full(new[] { n, 1 }, NumOps.One);

        for (int step = 0; step < n; step++)
        {
            // Closest unsettled vertex for every source (first index among equal distances)
            var open = engine.TensorWhere(settled, infinite, dist);
            var closest = engine.Reshape(AiDotNet.AnomalyDetection.AnomalyTensorOps.FirstArgMinRows(open), new[] { n, 1 });
            settled = engine.TensorScatter(settled, closest, settleMark, 1);

            // Relax through the settled vertex; an unreachable vertex (infinite distance) changes nothing
            var throughClosest = engine.TensorAdd(
                engine.TensorTakeAlongDim(dist, closest, 1),
                engine.TensorIndexSelect(graph, engine.Reshape(closest, new[] { n }), 0));
            dist = engine.TensorMin(dist, throughClosest);
        }

        return dist;
    }

    /// <summary>
    /// Returns the embedding computed during Fit.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_embedding is null)
        {
            throw new InvalidOperationException("Isomap has not been fitted.");
        }

        // Isomap doesn't naturally support out-of-sample transformation
        if (data.Rows != _nSamples)
        {
            throw new InvalidOperationException(
                "Isomap does not support out-of-sample transformation. " +
                "Use FitTransform() on the complete dataset.");
        }

        return _embedding.Clone();
    }

    /// <summary>
    /// Inverse transformation is not supported.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("Isomap does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        var names = new string[_nComponents];
        for (int i = 0; i < _nComponents; i++)
        {
            names[i] = $"Isomap{i + 1}";
        }
        return names;
    }
}

/// <summary>
/// Specifies the neighbor search algorithm for Isomap.
/// </summary>
public enum IsomapNeighborAlgorithm
{
    /// <summary>
    /// Automatically choose the best algorithm.
    /// </summary>
    Auto,

    /// <summary>
    /// Brute force search (O(n²)).
    /// </summary>
    BruteForce,

    /// <summary>
    /// KD-tree for efficient neighbor search.
    /// </summary>
    KDTree
}

/// <summary>
/// Specifies the shortest path algorithm for Isomap.
/// </summary>
public enum IsomapPathAlgorithm
{
    /// <summary>
    /// Floyd-Warshall algorithm (O(n³), finds all pairs).
    /// </summary>
    FloydWarshall,

    /// <summary>
    /// Dijkstra's algorithm (O(n² log n) for all pairs).
    /// </summary>
    Dijkstra
}
