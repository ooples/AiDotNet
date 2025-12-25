using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

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
public class Isomap<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nComponents;
    private readonly int _nNeighbors;
    private readonly IsomapNeighborAlgorithm _neighborAlgorithm;
    private readonly IsomapPathAlgorithm _pathAlgorithm;

    // Fitted parameters
    private double[,]? _embedding;
    private double[,]? _geodesicDistances;
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
    public double[,]? GeodesicDistances => _geodesicDistances;

    /// <summary>
    /// Gets the embedding result.
    /// </summary>
    public double[,]? Embedding => _embedding;

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
        int n = data.Rows;
        int p = data.Columns;

        // Convert to double array
        var X = new double[n, p];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < p; j++)
            {
                X[i, j] = NumOps.ToDouble(data[i, j]);
            }
        }

        // Step 1: Compute pairwise Euclidean distances
        var distances = ComputeDistanceMatrix(X, n, p);

        // Step 2: Build k-nearest neighbors graph
        var graph = BuildNeighborGraph(distances, n);

        // Step 3: Compute shortest path distances (geodesic)
        _geodesicDistances = _pathAlgorithm == IsomapPathAlgorithm.FloydWarshall
            ? FloydWarshall(graph, n)
            : Dijkstra(graph, n);

        // Check for disconnected graph
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (double.IsPositiveInfinity(_geodesicDistances[i, j]))
                {
                    throw new InvalidOperationException(
                        "The neighborhood graph is disconnected. Try increasing n_neighbors.");
                }
            }
        }

        // Step 4: Apply classical MDS to geodesic distance matrix
        _embedding = ClassicalMDS(_geodesicDistances, n);
    }

    private double[,] ComputeDistanceMatrix(double[,] X, int n, int p)
    {
        var distances = new double[n, n];

        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                double dist = 0;
                for (int k = 0; k < p; k++)
                {
                    double diff = X[i, k] - X[j, k];
                    dist += diff * diff;
                }
                dist = Math.Sqrt(dist);

                distances[i, j] = dist;
                distances[j, i] = dist;
            }
        }

        return distances;
    }

    private double[,] BuildNeighborGraph(double[,] distances, int n)
    {
        // Initialize graph with infinity
        var graph = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                graph[i, j] = i == j ? 0 : double.PositiveInfinity;
            }
        }

        // Find k-nearest neighbors for each point
        for (int i = 0; i < n; i++)
        {
            // Get indices sorted by distance
            var neighbors = Enumerable.Range(0, n)
                .Where(j => j != i)
                .OrderBy(j => distances[i, j])
                .Take(_nNeighbors)
                .ToList();

            // Add edges to nearest neighbors
            foreach (int j in neighbors)
            {
                graph[i, j] = distances[i, j];
                graph[j, i] = distances[j, i]; // Make symmetric
            }
        }

        return graph;
    }

    private double[,] FloydWarshall(double[,] graph, int n)
    {
        var dist = (double[,])graph.Clone();

        for (int k = 0; k < n; k++)
        {
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    if (dist[i, k] + dist[k, j] < dist[i, j])
                    {
                        dist[i, j] = dist[i, k] + dist[k, j];
                    }
                }
            }
        }

        return dist;
    }

    private double[,] Dijkstra(double[,] graph, int n)
    {
        var dist = new double[n, n];

        for (int source = 0; source < n; source++)
        {
            // Initialize distances
            var d = new double[n];
            var visited = new bool[n];

            for (int i = 0; i < n; i++)
            {
                d[i] = double.PositiveInfinity;
            }
            d[source] = 0;

            // Priority queue simulation using linear search
            for (int count = 0; count < n; count++)
            {
                // Find minimum distance unvisited vertex
                int u = -1;
                double minDist = double.PositiveInfinity;
                for (int i = 0; i < n; i++)
                {
                    if (!visited[i] && d[i] < minDist)
                    {
                        minDist = d[i];
                        u = i;
                    }
                }

                if (u == -1) break;
                visited[u] = true;

                // Update distances
                for (int v = 0; v < n; v++)
                {
                    if (!visited[v] && graph[u, v] < double.PositiveInfinity)
                    {
                        double newDist = d[u] + graph[u, v];
                        if (newDist < d[v])
                        {
                            d[v] = newDist;
                        }
                    }
                }
            }

            for (int i = 0; i < n; i++)
            {
                dist[source, i] = d[i];
            }
        }

        return dist;
    }

    private double[,] ClassicalMDS(double[,] D, int n)
    {
        // Convert distance matrix to squared distances
        var D2 = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                D2[i, j] = D[i, j] * D[i, j];
            }
        }

        // Double centering: B = -0.5 * J * D^2 * J where J = I - 1/n * 1*1'
        var B = new double[n, n];

        // Compute row means and grand mean
        var rowMeans = new double[n];
        double grandMean = 0;

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                rowMeans[i] += D2[i, j];
            }
            rowMeans[i] /= n;
            grandMean += rowMeans[i];
        }
        grandMean /= n;

        var colMeans = new double[n];
        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < n; i++)
            {
                colMeans[j] += D2[i, j];
            }
            colMeans[j] /= n;
        }

        // B = -0.5 * (D^2 - row_mean - col_mean + grand_mean)
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                B[i, j] = -0.5 * (D2[i, j] - rowMeans[i] - colMeans[j] + grandMean);
            }
        }

        // Eigenvalue decomposition of B
        var (eigenvalues, eigenvectors) = ComputeEigen(B, n);

        // Sort by eigenvalue descending
        var indices = Enumerable.Range(0, n)
            .OrderByDescending(i => eigenvalues[i])
            .ToArray();

        // Take top k components
        int k = Math.Min(_nComponents, n);
        var embedding = new double[n, k];

        for (int d = 0; d < k; d++)
        {
            int idx = indices[d];
            double scale = Math.Sqrt(Math.Max(eigenvalues[idx], 0));

            for (int i = 0; i < n; i++)
            {
                embedding[i, d] = eigenvectors[idx, i] * scale;
            }
        }

        return embedding;
    }

    private (double[] Eigenvalues, double[,] Eigenvectors) ComputeEigen(double[,] matrix, int n)
    {
        var eigenvalues = new double[n];
        var eigenvectors = new double[n, n];
        var A = (double[,])matrix.Clone();

        for (int k = 0; k < n; k++)
        {
            var v = new double[n];
            for (int i = 0; i < n; i++)
            {
                v[i] = 1.0 / Math.Sqrt(n);
            }

            for (int iter = 0; iter < 100; iter++)
            {
                var Av = new double[n];
                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        Av[i] += A[i, j] * v[j];
                    }
                }

                double norm = 0;
                for (int i = 0; i < n; i++)
                {
                    norm += Av[i] * Av[i];
                }
                norm = Math.Sqrt(norm);

                if (norm < 1e-10) break;

                for (int i = 0; i < n; i++)
                {
                    v[i] = Av[i] / norm;
                }
            }

            var Av2 = new double[n];
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    Av2[i] += A[i, j] * v[j];
                }
            }

            double eigenvalue = 0;
            for (int i = 0; i < n; i++)
            {
                eigenvalue += v[i] * Av2[i];
            }

            eigenvalues[k] = eigenvalue;
            for (int i = 0; i < n; i++)
            {
                eigenvectors[k, i] = v[i];
            }

            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    A[i, j] -= eigenvalue * v[i] * v[j];
                }
            }
        }

        return (eigenvalues, eigenvectors);
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

        int n = _embedding.GetLength(0);
        int k = _embedding.GetLength(1);
        var result = new T[n, k];

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < k; j++)
            {
                result[i, j] = NumOps.FromDouble(_embedding[i, j]);
            }
        }

        return new Matrix<T>(result);
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
