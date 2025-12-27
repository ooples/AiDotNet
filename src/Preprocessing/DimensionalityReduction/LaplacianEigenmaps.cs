using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DimensionalityReduction;

/// <summary>
/// Laplacian Eigenmaps for nonlinear dimensionality reduction.
/// </summary>
/// <remarks>
/// <para>
/// Laplacian Eigenmaps constructs a weighted graph from the data and finds a low-dimensional
/// embedding that preserves local neighborhood relationships by minimizing a cost function
/// based on the graph Laplacian.
/// </para>
/// <para>
/// The algorithm:
/// 1. Constructs a k-nearest neighbor or epsilon-neighborhood graph
/// 2. Computes edge weights using a kernel (e.g., heat kernel)
/// 3. Computes the graph Laplacian: L = D - W
/// 4. Finds eigenvectors of the generalized eigenvalue problem: L*y = λ*D*y
/// </para>
/// <para><b>For Beginners:</b> Laplacian Eigenmaps finds a low-dimensional representation where:
/// - Connected points in the graph stay close together
/// - The embedding respects the local geometry of the data
/// - It's similar to spectral clustering but for dimensionality reduction
///
/// Use cases:
/// - Manifold learning when data lies on a curved surface
/// - Image segmentation and clustering
/// - When you want to preserve local connectivity
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class LaplacianEigenmaps<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nComponents;
    private readonly int _nNeighbors;
    private readonly double? _radius;
    private readonly LaplacianAffinityType _affinity;
    private readonly double _gamma;
    private readonly int? _randomState;

    // Fitted parameters
    private double[,]? _embedding;
    private int _nSamples;

    /// <summary>
    /// Gets the number of components (dimensions).
    /// </summary>
    public int NComponents => _nComponents;

    /// <summary>
    /// Gets the number of neighbors.
    /// </summary>
    public int NNeighbors => _nNeighbors;

    /// <summary>
    /// Gets the embedding result.
    /// </summary>
    public double[,]? Embedding => _embedding;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="LaplacianEigenmaps{T}"/>.
    /// </summary>
    /// <param name="nComponents">Target dimensionality. Defaults to 2.</param>
    /// <param name="nNeighbors">Number of neighbors for graph construction. Defaults to 5.</param>
    /// <param name="radius">Radius for epsilon-neighborhood (if null, uses k-NN). Defaults to null.</param>
    /// <param name="affinity">Affinity type for edge weights. Defaults to NearestNeighbors.</param>
    /// <param name="gamma">Kernel coefficient for RBF affinity. Defaults to 1.0.</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <param name="columnIndices">The column indices to use, or null for all columns.</param>
    public LaplacianEigenmaps(
        int nComponents = 2,
        int nNeighbors = 5,
        double? radius = null,
        LaplacianAffinityType affinity = LaplacianAffinityType.NearestNeighbors,
        double gamma = 1.0,
        int? randomState = null,
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
        _radius = radius;
        _affinity = affinity;
        _gamma = gamma;
        _randomState = randomState;
    }

    /// <summary>
    /// Fits Laplacian Eigenmaps and computes the embedding.
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        _nSamples = data.Rows;
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

        // Step 1: Compute affinity matrix (weighted adjacency)
        var W = ComputeAffinityMatrix(X, n, p);

        // Step 2: Compute degree matrix D
        var D = new double[n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                D[i] += W[i, j];
            }
        }

        // Step 3: Compute normalized Laplacian: L = D^(-1/2) * (D - W) * D^(-1/2)
        // Or solve generalized eigenvalue problem: (D - W) * y = λ * D * y
        var L = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (i == j)
                {
                    L[i, j] = 1.0; // D[i] - W[i,i] normalized
                }
                else if (W[i, j] > 0 && D[i] > 0 && D[j] > 0)
                {
                    L[i, j] = -W[i, j] / Math.Sqrt(D[i] * D[j]);
                }
            }
        }

        // Step 4: Compute eigenvectors of L (smallest eigenvalues)
        var (eigenvalues, eigenvectors) = ComputeSmallestEigenvectors(L, n);

        // Step 5: Select eigenvectors (skip first one which is constant)
        _embedding = new double[n, _nComponents];
        for (int k = 0; k < _nComponents; k++)
        {
            // Use eigenvector k+1 (skip the trivial constant eigenvector)
            int eigIdx = k + 1;
            if (eigIdx >= n) eigIdx = k;

            for (int i = 0; i < n; i++)
            {
                _embedding[i, k] = eigenvectors[eigIdx, i];
            }
        }
    }

    private double[,] ComputeAffinityMatrix(double[,] X, int n, int p)
    {
        var W = new double[n, n];

        // Compute pairwise distances
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

        if (_affinity == LaplacianAffinityType.RBF)
        {
            // Full RBF kernel
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    if (i != j)
                    {
                        W[i, j] = Math.Exp(-_gamma * distances[i, j] * distances[i, j]);
                    }
                }
            }
        }
        else
        {
            // k-NN based connectivity
            for (int i = 0; i < n; i++)
            {
                // Find k nearest neighbors
                var neighborDists = new (double dist, int idx)[n];
                for (int j = 0; j < n; j++)
                {
                    neighborDists[j] = (j == i ? double.MaxValue : distances[i, j], j);
                }
                Array.Sort(neighborDists, (a, b) => a.dist.CompareTo(b.dist));

                // Set connections to k nearest
                for (int k = 0; k < Math.Min(_nNeighbors, n - 1); k++)
                {
                    int j = neighborDists[k].idx;
                    double weight = _affinity == LaplacianAffinityType.NearestNeighbors
                        ? 1.0
                        : Math.Exp(-_gamma * distances[i, j] * distances[i, j]);
                    W[i, j] = weight;
                }
            }

            // Symmetrize
            for (int i = 0; i < n; i++)
            {
                for (int j = i + 1; j < n; j++)
                {
                    double sym = Math.Max(W[i, j], W[j, i]);
                    W[i, j] = sym;
                    W[j, i] = sym;
                }
            }
        }

        return W;
    }

    private (double[] Eigenvalues, double[,] Eigenvectors) ComputeSmallestEigenvectors(double[,] L, int n)
    {
        // Power iteration on (I - L) to get largest eigenvalues of (I - L),
        // which correspond to smallest eigenvalues of L
        var eigenvalues = new double[n];
        var eigenvectors = new double[n, n];

        // Create I - L (shift to get smallest eigenvalues as largest)
        var shiftedL = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                shiftedL[i, j] = (i == j ? 2.0 : 0.0) - L[i, j];
            }
        }

        var A = (double[,])shiftedL.Clone();

        for (int k = 0; k < Math.Min(n, _nComponents + 2); k++)
        {
            var v = new double[n];
            var random = _randomState.HasValue
                ? RandomHelper.CreateSeededRandom(_randomState.Value + k)
                : RandomHelper.CreateSeededRandom(42 + k);

            for (int i = 0; i < n; i++)
            {
                v[i] = random.NextDouble() - 0.5;
            }

            // Power iteration
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

            // Compute eigenvalue
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

            eigenvalues[k] = 2.0 - eigenvalue; // Convert back to L eigenvalue
            for (int i = 0; i < n; i++)
            {
                eigenvectors[k, i] = v[i];
            }

            // Deflate
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
            throw new InvalidOperationException("LaplacianEigenmaps has not been fitted.");
        }

        if (data.Rows != _nSamples)
        {
            throw new InvalidOperationException(
                "LaplacianEigenmaps does not support out-of-sample transformation. " +
                "Use FitTransform() on the complete dataset.");
        }

        int n = _embedding.GetLength(0);
        int d = _embedding.GetLength(1);
        var result = new T[n, d];

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < d; j++)
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
        throw new NotSupportedException("LaplacianEigenmaps does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        var names = new string[_nComponents];
        for (int i = 0; i < _nComponents; i++)
        {
            names[i] = $"LE{i + 1}";
        }
        return names;
    }
}
