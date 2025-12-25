using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DimensionalityReduction;

/// <summary>
/// Spectral Embedding for nonlinear dimensionality reduction.
/// </summary>
/// <remarks>
/// <para>
/// Spectral Embedding forms an affinity matrix from the data and computes
/// the eigenvectors of the graph Laplacian. This provides a low-dimensional
/// representation that preserves local connectivity.
/// </para>
/// <para>
/// The algorithm constructs a similarity graph and uses spectral decomposition
/// of the Laplacian matrix to find coordinates that respect graph structure.
/// </para>
/// <para><b>For Beginners:</b> Spectral Embedding uses graph theory:
/// - Build a graph where similar points are connected
/// - Use the graph's structure to find good coordinates
/// - Similar to what's used in spectral clustering
/// - Good for data with cluster structure
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class SpectralEmbedding<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nComponents;
    private readonly SpectralAffinity _affinity;
    private readonly double _gamma;
    private readonly int _nNeighbors;
    private readonly int? _randomState;

    // Fitted parameters
    private double[,]? _embedding;
    private double[,]? _affinityMatrix;
    private int _nSamples;
    private int _nFeaturesIn;

    /// <summary>
    /// Gets the number of components.
    /// </summary>
    public int NComponents => _nComponents;

    /// <summary>
    /// Gets the affinity type.
    /// </summary>
    public SpectralAffinity Affinity => _affinity;

    /// <summary>
    /// Gets the gamma parameter for RBF kernel.
    /// </summary>
    public double Gamma => _gamma;

    /// <summary>
    /// Gets the number of neighbors for nearest neighbors affinity.
    /// </summary>
    public int NNeighbors => _nNeighbors;

    /// <summary>
    /// Gets the affinity matrix.
    /// </summary>
    public double[,]? AffinityMatrix => _affinityMatrix;

    /// <summary>
    /// Gets the embedding result.
    /// </summary>
    public double[,]? Embedding => _embedding;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="SpectralEmbedding{T}"/>.
    /// </summary>
    /// <param name="nComponents">Target dimensionality. Defaults to 2.</param>
    /// <param name="affinity">How to construct the affinity matrix. Defaults to NearestNeighbors.</param>
    /// <param name="gamma">Kernel coefficient for RBF. If null, uses 1/n_features.</param>
    /// <param name="nNeighbors">Number of neighbors for NN affinity. Defaults to 10.</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <param name="columnIndices">The column indices to use, or null for all columns.</param>
    public SpectralEmbedding(
        int nComponents = 2,
        SpectralAffinity affinity = SpectralAffinity.NearestNeighbors,
        double? gamma = null,
        int nNeighbors = 10,
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
        _affinity = affinity;
        _gamma = gamma ?? 0; // Will be computed during fit
        _nNeighbors = nNeighbors;
        _randomState = randomState;
    }

    /// <summary>
    /// Fits Spectral Embedding by computing the graph Laplacian eigenvectors.
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

        // Compute gamma if not provided
        double gamma = _gamma > 0 ? _gamma : 1.0 / p;

        // Step 1: Compute affinity matrix
        _affinityMatrix = ComputeAffinityMatrix(X, n, p, gamma);

        // Step 2: Compute normalized graph Laplacian
        var (L, D) = ComputeNormalizedLaplacian(_affinityMatrix, n);

        // Step 3: Find smallest eigenvectors of Laplacian
        _embedding = ComputeEmbedding(L, n);
    }

    private double[,] ComputeAffinityMatrix(double[,] X, int n, int p, double gamma)
    {
        var affinity = new double[n, n];

        switch (_affinity)
        {
            case SpectralAffinity.RBF:
                // Radial Basis Function (Gaussian) kernel
                for (int i = 0; i < n; i++)
                {
                    for (int j = i; j < n; j++)
                    {
                        if (i == j)
                        {
                            affinity[i, j] = 1.0;
                        }
                        else
                        {
                            double dist = 0;
                            for (int k = 0; k < p; k++)
                            {
                                double diff = X[i, k] - X[j, k];
                                dist += diff * diff;
                            }
                            double sim = Math.Exp(-gamma * dist);
                            affinity[i, j] = sim;
                            affinity[j, i] = sim;
                        }
                    }
                }
                break;

            case SpectralAffinity.NearestNeighbors:
                // First compute distances
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
                        distances[i, j] = Math.Sqrt(dist);
                        distances[j, i] = distances[i, j];
                    }
                }

                // Build symmetric KNN graph
                for (int i = 0; i < n; i++)
                {
                    // Find k nearest neighbors
                    var neighbors = Enumerable.Range(0, n)
                        .Where(j => j != i)
                        .OrderBy(j => distances[i, j])
                        .Take(_nNeighbors)
                        .ToList();

                    foreach (int j in neighbors)
                    {
                        // Use distance-based weight
                        double weight = Math.Exp(-distances[i, j] * distances[i, j] * gamma);
                        affinity[i, j] = Math.Max(affinity[i, j], weight);
                        affinity[j, i] = affinity[i, j]; // Symmetrize
                    }
                }

                // Set diagonal to 1
                for (int i = 0; i < n; i++)
                {
                    affinity[i, i] = 1.0;
                }
                break;

            case SpectralAffinity.Precomputed:
                throw new InvalidOperationException(
                    "Precomputed affinity requires passing the affinity matrix directly.");
        }

        return affinity;
    }

    private (double[,] L, double[] D) ComputeNormalizedLaplacian(double[,] W, int n)
    {
        // Compute degree matrix D
        var D = new double[n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                D[i] += W[i, j];
            }
        }

        // Compute D^(-1/2)
        var Dinv = new double[n];
        for (int i = 0; i < n; i++)
        {
            Dinv[i] = D[i] > 1e-10 ? 1.0 / Math.Sqrt(D[i]) : 0;
        }

        // Compute normalized Laplacian: L = I - D^(-1/2) * W * D^(-1/2)
        var L = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double normalized = Dinv[i] * W[i, j] * Dinv[j];
                L[i, j] = (i == j ? 1.0 : 0.0) - normalized;
            }
        }

        return (L, D);
    }

    private double[,] ComputeEmbedding(double[,] L, int n)
    {
        // Find smallest eigenvectors of Laplacian
        var (eigenvalues, eigenvectors) = ComputeEigen(L, n);

        // Sort by eigenvalue ascending
        var indices = Enumerable.Range(0, n)
            .OrderBy(i => eigenvalues[i])
            .ToArray();

        // Take eigenvectors 1 to nComponents (skip the first which is constant)
        int nComp = Math.Min(_nComponents, n - 1);
        var embedding = new double[n, nComp];

        for (int d = 0; d < nComp; d++)
        {
            // Skip the first (zero) eigenvalue
            int idx = indices[d + 1];

            for (int i = 0; i < n; i++)
            {
                embedding[i, d] = eigenvectors[idx, i];
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
            var random = _randomState.HasValue
                ? RandomHelper.CreateSeededRandom(_randomState.Value + k)
                : RandomHelper.CreateSeededRandom(42 + k);

            // Random initialization for better convergence
            for (int i = 0; i < n; i++)
            {
                v[i] = random.NextDouble() - 0.5;
            }

            // Normalize
            double initNorm = 0;
            for (int i = 0; i < n; i++)
            {
                initNorm += v[i] * v[i];
            }
            initNorm = Math.Sqrt(initNorm);
            for (int i = 0; i < n; i++)
            {
                v[i] /= initNorm;
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
            throw new InvalidOperationException("SpectralEmbedding has not been fitted.");
        }

        if (data.Rows != _nSamples)
        {
            throw new InvalidOperationException(
                "SpectralEmbedding does not support out-of-sample transformation. " +
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
        throw new NotSupportedException("SpectralEmbedding does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        var names = new string[_nComponents];
        for (int i = 0; i < _nComponents; i++)
        {
            names[i] = $"Spectral{i + 1}";
        }
        return names;
    }
}

/// <summary>
/// Specifies how to construct the affinity matrix for Spectral Embedding.
/// </summary>
public enum SpectralAffinity
{
    /// <summary>
    /// K-nearest neighbors graph.
    /// </summary>
    NearestNeighbors,

    /// <summary>
    /// Radial Basis Function (Gaussian) kernel.
    /// </summary>
    RBF,

    /// <summary>
    /// Use a precomputed affinity matrix.
    /// </summary>
    Precomputed
}
