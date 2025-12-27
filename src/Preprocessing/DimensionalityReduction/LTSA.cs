using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DimensionalityReduction;

/// <summary>
/// Local Tangent Space Alignment for nonlinear dimensionality reduction.
/// </summary>
/// <remarks>
/// <para>
/// LTSA is a manifold learning algorithm that uses local tangent spaces to compute
/// a global embedding. It estimates the tangent space at each point using PCA on
/// neighbors, then aligns these local coordinate systems.
/// </para>
/// <para>
/// The algorithm:
/// 1. Find k-nearest neighbors for each point
/// 2. Compute local tangent space via PCA on each neighborhood
/// 3. Compute local coordinates in tangent space
/// 4. Align tangent spaces by minimizing reconstruction error
/// </para>
/// <para><b>For Beginners:</b> LTSA is an improved version of LLE that:
/// - Uses tangent spaces (local linear approximations) at each point
/// - Better handles points with different local geometries
/// - More mathematically principled than standard LLE
///
/// Use cases:
/// - When standard LLE produces poor results
/// - Manifolds with varying curvature
/// - When you need more stable embeddings
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class LTSA<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nComponents;
    private readonly int _nNeighbors;
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
    /// Creates a new instance of <see cref="LTSA{T}"/>.
    /// </summary>
    /// <param name="nComponents">Target dimensionality. Defaults to 2.</param>
    /// <param name="nNeighbors">Number of neighbors. Defaults to 5.</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <param name="columnIndices">The column indices to use, or null for all columns.</param>
    public LTSA(
        int nComponents = 2,
        int nNeighbors = 5,
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
        _randomState = randomState;
    }

    /// <summary>
    /// Fits LTSA and computes the embedding.
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        _nSamples = data.Rows;
        int n = data.Rows;
        int p = data.Columns;
        int k = Math.Min(_nNeighbors, n - 1);

        // Convert to double array
        var X = new double[n, p];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < p; j++)
            {
                X[i, j] = NumOps.ToDouble(data[i, j]);
            }
        }

        // Step 1: Find k-nearest neighbors
        var neighbors = FindKNearestNeighbors(X, n, p, k);

        // Step 2: Compute alignment matrix B
        var B = new double[n, n];

        for (int i = 0; i < n; i++)
        {
            // Get neighborhood (including point i)
            var neighborIdx = new int[k + 1];
            neighborIdx[0] = i;
            for (int j = 0; j < k; j++)
            {
                neighborIdx[j + 1] = neighbors[i, j];
            }

            // Extract neighborhood data and center
            var Xi = new double[k + 1, p];
            var mean = new double[p];

            for (int j = 0; j <= k; j++)
            {
                for (int d = 0; d < p; d++)
                {
                    Xi[j, d] = X[neighborIdx[j], d];
                    mean[d] += Xi[j, d];
                }
            }

            for (int d = 0; d < p; d++)
            {
                mean[d] /= (k + 1);
            }

            for (int j = 0; j <= k; j++)
            {
                for (int d = 0; d < p; d++)
                {
                    Xi[j, d] -= mean[d];
                }
            }

            // Compute local tangent space via SVD/PCA
            // Get top _nComponents principal directions
            var (_, V) = ComputeTangentSpace(Xi, k + 1, p, _nComponents);

            // Compute local coordinates in tangent space
            var theta = new double[k + 1, _nComponents];
            for (int j = 0; j <= k; j++)
            {
                for (int d = 0; d < _nComponents; d++)
                {
                    for (int dd = 0; dd < p; dd++)
                    {
                        theta[j, d] += Xi[j, dd] * V[dd, d];
                    }
                }
            }

            // Compute Gi = [1, theta] and Wi = I - Gi * pinv(Gi)
            var Gi = new double[k + 1, _nComponents + 1];
            for (int j = 0; j <= k; j++)
            {
                Gi[j, 0] = 1.0 / Math.Sqrt(k + 1); // Normalized ones
                for (int d = 0; d < _nComponents; d++)
                {
                    Gi[j, d + 1] = theta[j, d];
                }
            }

            // Gram-Schmidt orthogonalize columns of Gi
            var Q = GramSchmidt(Gi, k + 1, _nComponents + 1);

            // Wi = I - Q * Q^T
            var Wi = new double[k + 1, k + 1];
            for (int j1 = 0; j1 <= k; j1++)
            {
                for (int j2 = 0; j2 <= k; j2++)
                {
                    Wi[j1, j2] = (j1 == j2 ? 1.0 : 0.0);
                    for (int d = 0; d <= _nComponents; d++)
                    {
                        Wi[j1, j2] -= Q[j1, d] * Q[j2, d];
                    }
                }
            }

            // Add to B: B[I, I] += Wi
            for (int j1 = 0; j1 <= k; j1++)
            {
                for (int j2 = 0; j2 <= k; j2++)
                {
                    B[neighborIdx[j1], neighborIdx[j2]] += Wi[j1, j2];
                }
            }
        }

        // Step 3: Compute eigenvectors of B (smallest eigenvalues)
        var (eigenvalues, eigenvectors) = ComputeSmallestEigenvectors(B, n);

        // Step 4: Create embedding (skip first eigenvector)
        _embedding = new double[n, _nComponents];
        for (int d = 0; d < _nComponents; d++)
        {
            int eigIdx = d + 1;
            if (eigIdx >= n) eigIdx = d;

            for (int i = 0; i < n; i++)
            {
                _embedding[i, d] = eigenvectors[eigIdx, i];
            }
        }
    }

    private int[,] FindKNearestNeighbors(double[,] X, int n, int p, int k)
    {
        var neighbors = new int[n, k];

        for (int i = 0; i < n; i++)
        {
            var distances = new (double dist, int idx)[n];
            for (int j = 0; j < n; j++)
            {
                if (i == j)
                {
                    distances[j] = (double.MaxValue, j);
                    continue;
                }

                double dist = 0;
                for (int d = 0; d < p; d++)
                {
                    double diff = X[i, d] - X[j, d];
                    dist += diff * diff;
                }
                distances[j] = (dist, j);
            }

            Array.Sort(distances, (a, b) => a.dist.CompareTo(b.dist));

            for (int j = 0; j < k; j++)
            {
                neighbors[i, j] = distances[j].idx;
            }
        }

        return neighbors;
    }

    private (double[,] U, double[,] V) ComputeTangentSpace(double[,] Xi, int m, int p, int nDims)
    {
        // Simple PCA to get principal directions
        // Compute X^T * X
        var XtX = new double[p, p];
        for (int i = 0; i < p; i++)
        {
            for (int j = 0; j < p; j++)
            {
                for (int k = 0; k < m; k++)
                {
                    XtX[i, j] += Xi[k, i] * Xi[k, j];
                }
            }
        }

        // Power iteration to get top eigenvectors
        var V = new double[p, nDims];
        var A = (double[,])XtX.Clone();

        for (int d = 0; d < nDims; d++)
        {
            var v = new double[p];
            for (int i = 0; i < p; i++) v[i] = 1.0 / Math.Sqrt(p);

            for (int iter = 0; iter < 50; iter++)
            {
                var Av = new double[p];
                for (int i = 0; i < p; i++)
                {
                    for (int j = 0; j < p; j++)
                    {
                        Av[i] += A[i, j] * v[j];
                    }
                }

                double norm = 0;
                for (int i = 0; i < p; i++) norm += Av[i] * Av[i];
                norm = Math.Sqrt(norm);
                if (norm < 1e-10) break;

                for (int i = 0; i < p; i++) v[i] = Av[i] / norm;
            }

            for (int i = 0; i < p; i++) V[i, d] = v[i];

            // Deflate
            var Av2 = new double[p];
            double lambda = 0;
            for (int i = 0; i < p; i++)
            {
                for (int j = 0; j < p; j++)
                {
                    Av2[i] += A[i, j] * v[j];
                }
                lambda += v[i] * Av2[i];
            }

            for (int i = 0; i < p; i++)
            {
                for (int j = 0; j < p; j++)
                {
                    A[i, j] -= lambda * v[i] * v[j];
                }
            }
        }

        return (new double[m, nDims], V);
    }

    private double[,] GramSchmidt(double[,] A, int m, int n)
    {
        var Q = new double[m, n];

        for (int j = 0; j < n; j++)
        {
            // Copy column j
            for (int i = 0; i < m; i++)
            {
                Q[i, j] = A[i, j];
            }

            // Subtract projections onto previous columns
            for (int k = 0; k < j; k++)
            {
                double dot = 0;
                for (int i = 0; i < m; i++)
                {
                    dot += Q[i, k] * A[i, j];
                }

                for (int i = 0; i < m; i++)
                {
                    Q[i, j] -= dot * Q[i, k];
                }
            }

            // Normalize
            double norm = 0;
            for (int i = 0; i < m; i++)
            {
                norm += Q[i, j] * Q[i, j];
            }
            norm = Math.Sqrt(norm);

            if (norm > 1e-10)
            {
                for (int i = 0; i < m; i++)
                {
                    Q[i, j] /= norm;
                }
            }
        }

        return Q;
    }

    private (double[] Eigenvalues, double[,] Eigenvectors) ComputeSmallestEigenvectors(double[,] B, int n)
    {
        // Shift and invert to get smallest eigenvalues
        var eigenvalues = new double[n];
        var eigenvectors = new double[n, n];

        // Shift: A = max_eigenvalue * I - B
        double shift = 0;
        for (int i = 0; i < n; i++) shift = Math.Max(shift, B[i, i]);
        shift *= 2;

        var A = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                A[i, j] = (i == j ? shift : 0) - B[i, j];
            }
        }

        for (int k = 0; k < Math.Min(n, _nComponents + 2); k++)
        {
            var v = new double[n];
            var random = _randomState.HasValue
                ? RandomHelper.CreateSeededRandom(_randomState.Value + k)
                : RandomHelper.CreateSeededRandom(42 + k);

            for (int i = 0; i < n; i++) v[i] = random.NextDouble() - 0.5;

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
                for (int i = 0; i < n; i++) norm += Av[i] * Av[i];
                norm = Math.Sqrt(norm);
                if (norm < 1e-10) break;

                for (int i = 0; i < n; i++) v[i] = Av[i] / norm;
            }

            var Av2 = new double[n];
            double lambda = 0;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    Av2[i] += A[i, j] * v[j];
                }
                lambda += v[i] * Av2[i];
            }

            eigenvalues[k] = shift - lambda;
            for (int i = 0; i < n; i++) eigenvectors[k, i] = v[i];

            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    A[i, j] -= lambda * v[i] * v[j];
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
            throw new InvalidOperationException("LTSA has not been fitted.");
        }

        if (data.Rows != _nSamples)
        {
            throw new InvalidOperationException(
                "LTSA does not support out-of-sample transformation. " +
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
        throw new NotSupportedException("LTSA does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        var names = new string[_nComponents];
        for (int i = 0; i < _nComponents; i++)
        {
            names[i] = $"LTSA{i + 1}";
        }
        return names;
    }
}
