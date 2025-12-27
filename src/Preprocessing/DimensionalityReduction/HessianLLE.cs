using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DimensionalityReduction;

/// <summary>
/// Hessian Locally Linear Embedding for nonlinear dimensionality reduction.
/// </summary>
/// <remarks>
/// <para>
/// Hessian LLE is an improvement over standard LLE that estimates the Hessian
/// (second derivative) of the embedding function. It uses a quadratic form to
/// measure local curvature and produces more globally coherent embeddings.
/// </para>
/// <para>
/// The algorithm:
/// 1. Find k-nearest neighbors for each point
/// 2. Compute local Hessian estimator using quadratic polynomials
/// 3. Build global Hessian matrix
/// 4. Find embedding by minimizing Hessian-based cost function
/// </para>
/// <para><b>For Beginners:</b> Hessian LLE improves on LLE by:
/// - Using curvature information (second derivatives)
/// - Producing more faithful global embeddings
/// - Better handling of manifolds with varying curvature
///
/// Use cases:
/// - When standard LLE produces distorted embeddings
/// - Manifolds with non-uniform curvature
/// - When you need more accurate distance preservation
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class HessianLLE<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
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
    /// Creates a new instance of <see cref="HessianLLE{T}"/>.
    /// </summary>
    /// <param name="nComponents">Target dimensionality. Defaults to 2.</param>
    /// <param name="nNeighbors">Number of neighbors. Should be > (n_components * (n_components + 3) / 2). Defaults to 10.</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <param name="columnIndices">The column indices to use, or null for all columns.</param>
    public HessianLLE(
        int nComponents = 2,
        int nNeighbors = 10,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nComponents < 1)
        {
            throw new ArgumentException("Number of components must be at least 1.", nameof(nComponents));
        }

        // Hessian LLE requires k >= d(d+3)/2 + 1
        int minNeighbors = nComponents * (nComponents + 3) / 2 + 1;
        if (nNeighbors < minNeighbors)
        {
            throw new ArgumentException(
                $"For {nComponents} components, at least {minNeighbors} neighbors are required.",
                nameof(nNeighbors));
        }

        _nComponents = nComponents;
        _nNeighbors = nNeighbors;
        _randomState = randomState;
    }

    /// <summary>
    /// Fits Hessian LLE and computes the embedding.
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        _nSamples = data.Rows;
        int n = data.Rows;
        int p = data.Columns;
        int k = Math.Min(_nNeighbors, n - 1);
        int d = _nComponents;

        // Number of Hessian estimator columns: d(d+1)/2
        int dp = d * (d + 1) / 2;

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

        // Step 2: Build Hessian estimator matrix H
        var H = new double[n, n];

        for (int i = 0; i < n; i++)
        {
            // Get neighborhood data
            var Xi = new double[k, p];
            var mean = new double[p];

            for (int j = 0; j < k; j++)
            {
                int idx = neighbors[i, j];
                for (int dd = 0; dd < p; dd++)
                {
                    Xi[j, dd] = X[idx, dd];
                    mean[dd] += Xi[j, dd];
                }
            }

            for (int dd = 0; dd < p; dd++) mean[dd] /= k;

            // Center the neighborhood
            for (int j = 0; j < k; j++)
            {
                for (int dd = 0; dd < p; dd++)
                {
                    Xi[j, dd] -= mean[dd];
                }
            }

            // PCA to get local coordinates
            var V = ComputeTopPrincipalComponents(Xi, k, p, d);

            // Project onto tangent space
            var Yi = new double[k, d];
            for (int j = 0; j < k; j++)
            {
                for (int dd = 0; dd < d; dd++)
                {
                    for (int pp = 0; pp < p; pp++)
                    {
                        Yi[j, dd] += Xi[j, pp] * V[pp, dd];
                    }
                }
            }

            // Build Hessian estimator matrix
            // Columns: 1, y1, y2, y1^2, y1*y2, y2^2 (for d=2)
            int nCols = 1 + d + dp;
            var G = new double[k, nCols];

            for (int j = 0; j < k; j++)
            {
                int col = 0;
                G[j, col++] = 1; // Constant

                for (int d1 = 0; d1 < d; d1++)
                {
                    G[j, col++] = Yi[j, d1]; // Linear terms
                }

                for (int d1 = 0; d1 < d; d1++)
                {
                    for (int d2 = d1; d2 < d; d2++)
                    {
                        G[j, col++] = Yi[j, d1] * Yi[j, d2]; // Quadratic terms
                    }
                }
            }

            // QR decomposition of G
            var (Q, _) = QRDecomposition(G, k, nCols);

            // Extract null space (last columns correspond to Hessian)
            var nullSpace = new double[k, dp];
            for (int j = 0; j < k; j++)
            {
                for (int dd = 0; dd < dp; dd++)
                {
                    int colIdx = nCols - dp + dd;
                    if (colIdx >= 0 && colIdx < k)
                    {
                        nullSpace[j, dd] = Q[j, colIdx];
                    }
                }
            }

            // Compute local Hessian piece: Yi = nullSpace * nullSpace^T
            var Wi = new double[k, k];
            for (int j1 = 0; j1 < k; j1++)
            {
                for (int j2 = 0; j2 < k; j2++)
                {
                    for (int dd = 0; dd < dp; dd++)
                    {
                        Wi[j1, j2] += nullSpace[j1, dd] * nullSpace[j2, dd];
                    }
                }
            }

            // Add to global matrix H
            for (int j1 = 0; j1 < k; j1++)
            {
                for (int j2 = 0; j2 < k; j2++)
                {
                    H[neighbors[i, j1], neighbors[i, j2]] += Wi[j1, j2];
                }
            }
        }

        // Step 3: Compute eigenvectors of H
        var (eigenvalues, eigenvectors) = ComputeSmallestEigenvectors(H, n);

        // Step 4: Create embedding
        _embedding = new double[n, _nComponents];
        for (int dd = 0; dd < _nComponents; dd++)
        {
            int eigIdx = dd + 1; // Skip null eigenvector
            if (eigIdx >= n) eigIdx = dd;

            for (int i = 0; i < n; i++)
            {
                _embedding[i, dd] = eigenvectors[eigIdx, i];
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

    private double[,] ComputeTopPrincipalComponents(double[,] Xi, int m, int p, int nDims)
    {
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

            var Av2 = new double[p];
            double lambda = 0;
            for (int i = 0; i < p; i++)
            {
                for (int j = 0; j < p; j++) Av2[i] += A[i, j] * v[j];
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

        return V;
    }

    private (double[,] Q, double[,] R) QRDecomposition(double[,] A, int m, int n)
    {
        var Q = new double[m, m];
        var R = new double[m, n];

        // Initialize Q as identity
        for (int i = 0; i < m; i++) Q[i, i] = 1;

        for (int j = 0; j < Math.Min(m, n); j++)
        {
            // Compute column j of Q
            var v = new double[m];
            for (int i = j; i < m; i++) v[i] = A[i, j];

            // Subtract projections
            for (int k = 0; k < j; k++)
            {
                double dot = 0;
                for (int i = 0; i < m; i++) dot += Q[i, k] * A[i, j];
                R[k, j] = dot;
            }

            for (int k = 0; k < j; k++)
            {
                for (int i = 0; i < m; i++)
                {
                    v[i] -= R[k, j] * Q[i, k];
                }
            }

            double norm = 0;
            for (int i = 0; i < m; i++) norm += v[i] * v[i];
            norm = Math.Sqrt(norm);

            R[j, j] = norm;
            if (norm > 1e-10)
            {
                for (int i = 0; i < m; i++) Q[i, j] = v[i] / norm;
            }
        }

        return (Q, R);
    }

    private (double[] Eigenvalues, double[,] Eigenvectors) ComputeSmallestEigenvectors(double[,] H, int n)
    {
        var eigenvalues = new double[n];
        var eigenvectors = new double[n, n];

        double shift = 0;
        for (int i = 0; i < n; i++) shift = Math.Max(shift, Math.Abs(H[i, i]));
        shift *= 2;

        var A = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                A[i, j] = (i == j ? shift : 0) - H[i, j];
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
                    for (int j = 0; j < n; j++) Av[i] += A[i, j] * v[j];
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
                for (int j = 0; j < n; j++) Av2[i] += A[i, j] * v[j];
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
            throw new InvalidOperationException("HessianLLE has not been fitted.");
        }

        if (data.Rows != _nSamples)
        {
            throw new InvalidOperationException(
                "HessianLLE does not support out-of-sample transformation. " +
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
        throw new NotSupportedException("HessianLLE does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        var names = new string[_nComponents];
        for (int i = 0; i < _nComponents; i++)
        {
            names[i] = $"HLLE{i + 1}";
        }
        return names;
    }
}
