using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DimensionalityReduction;

/// <summary>
/// Modified Locally Linear Embedding with regularization.
/// </summary>
/// <remarks>
/// <para>
/// Modified LLE adds regularization to the standard LLE algorithm to improve
/// numerical stability when the number of neighbors exceeds the input dimensionality.
/// </para>
/// <para>
/// The algorithm:
/// 1. Find k-nearest neighbors for each point
/// 2. Compute reconstruction weights with regularization
/// 3. Compute embedding by minimizing reconstruction error
/// </para>
/// <para><b>For Beginners:</b> Modified LLE is more stable than standard LLE:
/// - Works better when you have many neighbors
/// - Less sensitive to noise
/// - Produces more consistent results
///
/// Use cases:
/// - When standard LLE is unstable
/// - High number of neighbors relative to dimensions
/// - Noisy data
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class ModifiedLLE<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nComponents;
    private readonly int _nNeighbors;
    private readonly double _regParam;
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
    /// Creates a new instance of <see cref="ModifiedLLE{T}"/>.
    /// </summary>
    /// <param name="nComponents">Target dimensionality. Defaults to 2.</param>
    /// <param name="nNeighbors">Number of neighbors. Defaults to 5.</param>
    /// <param name="regParam">Regularization parameter. Defaults to 1e-3.</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <param name="columnIndices">The column indices to use, or null for all columns.</param>
    public ModifiedLLE(
        int nComponents = 2,
        int nNeighbors = 5,
        double regParam = 1e-3,
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

        if (regParam < 0)
        {
            throw new ArgumentException("Regularization parameter must be non-negative.", nameof(regParam));
        }

        _nComponents = nComponents;
        _nNeighbors = nNeighbors;
        _regParam = regParam;
        _randomState = randomState;
    }

    /// <summary>
    /// Fits Modified LLE and computes the embedding.
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

        // Step 2: Compute reconstruction weights with regularization
        var W = new double[n, n];

        for (int i = 0; i < n; i++)
        {
            // Compute local covariance matrix
            var C = new double[k, k];
            for (int j1 = 0; j1 < k; j1++)
            {
                for (int j2 = 0; j2 < k; j2++)
                {
                    for (int d = 0; d < p; d++)
                    {
                        double diff1 = X[neighbors[i, j1], d] - X[i, d];
                        double diff2 = X[neighbors[i, j2], d] - X[i, d];
                        C[j1, j2] += diff1 * diff2;
                    }
                }
            }

            // Add regularization
            double trace = 0;
            for (int j = 0; j < k; j++) trace += C[j, j];
            double reg = _regParam * trace / k;
            if (reg < 1e-10) reg = 1e-3;

            for (int j = 0; j < k; j++)
            {
                C[j, j] += reg;
            }

            // Solve C * w = 1
            var w = SolveLinearSystem(C, k);

            // Normalize weights
            double sum = 0;
            for (int j = 0; j < k; j++) sum += w[j];
            if (Math.Abs(sum) > 1e-10)
            {
                for (int j = 0; j < k; j++) w[j] /= sum;
            }

            // Store weights
            for (int j = 0; j < k; j++)
            {
                W[i, neighbors[i, j]] = w[j];
            }
        }

        // Step 3: Compute M = (I - W)^T * (I - W)
        var M = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double sum = 0;
                for (int m = 0; m < n; m++)
                {
                    double Imi = (m == i ? 1 : 0) - W[m, i];
                    double Imj = (m == j ? 1 : 0) - W[m, j];
                    sum += Imi * Imj;
                }
                M[i, j] = sum;
            }
        }

        // Step 4: Compute eigenvectors (smallest eigenvalues)
        var (eigenvalues, eigenvectors) = ComputeSmallestEigenvectors(M, n);

        // Step 5: Create embedding
        _embedding = new double[n, _nComponents];
        for (int d = 0; d < _nComponents; d++)
        {
            int eigIdx = d + 1; // Skip null eigenvector
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

    private double[] SolveLinearSystem(double[,] A, int n)
    {
        // Solve A * x = 1 using Cholesky-like decomposition
        var result = new double[n];
        var b = new double[n];
        for (int i = 0; i < n; i++) b[i] = 1;

        // Simple Gauss-Seidel iteration
        for (int iter = 0; iter < 100; iter++)
        {
            for (int i = 0; i < n; i++)
            {
                double sum = b[i];
                for (int j = 0; j < n; j++)
                {
                    if (j != i)
                    {
                        sum -= A[i, j] * result[j];
                    }
                }
                if (Math.Abs(A[i, i]) > 1e-10)
                {
                    result[i] = sum / A[i, i];
                }
            }
        }

        return result;
    }

    private (double[] Eigenvalues, double[,] Eigenvectors) ComputeSmallestEigenvectors(double[,] M, int n)
    {
        var eigenvalues = new double[n];
        var eigenvectors = new double[n, n];

        double shift = 0;
        for (int i = 0; i < n; i++) shift = Math.Max(shift, M[i, i]);
        shift *= 2;

        var A = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                A[i, j] = (i == j ? shift : 0) - M[i, j];
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
            throw new InvalidOperationException("ModifiedLLE has not been fitted.");
        }

        if (data.Rows != _nSamples)
        {
            throw new InvalidOperationException(
                "ModifiedLLE does not support out-of-sample transformation. " +
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
        throw new NotSupportedException("ModifiedLLE does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        var names = new string[_nComponents];
        for (int i = 0; i < _nComponents; i++)
        {
            names[i] = $"MLLE{i + 1}";
        }
        return names;
    }
}
