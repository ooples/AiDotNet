using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DimensionalityReduction;

/// <summary>
/// Kernel Principal Component Analysis for non-linear dimensionality reduction.
/// </summary>
/// <remarks>
/// <para>
/// Kernel PCA is an extension of PCA that uses a kernel function to map
/// data into a higher-dimensional feature space where non-linear relationships
/// become linear, then performs standard PCA in that space.
/// </para>
/// <para>
/// This allows capturing non-linear relationships that standard PCA cannot.
/// Common kernels include RBF (Gaussian), polynomial, and sigmoid.
/// </para>
/// <para><b>For Beginners:</b> Regular PCA finds straight-line patterns.
/// Kernel PCA can find curved patterns by mathematically "bending" the data:
/// - RBF kernel: Good for data with clusters or blobs
/// - Polynomial: Good for polynomial relationships
/// - Linear: Same as regular PCA
///
/// Think of it as finding principal components in a transformed space.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class KernelPCA<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nComponents;
    private readonly KernelType _kernel;
    private readonly double _gamma;
    private readonly double _degree;
    private readonly double _coef0;
    private readonly bool _fitInverseTransform;
    private readonly double _alpha;

    // Fitted parameters
    private double[,]? _trainingData;
    private double[,]? _alphas; // Eigenvectors in kernel space
    private double[]? _lambdas; // Eigenvalues
    private double[,]? _inverseTransformMatrix;
    private int _nFeaturesIn;

    /// <summary>
    /// Gets the number of components.
    /// </summary>
    public int NComponents => _nComponents;

    /// <summary>
    /// Gets the kernel type.
    /// </summary>
    public KernelType Kernel => _kernel;

    /// <summary>
    /// Gets the gamma parameter for RBF and polynomial kernels.
    /// </summary>
    public double Gamma => _gamma;

    /// <summary>
    /// Gets the eigenvalues.
    /// </summary>
    public double[]? Lambdas => _lambdas;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => _fitInverseTransform;

    /// <summary>
    /// Creates a new instance of <see cref="KernelPCA{T}"/>.
    /// </summary>
    /// <param name="nComponents">Number of components to keep. Defaults to 2.</param>
    /// <param name="kernel">The kernel function to use. Defaults to RBF.</param>
    /// <param name="gamma">Kernel coefficient for RBF/poly/sigmoid. Defaults to 1.0.</param>
    /// <param name="degree">Polynomial degree. Defaults to 3.</param>
    /// <param name="coef0">Independent term in polynomial/sigmoid. Defaults to 1.0.</param>
    /// <param name="fitInverseTransform">Whether to learn inverse transformation. Defaults to false.</param>
    /// <param name="alpha">Regularization for inverse transform. Defaults to 1e-3.</param>
    /// <param name="columnIndices">The column indices to use, or null for all columns.</param>
    public KernelPCA(
        int nComponents = 2,
        KernelType kernel = KernelType.RBF,
        double gamma = 1.0,
        double degree = 3.0,
        double coef0 = 1.0,
        bool fitInverseTransform = false,
        double alpha = 1e-3,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nComponents < 1)
        {
            throw new ArgumentException("Number of components must be at least 1.", nameof(nComponents));
        }

        _nComponents = nComponents;
        _kernel = kernel;
        _gamma = gamma;
        _degree = degree;
        _coef0 = coef0;
        _fitInverseTransform = fitInverseTransform;
        _alpha = alpha;
    }

    /// <summary>
    /// Fits KernelPCA by computing the kernel matrix and its eigenvectors.
    /// </summary>
    /// <param name="data">The training data matrix.</param>
    protected override void FitCore(Matrix<T> data)
    {
        _nFeaturesIn = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        int k = Math.Min(_nComponents, n);

        // Store training data for transformation
        _trainingData = new double[n, p];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < p; j++)
            {
                _trainingData[i, j] = NumOps.ToDouble(data[i, j]);
            }
        }

        // Compute kernel matrix
        var K = ComputeKernelMatrix(_trainingData, _trainingData);

        // Center the kernel matrix
        CenterKernelMatrix(K, n);

        // Compute eigenvectors of centered kernel matrix
        var (eigenvalues, eigenvectors) = ComputeEigen(K, n, k);

        // Sort by eigenvalue descending
        var indices = Enumerable.Range(0, k)
            .OrderByDescending(i => eigenvalues[i])
            .ToArray();

        _lambdas = new double[k];
        _alphas = new double[n, k];

        for (int i = 0; i < k; i++)
        {
            _lambdas[i] = Math.Max(0, eigenvalues[indices[i]]);

            // Scale eigenvectors by 1/sqrt(lambda)
            double scale = _lambdas[i] > 1e-10 ? 1.0 / Math.Sqrt(_lambdas[i]) : 0;
            for (int j = 0; j < n; j++)
            {
                _alphas[j, i] = eigenvectors[indices[i], j] * scale;
            }
        }

        // Fit inverse transformation if requested
        if (_fitInverseTransform)
        {
            FitInverseTransform(data);
        }
    }

    private double[,] ComputeKernelMatrix(double[,] X1, double[,] X2)
    {
        int n1 = X1.GetLength(0);
        int n2 = X2.GetLength(0);
        int p = X1.GetLength(1);
        var K = new double[n1, n2];

        for (int i = 0; i < n1; i++)
        {
            for (int j = 0; j < n2; j++)
            {
                K[i, j] = ComputeKernel(X1, i, X2, j, p);
            }
        }

        return K;
    }

    private double ComputeKernel(double[,] X1, int i, double[,] X2, int j, int p)
    {
        switch (_kernel)
        {
            case KernelType.Linear:
            {
                double dot = 0;
                for (int k = 0; k < p; k++)
                {
                    dot += X1[i, k] * X2[j, k];
                }
                return dot;
            }

            case KernelType.RBF:
            {
                double sqDist = 0;
                for (int k = 0; k < p; k++)
                {
                    double diff = X1[i, k] - X2[j, k];
                    sqDist += diff * diff;
                }
                return Math.Exp(-_gamma * sqDist);
            }

            case KernelType.Polynomial:
            {
                double dot = 0;
                for (int k = 0; k < p; k++)
                {
                    dot += X1[i, k] * X2[j, k];
                }
                return Math.Pow(_gamma * dot + _coef0, _degree);
            }

            case KernelType.Sigmoid:
            {
                double dot = 0;
                for (int k = 0; k < p; k++)
                {
                    dot += X1[i, k] * X2[j, k];
                }
                return Math.Tanh(_gamma * dot + _coef0);
            }

            default:
                throw new ArgumentException($"Unknown kernel type: {_kernel}");
        }
    }

    private void CenterKernelMatrix(double[,] K, int n)
    {
        // K_c = K - 1_n K - K 1_n + 1_n K 1_n
        // where 1_n is the n×n matrix of 1/n

        // Compute row means
        var rowMeans = new double[n];
        double totalMean = 0;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                rowMeans[i] += K[i, j];
            }
            rowMeans[i] /= n;
            totalMean += rowMeans[i];
        }
        totalMean /= n;

        // Compute column means (symmetric, so same as row means)
        var colMeans = rowMeans;

        // Center
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                K[i, j] = K[i, j] - rowMeans[i] - colMeans[j] + totalMean;
            }
        }
    }

    private (double[] Eigenvalues, double[,] Eigenvectors) ComputeEigen(double[,] matrix, int n, int k)
    {
        // Power iteration with deflation
        var eigenvalues = new double[k];
        var eigenvectors = new double[k, n];
        var A = (double[,])matrix.Clone();
        var random = RandomHelper.CreateSeededRandom(42);

        for (int m = 0; m < k; m++)
        {
            var v = new double[n];
            double norm = 0;
            for (int i = 0; i < n; i++)
            {
                v[i] = random.NextDouble() - 0.5;
                norm += v[i] * v[i];
            }
            norm = Math.Sqrt(norm);
            for (int i = 0; i < n; i++)
            {
                v[i] /= norm;
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

                norm = 0;
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

            eigenvalues[m] = Math.Max(0, eigenvalue);

            for (int i = 0; i < n; i++)
            {
                eigenvectors[m, i] = v[i];
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

    private void FitInverseTransform(Matrix<T> data)
    {
        // Approximate inverse using ridge regression
        // X ≈ A @ X_transformed + b
        // where X_transformed = Transform(X)
        if (_trainingData is null || _alphas is null)
        {
            return;
        }

        int n = data.Rows;
        int p = data.Columns;
        int k = _alphas.GetLength(1);

        // Compute transformed data
        var transformed = TransformInternal(_trainingData);

        // Solve (A^T A + alpha I) W = A^T X
        var ATA = new double[k, k];
        for (int i = 0; i < k; i++)
        {
            for (int j = 0; j < k; j++)
            {
                double sum = 0;
                for (int r = 0; r < n; r++)
                {
                    sum += transformed[r, i] * transformed[r, j];
                }
                ATA[i, j] = sum;
                if (i == j)
                {
                    ATA[i, j] += _alpha;
                }
            }
        }

        var ATX = new double[k, p];
        for (int i = 0; i < k; i++)
        {
            for (int j = 0; j < p; j++)
            {
                double sum = 0;
                for (int r = 0; r < n; r++)
                {
                    sum += transformed[r, i] * _trainingData[r, j];
                }
                ATX[i, j] = sum;
            }
        }

        // Solve using simple inversion (for small k)
        _inverseTransformMatrix = SolveLinearSystem(ATA, ATX, k, p);
    }

    private double[,] SolveLinearSystem(double[,] A, double[,] B, int n, int m)
    {
        // Solve A @ X = B for each column of B
        var result = new double[n, m];

        // Simple Gaussian elimination
        for (int col = 0; col < m; col++)
        {
            var augmented = new double[n, n + 1];
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    augmented[i, j] = A[i, j];
                }
                augmented[i, n] = B[i, col];
            }

            // Forward elimination
            for (int c = 0; c < n; c++)
            {
                int maxRow = c;
                for (int row = c + 1; row < n; row++)
                {
                    if (Math.Abs(augmented[row, c]) > Math.Abs(augmented[maxRow, c]))
                    {
                        maxRow = row;
                    }
                }

                for (int j = c; j <= n; j++)
                {
                    double temp = augmented[c, j];
                    augmented[c, j] = augmented[maxRow, j];
                    augmented[maxRow, j] = temp;
                }

                if (Math.Abs(augmented[c, c]) > 1e-10)
                {
                    for (int row = c + 1; row < n; row++)
                    {
                        double factor = augmented[row, c] / augmented[c, c];
                        for (int j = c; j <= n; j++)
                        {
                            augmented[row, j] -= factor * augmented[c, j];
                        }
                    }
                }
            }

            // Back substitution
            for (int i = n - 1; i >= 0; i--)
            {
                result[i, col] = augmented[i, n];
                for (int j = i + 1; j < n; j++)
                {
                    result[i, col] -= augmented[i, j] * result[j, col];
                }
                if (Math.Abs(augmented[i, i]) > 1e-10)
                {
                    result[i, col] /= augmented[i, i];
                }
            }
        }

        return result;
    }

    private double[,] TransformInternal(double[,] X)
    {
        if (_trainingData is null || _alphas is null)
        {
            throw new InvalidOperationException("KernelPCA has not been fitted.");
        }

        int n = X.GetLength(0);
        int nTrain = _trainingData.GetLength(0);
        int k = _alphas.GetLength(1);
        var result = new double[n, k];

        // Compute kernel matrix between X and training data
        var K = ComputeKernelMatrix(X, _trainingData);

        // Center using training data statistics
        // (simplified centering for new data)
        for (int i = 0; i < n; i++)
        {
            double rowMean = 0;
            for (int j = 0; j < nTrain; j++)
            {
                rowMean += K[i, j];
            }
            rowMean /= nTrain;

            for (int j = 0; j < nTrain; j++)
            {
                K[i, j] -= rowMean;
            }
        }

        // Project onto principal components
        for (int i = 0; i < n; i++)
        {
            for (int c = 0; c < k; c++)
            {
                double sum = 0;
                for (int j = 0; j < nTrain; j++)
                {
                    sum += K[i, j] * _alphas[j, c];
                }
                result[i, c] = sum;
            }
        }

        return result;
    }

    /// <summary>
    /// Transforms the data using kernel PCA.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The transformed data.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        int n = data.Rows;
        int p = data.Columns;

        var X = new double[n, p];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < p; j++)
            {
                X[i, j] = NumOps.ToDouble(data[i, j]);
            }
        }

        var transformed = TransformInternal(X);

        int k = transformed.GetLength(1);
        var result = new T[n, k];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < k; j++)
            {
                result[i, j] = NumOps.FromDouble(transformed[i, j]);
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Transforms data back to approximate original space.
    /// </summary>
    /// <param name="data">The transformed data.</param>
    /// <returns>Approximate reconstruction in original space.</returns>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        if (!_fitInverseTransform || _inverseTransformMatrix is null)
        {
            throw new NotSupportedException(
                "Inverse transform was not fitted. Set fitInverseTransform=true in constructor.");
        }

        int n = data.Rows;
        int k = _inverseTransformMatrix.GetLength(0);
        int p = _inverseTransformMatrix.GetLength(1);
        var result = new T[n, p];

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < p; j++)
            {
                double sum = 0;
                for (int c = 0; c < k; c++)
                {
                    sum += NumOps.ToDouble(data[i, c]) * _inverseTransformMatrix[c, j];
                }
                result[i, j] = NumOps.FromDouble(sum);
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        int k = _alphas?.GetLength(1) ?? _nComponents;
        var names = new string[k];
        for (int i = 0; i < k; i++)
        {
            names[i] = $"KPC{i + 1}";
        }
        return names;
    }
}

/// <summary>
/// Specifies the kernel type for Kernel PCA.
/// </summary>
public enum KernelType
{
    /// <summary>
    /// Linear kernel: K(x, y) = x · y
    /// </summary>
    Linear,

    /// <summary>
    /// Radial Basis Function (Gaussian): K(x, y) = exp(-γ||x-y||²)
    /// </summary>
    RBF,

    /// <summary>
    /// Polynomial kernel: K(x, y) = (γ(x · y) + c₀)^d
    /// </summary>
    Polynomial,

    /// <summary>
    /// Sigmoid (hyperbolic tangent): K(x, y) = tanh(γ(x · y) + c₀)
    /// </summary>
    Sigmoid
}
