using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DimensionalityReduction;

/// <summary>
/// Sparse Principal Component Analysis using L1 regularization.
/// </summary>
/// <remarks>
/// <para>
/// SparsePCA finds sparse principal components by applying L1 regularization
/// (LASSO-like penalty) to the component loadings. This results in components
/// where many loadings are exactly zero, making them more interpretable.
/// </para>
/// <para>
/// Unlike standard PCA where each component is a combination of ALL features,
/// sparse PCA produces components that depend on only a subset of features.
/// </para>
/// <para><b>For Beginners:</b> Sparse PCA creates "simpler" principal components:
/// - Standard PCA: Component = 0.3*Feature1 + 0.2*Feature2 + 0.1*Feature3 + ...
/// - Sparse PCA: Component = 0.5*Feature1 + 0*Feature2 + 0.4*Feature3 + 0*...
/// - Zeros make it easier to interpret what each component represents
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class SparsePCA<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nComponents;
    private readonly double _alpha;
    private readonly double _ridge;
    private readonly int _maxIter;
    private readonly double _tol;
    private readonly int? _randomState;

    // Fitted parameters
    private double[]? _mean;
    private double[,]? _components;
    private double[]? _error;
    private int _nFeaturesIn;

    /// <summary>
    /// Gets the number of components.
    /// </summary>
    public int NComponents => _nComponents;

    /// <summary>
    /// Gets the sparsity regularization parameter.
    /// </summary>
    public double Alpha => _alpha;

    /// <summary>
    /// Gets the ridge regularization parameter.
    /// </summary>
    public double Ridge => _ridge;

    /// <summary>
    /// Gets the mean of each feature.
    /// </summary>
    public double[]? Mean => _mean;

    /// <summary>
    /// Gets the sparse components (each row is a component).
    /// </summary>
    public double[,]? Components => _components;

    /// <summary>
    /// Gets the reconstruction error for each iteration.
    /// </summary>
    public double[]? Error => _error;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => true;

    /// <summary>
    /// Creates a new instance of <see cref="SparsePCA{T}"/>.
    /// </summary>
    /// <param name="nComponents">Number of sparse components. Defaults to 2.</param>
    /// <param name="alpha">Sparsity regularization parameter. Higher values produce sparser components. Defaults to 1.0.</param>
    /// <param name="ridge">Ridge regularization for stability. Defaults to 0.01.</param>
    /// <param name="maxIter">Maximum number of iterations. Defaults to 100.</param>
    /// <param name="tol">Convergence tolerance. Defaults to 1e-6.</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <param name="columnIndices">The column indices to use, or null for all columns.</param>
    public SparsePCA(
        int nComponents = 2,
        double alpha = 1.0,
        double ridge = 0.01,
        int maxIter = 100,
        double tol = 1e-6,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nComponents < 1)
        {
            throw new ArgumentException("Number of components must be at least 1.", nameof(nComponents));
        }

        if (alpha < 0)
        {
            throw new ArgumentException("Alpha must be non-negative.", nameof(alpha));
        }

        if (ridge < 0)
        {
            throw new ArgumentException("Ridge must be non-negative.", nameof(ridge));
        }

        _nComponents = nComponents;
        _alpha = alpha;
        _ridge = ridge;
        _maxIter = maxIter;
        _tol = tol;
        _randomState = randomState;
    }

    /// <summary>
    /// Fits Sparse PCA using coordinate descent with L1 regularization.
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        _nFeaturesIn = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        int k = Math.Min(_nComponents, Math.Min(n, p));

        // Center the data
        _mean = new double[p];
        var centered = new double[n, p];

        for (int j = 0; j < p; j++)
        {
            double sum = 0;
            for (int i = 0; i < n; i++)
            {
                sum += NumOps.ToDouble(data[i, j]);
            }
            _mean[j] = sum / n;
        }

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < p; j++)
            {
                centered[i, j] = NumOps.ToDouble(data[i, j]) - _mean[j];
            }
        }

        // Initialize components using SVD-like approach (PCA initialization)
        _components = new double[k, p];
        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSeededRandom(42);

        // Initialize with random unit vectors
        for (int c = 0; c < k; c++)
        {
            double norm = 0;
            for (int j = 0; j < p; j++)
            {
                _components[c, j] = random.NextDouble() - 0.5;
                norm += _components[c, j] * _components[c, j];
            }
            norm = Math.Sqrt(norm);
            if (norm > 1e-10)
            {
                for (int j = 0; j < p; j++)
                {
                    _components[c, j] /= norm;
                }
            }
        }

        // Alternating minimization: solve for codes and dictionary
        var errorHistory = new List<double>();
        var codes = new double[n, k];

        for (int iter = 0; iter < _maxIter; iter++)
        {
            // Step 1: Update codes (projections) - ridge regression
            UpdateCodes(centered, codes, n, p, k);

            // Step 2: Update dictionary (components) with L1 regularization
            double[] oldComponents = new double[k * p];
            for (int c = 0; c < k; c++)
            {
                for (int j = 0; j < p; j++)
                {
                    oldComponents[c * p + j] = _components[c, j];
                }
            }

            UpdateDictionary(centered, codes, n, p, k);

            // Compute reconstruction error
            double error = ComputeReconstructionError(centered, codes, n, p, k);
            errorHistory.Add(error);

            // Check convergence
            double maxChange = 0;
            for (int c = 0; c < k; c++)
            {
                for (int j = 0; j < p; j++)
                {
                    double change = Math.Abs(_components[c, j] - oldComponents[c * p + j]);
                    maxChange = Math.Max(maxChange, change);
                }
            }

            if (maxChange < _tol)
            {
                break;
            }
        }

        _error = errorHistory.ToArray();
    }

    private void UpdateCodes(double[,] data, double[,] codes, int n, int p, int k)
    {
        // Solve: codes = data * components^T * (components * components^T + ridge*I)^-1
        // For simplicity, using simple projection with ridge regularization

        // Compute components * components^T + ridge*I
        var gram = new double[k, k];
        for (int i = 0; i < k; i++)
        {
            for (int j = 0; j < k; j++)
            {
                double sum = 0;
                for (int f = 0; f < p; f++)
                {
                    sum += _components![i, f] * _components[j, f];
                }
                gram[i, j] = sum;
                if (i == j)
                {
                    gram[i, j] += _ridge;
                }
            }
        }

        // Invert gram matrix (simple Gauss-Jordan for small matrices)
        var gramInv = InvertMatrix(gram, k);

        // Compute data * components^T
        var dataProj = new double[n, k];
        for (int i = 0; i < n; i++)
        {
            for (int c = 0; c < k; c++)
            {
                double sum = 0;
                for (int j = 0; j < p; j++)
                {
                    sum += data[i, j] * _components![c, j];
                }
                dataProj[i, c] = sum;
            }
        }

        // Compute codes = dataProj * gramInv
        for (int i = 0; i < n; i++)
        {
            for (int c = 0; c < k; c++)
            {
                double sum = 0;
                for (int j = 0; j < k; j++)
                {
                    sum += dataProj[i, j] * gramInv[j, c];
                }
                codes[i, c] = sum;
            }
        }
    }

    private void UpdateDictionary(double[,] data, double[,] codes, int n, int p, int k)
    {
        // Update each component using coordinate descent with L1 regularization
        for (int c = 0; c < k; c++)
        {
            // Compute residual without component c
            var residual = new double[n, p];
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < p; j++)
                {
                    residual[i, j] = data[i, j];
                    for (int cc = 0; cc < k; cc++)
                    {
                        if (cc != c)
                        {
                            residual[i, j] -= codes[i, cc] * _components![cc, j];
                        }
                    }
                }
            }

            // Compute sum of squared codes for component c
            double codesSquared = 0;
            for (int i = 0; i < n; i++)
            {
                codesSquared += codes[i, c] * codes[i, c];
            }

            if (codesSquared < 1e-10)
            {
                continue;
            }

            // Update each loading with soft thresholding (L1)
            for (int j = 0; j < p; j++)
            {
                double correlation = 0;
                for (int i = 0; i < n; i++)
                {
                    correlation += codes[i, c] * residual[i, j];
                }

                // Soft thresholding
                double newValue = SoftThreshold(correlation / codesSquared, _alpha / codesSquared);
                _components![c, j] = newValue;
            }

            // Normalize component
            double norm = 0;
            for (int j = 0; j < p; j++)
            {
                norm += _components![c, j] * _components[c, j];
            }
            norm = Math.Sqrt(norm);

            if (norm > 1e-10)
            {
                for (int j = 0; j < p; j++)
                {
                    _components![c, j] /= norm;
                }
            }
        }
    }

    private static double SoftThreshold(double x, double lambda)
    {
        if (x > lambda)
        {
            return x - lambda;
        }
        else if (x < -lambda)
        {
            return x + lambda;
        }
        else
        {
            return 0;
        }
    }

    private double ComputeReconstructionError(double[,] data, double[,] codes, int n, int p, int k)
    {
        double error = 0;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < p; j++)
            {
                double reconstructed = 0;
                for (int c = 0; c < k; c++)
                {
                    reconstructed += codes[i, c] * _components![c, j];
                }
                double diff = data[i, j] - reconstructed;
                error += diff * diff;
            }
        }
        return error / (n * p);
    }

    private static double[,] InvertMatrix(double[,] matrix, int n)
    {
        var result = new double[n, n];
        var temp = new double[n, 2 * n];

        // Create augmented matrix [A | I]
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                temp[i, j] = matrix[i, j];
                temp[i, j + n] = (i == j) ? 1.0 : 0.0;
            }
        }

        // Gauss-Jordan elimination
        for (int i = 0; i < n; i++)
        {
            // Find pivot
            double maxVal = Math.Abs(temp[i, i]);
            int maxRow = i;
            for (int k = i + 1; k < n; k++)
            {
                if (Math.Abs(temp[k, i]) > maxVal)
                {
                    maxVal = Math.Abs(temp[k, i]);
                    maxRow = k;
                }
            }

            // Swap rows
            if (maxRow != i)
            {
                for (int j = 0; j < 2 * n; j++)
                {
                    (temp[i, j], temp[maxRow, j]) = (temp[maxRow, j], temp[i, j]);
                }
            }

            // Scale row
            double pivot = temp[i, i];
            if (Math.Abs(pivot) < 1e-10)
            {
                pivot = 1e-10;
            }

            for (int j = 0; j < 2 * n; j++)
            {
                temp[i, j] /= pivot;
            }

            // Eliminate column
            for (int k = 0; k < n; k++)
            {
                if (k != i)
                {
                    double factor = temp[k, i];
                    for (int j = 0; j < 2 * n; j++)
                    {
                        temp[k, j] -= factor * temp[i, j];
                    }
                }
            }
        }

        // Extract inverse
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                result[i, j] = temp[i, j + n];
            }
        }

        return result;
    }

    /// <summary>
    /// Transforms the data by projecting onto sparse components.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_mean is null || _components is null)
        {
            throw new InvalidOperationException("SparsePCA has not been fitted.");
        }

        int n = data.Rows;
        int p = data.Columns;
        int k = _components.GetLength(0);
        var result = new T[n, k];

        for (int i = 0; i < n; i++)
        {
            for (int c = 0; c < k; c++)
            {
                double sum = 0;
                for (int j = 0; j < p; j++)
                {
                    double centered = NumOps.ToDouble(data[i, j]) - _mean[j];
                    sum += centered * _components[c, j];
                }
                result[i, c] = NumOps.FromDouble(sum);
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Transforms data back to original space.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        if (_mean is null || _components is null)
        {
            throw new InvalidOperationException("SparsePCA has not been fitted.");
        }

        int n = data.Rows;
        int p = _nFeaturesIn;
        int k = _components.GetLength(0);
        var result = new T[n, p];

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < p; j++)
            {
                double sum = _mean[j];
                for (int c = 0; c < k; c++)
                {
                    sum += NumOps.ToDouble(data[i, c]) * _components[c, j];
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
        int k = _components?.GetLength(0) ?? _nComponents;
        var names = new string[k];
        for (int i = 0; i < k; i++)
        {
            names[i] = $"SparsePC{i + 1}";
        }
        return names;
    }
}
