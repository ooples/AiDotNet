using AiDotNet.Extensions;
using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DimensionalityReduction;

/// <summary>
/// Randomized PCA using randomized SVD for efficient computation.
/// </summary>
/// <remarks>
/// <para>
/// Randomized PCA uses randomized algorithms to efficiently compute principal components
/// without computing the full SVD. It's much faster than standard PCA for large datasets
/// while providing accurate approximations of the top components.
/// </para>
/// <para>
/// The algorithm:
/// 1. Generate random projection matrix
/// 2. Form sample matrix Y = A * Ω (project data to random subspace)
/// 3. Orthonormalize Y using QR decomposition
/// 4. Form B = Q^T * A (project to Q's range)
/// 5. Compute SVD of B to get principal components
/// </para>
/// <para><b>For Beginners:</b> Randomized PCA is faster because:
/// - It only computes the components you need (not all)
/// - Random projection preserves structure efficiently
/// - Power iteration improves accuracy if needed
/// - Works well for low-rank data
///
/// Use cases:
/// - Very large datasets where standard PCA is slow
/// - When you only need top few components
/// - Streaming or online scenarios
/// - Memory-constrained environments
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class RandomizedPCA<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nComponents;
    private readonly int _nOversamples;
    private readonly int _nPowerIter;
    private readonly int? _randomState;

    // Fitted parameters
    private double[,]? _components; // Principal components (d x p)
    private double[]? _mean;
    private double[]? _explainedVariance;
    private double[]? _singularValues;
    private double _totalVariance;
    private int _nFeatures;

    /// <summary>
    /// Gets the number of components (dimensions).
    /// </summary>
    public int NComponents => _nComponents;

    /// <summary>
    /// Gets the principal components (each row is a component).
    /// </summary>
    public double[,]? Components => _components;

    /// <summary>
    /// Gets the explained variance for each component.
    /// </summary>
    public double[]? ExplainedVariance => _explainedVariance;

    /// <summary>
    /// Gets the singular values.
    /// </summary>
    public double[]? SingularValues => _singularValues;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => true;

    /// <summary>
    /// Creates a new instance of <see cref="RandomizedPCA{T}"/>.
    /// </summary>
    /// <param name="nComponents">Target dimensionality. Defaults to 2.</param>
    /// <param name="nOversamples">Additional random vectors for accuracy. Defaults to 10.</param>
    /// <param name="nPowerIter">Number of power iterations for accuracy. Defaults to 2.</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <param name="columnIndices">The column indices to use, or null for all columns.</param>
    public RandomizedPCA(
        int nComponents = 2,
        int nOversamples = 10,
        int nPowerIter = 2,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nComponents < 1)
        {
            throw new ArgumentException("Number of components must be at least 1.", nameof(nComponents));
        }

        _nComponents = nComponents;
        _nOversamples = nOversamples;
        _nPowerIter = nPowerIter;
        _randomState = randomState;
    }

    /// <summary>
    /// Fits Randomized PCA using randomized SVD.
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        int n = data.Rows;
        int p = data.Columns;
        _nFeatures = p;

        // Validate minimum sample count (need at least 2 for variance calculation)
        if (n < 2)
        {
            throw new ArgumentException("RandomizedPCA requires at least 2 samples.", nameof(data));
        }

        // Validate that we can extract the requested number of components
        int maxComponents = Math.Min(n, p);
        if (_nComponents > maxComponents)
        {
            throw new ArgumentException(
                $"Cannot extract {_nComponents} components from data with dimensions {n}x{p}. " +
                $"Maximum available components: {maxComponents}.");
        }

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSeededRandom(42);

        // Convert to double array
        var X = new double[n, p];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < p; j++)
            {
                X[i, j] = NumOps.ToDouble(data[i, j]);
            }
        }

        // Step 1: Center the data
        _mean = new double[p];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < p; j++)
            {
                _mean[j] += X[i, j];
            }
        }
        for (int j = 0; j < p; j++) _mean[j] /= n;

        var Xc = new double[n, p];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < p; j++)
            {
                Xc[i, j] = X[i, j] - _mean[j];
            }
        }

        // Compute total variance (sum of variances of all features)
        _totalVariance = 0;
        for (int j = 0; j < p; j++)
        {
            double colVariance = 0;
            for (int i = 0; i < n; i++)
            {
                colVariance += Xc[i, j] * Xc[i, j];
            }
            _totalVariance += colVariance / (n - 1);
        }

        // Step 2: Randomized range finder
        int k = _nComponents + _nOversamples;
        k = Math.Min(k, Math.Min(n, p));

        // Generate random Gaussian matrix Ω (p x k)
        var Omega = new double[p, k];
        for (int i = 0; i < p; i++)
        {
            for (int j = 0; j < k; j++)
            {
                Omega[i, j] = random.NextGaussian();
            }
        }

        // Y = X * Ω (n x k)
        var Y = new double[n, k];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < k; j++)
            {
                for (int l = 0; l < p; l++)
                {
                    Y[i, j] += Xc[i, l] * Omega[l, j];
                }
            }
        }

        // Step 3: Power iteration for improved accuracy
        for (int iter = 0; iter < _nPowerIter; iter++)
        {
            // Y = X * X^T * Y
            // First compute Z = X^T * Y (p x k)
            var Z = new double[p, k];
            for (int i = 0; i < p; i++)
            {
                for (int j = 0; j < k; j++)
                {
                    for (int l = 0; l < n; l++)
                    {
                        Z[i, j] += Xc[l, i] * Y[l, j];
                    }
                }
            }

            // Orthonormalize Z
            Z = QROrthonormalize(Z, p, k);

            // Y = X * Z
            Y = new double[n, k];
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < k; j++)
                {
                    for (int l = 0; l < p; l++)
                    {
                        Y[i, j] += Xc[i, l] * Z[l, j];
                    }
                }
            }
        }

        // Step 4: Orthonormalize Y to get Q (n x k)
        var Q = QROrthonormalize(Y, n, k);

        // Step 5: Form B = Q^T * X (k x p)
        var B = new double[k, p];
        for (int i = 0; i < k; i++)
        {
            for (int j = 0; j < p; j++)
            {
                for (int l = 0; l < n; l++)
                {
                    B[i, j] += Q[l, i] * Xc[l, j];
                }
            }
        }

        // Step 6: SVD of B to get right singular vectors
        var (singularValues, rightVectors) = ComputeSVD(B, k, p, random);

        // Step 7: Extract top components
        _components = new double[_nComponents, p];
        _singularValues = new double[_nComponents];
        _explainedVariance = new double[_nComponents];

        for (int d = 0; d < _nComponents; d++)
        {
            _singularValues[d] = singularValues[d];
            _explainedVariance[d] = (singularValues[d] * singularValues[d]) / (n - 1);

            for (int j = 0; j < p; j++)
            {
                _components[d, j] = rightVectors[d, j];
            }
        }
    }

    private double[,] QROrthonormalize(double[,] A, int m, int n)
    {
        var Q = new double[m, n];

        // Copy A to Q
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                Q[i, j] = A[i, j];
            }
        }

        // Modified Gram-Schmidt
        for (int j = 0; j < n; j++)
        {
            // Normalize column j
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

            // Orthogonalize remaining columns
            for (int k = j + 1; k < n; k++)
            {
                double dot = 0;
                for (int i = 0; i < m; i++)
                {
                    dot += Q[i, j] * Q[i, k];
                }

                for (int i = 0; i < m; i++)
                {
                    Q[i, k] -= dot * Q[i, j];
                }
            }
        }

        return Q;
    }

    private (double[] singularValues, double[,] rightVectors) ComputeSVD(
        double[,] B, int m, int n, Random random)
    {
        // Compute B^T * B for right singular vectors
        var BtB = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                for (int k = 0; k < m; k++)
                {
                    BtB[i, j] += B[k, i] * B[k, j];
                }
            }
        }

        // Power iteration to get top eigenvectors of B^T * B
        var singularValues = new double[_nComponents];
        var rightVectors = new double[_nComponents, n];
        var A = (double[,])BtB.Clone();

        for (int d = 0; d < _nComponents; d++)
        {
            var v = new double[n];
            for (int i = 0; i < n; i++) v[i] = random.NextDouble() - 0.5;

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
                for (int i = 0; i < n; i++) norm += Av[i] * Av[i];
                norm = Math.Sqrt(norm);

                if (norm < 1e-10) break;

                double change = 0;
                for (int i = 0; i < n; i++)
                {
                    double newV = Av[i] / norm;
                    change += Math.Abs(newV - v[i]);
                    v[i] = newV;
                }

                if (change < 1e-10) break;
            }

            // Store right singular vector
            for (int i = 0; i < n; i++)
            {
                rightVectors[d, i] = v[i];
            }

            // Compute eigenvalue (squared singular value)
            var Av2 = new double[n];
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    Av2[i] += A[i, j] * v[j];
                }
            }

            double lambda = 0;
            for (int i = 0; i < n; i++)
            {
                lambda += v[i] * Av2[i];
            }

            singularValues[d] = Math.Sqrt(Math.Max(lambda, 0));

            // Deflate
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    A[i, j] -= lambda * v[i] * v[j];
                }
            }
        }

        return (singularValues, rightVectors);
    }

    /// <summary>
    /// Transforms data by projecting onto principal components.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_components is null || _mean is null)
        {
            throw new InvalidOperationException("RandomizedPCA has not been fitted.");
        }

        int n = data.Rows;
        int p = data.Columns;

        if (p != _nFeatures)
        {
            throw new ArgumentException(
                $"Input has {p} features but RandomizedPCA was fitted with {_nFeatures} features.",
                nameof(data));
        }

        var result = new T[n, _nComponents];

        for (int i = 0; i < n; i++)
        {
            for (int d = 0; d < _nComponents; d++)
            {
                double val = 0;
                for (int j = 0; j < p; j++)
                {
                    double xij = NumOps.ToDouble(data[i, j]) - _mean[j];
                    val += xij * _components[d, j];
                }
                result[i, d] = NumOps.FromDouble(val);
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Reconstructs data from the reduced representation.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        if (_components is null || _mean is null)
        {
            throw new InvalidOperationException("RandomizedPCA has not been fitted.");
        }

        int n = data.Rows;
        int d = data.Columns;

        if (d != _nComponents)
        {
            throw new ArgumentException(
                $"Input has {d} dimensions but RandomizedPCA was fitted with {_nComponents} components.",
                nameof(data));
        }

        var result = new T[n, _nFeatures];

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < _nFeatures; j++)
            {
                double val = _mean[j];
                for (int k = 0; k < _nComponents; k++)
                {
                    val += NumOps.ToDouble(data[i, k]) * _components[k, j];
                }
                result[i, j] = NumOps.FromDouble(val);
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Gets the total explained variance ratio (proportion of total variance explained by selected components).
    /// </summary>
    /// <returns>A value between 0 and 1 representing the fraction of total variance captured.</returns>
    public double GetExplainedVarianceRatio()
    {
        if (_explainedVariance is null)
        {
            throw new InvalidOperationException("RandomizedPCA has not been fitted.");
        }

        double explainedSum = 0;
        foreach (var v in _explainedVariance)
        {
            explainedSum += v;
        }

        return _totalVariance > 0 ? explainedSum / _totalVariance : 0;
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        var names = new string[_nComponents];
        for (int i = 0; i < _nComponents; i++)
        {
            names[i] = $"RPCA{i + 1}";
        }
        return names;
    }
}
