using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DimensionalityReduction;

/// <summary>
/// Principal Component Analysis for dimensionality reduction.
/// </summary>
/// <remarks>
/// <para>
/// PCA finds the directions of maximum variance in the data and projects the data
/// onto these principal components. This reduces dimensionality while preserving
/// as much variance as possible.
/// </para>
/// <para>
/// PCA is useful for:
/// - Reducing the number of features while retaining most information
/// - Removing multicollinearity between features
/// - Visualizing high-dimensional data
/// - Noise reduction
/// </para>
/// <para><b>For Beginners:</b> PCA transforms your features into new features called
/// "principal components" that are:
/// - Uncorrelated with each other
/// - Ordered by importance (how much variance they explain)
/// - Linear combinations of your original features
///
/// Example: 100 features might be reduced to 10 principal components that capture
/// 95% of the information in your data.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class PCA<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int? _nComponents;
    private readonly double? _varianceRatio;
    private readonly bool _whiten;

    // Fitted parameters
    private double[]? _mean;
    private double[]? _std;
    private double[,]? _components; // Principal component vectors (eigenvectors)
    private double[]? _explainedVariance;
    private double[]? _explainedVarianceRatio;
    private double[]? _singularValues;
    private int _nFeaturesIn;
    private int _nComponentsOut;

    /// <summary>
    /// Gets the number of components to keep.
    /// </summary>
    public int? NComponents => _nComponents;

    /// <summary>
    /// Gets the target variance ratio to retain.
    /// </summary>
    public double? VarianceRatio => _varianceRatio;

    /// <summary>
    /// Gets whether whitening is applied.
    /// </summary>
    public bool Whiten => _whiten;

    /// <summary>
    /// Gets the mean of each feature.
    /// </summary>
    public double[]? Mean => _mean;

    /// <summary>
    /// Gets the principal components (each row is a component).
    /// </summary>
    public double[,]? Components => _components;

    /// <summary>
    /// Gets the explained variance for each component.
    /// </summary>
    public double[]? ExplainedVariance => _explainedVariance;

    /// <summary>
    /// Gets the explained variance ratio for each component.
    /// </summary>
    public double[]? ExplainedVarianceRatio => _explainedVarianceRatio;

    /// <summary>
    /// Gets the singular values.
    /// </summary>
    public double[]? SingularValues => _singularValues;

    /// <summary>
    /// Gets the number of components after fitting.
    /// </summary>
    public int NComponentsOut => _nComponentsOut;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => true;

    /// <summary>
    /// Creates a new instance of <see cref="PCA{T}"/>.
    /// </summary>
    /// <param name="nComponents">Number of components to keep. If null, keeps all.</param>
    /// <param name="varianceRatio">Keep enough components to explain this variance ratio (0-1). Overrides nComponents if set.</param>
    /// <param name="whiten">If true, scale components to unit variance. Defaults to false.</param>
    /// <param name="columnIndices">The column indices to use, or null for all columns.</param>
    public PCA(
        int? nComponents = null,
        double? varianceRatio = null,
        bool whiten = false,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nComponents.HasValue && nComponents.Value < 1)
        {
            throw new ArgumentException("Number of components must be at least 1.", nameof(nComponents));
        }

        if (varianceRatio.HasValue && (varianceRatio.Value <= 0 || varianceRatio.Value > 1))
        {
            throw new ArgumentException("Variance ratio must be between 0 and 1.", nameof(varianceRatio));
        }

        _nComponents = nComponents;
        _varianceRatio = varianceRatio;
        _whiten = whiten;
    }

    /// <summary>
    /// Fits PCA by computing principal components.
    /// </summary>
    /// <param name="data">The training data matrix.</param>
    protected override void FitCore(Matrix<T> data)
    {
        _nFeaturesIn = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Convert to double array and center
        _mean = new double[p];
        _std = new double[p];
        var centered = new double[n, p];

        // Compute mean
        for (int j = 0; j < p; j++)
        {
            double sum = 0;
            for (int i = 0; i < n; i++)
            {
                sum += NumOps.ToDouble(data[i, j]);
            }
            _mean[j] = sum / n;
        }

        // Center the data and compute std
        for (int j = 0; j < p; j++)
        {
            double sumSq = 0;
            for (int i = 0; i < n; i++)
            {
                centered[i, j] = NumOps.ToDouble(data[i, j]) - _mean[j];
                sumSq += centered[i, j] * centered[i, j];
            }
            _std[j] = Math.Sqrt(sumSq / n);
        }

        // Compute covariance matrix
        var covariance = new double[p, p];
        for (int i = 0; i < p; i++)
        {
            for (int j = i; j < p; j++)
            {
                double sum = 0;
                for (int k = 0; k < n; k++)
                {
                    sum += centered[k, i] * centered[k, j];
                }
                covariance[i, j] = sum / (n - 1);
                covariance[j, i] = covariance[i, j];
            }
        }

        // Compute eigenvalues and eigenvectors using power iteration
        var (eigenvalues, eigenvectors) = ComputeEigen(covariance, p);

        // Sort by eigenvalue descending
        var indices = Enumerable.Range(0, p)
            .OrderByDescending(i => eigenvalues[i])
            .ToArray();

        var sortedEigenvalues = new double[p];
        var sortedEigenvectors = new double[p, p];

        for (int i = 0; i < p; i++)
        {
            sortedEigenvalues[i] = eigenvalues[indices[i]];
            for (int j = 0; j < p; j++)
            {
                sortedEigenvectors[i, j] = eigenvectors[indices[i], j];
            }
        }

        // Determine number of components to keep
        double totalVariance = sortedEigenvalues.Sum();
        _explainedVariance = sortedEigenvalues;
        _explainedVarianceRatio = new double[p];

        for (int i = 0; i < p; i++)
        {
            _explainedVarianceRatio[i] = sortedEigenvalues[i] / totalVariance;
        }

        if (_varianceRatio.HasValue)
        {
            // Keep enough components to explain target variance
            double cumulative = 0;
            _nComponentsOut = 0;
            for (int i = 0; i < p; i++)
            {
                cumulative += _explainedVarianceRatio[i];
                _nComponentsOut++;
                if (cumulative >= _varianceRatio.Value)
                {
                    break;
                }
            }
        }
        else if (_nComponents.HasValue)
        {
            _nComponentsOut = Math.Min(_nComponents.Value, p);
        }
        else
        {
            _nComponentsOut = p;
        }

        // Store the selected components
        _components = new double[_nComponentsOut, p];
        _singularValues = new double[_nComponentsOut];

        for (int i = 0; i < _nComponentsOut; i++)
        {
            _singularValues[i] = Math.Sqrt(sortedEigenvalues[i] * (n - 1));
            for (int j = 0; j < p; j++)
            {
                _components[i, j] = sortedEigenvectors[i, j];
            }
        }
    }

    private (double[] Eigenvalues, double[,] Eigenvectors) ComputeEigen(double[,] matrix, int n)
    {
        // Power iteration method with deflation for eigenvalue decomposition
        var eigenvalues = new double[n];
        var eigenvectors = new double[n, n];
        var A = (double[,])matrix.Clone();

        for (int k = 0; k < n; k++)
        {
            // Initialize random vector
            var v = new double[n];
            for (int i = 0; i < n; i++)
            {
                v[i] = 1.0 / Math.Sqrt(n);
            }

            // Power iteration
            for (int iter = 0; iter < 100; iter++)
            {
                // Multiply A * v
                var Av = new double[n];
                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        Av[i] += A[i, j] * v[j];
                    }
                }

                // Compute eigenvalue (Rayleigh quotient)
                double lambda = 0;
                double norm = 0;
                for (int i = 0; i < n; i++)
                {
                    lambda += v[i] * Av[i];
                    norm += Av[i] * Av[i];
                }

                // Normalize
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

            eigenvalues[k] = Math.Max(0, eigenvalue);

            for (int i = 0; i < n; i++)
            {
                eigenvectors[k, i] = v[i];
            }

            // Deflate: A = A - lambda * v * v^T
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
    /// Transforms the data by projecting onto principal components.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The transformed data.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_mean is null || _components is null)
        {
            throw new InvalidOperationException("PCA has not been fitted.");
        }

        int n = data.Rows;
        int p = data.Columns;
        var result = new T[n, _nComponentsOut];

        for (int i = 0; i < n; i++)
        {
            for (int k = 0; k < _nComponentsOut; k++)
            {
                double sum = 0;
                for (int j = 0; j < p; j++)
                {
                    double centered = NumOps.ToDouble(data[i, j]) - _mean[j];
                    sum += centered * _components[k, j];
                }

                if (_whiten && _singularValues is not null && _singularValues[k] > 1e-10)
                {
                    sum /= _singularValues[k];
                }

                result[i, k] = NumOps.FromDouble(sum);
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Transforms data back to original space.
    /// </summary>
    /// <param name="data">The transformed data.</param>
    /// <returns>Data in original feature space.</returns>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        if (_mean is null || _components is null)
        {
            throw new InvalidOperationException("PCA has not been fitted.");
        }

        int n = data.Rows;
        int p = _nFeaturesIn;
        var result = new T[n, p];

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < p; j++)
            {
                double sum = _mean[j];
                for (int k = 0; k < _nComponentsOut; k++)
                {
                    double val = NumOps.ToDouble(data[i, k]);
                    if (_whiten && _singularValues is not null && _singularValues[k] > 1e-10)
                    {
                        val *= _singularValues[k];
                    }
                    sum += val * _components[k, j];
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
        var names = new string[_nComponentsOut];
        for (int i = 0; i < _nComponentsOut; i++)
        {
            names[i] = $"PC{i + 1}";
        }
        return names;
    }
}
