using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DimensionalityReduction;

/// <summary>
/// Truncated Singular Value Decomposition for dimensionality reduction.
/// </summary>
/// <remarks>
/// <para>
/// TruncatedSVD performs dimensionality reduction by computing the truncated
/// singular value decomposition. Unlike PCA, it does not center the data,
/// making it suitable for sparse matrices (e.g., TF-IDF from text).
/// </para>
/// <para>
/// Also known as Latent Semantic Analysis (LSA) when applied to document-term matrices.
/// </para>
/// <para><b>For Beginners:</b> SVD is similar to PCA but:
/// - Doesn't center the data (preserves sparsity in sparse matrices)
/// - Can be more memory-efficient for large sparse datasets
/// - Often used for text analysis (finding hidden topics in documents)
///
/// Example: In text analysis, TruncatedSVD can find that "car" and "automobile"
/// are related even if they never appear together in the same document.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class TruncatedSVD<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nComponents;
    private readonly int _nIterations;
    private readonly int _randomState;

    // Fitted parameters
    private double[,]? _components; // Right singular vectors (V^T)
    private double[]? _singularValues;
    private double[]? _explainedVariance;
    private double[]? _explainedVarianceRatio;
    private int _nFeaturesIn;

    /// <summary>
    /// Gets the number of components.
    /// </summary>
    public int NComponents => _nComponents;

    /// <summary>
    /// Gets the number of iterations for randomized SVD.
    /// </summary>
    public int NIterations => _nIterations;

    /// <summary>
    /// Gets the components (right singular vectors).
    /// </summary>
    public double[,]? Components => _components;

    /// <summary>
    /// Gets the singular values.
    /// </summary>
    public double[]? SingularValues => _singularValues;

    /// <summary>
    /// Gets the explained variance for each component.
    /// </summary>
    public double[]? ExplainedVariance => _explainedVariance;

    /// <summary>
    /// Gets the explained variance ratio for each component.
    /// </summary>
    public double[]? ExplainedVarianceRatio => _explainedVarianceRatio;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => true;

    /// <summary>
    /// Creates a new instance of <see cref="TruncatedSVD{T}"/>.
    /// </summary>
    /// <param name="nComponents">Number of components to keep. Defaults to 2.</param>
    /// <param name="nIterations">Number of iterations for randomized SVD. Defaults to 5.</param>
    /// <param name="randomState">Random seed for reproducibility. Defaults to 0.</param>
    /// <param name="columnIndices">The column indices to use, or null for all columns.</param>
    public TruncatedSVD(
        int nComponents = 2,
        int nIterations = 5,
        int randomState = 0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nComponents < 1)
        {
            throw new ArgumentException("Number of components must be at least 1.", nameof(nComponents));
        }

        if (nIterations < 1)
        {
            throw new ArgumentException("Number of iterations must be at least 1.", nameof(nIterations));
        }

        _nComponents = nComponents;
        _nIterations = nIterations;
        _randomState = randomState;
    }

    /// <summary>
    /// Fits TruncatedSVD by computing singular value decomposition.
    /// </summary>
    /// <param name="data">The training data matrix.</param>
    protected override void FitCore(Matrix<T> data)
    {
        _nFeaturesIn = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        int k = Math.Min(_nComponents, Math.Min(n, p));

        // Convert to double array
        var X = new double[n, p];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < p; j++)
            {
                X[i, j] = NumOps.ToDouble(data[i, j]);
            }
        }

        // Compute X^T * X for right singular vectors
        var XTX = new double[p, p];
        for (int i = 0; i < p; i++)
        {
            for (int j = i; j < p; j++)
            {
                double sum = 0;
                for (int l = 0; l < n; l++)
                {
                    sum += X[l, i] * X[l, j];
                }
                XTX[i, j] = sum;
                XTX[j, i] = sum;
            }
        }

        // Compute eigenvalues and eigenvectors of X^T * X
        var (eigenvalues, eigenvectors) = ComputeEigen(XTX, p, k);

        // Sort by eigenvalue descending
        var indices = Enumerable.Range(0, k)
            .OrderByDescending(i => eigenvalues[i])
            .ToArray();

        _singularValues = new double[k];
        _components = new double[k, p];
        _explainedVariance = new double[k];

        double totalVariance = 0;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < p; j++)
            {
                totalVariance += X[i, j] * X[i, j];
            }
        }

        for (int i = 0; i < k; i++)
        {
            double eigenvalue = Math.Max(0, eigenvalues[indices[i]]);
            _singularValues[i] = Math.Sqrt(eigenvalue);
            _explainedVariance[i] = eigenvalue;

            for (int j = 0; j < p; j++)
            {
                _components[i, j] = eigenvectors[indices[i], j];
            }
        }

        // Compute explained variance ratio
        _explainedVarianceRatio = new double[k];
        if (totalVariance > 1e-10)
        {
            for (int i = 0; i < k; i++)
            {
                _explainedVarianceRatio[i] = _explainedVariance[i] / totalVariance;
            }
        }
    }

    private (double[] Eigenvalues, double[,] Eigenvectors) ComputeEigen(double[,] matrix, int n, int k)
    {
        // Power iteration with deflation
        var eigenvalues = new double[k];
        var eigenvectors = new double[k, n];
        var A = (double[,])matrix.Clone();
        var random = new Random(_randomState);

        for (int m = 0; m < k; m++)
        {
            // Initialize random vector
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

            // Power iteration
            for (int iter = 0; iter < 50 + _nIterations * 10; iter++)
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

                // Normalize
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

            eigenvalues[m] = eigenvalue;

            for (int i = 0; i < n; i++)
            {
                eigenvectors[m, i] = v[i];
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
    /// Transforms the data by projecting onto singular vectors.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The transformed data.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_components is null)
        {
            throw new InvalidOperationException("TruncatedSVD has not been fitted.");
        }

        int n = data.Rows;
        int p = data.Columns;
        int k = _components.GetLength(0);
        var result = new T[n, k];

        for (int i = 0; i < n; i++)
        {
            for (int m = 0; m < k; m++)
            {
                double sum = 0;
                for (int j = 0; j < p; j++)
                {
                    sum += NumOps.ToDouble(data[i, j]) * _components[m, j];
                }
                result[i, m] = NumOps.FromDouble(sum);
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
        if (_components is null)
        {
            throw new InvalidOperationException("TruncatedSVD has not been fitted.");
        }

        int n = data.Rows;
        int p = _nFeaturesIn;
        int k = _components.GetLength(0);
        var result = new T[n, p];

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < p; j++)
            {
                double sum = 0;
                for (int m = 0; m < k; m++)
                {
                    sum += NumOps.ToDouble(data[i, m]) * _components[m, j];
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
            names[i] = $"Component{i + 1}";
        }
        return names;
    }
}
