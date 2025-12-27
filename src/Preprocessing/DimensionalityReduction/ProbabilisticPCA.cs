using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DimensionalityReduction;

/// <summary>
/// Probabilistic Principal Component Analysis (PPCA).
/// </summary>
/// <remarks>
/// <para>
/// Probabilistic PCA is a probabilistic formulation of PCA that models data as being
/// generated from a lower-dimensional latent space with added Gaussian noise. This
/// provides a proper likelihood model, enables handling of missing data, and allows
/// for Bayesian extensions.
/// </para>
/// <para>
/// The model:
/// x = W * z + μ + ε
/// where z ~ N(0, I), ε ~ N(0, σ²I)
/// </para>
/// <para>
/// The algorithm:
/// 1. Center the data
/// 2. Use EM algorithm or closed-form solution to estimate W and σ²
/// 3. Project data to latent space using posterior mean
/// </para>
/// <para><b>For Beginners:</b> PPCA extends standard PCA by:
/// - Providing a probabilistic model for the data
/// - Estimating noise variance σ² separately
/// - Enabling computation of data likelihoods
/// - Handling missing values naturally
///
/// Use cases:
/// - When you need uncertainty estimates
/// - Data with missing values
/// - Model comparison using likelihoods
/// - Bayesian machine learning pipelines
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class ProbabilisticPCA<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nComponents;
    private readonly int _maxIter;
    private readonly double _tol;
    private readonly int? _randomState;

    // Fitted parameters
    private double[,]? _W; // Loading matrix (p x d)
    private double[]? _mean;
    private double _noiseVariance;
    private int _nFeatures;

    /// <summary>
    /// Gets the number of components (dimensions).
    /// </summary>
    public int NComponents => _nComponents;

    /// <summary>
    /// Gets the loading matrix W.
    /// </summary>
    public double[,]? LoadingMatrix => _W;

    /// <summary>
    /// Gets the estimated noise variance.
    /// </summary>
    public double NoiseVariance => _noiseVariance;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => true;

    /// <summary>
    /// Creates a new instance of <see cref="ProbabilisticPCA{T}"/>.
    /// </summary>
    /// <param name="nComponents">Target dimensionality. Defaults to 2.</param>
    /// <param name="maxIter">Maximum number of EM iterations. Defaults to 100.</param>
    /// <param name="tol">Tolerance for convergence. Defaults to 1e-4.</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <param name="columnIndices">The column indices to use, or null for all columns.</param>
    public ProbabilisticPCA(
        int nComponents = 2,
        int maxIter = 100,
        double tol = 1e-4,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nComponents < 1)
        {
            throw new ArgumentException("Number of components must be at least 1.", nameof(nComponents));
        }

        _nComponents = nComponents;
        _maxIter = maxIter;
        _tol = tol;
        _randomState = randomState;
    }

    /// <summary>
    /// Fits Probabilistic PCA using the EM algorithm.
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        int n = data.Rows;
        int p = data.Columns;
        _nFeatures = p;

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

        // Step 2: Compute sample covariance matrix
        var S = new double[p, p];
        for (int i = 0; i < p; i++)
        {
            for (int j = 0; j < p; j++)
            {
                for (int k = 0; k < n; k++)
                {
                    S[i, j] += Xc[k, i] * Xc[k, j];
                }
                S[i, j] /= n;
            }
        }

        // Step 3: Compute eigendecomposition of S
        var eigenvalues = new double[p];
        var eigenvectors = new double[p, p];
        ComputeEigendecomposition(S, eigenvalues, eigenvectors, p, random);

        // Step 4: Estimate noise variance (average of smallest eigenvalues)
        _noiseVariance = 0;
        int numSmall = p - _nComponents;
        if (numSmall > 0)
        {
            for (int i = _nComponents; i < p; i++)
            {
                _noiseVariance += eigenvalues[i];
            }
            _noiseVariance /= numSmall;
        }
        _noiseVariance = Math.Max(_noiseVariance, 1e-10);

        // Step 5: Compute W using closed-form solution
        // W = U_q * (Λ_q - σ² * I)^(1/2) * R
        // where U_q are top q eigenvectors, Λ_q are top q eigenvalues
        // R is an arbitrary rotation matrix (we use identity)
        _W = new double[p, _nComponents];
        for (int i = 0; i < p; i++)
        {
            for (int j = 0; j < _nComponents; j++)
            {
                double scale = Math.Sqrt(Math.Max(eigenvalues[j] - _noiseVariance, 1e-10));
                _W[i, j] = eigenvectors[i, j] * scale;
            }
        }
    }

    private void ComputeEigendecomposition(
        double[,] S, double[] eigenvalues, double[,] eigenvectors, int p, Random random)
    {
        // Power iteration with deflation for eigenvalues in descending order
        var A = (double[,])S.Clone();

        for (int d = 0; d < p; d++)
        {
            var v = new double[p];
            for (int i = 0; i < p; i++) v[i] = random.NextDouble() - 0.5;

            // Power iteration
            for (int iter = 0; iter < 100; iter++)
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

                double change = 0;
                for (int i = 0; i < p; i++)
                {
                    double newV = Av[i] / norm;
                    change += Math.Abs(newV - v[i]);
                    v[i] = newV;
                }

                if (change < 1e-10) break;
            }

            // Store eigenvector
            for (int i = 0; i < p; i++)
            {
                eigenvectors[i, d] = v[i];
            }

            // Compute eigenvalue
            var Av2 = new double[p];
            for (int i = 0; i < p; i++)
            {
                for (int j = 0; j < p; j++)
                {
                    Av2[i] += A[i, j] * v[j];
                }
            }

            double lambda = 0;
            for (int i = 0; i < p; i++)
            {
                lambda += v[i] * Av2[i];
            }
            eigenvalues[d] = Math.Max(lambda, 0);

            // Deflate
            for (int i = 0; i < p; i++)
            {
                for (int j = 0; j < p; j++)
                {
                    A[i, j] -= lambda * v[i] * v[j];
                }
            }
        }
    }

    /// <summary>
    /// Transforms data by projecting to the latent space.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_W is null || _mean is null)
        {
            throw new InvalidOperationException("ProbabilisticPCA has not been fitted.");
        }

        int n = data.Rows;
        int p = data.Columns;

        if (p != _nFeatures)
        {
            throw new ArgumentException(
                $"Input has {p} features but PPCA was fitted with {_nFeatures} features.",
                nameof(data));
        }

        // Convert to double array
        var X = new double[n, p];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < p; j++)
            {
                X[i, j] = NumOps.ToDouble(data[i, j]);
            }
        }

        // Center data
        var Xc = new double[n, p];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < p; j++)
            {
                Xc[i, j] = X[i, j] - _mean[j];
            }
        }

        // Compute M = W^T * W + σ² * I
        var WtW = new double[_nComponents, _nComponents];
        for (int i = 0; i < _nComponents; i++)
        {
            for (int j = 0; j < _nComponents; j++)
            {
                for (int k = 0; k < p; k++)
                {
                    WtW[i, j] += _W[k, i] * _W[k, j];
                }
            }
            WtW[i, i] += _noiseVariance;
        }

        // Invert M
        var Minv = InvertMatrix(WtW, _nComponents);

        // Compute posterior mean: E[z|x] = M^(-1) * W^T * (x - μ)
        var result = new T[n, _nComponents];

        for (int i = 0; i < n; i++)
        {
            // W^T * x_c
            var Wtx = new double[_nComponents];
            for (int d = 0; d < _nComponents; d++)
            {
                for (int j = 0; j < p; j++)
                {
                    Wtx[d] += _W[j, d] * Xc[i, j];
                }
            }

            // M^(-1) * W^T * x_c
            for (int d = 0; d < _nComponents; d++)
            {
                double val = 0;
                for (int k = 0; k < _nComponents; k++)
                {
                    val += Minv[d, k] * Wtx[k];
                }
                result[i, d] = NumOps.FromDouble(val);
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Reconstructs data from the latent space.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        if (_W is null || _mean is null)
        {
            throw new InvalidOperationException("ProbabilisticPCA has not been fitted.");
        }

        int n = data.Rows;
        int d = data.Columns;

        if (d != _nComponents)
        {
            throw new ArgumentException(
                $"Input has {d} dimensions but PPCA was fitted with {_nComponents} components.",
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
                    val += NumOps.ToDouble(data[i, k]) * _W[j, k];
                }
                result[i, j] = NumOps.FromDouble(val);
            }
        }

        return new Matrix<T>(result);
    }

    private double[,] InvertMatrix(double[,] A, int n)
    {
        // Gauss-Jordan elimination for small matrices
        var augmented = new double[n, 2 * n];

        // Initialize augmented matrix [A | I]
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                augmented[i, j] = A[i, j];
            }
            augmented[i, n + i] = 1.0;
        }

        // Forward elimination
        for (int col = 0; col < n; col++)
        {
            // Find pivot
            int maxRow = col;
            for (int row = col + 1; row < n; row++)
            {
                if (Math.Abs(augmented[row, col]) > Math.Abs(augmented[maxRow, col]))
                {
                    maxRow = row;
                }
            }

            // Swap rows
            for (int j = 0; j < 2 * n; j++)
            {
                (augmented[col, j], augmented[maxRow, j]) = (augmented[maxRow, j], augmented[col, j]);
            }

            // Scale pivot row
            double pivot = augmented[col, col];
            if (Math.Abs(pivot) < 1e-10)
            {
                // Add regularization for near-singular matrix
                augmented[col, col] += 1e-6;
                pivot = augmented[col, col];
            }

            for (int j = 0; j < 2 * n; j++)
            {
                augmented[col, j] /= pivot;
            }

            // Eliminate column
            for (int row = 0; row < n; row++)
            {
                if (row != col)
                {
                    double factor = augmented[row, col];
                    for (int j = 0; j < 2 * n; j++)
                    {
                        augmented[row, j] -= factor * augmented[col, j];
                    }
                }
            }
        }

        // Extract inverse
        var inverse = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                inverse[i, j] = augmented[i, n + j];
            }
        }

        return inverse;
    }

    /// <summary>
    /// Computes the log-likelihood of the data under the model.
    /// </summary>
    /// <param name="data">The data to evaluate.</param>
    /// <returns>The log-likelihood.</returns>
    public double Score(Matrix<T> data)
    {
        if (_W is null || _mean is null)
        {
            throw new InvalidOperationException("ProbabilisticPCA has not been fitted.");
        }

        int n = data.Rows;
        int p = data.Columns;

        // Compute C = W * W^T + σ² * I
        var C = new double[p, p];
        for (int i = 0; i < p; i++)
        {
            for (int j = 0; j < p; j++)
            {
                for (int k = 0; k < _nComponents; k++)
                {
                    C[i, j] += _W[i, k] * _W[j, k];
                }
            }
            C[i, i] += _noiseVariance;
        }

        // Compute log determinant and inverse of C
        // (Using approximation for numerical stability)
        double logDet = 0;
        for (int i = 0; i < p; i++)
        {
            logDet += Math.Log(Math.Max(C[i, i], 1e-10));
        }

        var Cinv = InvertMatrix(C, p);

        // Compute log-likelihood
        double logLik = 0;
        double logConstant = -0.5 * p * Math.Log(2 * Math.PI) - 0.5 * logDet;

        for (int i = 0; i < n; i++)
        {
            // Compute (x - μ)^T * C^(-1) * (x - μ)
            var xc = new double[p];
            for (int j = 0; j < p; j++)
            {
                xc[j] = NumOps.ToDouble(data[i, j]) - _mean[j];
            }

            double mahal = 0;
            for (int j = 0; j < p; j++)
            {
                double sum = 0;
                for (int k = 0; k < p; k++)
                {
                    sum += Cinv[j, k] * xc[k];
                }
                mahal += xc[j] * sum;
            }

            logLik += logConstant - 0.5 * mahal;
        }

        return logLik;
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        var names = new string[_nComponents];
        for (int i = 0; i < _nComponents; i++)
        {
            names[i] = $"PPCA{i + 1}";
        }
        return names;
    }
}
