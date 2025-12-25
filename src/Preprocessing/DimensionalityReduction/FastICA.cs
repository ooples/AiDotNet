using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DimensionalityReduction;

/// <summary>
/// Fast Independent Component Analysis (FastICA) for blind source separation.
/// </summary>
/// <remarks>
/// <para>
/// FastICA separates a multivariate signal into additive, independent non-Gaussian
/// components. It is commonly used for blind source separation (e.g., separating
/// mixed audio signals) and feature extraction.
/// </para>
/// <para>
/// The algorithm:
/// 1. Centers and whitens the data
/// 2. Uses fixed-point iteration to find independent components
/// 3. Each component is found by maximizing non-Gaussianity
/// </para>
/// <para><b>For Beginners:</b> Imagine you have recordings from multiple microphones
/// in a room with multiple speakers. ICA can separate the individual speakers:
/// - It finds components that are statistically independent
/// - Unlike PCA which finds uncorrelated components, ICA finds truly independent ones
/// - Works best when source signals are non-Gaussian (most real-world signals are)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class FastICA<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nComponents;
    private readonly ICAAlgorithm _algorithm;
    private readonly ICAFunction _fun;
    private readonly int _maxIterations;
    private readonly double _tolerance;
    private readonly int _randomState;
    private readonly bool _whiten;

    // Fitted parameters
    private double[]? _mean;
    private double[,]? _whitening;
    private double[,]? _components; // Unmixing matrix
    private double[,]? _mixing; // Mixing matrix (inverse of unmixing)
    private int _nFeaturesIn;

    /// <summary>
    /// Gets the number of independent components.
    /// </summary>
    public int NComponents => _nComponents;

    /// <summary>
    /// Gets the algorithm type (parallel or deflation).
    /// </summary>
    public ICAAlgorithm Algorithm => _algorithm;

    /// <summary>
    /// Gets the non-linearity function used.
    /// </summary>
    public ICAFunction Fun => _fun;

    /// <summary>
    /// Gets the unmixing matrix (components).
    /// </summary>
    public double[,]? Components => _components;

    /// <summary>
    /// Gets the mixing matrix.
    /// </summary>
    public double[,]? Mixing => _mixing;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => true;

    /// <summary>
    /// Creates a new instance of <see cref="FastICA{T}"/>.
    /// </summary>
    /// <param name="nComponents">Number of components to extract. Defaults to 2.</param>
    /// <param name="algorithm">Algorithm: parallel or deflation. Defaults to Parallel.</param>
    /// <param name="fun">Non-linearity function: logcosh, exp, or cube. Defaults to LogCosh.</param>
    /// <param name="maxIterations">Maximum iterations. Defaults to 200.</param>
    /// <param name="tolerance">Convergence tolerance. Defaults to 1e-4.</param>
    /// <param name="whiten">Whether to whiten data before ICA. Defaults to true.</param>
    /// <param name="randomState">Random seed. Defaults to 0.</param>
    /// <param name="columnIndices">The column indices to use, or null for all columns.</param>
    public FastICA(
        int nComponents = 2,
        ICAAlgorithm algorithm = ICAAlgorithm.Parallel,
        ICAFunction fun = ICAFunction.LogCosh,
        int maxIterations = 200,
        double tolerance = 1e-4,
        bool whiten = true,
        int randomState = 0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nComponents < 1)
        {
            throw new ArgumentException("Number of components must be at least 1.", nameof(nComponents));
        }

        _nComponents = nComponents;
        _algorithm = algorithm;
        _fun = fun;
        _maxIterations = maxIterations;
        _tolerance = tolerance;
        _whiten = whiten;
        _randomState = randomState;
    }

    /// <summary>
    /// Fits FastICA to extract independent components.
    /// </summary>
    /// <param name="data">The training data matrix.</param>
    protected override void FitCore(Matrix<T> data)
    {
        _nFeaturesIn = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        int k = Math.Min(_nComponents, Math.Min(n, p));

        // Convert to double and center
        var X = new double[n, p];
        _mean = new double[p];

        for (int j = 0; j < p; j++)
        {
            double sum = 0;
            for (int i = 0; i < n; i++)
            {
                X[i, j] = NumOps.ToDouble(data[i, j]);
                sum += X[i, j];
            }
            _mean[j] = sum / n;
        }

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < p; j++)
            {
                X[i, j] -= _mean[j];
            }
        }

        // Whiten the data
        double[,] X1;
        if (_whiten)
        {
            X1 = WhitenData(X, n, p, out _whitening);
        }
        else
        {
            X1 = X;
            _whitening = null;
        }

        // Find independent components
        if (_algorithm == ICAAlgorithm.Deflation)
        {
            _components = FastICADeflation(X1, n, p, k);
        }
        else
        {
            _components = FastICAParallel(X1, n, p, k);
        }

        // Compute mixing matrix
        ComputeMixingMatrix(k);
    }

    private double[,] WhitenData(double[,] X, int n, int p, out double[,] whitening)
    {
        // Compute covariance matrix
        var cov = new double[p, p];
        for (int i = 0; i < p; i++)
        {
            for (int j = i; j < p; j++)
            {
                double sum = 0;
                for (int k = 0; k < n; k++)
                {
                    sum += X[k, i] * X[k, j];
                }
                cov[i, j] = sum / n;
                cov[j, i] = cov[i, j];
            }
        }

        // Eigendecomposition of covariance
        var (eigenvalues, eigenvectors) = ComputeEigen(cov, p, p);

        // Sort by eigenvalue descending
        var indices = Enumerable.Range(0, p)
            .OrderByDescending(i => eigenvalues[i])
            .ToArray();

        // Compute whitening matrix: W = D^(-1/2) @ V^T
        whitening = new double[p, p];
        for (int i = 0; i < p; i++)
        {
            double scale = eigenvalues[indices[i]] > 1e-10 ? 1.0 / Math.Sqrt(eigenvalues[indices[i]]) : 0;
            for (int j = 0; j < p; j++)
            {
                whitening[i, j] = eigenvectors[indices[i], j] * scale;
            }
        }

        // Apply whitening: X_white = X @ W^T
        var X1 = new double[n, p];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < p; j++)
            {
                double sum = 0;
                for (int k = 0; k < p; k++)
                {
                    sum += X[i, k] * whitening[j, k];
                }
                X1[i, j] = sum;
            }
        }

        return X1;
    }

    private double[,] FastICADeflation(double[,] X, int n, int p, int nComponents)
    {
        var W = new double[nComponents, p];
        var random = RandomHelper.CreateSeededRandom(_randomState);

        for (int c = 0; c < nComponents; c++)
        {
            // Initialize random vector
            var w = new double[p];
            double norm = 0;
            for (int i = 0; i < p; i++)
            {
                w[i] = random.NextDouble() - 0.5;
                norm += w[i] * w[i];
            }
            norm = Math.Sqrt(norm);
            for (int i = 0; i < p; i++)
            {
                w[i] /= norm;
            }

            // Fixed-point iteration
            for (int iter = 0; iter < _maxIterations; iter++)
            {
                var wOld = (double[])w.Clone();

                // w_new = E{x * g(w^T * x)} - E{g'(w^T * x)} * w
                var gWx = new double[n];
                var gPrimeWx = new double[n];

                for (int i = 0; i < n; i++)
                {
                    double wx = 0;
                    for (int j = 0; j < p; j++)
                    {
                        wx += w[j] * X[i, j];
                    }
                    var (g, gPrime) = ApplyNonLinearity(wx);
                    gWx[i] = g;
                    gPrimeWx[i] = gPrime;
                }

                // Compute E{x * g(w^T * x)}
                var E_xg = new double[p];
                for (int j = 0; j < p; j++)
                {
                    double sum = 0;
                    for (int i = 0; i < n; i++)
                    {
                        sum += X[i, j] * gWx[i];
                    }
                    E_xg[j] = sum / n;
                }

                // Compute E{g'(w^T * x)}
                double E_gPrime = gPrimeWx.Average();

                // Update w
                for (int j = 0; j < p; j++)
                {
                    w[j] = E_xg[j] - E_gPrime * wOld[j];
                }

                // Orthogonalize against previous components
                for (int prevC = 0; prevC < c; prevC++)
                {
                    double dot = 0;
                    for (int j = 0; j < p; j++)
                    {
                        dot += w[j] * W[prevC, j];
                    }
                    for (int j = 0; j < p; j++)
                    {
                        w[j] -= dot * W[prevC, j];
                    }
                }

                // Normalize
                norm = 0;
                for (int j = 0; j < p; j++)
                {
                    norm += w[j] * w[j];
                }
                norm = Math.Sqrt(norm);
                if (norm < 1e-10) break;

                for (int j = 0; j < p; j++)
                {
                    w[j] /= norm;
                }

                // Check convergence
                double change = 0;
                for (int j = 0; j < p; j++)
                {
                    change += Math.Abs(Math.Abs(w[j]) - Math.Abs(wOld[j]));
                }

                if (change < _tolerance) break;
            }

            // Store component
            for (int j = 0; j < p; j++)
            {
                W[c, j] = w[j];
            }
        }

        return W;
    }

    private double[,] FastICAParallel(double[,] X, int n, int p, int nComponents)
    {
        var W = new double[nComponents, p];
        var random = RandomHelper.CreateSeededRandom(_randomState);

        // Initialize with random orthogonal matrix
        for (int c = 0; c < nComponents; c++)
        {
            for (int j = 0; j < p; j++)
            {
                W[c, j] = random.NextDouble() - 0.5;
            }
        }
        OrthogonalizeMatrix(W, nComponents, p);

        // Fixed-point iteration
        for (int iter = 0; iter < _maxIterations; iter++)
        {
            var WOld = (double[,])W.Clone();

            // Update all components simultaneously
            for (int c = 0; c < nComponents; c++)
            {
                var gWx = new double[n];
                var gPrimeWx = new double[n];

                for (int i = 0; i < n; i++)
                {
                    double wx = 0;
                    for (int j = 0; j < p; j++)
                    {
                        wx += W[c, j] * X[i, j];
                    }
                    var (g, gPrime) = ApplyNonLinearity(wx);
                    gWx[i] = g;
                    gPrimeWx[i] = gPrime;
                }

                // Update
                double E_gPrime = gPrimeWx.Average();
                for (int j = 0; j < p; j++)
                {
                    double E_xg = 0;
                    for (int i = 0; i < n; i++)
                    {
                        E_xg += X[i, j] * gWx[i];
                    }
                    E_xg /= n;
                    W[c, j] = E_xg - E_gPrime * WOld[c, j];
                }
            }

            // Orthogonalize
            OrthogonalizeMatrix(W, nComponents, p);

            // Check convergence
            double maxChange = 0;
            for (int c = 0; c < nComponents; c++)
            {
                for (int j = 0; j < p; j++)
                {
                    double change = Math.Abs(Math.Abs(W[c, j]) - Math.Abs(WOld[c, j]));
                    maxChange = Math.Max(maxChange, change);
                }
            }

            if (maxChange < _tolerance) break;
        }

        return W;
    }

    private void OrthogonalizeMatrix(double[,] W, int rows, int cols)
    {
        // Gram-Schmidt orthogonalization
        for (int i = 0; i < rows; i++)
        {
            // Subtract projections onto previous rows
            for (int j = 0; j < i; j++)
            {
                double dot = 0;
                for (int k = 0; k < cols; k++)
                {
                    dot += W[i, k] * W[j, k];
                }
                for (int k = 0; k < cols; k++)
                {
                    W[i, k] -= dot * W[j, k];
                }
            }

            // Normalize
            double norm = 0;
            for (int k = 0; k < cols; k++)
            {
                norm += W[i, k] * W[i, k];
            }
            norm = Math.Sqrt(norm);
            if (norm > 1e-10)
            {
                for (int k = 0; k < cols; k++)
                {
                    W[i, k] /= norm;
                }
            }
        }
    }

    private (double G, double GPrime) ApplyNonLinearity(double u)
    {
        switch (_fun)
        {
            case ICAFunction.LogCosh:
                double tanh_u = Math.Tanh(u);
                return (tanh_u, 1 - tanh_u * tanh_u);

            case ICAFunction.Exp:
                double exp_u2 = Math.Exp(-u * u / 2);
                return (u * exp_u2, (1 - u * u) * exp_u2);

            case ICAFunction.Cube:
                return (u * u * u, 3 * u * u);

            default:
                return (Math.Tanh(u), 1 - Math.Tanh(u) * Math.Tanh(u));
        }
    }

    private (double[] Eigenvalues, double[,] Eigenvectors) ComputeEigen(double[,] matrix, int n, int k)
    {
        var eigenvalues = new double[k];
        var eigenvectors = new double[k, n];
        var A = (double[,])matrix.Clone();
        var random = RandomHelper.CreateSeededRandom(_randomState);

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

            for (int iter = 0; iter < 50; iter++)
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

            double eigenvalue = 0;
            var Av2 = new double[n];
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    Av2[i] += A[i, j] * v[j];
                }
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

    private void ComputeMixingMatrix(int k)
    {
        if (_components is null) return;

        int p = _components.GetLength(1);
        _mixing = new double[p, k];

        // Mixing = pinv(Unmixing)
        // For orthogonal W, this is just W^T
        for (int i = 0; i < p; i++)
        {
            for (int j = 0; j < k; j++)
            {
                _mixing[i, j] = _components[j, i];
            }
        }

        // If whitening was used, adjust mixing matrix
        if (_whitening is not null)
        {
            // mixing = whitening^T @ mixing
            var newMixing = new double[p, k];
            for (int i = 0; i < p; i++)
            {
                for (int j = 0; j < k; j++)
                {
                    double sum = 0;
                    for (int l = 0; l < p; l++)
                    {
                        sum += _whitening[l, i] * _mixing[l, j];
                    }
                    newMixing[i, j] = sum;
                }
            }
            _mixing = newMixing;
        }
    }

    /// <summary>
    /// Transforms the data to independent components.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The independent components.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_mean is null || _components is null)
        {
            throw new InvalidOperationException("FastICA has not been fitted.");
        }

        if (data.Columns != _nFeaturesIn)
        {
            throw new ArgumentException(
                $"Input data has {data.Columns} features, but FastICA was fitted with {_nFeaturesIn} features.",
                nameof(data));
        }

        int n = data.Rows;
        int p = data.Columns;
        int k = _components.GetLength(0);
        var result = new T[n, k];

        for (int i = 0; i < n; i++)
        {
            // Center
            var x = new double[p];
            for (int j = 0; j < p; j++)
            {
                x[j] = NumOps.ToDouble(data[i, j]) - _mean[j];
            }

            // Whiten if applicable
            if (_whitening is not null)
            {
                var xw = new double[p];
                for (int j = 0; j < p; j++)
                {
                    for (int l = 0; l < p; l++)
                    {
                        xw[j] += _whitening[j, l] * x[l];
                    }
                }
                x = xw;
            }

            // Apply unmixing matrix
            for (int c = 0; c < k; c++)
            {
                double sum = 0;
                for (int j = 0; j < p; j++)
                {
                    sum += _components[c, j] * x[j];
                }
                result[i, c] = NumOps.FromDouble(sum);
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Transforms independent components back to original space.
    /// </summary>
    /// <param name="data">The independent components.</param>
    /// <returns>Reconstructed data in original space.</returns>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        if (_mean is null || _mixing is null)
        {
            throw new InvalidOperationException("FastICA has not been fitted.");
        }

        int n = data.Rows;
        int p = _nFeaturesIn;
        int k = _mixing.GetLength(1);

        if (data.Columns != k)
        {
            throw new ArgumentException(
                $"Input data has {data.Columns} columns, but expected {k} independent components.",
                nameof(data));
        }

        var result = new T[n, p];

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < p; j++)
            {
                double sum = _mean[j];
                for (int c = 0; c < k; c++)
                {
                    sum += _mixing[j, c] * NumOps.ToDouble(data[i, c]);
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
            names[i] = $"IC{i + 1}";
        }
        return names;
    }
}

/// <summary>
/// Specifies the ICA algorithm type.
/// </summary>
public enum ICAAlgorithm
{
    /// <summary>
    /// Parallel (symmetric) algorithm - all components extracted simultaneously.
    /// </summary>
    Parallel,

    /// <summary>
    /// Deflation algorithm - components extracted one at a time.
    /// </summary>
    Deflation
}

/// <summary>
/// Specifies the non-linearity function for ICA.
/// </summary>
public enum ICAFunction
{
    /// <summary>
    /// Log cosh function: g(u) = tanh(u). Good general-purpose choice.
    /// </summary>
    LogCosh,

    /// <summary>
    /// Exponential function: g(u) = u * exp(-u²/2). Good for super-Gaussian sources.
    /// </summary>
    Exp,

    /// <summary>
    /// Cubic function: g(u) = u³. Fast but less robust.
    /// </summary>
    Cube
}
