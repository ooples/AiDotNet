using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DimensionalityReduction;

/// <summary>
/// Factor Analysis for dimensionality reduction with noise modeling.
/// </summary>
/// <remarks>
/// <para>
/// Factor Analysis assumes that the observed data X is generated from a set of
/// latent factors F plus feature-specific noise: X = F * W + noise.
/// Unlike PCA, Factor Analysis explicitly models unique variance (noise) for
/// each feature.
/// </para>
/// <para>
/// The model assumes:
/// - X = W * F + ε
/// - Where ε ~ N(0, Ψ) and Ψ is diagonal (unique variances)
/// - F ~ N(0, I) are the latent factors
/// </para>
/// <para><b>For Beginners:</b> Factor Analysis is like PCA but smarter about noise:
/// - PCA assumes all variance is signal
/// - Factor Analysis separates "common variance" (shared across features)
///   from "unique variance" (noise specific to each feature)
/// - Use when your features have different noise levels
/// - Popular in psychology, social sciences, and survey analysis
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class FactorAnalysis<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nComponents;
    private readonly int _maxIter;
    private readonly double _tol;
    private readonly FactorRotation _rotation;
    private readonly int? _randomState;

    // Fitted parameters
    private double[]? _mean;
    private double[,]? _components; // Factor loadings (features x factors)
    private double[]? _noiseVariance; // Unique variances (psi)
    private double[,]? _rotatedComponents;
    private int _nFeaturesIn;

    /// <summary>
    /// Gets the number of factors.
    /// </summary>
    public int NComponents => _nComponents;

    /// <summary>
    /// Gets the rotation method.
    /// </summary>
    public FactorRotation Rotation => _rotation;

    /// <summary>
    /// Gets the mean of each feature.
    /// </summary>
    public double[]? Mean => _mean;

    /// <summary>
    /// Gets the factor loadings (each column is a factor).
    /// </summary>
    public double[,]? Components => _rotation == FactorRotation.None ? _components : _rotatedComponents;

    /// <summary>
    /// Gets the unique variance (noise) for each feature.
    /// </summary>
    public double[]? NoiseVariance => _noiseVariance;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="FactorAnalysis{T}"/>.
    /// </summary>
    /// <param name="nComponents">Number of latent factors. Defaults to 2.</param>
    /// <param name="maxIter">Maximum EM iterations. Defaults to 1000.</param>
    /// <param name="tol">Convergence tolerance. Defaults to 1e-4.</param>
    /// <param name="rotation">Factor rotation method. Defaults to None.</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <param name="columnIndices">The column indices to use, or null for all columns.</param>
    public FactorAnalysis(
        int nComponents = 2,
        int maxIter = 1000,
        double tol = 1e-4,
        FactorRotation rotation = FactorRotation.None,
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
        _rotation = rotation;
        _randomState = randomState;
    }

    /// <summary>
    /// Fits Factor Analysis using the EM algorithm.
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        _nFeaturesIn = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        int k = Math.Min(_nComponents, p);

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

        // Compute sample covariance
        var S = new double[p, p];
        for (int i = 0; i < p; i++)
        {
            for (int j = i; j < p; j++)
            {
                double sum = 0;
                for (int s = 0; s < n; s++)
                {
                    sum += centered[s, i] * centered[s, j];
                }
                S[i, j] = sum / (n - 1);
                S[j, i] = S[i, j];
            }
        }

        // Initialize using PCA
        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSeededRandom(42);

        // Get initial factor loadings from PCA
        var (eigenvalues, eigenvectors) = ComputeEigen(S, p);

        // Sort by eigenvalue descending
        var indices = Enumerable.Range(0, p)
            .OrderByDescending(i => eigenvalues[i])
            .ToArray();

        // Initialize loadings
        _components = new double[p, k];
        for (int j = 0; j < k; j++)
        {
            int idx = indices[j];
            double scale = Math.Sqrt(Math.Max(eigenvalues[idx], 1e-10));
            for (int i = 0; i < p; i++)
            {
                _components[i, j] = eigenvectors[idx, i] * scale;
            }
        }

        // Initialize noise variances
        _noiseVariance = new double[p];
        for (int i = 0; i < p; i++)
        {
            double communality = 0;
            for (int j = 0; j < k; j++)
            {
                communality += _components[i, j] * _components[i, j];
            }
            _noiseVariance[i] = Math.Max(S[i, i] - communality, 1e-6);
        }

        // EM algorithm
        for (int iter = 0; iter < _maxIter; iter++)
        {
            var oldLoadings = new double[p, k];
            for (int i = 0; i < p; i++)
            {
                for (int j = 0; j < k; j++)
                {
                    oldLoadings[i, j] = _components[i, j];
                }
            }

            // E-step: Compute expected latent factors
            // E[F|X] = (W^T Ψ^-1 W + I)^-1 W^T Ψ^-1 X
            var WtPsiInv = new double[k, p];
            for (int j = 0; j < k; j++)
            {
                for (int i = 0; i < p; i++)
                {
                    WtPsiInv[j, i] = _components[i, j] / _noiseVariance[i];
                }
            }

            // WtPsiInvW + I
            var M = new double[k, k];
            for (int i = 0; i < k; i++)
            {
                for (int j = 0; j < k; j++)
                {
                    double sum = 0;
                    for (int l = 0; l < p; l++)
                    {
                        sum += WtPsiInv[i, l] * _components[l, j];
                    }
                    M[i, j] = sum;
                    if (i == j)
                    {
                        M[i, j] += 1.0;
                    }
                }
            }

            var MInv = InvertMatrix(M, k);

            // Compute expected factor scores E[F|X]
            var expectedF = new double[n, k];
            for (int s = 0; s < n; s++)
            {
                // WtPsiInv * x
                var temp = new double[k];
                for (int j = 0; j < k; j++)
                {
                    double sum = 0;
                    for (int i = 0; i < p; i++)
                    {
                        sum += WtPsiInv[j, i] * centered[s, i];
                    }
                    temp[j] = sum;
                }

                // MInv * temp
                for (int j = 0; j < k; j++)
                {
                    double sum = 0;
                    for (int l = 0; l < k; l++)
                    {
                        sum += MInv[j, l] * temp[l];
                    }
                    expectedF[s, j] = sum;
                }
            }

            // E[FF^T|X] = MInv + E[F|X]E[F|X]^T
            var expectedFFt = new double[k, k];
            for (int i = 0; i < k; i++)
            {
                for (int j = 0; j < k; j++)
                {
                    double sum = n * MInv[i, j];
                    for (int s = 0; s < n; s++)
                    {
                        sum += expectedF[s, i] * expectedF[s, j];
                    }
                    expectedFFt[i, j] = sum;
                }
            }

            // E[XF^T]
            var expectedXFt = new double[p, k];
            for (int i = 0; i < p; i++)
            {
                for (int j = 0; j < k; j++)
                {
                    double sum = 0;
                    for (int s = 0; s < n; s++)
                    {
                        sum += centered[s, i] * expectedF[s, j];
                    }
                    expectedXFt[i, j] = sum;
                }
            }

            // M-step: Update loadings and noise variances
            // W = E[XF^T] * E[FF^T]^-1
            var expectedFFtInv = InvertMatrix(expectedFFt, k);

            for (int i = 0; i < p; i++)
            {
                for (int j = 0; j < k; j++)
                {
                    double sum = 0;
                    for (int l = 0; l < k; l++)
                    {
                        sum += expectedXFt[i, l] * expectedFFtInv[l, j];
                    }
                    _components[i, j] = sum;
                }
            }

            // Update noise variances
            // Ψ = diag(S - W * E[XF^T]^T / n)
            for (int i = 0; i < p; i++)
            {
                double reconstructed = 0;
                for (int j = 0; j < k; j++)
                {
                    reconstructed += _components[i, j] * expectedXFt[i, j];
                }
                _noiseVariance[i] = Math.Max(S[i, i] - reconstructed / n, 1e-6);
            }

            // Check convergence
            double maxChange = 0;
            for (int i = 0; i < p; i++)
            {
                for (int j = 0; j < k; j++)
                {
                    double change = Math.Abs(_components[i, j] - oldLoadings[i, j]);
                    maxChange = Math.Max(maxChange, change);
                }
            }

            if (maxChange < _tol)
            {
                break;
            }
        }

        // Apply rotation if requested
        if (_rotation != FactorRotation.None)
        {
            _rotatedComponents = ApplyRotation(_components, p, k);
        }
    }

    private double[,] ApplyRotation(double[,] loadings, int p, int k)
    {
        var rotated = new double[p, k];

        switch (_rotation)
        {
            case FactorRotation.Varimax:
                return ApplyVarimax(loadings, p, k);
            case FactorRotation.Quartimax:
                return ApplyQuartimax(loadings, p, k);
            default:
                // Copy original
                for (int i = 0; i < p; i++)
                {
                    for (int j = 0; j < k; j++)
                    {
                        rotated[i, j] = loadings[i, j];
                    }
                }
                return rotated;
        }
    }

    private double[,] ApplyVarimax(double[,] loadings, int p, int k)
    {
        // Varimax rotation maximizes variance of squared loadings
        var rotated = (double[,])loadings.Clone();
        double gamma = 1.0; // Varimax uses gamma = 1

        for (int iter = 0; iter < 100; iter++)
        {
            for (int i = 0; i < k - 1; i++)
            {
                for (int j = i + 1; j < k; j++)
                {
                    // Compute rotation angle for columns i and j
                    double a = 0, b = 0, c = 0, d = 0;

                    for (int l = 0; l < p; l++)
                    {
                        double u = rotated[l, i] * rotated[l, i] - rotated[l, j] * rotated[l, j];
                        double v = 2 * rotated[l, i] * rotated[l, j];
                        a += u;
                        b += v;
                        c += u * u - v * v;
                        d += 2 * u * v;
                    }

                    double num = d - 2 * gamma * a * b / p;
                    double den = c - gamma * (a * a - b * b) / p;
                    double angle = 0.25 * Math.Atan2(num, den);

                    // Rotate columns i and j
                    double cos = Math.Cos(angle);
                    double sin = Math.Sin(angle);

                    for (int l = 0; l < p; l++)
                    {
                        double temp = rotated[l, i] * cos + rotated[l, j] * sin;
                        rotated[l, j] = -rotated[l, i] * sin + rotated[l, j] * cos;
                        rotated[l, i] = temp;
                    }
                }
            }
        }

        return rotated;
    }

    private double[,] ApplyQuartimax(double[,] loadings, int p, int k)
    {
        // Quartimax is like varimax but with gamma = 0
        var rotated = (double[,])loadings.Clone();

        for (int iter = 0; iter < 100; iter++)
        {
            for (int i = 0; i < k - 1; i++)
            {
                for (int j = i + 1; j < k; j++)
                {
                    double c = 0, d = 0;

                    for (int l = 0; l < p; l++)
                    {
                        double u = rotated[l, i] * rotated[l, i] - rotated[l, j] * rotated[l, j];
                        double v = 2 * rotated[l, i] * rotated[l, j];
                        c += u * u - v * v;
                        d += 2 * u * v;
                    }

                    double angle = 0.25 * Math.Atan2(d, c);

                    double cos = Math.Cos(angle);
                    double sin = Math.Sin(angle);

                    for (int l = 0; l < p; l++)
                    {
                        double temp = rotated[l, i] * cos + rotated[l, j] * sin;
                        rotated[l, j] = -rotated[l, i] * sin + rotated[l, j] * cos;
                        rotated[l, i] = temp;
                    }
                }
            }
        }

        return rotated;
    }

    private (double[] Eigenvalues, double[,] Eigenvectors) ComputeEigen(double[,] matrix, int n)
    {
        var eigenvalues = new double[n];
        var eigenvectors = new double[n, n];
        var A = (double[,])matrix.Clone();

        for (int k = 0; k < n; k++)
        {
            var v = new double[n];
            for (int i = 0; i < n; i++)
            {
                v[i] = 1.0 / Math.Sqrt(n);
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

                double norm = 0;
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

            eigenvalues[k] = Math.Max(0, eigenvalue);
            for (int i = 0; i < n; i++)
            {
                eigenvectors[k, i] = v[i];
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

    private static double[,] InvertMatrix(double[,] matrix, int n)
    {
        var result = new double[n, n];
        var temp = new double[n, 2 * n];

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                temp[i, j] = matrix[i, j];
                temp[i, j + n] = (i == j) ? 1.0 : 0.0;
            }
        }

        for (int i = 0; i < n; i++)
        {
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

            if (maxRow != i)
            {
                for (int j = 0; j < 2 * n; j++)
                {
                    (temp[i, j], temp[maxRow, j]) = (temp[maxRow, j], temp[i, j]);
                }
            }

            double pivot = temp[i, i];
            if (Math.Abs(pivot) < 1e-10)
            {
                pivot = 1e-10;
            }

            for (int j = 0; j < 2 * n; j++)
            {
                temp[i, j] /= pivot;
            }

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
    /// Transforms the data by computing factor scores.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_mean is null || _components is null || _noiseVariance is null)
        {
            throw new InvalidOperationException("FactorAnalysis has not been fitted.");
        }

        int n = data.Rows;
        int p = data.Columns;
        int k = _components.GetLength(1);

        var loadings = _rotation == FactorRotation.None ? _components : _rotatedComponents!;

        // Compute factor scores using Bartlett method: F = (W^T Ψ^-1 W)^-1 W^T Ψ^-1 X
        var WtPsiInv = new double[k, p];
        for (int j = 0; j < k; j++)
        {
            for (int i = 0; i < p; i++)
            {
                WtPsiInv[j, i] = loadings[i, j] / _noiseVariance[i];
            }
        }

        var WtPsiInvW = new double[k, k];
        for (int i = 0; i < k; i++)
        {
            for (int j = 0; j < k; j++)
            {
                double sum = 0;
                for (int l = 0; l < p; l++)
                {
                    sum += WtPsiInv[i, l] * loadings[l, j];
                }
                WtPsiInvW[i, j] = sum;
            }
        }

        var WtPsiInvWInv = InvertMatrix(WtPsiInvW, k);

        var result = new T[n, k];
        for (int s = 0; s < n; s++)
        {
            // Center the sample
            var centered = new double[p];
            for (int j = 0; j < p; j++)
            {
                centered[j] = NumOps.ToDouble(data[s, j]) - _mean[j];
            }

            // WtPsiInv * centered
            var temp = new double[k];
            for (int j = 0; j < k; j++)
            {
                double sum = 0;
                for (int i = 0; i < p; i++)
                {
                    sum += WtPsiInv[j, i] * centered[i];
                }
                temp[j] = sum;
            }

            // WtPsiInvWInv * temp
            for (int j = 0; j < k; j++)
            {
                double sum = 0;
                for (int l = 0; l < k; l++)
                {
                    sum += WtPsiInvWInv[j, l] * temp[l];
                }
                result[s, j] = NumOps.FromDouble(sum);
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Inverse transformation is not supported for Factor Analysis.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("FactorAnalysis does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        int k = _components?.GetLength(1) ?? _nComponents;
        var names = new string[k];
        for (int i = 0; i < k; i++)
        {
            names[i] = $"Factor{i + 1}";
        }
        return names;
    }
}

/// <summary>
/// Specifies the rotation method for Factor Analysis.
/// </summary>
public enum FactorRotation
{
    /// <summary>
    /// No rotation (use raw factor loadings).
    /// </summary>
    None,

    /// <summary>
    /// Varimax rotation - maximizes variance of squared loadings within each factor.
    /// Produces simpler structure with high loadings on few variables per factor.
    /// </summary>
    Varimax,

    /// <summary>
    /// Quartimax rotation - maximizes variance of squared loadings within each variable.
    /// Tends to produce a general factor.
    /// </summary>
    Quartimax
}
