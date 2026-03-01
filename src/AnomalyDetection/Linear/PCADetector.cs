using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.Linear;

/// <summary>
/// Detects anomalies using Principal Component Analysis (PCA) reconstruction error.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> PCA-based anomaly detection works by projecting data onto
/// the main directions of variation (principal components) and measuring how well
/// points can be reconstructed. Anomalies have high reconstruction error.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Fit PCA on training data to find principal components
/// 2. Project each point onto these components
/// 3. Reconstruct the point from the projection
/// 4. Measure reconstruction error (anomaly score)
/// </para>
/// <para>
/// <b>When to use:</b>
/// - Linear relationships in data
/// - When anomalies deviate from the main data structure
/// - High-dimensional data that can be compressed
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Components: min(n_samples, n_features) or specified
/// - Variance threshold: 0.95 (keep 95% of variance)
/// - Contamination: 0.1 (10%)
/// </para>
/// </remarks>
public class PCADetector<T> : AnomalyDetectorBase<T>
{
    /// <summary>Eigenvalues below this threshold are treated as zero to avoid division by near-zero values in Mahalanobis distance.</summary>
    private const double EigenvalueFloor = 1e-10;

    private readonly int? _nComponents;
    private readonly double _varianceThreshold;
    private int _fittedComponents;
    private Vector<T>? _mean;
    private Matrix<T>? _components;
    private Vector<T>? _explainedVariance;

    /// <summary>
    /// Gets the number of components used.
    /// </summary>
    public int NComponents => _fittedComponents;

    /// <summary>
    /// Gets the variance threshold.
    /// </summary>
    public double VarianceThreshold => _varianceThreshold;

    /// <summary>
    /// Creates a new PCA anomaly detector.
    /// </summary>
    /// <param name="nComponents">
    /// Number of principal components. If null, determined by variance threshold.
    /// </param>
    /// <param name="varianceThreshold">
    /// Proportion of variance to retain. Default is 0.95 (95%).
    /// Used when nComponents is not specified.
    /// </param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public PCADetector(int? nComponents = null, double varianceThreshold = 0.95,
        double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (nComponents.HasValue && nComponents.Value < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(nComponents),
                "NComponents must be at least 1 if specified.");
        }

        if (varianceThreshold <= 0 || varianceThreshold > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(varianceThreshold),
                "VarianceThreshold must be between 0 (exclusive) and 1 (inclusive). Recommended is 0.95.");
        }

        _nComponents = nComponents;
        _varianceThreshold = varianceThreshold;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        int n = X.Rows;
        int d = X.Columns;

        // Compute mean
        _mean = new Vector<T>(d);
        for (int j = 0; j < d; j++)
        {
            T sum = NumOps.Zero;
            for (int i = 0; i < n; i++)
            {
                sum = NumOps.Add(sum, X[i, j]);
            }
            _mean[j] = NumOps.Divide(sum, NumOps.FromDouble(n));
        }

        // Center data
        var centered = new Matrix<T>(n, d);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < d; j++)
            {
                centered[i, j] = NumOps.Subtract(X[i, j], _mean[j]);
            }
        }

        // Compute covariance matrix
        var covariance = ComputeCovariance(centered, n, d);

        // Compute eigendecomposition (simplified power iteration)
        var (eigenvalues, eigenvectors) = ComputeEigenDecomposition(covariance, d);

        // Determine number of components
        _fittedComponents = _nComponents ?? DetermineComponents(eigenvalues);
        _fittedComponents = Math.Min(_fittedComponents, d);

        // Store components (top eigenvectors)
        _components = new Matrix<T>(_fittedComponents, d);
        _explainedVariance = new Vector<T>(_fittedComponents);

        for (int c = 0; c < _fittedComponents; c++)
        {
            _explainedVariance[c] = eigenvalues[c];
            for (int j = 0; j < d; j++)
            {
                _components[c, j] = eigenvectors[c, j];
            }
        }

        // Calculate scores for training data to set threshold
        var trainingScores = ScoreAnomaliesInternal(X);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
    }

    /// <inheritdoc/>
    public override Vector<T> ScoreAnomalies(Matrix<T> X)
    {
        EnsureFitted();
        return ScoreAnomaliesInternal(X);
    }

    private Vector<T> ScoreAnomaliesInternal(Matrix<T> X)
    {
        ValidateInput(X);

        var scores = new Vector<T>(X.Rows);
        int d = X.Columns;

        for (int i = 0; i < X.Rows; i++)
        {
            // Center the point
            var centered = new Vector<T>(d);
            for (int j = 0; j < d; j++)
            {
                centered[j] = NumOps.Subtract(X[i, j], _mean![j]);
            }

            // Project onto components
            var projected = new Vector<T>(_fittedComponents);
            for (int c = 0; c < _fittedComponents; c++)
            {
                T dot = NumOps.Zero;
                for (int j = 0; j < d; j++)
                {
                    dot = NumOps.Add(dot, NumOps.Multiply(centered[j], _components![c, j]));
                }
                projected[c] = dot;
            }

            // Reconstruct from projection
            var reconstructed = new Vector<T>(d);
            for (int j = 0; j < d; j++)
            {
                T sum = NumOps.Zero;
                for (int c = 0; c < _fittedComponents; c++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(projected[c], _components![c, j]));
                }
                reconstructed[j] = sum;
            }

            // Compute reconstruction error (residual from unretained components)
            double reconstructionError = 0;
            for (int j = 0; j < d; j++)
            {
                T diff = NumOps.Subtract(centered[j], reconstructed[j]);
                reconstructionError += NumOps.ToDouble(NumOps.Multiply(diff, diff));
            }

            // Compute Mahalanobis distance in PCA space (distance along retained components,
            // weighted by inverse eigenvalues). This catches outliers that lie far from the mean
            // along principal component directions, even when reconstruction error is low.
            double mahalanobis = 0;
            for (int c = 0; c < _fittedComponents; c++)
            {
                double eigenvalue = NumOps.ToDouble(_explainedVariance![c]);
                if (eigenvalue > EigenvalueFloor)
                {
                    double proj = NumOps.ToDouble(projected[c]);
                    mahalanobis += (proj * proj) / eigenvalue;
                }
            }

            // Combined score: Hotelling's T² (Mahalanobis in PC space) + SPE/Q (reconstruction error).
            // Both are kept as squared statistics, consistent with standard MSPC-based PCA anomaly scoring.
            scores[i] = NumOps.FromDouble(mahalanobis + reconstructionError);
        }

        return scores;
    }

    private Matrix<T> ComputeCovariance(Matrix<T> centered, int n, int d)
    {
        var cov = new Matrix<T>(d, d);

        for (int i = 0; i < d; i++)
        {
            for (int j = i; j < d; j++)
            {
                T sum = NumOps.Zero;
                for (int k = 0; k < n; k++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(centered[k, i], centered[k, j]));
                }
                cov[i, j] = NumOps.Divide(sum, NumOps.FromDouble(n - 1));
                cov[j, i] = cov[i, j];
            }
        }

        return cov;
    }

    private (Vector<T> eigenvalues, Matrix<T> eigenvectors) ComputeEigenDecomposition(Matrix<T> matrix, int d)
    {
        // Simplified power iteration method for eigendecomposition
        var random = new Random(_randomSeed);
        var eigenvalues = new Vector<T>(d);
        var eigenvectors = new Matrix<T>(d, d);
        var A = CopyMatrix(matrix, d);

        for (int e = 0; e < d; e++)
        {
            // Initialize random vector
            var v = new Vector<T>(d);
            for (int j = 0; j < d; j++)
            {
                v[j] = NumOps.FromDouble(random.NextDouble() - 0.5);
            }

            // Power iteration
            for (int iter = 0; iter < 100; iter++)
            {
                // Multiply A * v
                var Av = new Vector<T>(d);
                for (int i = 0; i < d; i++)
                {
                    T sum = NumOps.Zero;
                    for (int j = 0; j < d; j++)
                    {
                        sum = NumOps.Add(sum, NumOps.Multiply(A[i, j], v[j]));
                    }
                    Av[i] = sum;
                }

                // Normalize
                T norm = NumOps.Zero;
                for (int j = 0; j < d; j++)
                {
                    norm = NumOps.Add(norm, NumOps.Multiply(Av[j], Av[j]));
                }
                norm = NumOps.Sqrt(norm);

                if (NumOps.ToDouble(norm) < 1e-10) break;

                for (int j = 0; j < d; j++)
                {
                    v[j] = NumOps.Divide(Av[j], norm);
                }
            }

            // Compute eigenvalue: v'Av
            var Av2 = new Vector<T>(d);
            for (int i = 0; i < d; i++)
            {
                T sum = NumOps.Zero;
                for (int j = 0; j < d; j++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(A[i, j], v[j]));
                }
                Av2[i] = sum;
            }

            T eigenvalue = NumOps.Zero;
            for (int j = 0; j < d; j++)
            {
                eigenvalue = NumOps.Add(eigenvalue, NumOps.Multiply(v[j], Av2[j]));
            }

            eigenvalues[e] = eigenvalue;
            for (int j = 0; j < d; j++)
            {
                eigenvectors[e, j] = v[j];
            }

            // Deflate matrix: A = A - lambda * v * v'
            for (int i = 0; i < d; i++)
            {
                for (int j = 0; j < d; j++)
                {
                    A[i, j] = NumOps.Subtract(A[i, j],
                        NumOps.Multiply(eigenvalue, NumOps.Multiply(v[i], v[j])));
                }
            }
        }

        return (eigenvalues, eigenvectors);
    }

    private Matrix<T> CopyMatrix(Matrix<T> source, int d)
    {
        var copy = new Matrix<T>(d, d);
        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j < d; j++)
            {
                copy[i, j] = source[i, j];
            }
        }
        return copy;
    }

    private int DetermineComponents(Vector<T> eigenvalues)
    {
        // Find number of components to explain variance threshold
        double totalVariance = 0;
        for (int i = 0; i < eigenvalues.Length; i++)
        {
            totalVariance += Math.Max(0, NumOps.ToDouble(eigenvalues[i]));
        }

        double cumulativeVariance = 0;
        for (int i = 0; i < eigenvalues.Length; i++)
        {
            cumulativeVariance += Math.Max(0, NumOps.ToDouble(eigenvalues[i]));
            if (totalVariance > 0 && cumulativeVariance / totalVariance >= _varianceThreshold)
            {
                return i + 1;
            }
        }

        return eigenvalues.Length;
    }
}
