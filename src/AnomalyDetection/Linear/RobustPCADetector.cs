using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.Linear;

/// <summary>
/// Detects anomalies using Robust PCA (Principal Component Pursuit).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Robust PCA decomposes data into a low-rank component (normal patterns)
/// and a sparse component (anomalies). Unlike standard PCA, it's robust to outliers because
/// it explicitly models them as sparse corruptions.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Decompose data matrix M = L + S
/// 2. L is low-rank (captures normal patterns)
/// 3. S is sparse (captures anomalies)
/// 4. Solve via convex optimization (ADMM)
/// </para>
/// <para>
/// <b>When to use:</b>
/// - Data corrupted by sparse anomalies
/// - When standard PCA is affected by outliers
/// - Video surveillance (background subtraction)
/// - Network intrusion detection
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Lambda: 1/sqrt(max(n,m))
/// - Max iterations: 1000
/// - Tolerance: 1e-7
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Candès, E.J., et al. (2011). "Robust Principal Component Analysis?" JACM.
/// </para>
/// </remarks>
public class RobustPCADetector<T> : AnomalyDetectorBase<T>
{
    private readonly double _lambda;
    private readonly int _maxIterations;
    private readonly double _tolerance;
    private double[,]? _lowRank;
    private double[,]? _sparse;
    private double[]? _mean;
    private int _nSamples;
    private int _nFeatures;

    /// <summary>
    /// Gets the lambda parameter (sparsity penalty).
    /// </summary>
    public double Lambda => _lambda;

    /// <summary>
    /// Gets the maximum iterations.
    /// </summary>
    public int MaxIterations => _maxIterations;

    /// <summary>
    /// Creates a new Robust PCA anomaly detector.
    /// </summary>
    /// <param name="lambda">
    /// Sparsity penalty. -1 means auto (1/sqrt(max(n,m))). Default is -1.
    /// </param>
    /// <param name="maxIterations">Maximum iterations for ADMM. Default is 1000.</param>
    /// <param name="tolerance">Convergence tolerance. Default is 1e-7.</param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public RobustPCADetector(double lambda = -1, int maxIterations = 1000, double tolerance = 1e-7,
        double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (maxIterations < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(maxIterations),
                "MaxIterations must be at least 1. Recommended is 1000.");
        }

        if (tolerance <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(tolerance),
                "Tolerance must be positive. Recommended is 1e-7.");
        }

        _lambda = lambda;
        _maxIterations = maxIterations;
        _tolerance = tolerance;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        _nSamples = X.Rows;
        _nFeatures = X.Columns;

        // Convert to double matrix
        var data = new double[_nSamples, _nFeatures];
        _mean = new double[_nFeatures];

        for (int j = 0; j < _nFeatures; j++)
        {
            for (int i = 0; i < _nSamples; i++)
            {
                data[i, j] = NumOps.ToDouble(X[i, j]);
                _mean[j] += data[i, j];
            }
            _mean[j] /= _nSamples;
        }

        // Center data
        for (int i = 0; i < _nSamples; i++)
        {
            for (int j = 0; j < _nFeatures; j++)
            {
                data[i, j] -= _mean[j];
            }
        }

        // Set lambda if auto
        double effectiveLambda = _lambda > 0 ? _lambda : 1.0 / Math.Sqrt(Math.Max(_nSamples, _nFeatures));

        // Run Robust PCA via ADMM
        SolveADMM(data, effectiveLambda);

        // Calculate scores for training data to set threshold
        var trainingScores = ScoreAnomaliesInternal(X);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
    }

    private void SolveADMM(double[,] M, double lambda)
    {
        int n = _nSamples;
        int m = _nFeatures;

        // Initialize
        _lowRank = new double[n, m];
        _sparse = new double[n, m];
        var Y = new double[n, m]; // Dual variable

        double mu = n * m / (4 * NormL1(M));
        double muInv = 1.0 / mu;

        for (int iter = 0; iter < _maxIterations; iter++)
        {
            // Update L: minimize ||L||_* + (mu/2)||L - (M - S + mu^-1 Y)||_F^2
            // Soft-threshold singular values
            var LInput = new double[n, m];
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < m; j++)
                {
                    LInput[i, j] = M[i, j] - _sparse[i, j] + muInv * Y[i, j];
                }
            }

            _lowRank = SingularValueThreshold(LInput, muInv);

            // Update S: minimize lambda||S||_1 + (mu/2)||S - (M - L + mu^-1 Y)||_F^2
            // Soft-threshold entries
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < m; j++)
                {
                    double val = M[i, j] - _lowRank[i, j] + muInv * Y[i, j];
                    _sparse[i, j] = SoftThreshold(val, lambda * muInv);
                }
            }

            // Update Y: Y = Y + mu(M - L - S)
            double primalResidual = 0;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < m; j++)
                {
                    double residual = M[i, j] - _lowRank[i, j] - _sparse[i, j];
                    Y[i, j] += mu * residual;
                    primalResidual += residual * residual;
                }
            }

            // Check convergence
            if (Math.Sqrt(primalResidual) / (n * m) < _tolerance)
            {
                break;
            }
        }
    }

    private double[,] SingularValueThreshold(double[,] M, double tau)
    {
        int n = M.GetLength(0);
        int m = M.GetLength(1);
        int k = Math.Min(n, m);

        // Simplified SVD via power iteration for efficiency
        // For a full implementation, use a proper SVD library
        var U = new double[n, k];
        var S = new double[k];
        var V = new double[m, k];

        // Initialize with random vectors
        for (int i = 0; i < n; i++)
            for (int j = 0; j < k; j++)
                U[i, j] = _random.NextDouble() - 0.5;

        // Power iteration
        for (int iter = 0; iter < 20; iter++)
        {
            // V = M' * U
            for (int j = 0; j < k; j++)
            {
                for (int col = 0; col < m; col++)
                {
                    V[col, j] = 0;
                    for (int row = 0; row < n; row++)
                    {
                        V[col, j] += M[row, col] * U[row, j];
                    }
                }
            }

            // Orthogonalize V (simplified Gram-Schmidt)
            for (int j = 0; j < k; j++)
            {
                for (int prev = 0; prev < j; prev++)
                {
                    double dot = 0;
                    for (int row = 0; row < m; row++)
                        dot += V[row, j] * V[row, prev];
                    for (int row = 0; row < m; row++)
                        V[row, j] -= dot * V[row, prev];
                }
                double norm = 0;
                for (int row = 0; row < m; row++)
                    norm += V[row, j] * V[row, j];
                norm = Math.Sqrt(norm);
                if (norm > 1e-10)
                    for (int row = 0; row < m; row++)
                        V[row, j] /= norm;
            }

            // U = M * V
            for (int j = 0; j < k; j++)
            {
                for (int row = 0; row < n; row++)
                {
                    U[row, j] = 0;
                    for (int col = 0; col < m; col++)
                    {
                        U[row, j] += M[row, col] * V[col, j];
                    }
                }

                // Compute singular value
                S[j] = 0;
                for (int row = 0; row < n; row++)
                    S[j] += U[row, j] * U[row, j];
                S[j] = Math.Sqrt(S[j]);

                // Normalize U
                if (S[j] > 1e-10)
                    for (int row = 0; row < n; row++)
                        U[row, j] /= S[j];
            }
        }

        // Soft-threshold singular values and reconstruct
        var result = new double[n, m];
        for (int c = 0; c < k; c++)
        {
            double thresholdedS = Math.Max(0, S[c] - tau);
            if (thresholdedS > 0)
            {
                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j < m; j++)
                    {
                        result[i, j] += thresholdedS * U[i, c] * V[j, c];
                    }
                }
            }
        }

        return result;
    }

    private static double SoftThreshold(double x, double threshold)
    {
        if (x > threshold) return x - threshold;
        if (x < -threshold) return x + threshold;
        return 0;
    }

    private static double NormL1(double[,] M)
    {
        double sum = 0;
        for (int i = 0; i < M.GetLength(0); i++)
        {
            for (int j = 0; j < M.GetLength(1); j++)
            {
                sum += Math.Abs(M[i, j]);
            }
        }
        return sum;
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

        for (int i = 0; i < X.Rows; i++)
        {
            // For new points, project and compute sparse magnitude
            var point = new double[X.Columns];
            for (int j = 0; j < X.Columns; j++)
            {
                point[j] = NumOps.ToDouble(X[i, j]) - _mean![j];
            }

            // Score based on distance from low-rank subspace
            // Approximate using projection onto training low-rank basis
            double score = 0;
            for (int j = 0; j < X.Columns; j++)
            {
                score += point[j] * point[j];
            }

            // If this is training data, use sparse component magnitude
            if (i < _nSamples && _sparse != null)
            {
                score = 0;
                for (int j = 0; j < _nFeatures; j++)
                {
                    score += _sparse[i, j] * _sparse[i, j];
                }
            }

            scores[i] = NumOps.FromDouble(Math.Sqrt(score));
        }

        return scores;
    }
}
