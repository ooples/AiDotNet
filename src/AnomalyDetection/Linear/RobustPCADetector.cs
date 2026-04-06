using AiDotNet.Attributes;
using AiDotNet.Enums;
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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Linear)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ModelPaper("Robust Principal Component Analysis?", "https://doi.org/10.1145/1970392.1970395", Year = 2011, Authors = "Emmanuel J. Candes, Xiaodong Li, Yi Ma, John Wright")]
public class RobustPCADetector<T> : AnomalyDetectorBase<T>
{
    private readonly double _lambda;
    private readonly int _maxIterations;
    private readonly double _tolerance;
    private Matrix<T>? _lowRank;
    private Matrix<T>? _sparse;
    private Vector<T>? _mean;
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

        if (lambda != -1 && lambda <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(lambda),
                "Lambda must be -1 (auto) or a positive value.");
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

        // Compute mean and center data
        _mean = new Vector<T>(_nFeatures);
        T nT = NumOps.FromDouble(_nSamples);

        for (int j = 0; j < _nFeatures; j++)
        {
            T sum = NumOps.Zero;
            for (int i = 0; i < _nSamples; i++)
            {
                sum = NumOps.Add(sum, X[i, j]);
            }
            _mean[j] = NumOps.Divide(sum, nT);
        }

        var data = new Matrix<T>(_nSamples, _nFeatures);
        for (int i = 0; i < _nSamples; i++)
        {
            for (int j = 0; j < _nFeatures; j++)
            {
                data[i, j] = NumOps.Subtract(X[i, j], _mean[j]);
            }
        }

        // Set lambda if auto
        T effectiveLambda = _lambda > 0
            ? NumOps.FromDouble(_lambda)
            : NumOps.Divide(NumOps.One, NumOps.Sqrt(NumOps.FromDouble(Math.Max(_nSamples, _nFeatures))));

        // Run Robust PCA via ADMM
        SolveADMM(data, effectiveLambda);

        // Calculate scores for training data to set threshold.
        // Use isTrainingData: false so that the threshold is calibrated on the same
        // scoring method that ScoreAnomalies() will use for new data. This ensures
        // consistent score distributions between Fit and Predict.
        var trainingScores = ScoreAnomaliesInternal(X, isTrainingData: false);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
    }

    private void SolveADMM(Matrix<T> M, T lambda)
    {
        int n = _nSamples;
        int m = _nFeatures;

        // Initialize
        _lowRank = new Matrix<T>(n, m);
        _sparse = new Matrix<T>(n, m);
        var Y = new Matrix<T>(n, m); // Dual variable

        // Guard against zero-norm inputs to avoid NaN/Inf
        T normL1 = NormL1(M);
        T eps = NumOps.FromDouble(1e-10);
        if (NumOps.LessThan(normL1, eps))
        {
            return;
        }

        T mu = NumOps.Divide(NumOps.FromDouble(n * m), NumOps.Multiply(NumOps.FromDouble(4), normL1));
        T muInv = NumOps.Divide(NumOps.One, mu);
        T lambdaMuInv = NumOps.Multiply(lambda, muInv);
        T toleranceT = NumOps.FromDouble(_tolerance);
        T nm = NumOps.FromDouble(n * m);

        for (int iter = 0; iter < _maxIterations; iter++)
        {
            // Update L
            var LInput = new Matrix<T>(n, m);
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < m; j++)
                {
                    LInput[i, j] = NumOps.Add(NumOps.Subtract(M[i, j], _sparse[i, j]), NumOps.Multiply(muInv, Y[i, j]));
                }
            }

            _lowRank = SingularValueThreshold(LInput, muInv);

            // Update S
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < m; j++)
                {
                    T val = NumOps.Add(NumOps.Subtract(M[i, j], _lowRank[i, j]), NumOps.Multiply(muInv, Y[i, j]));
                    _sparse[i, j] = SoftThreshold(val, lambdaMuInv);
                }
            }

            // Update Y: Y = Y + mu(M - L - S)
            T primalResidual = NumOps.Zero;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < m; j++)
                {
                    T residual = NumOps.Subtract(NumOps.Subtract(M[i, j], _lowRank[i, j]), _sparse[i, j]);
                    Y[i, j] = NumOps.Add(Y[i, j], NumOps.Multiply(mu, residual));
                    primalResidual = NumOps.Add(primalResidual, NumOps.Multiply(residual, residual));
                }
            }

            if (NumOps.LessThan(NumOps.Divide(NumOps.Sqrt(primalResidual), nm), toleranceT))
            {
                break;
            }
        }
    }

    private Matrix<T> SingularValueThreshold(Matrix<T> M, T tau)
    {
        int n = M.Rows;
        int m = M.Columns;
        int k = Math.Min(n, m);
        T eps = NumOps.FromDouble(1e-10);

        var U = new Matrix<T>(n, k);
        var S = new Vector<T>(k);
        var V = new Matrix<T>(m, k);

        for (int i = 0; i < n; i++)
            for (int j = 0; j < k; j++)
                U[i, j] = NumOps.FromDouble(_random.NextDouble() - 0.5);

        for (int iter = 0; iter < 20; iter++)
        {
            // V = M' * U
            for (int j = 0; j < k; j++)
            {
                for (int col = 0; col < m; col++)
                {
                    T sum = NumOps.Zero;
                    for (int row = 0; row < n; row++)
                        sum = NumOps.Add(sum, NumOps.Multiply(M[row, col], U[row, j]));
                    V[col, j] = sum;
                }
            }

            // Orthogonalize V (Gram-Schmidt)
            for (int j = 0; j < k; j++)
            {
                for (int prev = 0; prev < j; prev++)
                {
                    T dot = NumOps.Zero;
                    for (int row = 0; row < m; row++)
                        dot = NumOps.Add(dot, NumOps.Multiply(V[row, j], V[row, prev]));
                    for (int row = 0; row < m; row++)
                        V[row, j] = NumOps.Subtract(V[row, j], NumOps.Multiply(dot, V[row, prev]));
                }
                T normSq = NumOps.Zero;
                for (int row = 0; row < m; row++)
                    normSq = NumOps.Add(normSq, NumOps.Multiply(V[row, j], V[row, j]));
                T norm = NumOps.Sqrt(normSq);
                if (NumOps.GreaterThan(norm, eps))
                    for (int row = 0; row < m; row++)
                        V[row, j] = NumOps.Divide(V[row, j], norm);
            }

            // U = M * V
            for (int j = 0; j < k; j++)
            {
                for (int row = 0; row < n; row++)
                {
                    T sum = NumOps.Zero;
                    for (int col = 0; col < m; col++)
                        sum = NumOps.Add(sum, NumOps.Multiply(M[row, col], V[col, j]));
                    U[row, j] = sum;
                }

                T sSq = NumOps.Zero;
                for (int row = 0; row < n; row++)
                    sSq = NumOps.Add(sSq, NumOps.Multiply(U[row, j], U[row, j]));
                S[j] = NumOps.Sqrt(sSq);

                if (NumOps.GreaterThan(S[j], eps))
                    for (int row = 0; row < n; row++)
                        U[row, j] = NumOps.Divide(U[row, j], S[j]);
            }
        }

        // Soft-threshold singular values and reconstruct
        var result = new Matrix<T>(n, m);
        for (int c = 0; c < k; c++)
        {
            T thresholdedS = NumOps.Subtract(S[c], tau);
            if (NumOps.LessThan(thresholdedS, NumOps.Zero)) thresholdedS = NumOps.Zero;
            if (NumOps.GreaterThan(thresholdedS, NumOps.Zero))
            {
                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j < m; j++)
                    {
                        result[i, j] = NumOps.Add(result[i, j],
                            NumOps.Multiply(thresholdedS, NumOps.Multiply(U[i, c], V[j, c])));
                    }
                }
            }
        }

        return result;
    }

    private T SoftThreshold(T x, T threshold)
    {
        if (NumOps.GreaterThan(x, threshold)) return NumOps.Subtract(x, threshold);
        T negThreshold = NumOps.Negate(threshold);
        if (NumOps.LessThan(x, negThreshold)) return NumOps.Add(x, threshold);
        return NumOps.Zero;
    }

    private T NormL1(Matrix<T> M)
    {
        T sum = NumOps.Zero;
        for (int i = 0; i < M.Rows; i++)
        {
            for (int j = 0; j < M.Columns; j++)
            {
                sum = NumOps.Add(sum, NumOps.Abs(M[i, j]));
            }
        }
        return sum;
    }

    /// <inheritdoc/>
    public override Vector<T> ScoreAnomalies(Matrix<T> X)
    {
        EnsureFitted();
        return ScoreAnomaliesInternal(X, isTrainingData: false);
    }

    private Vector<T> ScoreAnomaliesInternal(Matrix<T> X, bool isTrainingData)
    {
        ValidateInput(X);

        var mean = _mean;
        if (mean == null)
        {
            throw new InvalidOperationException("Model has not been fitted. Call Fit() first.");
        }

        if (X.Columns != _nFeatures)
        {
            throw new ArgumentException(
                $"Input has {X.Columns} features, but model was fitted with {_nFeatures} features.",
                nameof(X));
        }

        var scores = new Vector<T>(X.Rows);

        T nSamplesT = NumOps.FromDouble(_nSamples);

        for (int i = 0; i < X.Rows; i++)
        {
            T score;

            // For training data, use the precomputed sparse component magnitude
            if (isTrainingData && i < _nSamples && _sparse != null)
            {
                score = NumOps.Zero;
                for (int j = 0; j < _nFeatures; j++)
                {
                    score = NumOps.Add(score, NumOps.Multiply(_sparse[i, j], _sparse[i, j]));
                }
            }
            else
            {
                // For new points, center the point
                var centered = new Vector<T>(X.Columns);
                for (int j = 0; j < X.Columns; j++)
                {
                    centered[j] = NumOps.Subtract(X[i, j], mean[j]);
                }

                if (_lowRank != null && _nSamples > 0)
                {
                    score = NumOps.Zero;
                    for (int j = 0; j < X.Columns; j++)
                    {
                        T avgLowRank = NumOps.Zero;
                        for (int k = 0; k < _nSamples; k++)
                        {
                            avgLowRank = NumOps.Add(avgLowRank, _lowRank[k, j]);
                        }
                        avgLowRank = NumOps.Divide(avgLowRank, nSamplesT);

                        T sparseEst = NumOps.Subtract(centered[j], avgLowRank);
                        score = NumOps.Add(score, NumOps.Multiply(sparseEst, sparseEst));
                    }
                }
                else
                {
                    score = NumOps.Zero;
                    for (int j = 0; j < X.Columns; j++)
                    {
                        score = NumOps.Add(score, NumOps.Multiply(centered[j], centered[j]));
                    }
                }
            }

            scores[i] = NumOps.Sqrt(score);
        }

        return scores;
    }
}
