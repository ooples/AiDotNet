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
[ResearchPaper("Robust Principal Component Analysis?", "https://doi.org/10.1145/1970392.1970395", Year = 2011, Authors = "Emmanuel J. Candes, Xiaodong Li, Yi Ma, John Wright")]
public partial class RobustPCADetector<T> : AnomalyDetectorBase<T>
{
    private readonly double _lambda;
    private readonly int _maxIterations;
    private readonly double _tolerance;
    [AiDotNet.Attributes.FittedParameter]
    private Matrix<T>? _lowRank;
    [AiDotNet.Attributes.FittedParameter]
    private Matrix<T>? _sparse;
    [Buffer]
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
        var values = AnomalyTensorOps.ToTensor(X);
        var mean = Engine.ReduceMean(values, new[] { 0 }, false);
        _mean = mean.ToVector();
        var data = AnomalyTensorOps.SubtractRow(values, mean);

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

    private void SolveADMM(Tensor<T> M, T lambda)
    {
        var engine = Engine;
        int n = _nSamples;
        int m = _nFeatures;

        // Initialize
        var lowRank = Tensor<T>.CreateZeros(new[] { n, m });
        var sparse = Tensor<T>.CreateZeros(new[] { n, m });
        var Y = Tensor<T>.CreateZeros(new[] { n, m }); // Dual variable
        _lowRank = lowRank.ToMatrix();
        _sparse = sparse.ToMatrix();

        // Guard against zero-norm inputs to avoid NaN/Inf
        T normL1 = engine.TensorSum(engine.TensorAbs(M));
        T eps = NumOps.FromDouble(1e-10);
        if (NumOps.LessThan(normL1, eps))
            return;

        T mu = NumOps.Divide(NumOps.FromDouble(n * m), NumOps.Multiply(NumOps.FromDouble(4), normL1));
        T muInv = NumOps.Divide(NumOps.One, mu);
        T lambdaMuInv = NumOps.Multiply(lambda, muInv);
        T toleranceT = NumOps.FromDouble(_tolerance);
        T nm = NumOps.FromDouble(n * m);

        for (int iter = 0; iter < _maxIterations; iter++)
        {
            // Update L: singular value thresholding of M - S + Y / mu
            var scaledDual = engine.TensorMultiplyScalar(Y, muInv);
            lowRank = SingularValueThreshold(engine.TensorAdd(engine.TensorSubtract(M, sparse), scaledDual), muInv);

            // Update S: soft thresholding of M - L + Y / mu
            sparse = SoftThreshold(engine.TensorAdd(engine.TensorSubtract(M, lowRank), scaledDual), lambdaMuInv);

            // Update Y: Y = Y + mu(M - L - S)
            var residual = engine.TensorSubtract(engine.TensorSubtract(M, lowRank), sparse);
            Y = engine.TensorAdd(Y, engine.TensorMultiplyScalar(residual, mu));
            // Converged when sqrt(||M - L - S||_F^2) / (n m) < tolerance
            var primalResidual = engine.ReduceSum(engine.Reshape(engine.TensorSquare(residual), new[] { n * m }),
                new[] { 0 }, true);
            var converged = engine.TensorLessThan(engine.TensorDivideScalar(engine.TensorSqrt(primalResidual), nm), toleranceT);
            if (converged.Cast<int>().ToArray()[0] != 0)
                break;
        }

        _lowRank = lowRank.ToMatrix();
        _sparse = sparse.ToMatrix();
    }

    // Singular value thresholding: U diag(max(s - tau, 0)) V^T from the (exact) thin SVD of M.
    private Tensor<T> SingularValueThreshold(Tensor<T> M, T tau)
    {
        var engine = Engine;
        var (U, S, Vh) = LinalgDispatch<T>.Svd(M, false);
        var thresholded = engine.TensorMax(engine.TensorSubtractScalar(S, tau), NumOps.Zero);
        return engine.TensorMatMul(AnomalyTensorOps.MultiplyRow(U, thresholded), Vh);
    }

    // x - t where x > t, x + t where x < -t, else 0.
    private Tensor<T> SoftThreshold(Tensor<T> x, T threshold)
    {
        var engine = Engine;
        var zeros = Tensor<T>.CreateZeros(new[] { x.Shape[0], x.Shape[1] });
        return engine.TensorWhere(engine.TensorGreaterThan(x, threshold), engine.TensorSubtractScalar(x, threshold),
            engine.TensorWhere(engine.TensorLessThan(x, NumOps.Negate(threshold)), engine.TensorAddScalar(x, threshold), zeros));
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

        var engine = Engine;
        int rows = X.Rows;

        // For new points, center the point and subtract the average low-rank row (the sparse estimate)
        var centered = AnomalyTensorOps.SubtractRow(AnomalyTensorOps.ToTensor(X), mean.AsTensor());
        var lowRank = _lowRank;
        var sparseEstimate = lowRank != null && _nSamples > 0
            ? AnomalyTensorOps.SubtractRow(centered, engine.ReduceMean(AnomalyTensorOps.ToTensor(lowRank), new[] { 0 }, false))
            : centered;
        var score = engine.ReduceSum(engine.TensorSquare(sparseEstimate), new[] { 1 }, false);

        // For training data, use the precomputed sparse component magnitude
        var sparse = _sparse;
        if (isTrainingData && sparse != null)
        {
            int covered = Math.Min(rows, _nSamples);
            var sparseScore = engine.ReduceSum(engine.TensorSquare(
                engine.TensorSlice(AnomalyTensorOps.ToTensor(sparse), new[] { 0, 0 }, new[] { covered, _nFeatures })), new[] { 1 }, false);
            score = covered == rows
                ? sparseScore
                : engine.TensorConcatenate(new[] { sparseScore,
                    engine.TensorSlice(score, new[] { covered }, new[] { rows - covered }) }, 0);
        }

        return engine.TensorSqrt(score).ToVector();
    }
}
