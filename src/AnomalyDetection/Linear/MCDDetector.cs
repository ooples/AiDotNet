using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.Linear;

/// <summary>
/// Detects anomalies using Minimum Covariance Determinant (MCD).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> MCD is a robust method for estimating the center and spread of data.
/// Unlike standard mean and covariance, MCD is resistant to outliers by finding the subset
/// of points that minimizes the covariance determinant. Points far from this robust estimate
/// are anomalies.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Find the subset of h points with minimum covariance determinant
/// 2. Compute robust mean and covariance from this subset
/// 3. Compute Mahalanobis distances using robust estimates
/// 4. Points with large distances are anomalies
/// </para>
/// <para>
/// <b>When to use:</b>
/// - Data with outliers that corrupt standard statistics
/// - When you need robust location/scatter estimates
/// - Multivariate anomaly detection
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Support fraction: 0.5 (use 50% of data for robust estimate)
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Rousseeuw, P.J. (1984). "Least Median of Squares Regression."
/// Rousseeuw, P.J., Driessen, K.V. (1999). "A Fast Algorithm for the Minimum Covariance Determinant Estimator."
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelCategory(ModelCategory.AnomalyDetection)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("A Fast Algorithm for the Minimum Covariance Determinant Estimator",
    "https://doi.org/10.1080/00401706.1999.10485670",
    Year = 1999,
    Authors = "Peter J. Rousseeuw, Katrien Van Driessen")]
public partial class MCDDetector<T> : AnomalyDetectorBase<T>
{
    private readonly double _supportFraction;
    [Buffer]
    private Vector<T>? _robustMean;
    [Buffer]
    private Matrix<T>? _robustPrecision;
    private int _nFeatures;

    /// <summary>
    /// Gets the support fraction.
    /// </summary>
    public double SupportFraction => _supportFraction;

    /// <summary>
    /// Creates a new MCD anomaly detector.
    /// </summary>
    /// <param name="supportFraction">
    /// Fraction of data to use for robust estimate (0.5-1.0). Default is 0.5.
    /// </param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public MCDDetector(double supportFraction = 0.5, double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (supportFraction < 0.5 || supportFraction > 1.0)
        {
            throw new ArgumentOutOfRangeException(nameof(supportFraction),
                "SupportFraction must be between 0.5 and 1.0. Recommended is 0.5.");
        }

        _supportFraction = supportFraction;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        int n = X.Rows;
        _nFeatures = X.Columns;

        // Validate that we have more samples than features for covariance estimation
        if (n <= _nFeatures)
        {
            throw new ArgumentException(
                $"MCD requires more samples ({n}) than features ({_nFeatures}). " +
                "Consider reducing dimensions or adding more samples.",
                nameof(X));
        }

        // Compute h and ensure it's at least d + 1 for valid covariance estimation
        int h = Math.Max(_nFeatures + 1, (int)(n * _supportFraction));

        // Run Fast-MCD algorithm (simplified)
        var (robustMean, robustCov) = FastMCD(AnomalyTensorOps.ToTensor(X), h);

        // Compute precision matrix (inverse of covariance)
        _robustMean = robustMean.ToVector();
        _robustPrecision = LinalgDispatch<T>.Inv(robustCov).ToMatrix();

        // Calculate scores for training data to set threshold
        var trainingScores = ScoreAnomaliesInternal(X);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
    }

    private (Tensor<T> mean, Tensor<T> covariance) FastMCD(Tensor<T> data, int h)
    {
        var engine = Engine;
        int n = data.Shape[0];
        int nTrials = 10;
        var means = new Tensor<T>[nTrials];
        var covariances = new Tensor<T>[nTrials];
        var determinants = new Tensor<T>[nTrials];

        for (int trial = 0; trial < nTrials; trial++)
        {
            // Random initial subset
            var indices = Enumerable.Range(0, n)
                .OrderBy(_ => _random.NextDouble())
                .Take(h)
                .ToArray();

            // Compute mean and covariance of subset
            var (mean, cov) = ComputeMeanCovariance(data, Tensor<int>.FromVector(Vector<int>.Wrap(indices), new[] { h }));

            // C-step iterations: keep the h points with the smallest Mahalanobis distances (by index on ties)
            for (int iter = 0; iter < 10; iter++)
            {
                var distances = MahalanobisDistances(data, mean, LinalgDispatch<T>.Inv(cov));
                var selected = engine.TensorSlice(AnomalyTensorOps.StableOrder(distances, descending: false), new[] { 0 }, new[] { h });
                (mean, cov) = ComputeMeanCovariance(data, selected);
            }

            means[trial] = mean;
            covariances[trial] = cov;
            determinants[trial] = Determinant(cov);
        }

        // The solution with the smallest determinant above 1e-10 (the first on ties); fall back to the
        // sample mean/cov if MCD failed
        var dets = engine.TensorConcatenate(determinants, 0);
        var valid = engine.TensorGreaterThan(dets, NumOps.FromDouble(1e-10));
        if (!NumOps.GreaterThan(engine.TensorSum(valid), NumOps.Zero))
        {
            return ComputeMeanCovariance(data, AnomalyTensorOps.Range(n));
        }

        var candidates = engine.TensorWhere(valid, dets, engine.Fill(nTrials, NumOps.MaxValue).AsTensor());
        int best = AnomalyTensorOps.FirstArgMin(candidates);
        return (means[best], covariances[best]);
    }

    // Mean and covariance (divisor h - 1, plus 1e-6 on the diagonal) of the selected rows. Where T cannot
    // represent c_jj + 1e-6 (the ridge would round away, e.g. float with c_jj = 350), the ridge is scaled
    // to 1e-6 * c_jj instead so the regularized covariance stays invertible.
    private (Tensor<T> mean, Tensor<T> covariance) ComputeMeanCovariance(Tensor<T> data, Tensor<int> indices)
    {
        var engine = Engine;
        int h = indices.Length;
        var subset = engine.TensorGather(data, indices, 0);
        var mean = engine.ReduceMean(subset, new[] { 0 }, false);
        var cov = AnomalyTensorOps.Covariance(AnomalyTensorOps.SubtractRow(subset, mean), NumOps.FromDouble(h - 1));
        T ridge = NumOps.FromDouble(1e-6);
        var identity = AnomalyTensorOps.ScaledIdentity(_nFeatures, NumOps.One);
        var diagonal = engine.ReduceSum(engine.TensorMultiply(cov, identity), new[] { 1 }, false);
        var absorbed = engine.TensorEquals(engine.TensorAddScalar(diagonal, ridge), diagonal);
        var ridges = engine.TensorWhere(absorbed, engine.TensorMultiplyScalar(diagonal, ridge),
            engine.Fill(_nFeatures, ridge).AsTensor());
        return (mean, engine.TensorAdd(cov, AnomalyTensorOps.MultiplyRow(identity, ridges)));
    }

    // sqrt(max(0, (x - mean)' P (x - mean))) for every row.
    private Tensor<T> MahalanobisDistances(Tensor<T> data, Tensor<T> mean, Tensor<T> precision)
    {
        var engine = Engine;
        var squared = AnomalyTensorOps.MahalanobisSquared(AnomalyTensorOps.SubtractRow(data, mean), precision);
        return engine.TensorSqrt(engine.TensorMax(squared, NumOps.Zero));
    }

    // Determinant as a [1] tensor, 0 when an LU pivot (partial pivoting) is below 1e-10 in magnitude.
    private Tensor<T> Determinant(Tensor<T> matrix)
    {
        var engine = Engine;
        var (_, _, U) = LinalgDispatch<T>.LU(matrix);
        var minPivot = engine.TensorAmin(engine.TensorAbs(LinalgDispatch<T>.Diagonal(U)), new[] { 0 }, true);
        var det = engine.Reshape(LinalgDispatch<T>.Det(matrix), new[] { 1 });
        return engine.TensorWhere(engine.TensorLessThan(minPivot, NumOps.FromDouble(1e-10)),
            Tensor<T>.CreateZeros(new[] { 1 }), det);
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

        if (X.Columns != _nFeatures)
        {
            throw new ArgumentException(
                $"Input has {X.Columns} features, but model was fitted with {_nFeatures} features.",
                nameof(X));
        }

        var robustMean = _robustMean;
        var robustPrecision = _robustPrecision;

        if (robustMean == null || robustPrecision == null)
        {
            throw new InvalidOperationException("Model not properly fitted.");
        }

        // Mahalanobis distance in T-space
        return MahalanobisDistances(AnomalyTensorOps.ToTensor(X), robustMean.AsTensor(),
            AnomalyTensorOps.ToTensor(robustPrecision)).ToVector();
    }
}
