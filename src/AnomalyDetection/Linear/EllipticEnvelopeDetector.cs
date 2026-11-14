using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.Linear;

/// <summary>
/// Detects anomalies using Elliptic Envelope (robust covariance estimation).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Elliptic Envelope fits an ellipse (in 2D) or ellipsoid (in higher dimensions)
/// around the data using robust estimation. Points far from this envelope are anomalies.
/// It's like drawing the smallest ellipse that contains most of the data.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Estimate robust mean and covariance using Minimum Covariance Determinant (MCD)
/// 2. Compute Mahalanobis distance for each point
/// 3. Points with large distances are anomalies
/// </para>
/// <para>
/// <b>When to use:</b>
/// - Data is approximately Gaussian/elliptical
/// - You need robustness against outliers in the training data
/// - Multivariate anomaly detection
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Support fraction: 0.5 (use 50% of data for robust estimation)
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Rousseeuw, P.J., Van Driessen, K. (1999). "A Fast Algorithm for the Minimum
/// Covariance Determinant Estimator." Technometrics.
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
public partial class EllipticEnvelopeDetector<T> : AnomalyDetectorBase<T>
{
    private readonly double _supportFraction;
    private Vector<T>? _location;
    [AiDotNet.Attributes.FittedParameter]
    private Matrix<T>? _precisionMatrix;

    /// <summary>
    /// Gets the support fraction (proportion of data for robust estimation).
    /// </summary>
    public double SupportFraction => _supportFraction;

    /// <summary>
    /// Creates a new Elliptic Envelope anomaly detector.
    /// </summary>
    /// <param name="supportFraction">
    /// Proportion of data to use for robust estimation. Default is 0.5 (50%).
    /// Should be between 0.5 and 1.
    /// </param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public EllipticEnvelopeDetector(double supportFraction = 0.5, double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (supportFraction < 0.5 || supportFraction > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(supportFraction),
                "SupportFraction must be between 0.5 and 1. Recommended is 0.5.");
        }

        _supportFraction = supportFraction;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        int n = X.Rows;
        int d = X.Columns;

        if (n <= d)
        {
            throw new ArgumentException(
                $"Number of samples ({n}) must be greater than number of features ({d}).",
                nameof(X));
        }

        // Compute robust location and covariance using simplified MCD
        var (location, covariance) = ComputeRobustEstimates(AnomalyTensorOps.ToTensor(X));
        _location = location.ToVector();

        // Compute precision matrix (inverse of covariance)
        _precisionMatrix = LinalgDispatch<T>.Inv(covariance).ToMatrix();

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

        var location = _location ?? throw new InvalidOperationException("_location has not been initialized.");
        var precMat = _precisionMatrix ?? throw new InvalidOperationException("_precisionMatrix has not been initialized.");

        // Mahalanobis distance: sqrt((x - mu)' * Precision * (x - mu))
        var centered = AnomalyTensorOps.SubtractRow(AnomalyTensorOps.ToTensor(X), location.AsTensor());
        return Engine.TensorSqrt(AnomalyTensorOps.MahalanobisSquared(centered, AnomalyTensorOps.ToTensor(precMat))).ToVector();
    }

    private (Tensor<T> location, Tensor<T> covariance) ComputeRobustEstimates(Tensor<T> X)
    {
        var engine = Engine;
        int n = X.Shape[0];
        int d = X.Shape[1];
        int h = (int)(n * _supportFraction);
        h = Math.Max(h, d + 1);

        // Start with median-based initial estimates
        var location = AnomalyTensorOps.ColumnMedians(X);
        var covariance = RegularizedCovariance(AnomalyTensorOps.SubtractRow(X, location), n - 1);

        // Iterative refinement (C-steps): keep the h points with the smallest Mahalanobis distances
        // (by index on ties) and recompute location and covariance from that subset
        for (int iter = 0; iter < 10; iter++)
        {
            var precision = LinalgDispatch<T>.Inv(covariance);
            var distances = AnomalyTensorOps.MahalanobisSquared(AnomalyTensorOps.SubtractRow(X, location), precision);
            var order = AnomalyTensorOps.StableOrder(distances, descending: false);
            var subset = engine.TensorGather(X, engine.TensorSlice(order, new[] { 0 }, new[] { h }), 0);
            location = engine.ReduceMean(subset, new[] { 0 }, false);
            covariance = RegularizedCovariance(AnomalyTensorOps.SubtractRow(subset, location), h - 1);
        }

        return (location, covariance);
    }

    // centered' centered / divisor + 1e-6 I.
    private Tensor<T> RegularizedCovariance(Tensor<T> centered, int divisor)
    {
        int d = centered.Shape[1];
        return Engine.TensorAdd(AnomalyTensorOps.Covariance(centered, NumOps.FromDouble(divisor)),
            AnomalyTensorOps.ScaledIdentity(d, NumOps.FromDouble(1e-6)));
    }
}
