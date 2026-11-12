using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.Statistical;

/// <summary>
/// Detects anomalies using the Chi-Square test for multivariate data.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The Chi-Square (Mahalanobis distance) detector identifies outliers
/// by measuring how far a point is from the center of the data, accounting for correlations
/// between features. Points far from the center in Mahalanobis distance are anomalies.
/// </para>
/// <para>
/// The algorithm computes:
/// D^2 = (x - mean)' * Cov^(-1) * (x - mean)
/// This squared Mahalanobis distance follows a Chi-Square distribution with p degrees of freedom
/// (where p = number of features) under the assumption of multivariate normality.
/// </para>
/// <para>
/// <b>When to use:</b>
/// - Multivariate data (multiple correlated features)
/// - Data is approximately multivariate normal
/// - You want to account for correlations between features
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Alpha (significance level): 0.05 (5%)
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
    [ResearchPaper("Contributions to the Mathematical Theory of Evolution", "https://doi.org/10.1098/rsta.1900.0009")]
public partial class ChiSquareDetector<T> : AnomalyDetectorBase<T>
{
    private readonly double _alpha;
    [Buffer]
    private Vector<T>? _mean;
    [Buffer]
    private Matrix<T>? _covarianceInverse;

    private T _chiSquareCritical;

    private int _nFeatures;

    /// <summary>
    /// Gets the significance level (alpha) for the test.
    /// </summary>
    public double Alpha => _alpha;

    /// <summary>
    /// Gets the Chi-Square critical value based on the fitted degrees of freedom (number of features) and alpha.
    /// Use this for statistical threshold-based anomaly detection.
    /// </summary>
    public T ChiSquareCritical => _chiSquareCritical;

    /// <summary>
    /// Creates a new Chi-Square anomaly detector.
    /// </summary>
    /// <param name="alpha">
    /// The significance level for the test. Default is 0.05 (5%).
    /// </param>
    /// <param name="contamination">
    /// The expected proportion of anomalies in the data. Default is 0.1 (10%).
    /// </param>
    /// <param name="randomSeed">
    /// Random seed for reproducibility. Default is 42.
    /// </param>
    public ChiSquareDetector(double alpha = 0.05, double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        _chiSquareCritical = NumOps.Zero;
        if (alpha <= 0 || alpha >= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(alpha),
                "Alpha must be between 0 and 1. Recommended value is 0.05.");
        }

        _alpha = alpha;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        if (X.Rows <= X.Columns)
        {
            throw new ArgumentException(
                $"Number of samples ({X.Rows}) must be greater than number of features ({X.Columns}) for covariance estimation.",
                nameof(X));
        }

        int n = X.Rows;
        _nFeatures = X.Columns;
        int p = _nFeatures;

        // Calculate mean
        var engine = Engine;
        var data = AnomalyTensorOps.ToTensor(X);
        var mean = engine.ReduceMean(data, new[] { 0 }, false);
        _mean = mean.ToVector();

        // Calculate (sample) covariance matrix
        var covariance = AnomalyTensorOps.Covariance(AnomalyTensorOps.SubtractRow(data, mean), NumOps.FromDouble(n - 1));

        // Compute inverse of covariance matrix (with regularization for stability)
        _covarianceInverse = ComputeInverse(covariance);

        // Get Chi-Square critical value (approximation)
        _chiSquareCritical = NumOps.FromDouble(GetChiSquareCritical(p, 1 - _alpha));

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

        if (X.Columns != _nFeatures)
        {
            throw new ArgumentException(
                $"Input has {X.Columns} features, but model was fitted with {_nFeatures} features.",
                nameof(X));
        }

        // Capture nullable fields into local variables
        var mean = _mean;
        var covarianceInverse = _covarianceInverse;

        if (mean == null || covarianceInverse == null)
        {
            throw new InvalidOperationException("Model not properly fitted.");
        }

        // Mahalanobis distance squared: D^2 = (x - mean)' * Cov^(-1) * (x - mean)
        var centered = AnomalyTensorOps.SubtractRow(AnomalyTensorOps.ToTensor(X), mean.AsTensor());
        return AnomalyTensorOps.MahalanobisSquared(centered, AnomalyTensorOps.ToTensor(covarianceInverse)).ToVector();
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Uses the Chi-Square critical value for statistical anomaly classification.
    /// Points with Mahalanobis distance squared greater than the critical value are classified as anomalies.
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> X)
    {
        EnsureFitted();

        var scores = ScoreAnomalies(X);
        // Points with D^2 > chi-square critical are anomalies (-1), otherwise inliers (1)
        return AnomalyTensorOps.LabelsFromThreshold(scores, _chiSquareCritical);
    }

    private Matrix<T> ComputeInverse(Tensor<T> matrix)
    {
        // Small ridge on the diagonal for numerical stability, then an LU (partial pivoting) inverse
        int n = matrix.Shape[0];
        var regularized = Engine.TensorAdd(matrix, AnomalyTensorOps.ScaledIdentity(n, NumOps.FromDouble(1e-6)));
        return LinalgDispatch<T>.Inv(regularized).ToMatrix();
    }

    private double GetChiSquareCritical(int df, double p)
    {
        // Wilson-Hilferty approximation for Chi-Square quantile
        // X^2 ≈ df * (1 - 2/(9*df) + z * sqrt(2/(9*df)))^3
        double z = GetZCritical(1 - p);
        double term = 2.0 / (9 * df);
        double result = df * Math.Pow(1 - term + z * Math.Sqrt(term), 3);
        return Math.Max(0, result);
    }

    private double GetZCritical(double alpha)
    {
        // Approximation of inverse standard normal CDF
        double p = alpha;
        if (p > 0.5) p = 1 - p;

        double t = Math.Sqrt(-2 * Math.Log(p));
        double c0 = 2.515517;
        double c1 = 0.802853;
        double c2 = 0.010328;
        double d1 = 1.432788;
        double d2 = 0.189269;
        double d3 = 0.001308;

        double z = t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t);

        return alpha > 0.5 ? -z : z;
    }
}
