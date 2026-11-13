using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.Statistical;

/// <summary>
/// Detects anomalies using the Interquartile Range (IQR) method.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The IQR method is based on quartiles, which divide your data into four parts:
/// - Q1 (25th percentile): 25% of data is below this value
/// - Q3 (75th percentile): 75% of data is below this value
/// - IQR = Q3 - Q1: The range containing the middle 50% of data
///
/// Outliers are points below Q1 - k*IQR or above Q3 + k*IQR, where k is the multiplier (typically 1.5).
/// </para>
/// <para>
/// <b>When to use:</b>
/// - Your data may not be normally distributed
/// - You want a robust method that isn't affected by extreme outliers
/// - This is the same method used in box plots
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Multiplier: 1.5 (identifies "mild" outliers)
/// - Use 3.0 for "extreme" outliers only
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
    [ResearchPaper("Exploratory Data Analysis", "https://doi.org/10.1002/bimj.4710230408")]
public partial class IQRDetector<T> : AnomalyDetectorBase<T>
{
    private readonly double _multiplier;
    [Buffer]
    private Vector<T>? _q1;
    [Buffer]
    private Vector<T>? _q3;
    [Buffer]
    private Vector<T>? _iqr;
    [Buffer]
    private Vector<T>? _lowerBounds;
    [Buffer]
    private Vector<T>? _upperBounds;

    /// <summary>
    /// Gets the IQR multiplier for determining outlier boundaries.
    /// </summary>
    public double Multiplier => _multiplier;

    /// <summary>
    /// Gets the Q1 (25th percentile) values for each feature.
    /// </summary>
    public Vector<T>? Q1 => _q1;

    /// <summary>
    /// Gets the Q3 (75th percentile) values for each feature.
    /// </summary>
    public Vector<T>? Q3 => _q3;

    /// <summary>
    /// Gets the IQR values for each feature.
    /// </summary>
    public Vector<T>? IQR => _iqr;

    /// <summary>
    /// Gets the lower bounds (Q1 - multiplier * IQR) for each feature.
    /// </summary>
    public Vector<T>? LowerBounds => _lowerBounds;

    /// <summary>
    /// Gets the upper bounds (Q3 + multiplier * IQR) for each feature.
    /// </summary>
    public Vector<T>? UpperBounds => _upperBounds;

    /// <summary>
    /// Creates a new IQR-based anomaly detector.
    /// </summary>
    /// <param name="multiplier">
    /// The multiplier applied to the IQR to determine outlier boundaries. Default is 1.5 (industry standard).
    /// Common values:
    /// - 1.5: "Mild" outliers (standard for box plots)
    /// - 3.0: "Extreme" outliers only
    /// </param>
    /// <param name="contamination">
    /// Expected proportion of anomalies. Default is 0.1 (10%).
    /// </param>
    /// <param name="randomSeed">Random seed for reproducibility.</param>
    public IQRDetector(double multiplier = 1.5, double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (multiplier <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(multiplier),
                "Multiplier must be positive. Common values are 1.5 (mild outliers) or 3.0 (extreme outliers).");
        }

        _multiplier = multiplier;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        int nFeatures = X.Columns;
        T multiplierT = NumOps.FromDouble(_multiplier);

        // Calculate quartiles for each feature
        var engine = Engine;
        var (sorted, _) = engine.TensorSort(AnomalyTensorOps.ToTensor(X), 0, false);
        var q1 = AnomalyTensorOps.SortedColumnQuantile(sorted, 0.25);
        var q3 = AnomalyTensorOps.SortedColumnQuantile(sorted, 0.75);
        var iqr = engine.TensorSubtract(q3, q1);
        var spread = engine.TensorMultiplyScalar(iqr, multiplierT);
        _q1 = q1.ToVector();
        _q3 = q3.ToVector();
        _iqr = iqr.ToVector();

        // Lower bound = Q1 - k * IQR, upper bound = Q3 + k * IQR
        _lowerBounds = engine.TensorSubtract(q1, spread).ToVector();
        _upperBounds = engine.TensorAdd(q3, spread).ToVector();

        // Calculate anomaly scores for training data and set threshold
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

        var iqr = _iqr ?? throw new InvalidOperationException("_iqr has not been initialized.");
        var lowerBounds = _lowerBounds ?? throw new InvalidOperationException("_lowerBounds has not been initialized.");
        var upperBounds = _upperBounds ?? throw new InvalidOperationException("_upperBounds has not been initialized.");

        // Score = max deviation outside the IQR bounds normalized by IQR (constant, zero-IQR features skipped)
        return AnomalyTensorOps.MaxOutsideBand(X, lowerBounds.AsTensor(), upperBounds.AsTensor(), iqr.AsTensor());
    }
}
