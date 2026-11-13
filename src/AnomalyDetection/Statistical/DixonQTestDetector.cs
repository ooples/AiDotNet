using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.Statistical;

/// <summary>
/// Detects anomalies using Dixon's Q Test for small datasets.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Dixon's Q Test is designed for detecting a single outlier in small datasets
/// (typically n &lt; 25). It compares the gap between the suspect value and its nearest neighbor
/// to the range of the entire dataset.
/// </para>
/// <para>
/// The test statistic is: Q = gap / range
/// where gap = |suspect value - nearest value| and range = max - min.
/// </para>
/// <para>
/// <b>When to use:</b>
/// - Small datasets (n &lt; 25, ideally 3-10 samples)
/// - You suspect exactly one outlier
/// - Data is approximately normally distributed
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Alpha (significance level): 0.05 (5%)
/// </para>
/// <para>
/// Reference: Dixon, W. J. (1950). "Analysis of Extreme Values."
/// Annals of Mathematical Statistics.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Analysis of Extreme Values", "https://doi.org/10.1214/aoms/1177729747", Year = 1950, Authors = "Wilfrid J. Dixon")]
public partial class DixonQTestDetector<T> : AnomalyDetectorBase<T>
{
    private readonly double _alpha;
    [Buffer]
    private Vector<T>? _minValues;
    [Buffer]
    private Vector<T>? _maxValues;
    [Buffer]
    private Vector<T>? _ranges;
    [Buffer]
    private Vector<T>? _secondMin;
    [Buffer]
    private Vector<T>? _secondMax;
    private int _nFeatures;
    private int _nSamples;

    /// <summary>
    /// Gets the significance level (alpha) for the test.
    /// </summary>
    public double Alpha => _alpha;

    /// <summary>
    /// Creates a new Dixon's Q Test anomaly detector.
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
    public DixonQTestDetector(double alpha = 0.05, double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
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

        if (X.Rows < 3)
        {
            throw new ArgumentException(
                $"Dixon's Q Test requires at least 3 samples. Got {X.Rows}.",
                nameof(X));
        }

        _nFeatures = X.Columns;
        _nSamples = X.Rows;

        // Smallest, second smallest, second largest and largest value of each feature
        var engine = Engine;
        var (sorted, _) = engine.TensorSort(AnomalyTensorOps.ToTensor(X), 0, false);
        var minValues = engine.TensorSliceAxis(sorted, 0, 0);
        var maxValues = engine.TensorSliceAxis(sorted, 0, X.Rows - 1);
        _minValues = minValues.ToVector();
        _maxValues = maxValues.ToVector();
        _secondMin = engine.TensorSliceAxis(sorted, 0, 1).ToVector();
        _secondMax = engine.TensorSliceAxis(sorted, 0, X.Rows - 2).ToVector();
        _ranges = engine.TensorSubtract(maxValues, minValues).ToVector();

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

        var ranges = _ranges ?? throw new InvalidOperationException("_ranges has not been initialized.");
        var secondMin = _secondMin ?? throw new InvalidOperationException("_secondMin has not been initialized.");
        var secondMax = _secondMax ?? throw new InvalidOperationException("_secondMax has not been initialized.");

        // Dixon's Q measures how extreme a point is by comparing the gap between the suspect value and
        // its nearest neighbour to the total range: at or above the high end Q = (x - secondMax) / range,
        // at or below the low end Q = (secondMin - x) / range, 0 in the interior. Zero-range features are
        // skipped and the point's score is the largest Q across features.
        return AnomalyTensorOps.MaxOutsideBand(X, secondMin.AsTensor(), secondMax.AsTensor(), ranges.AsTensor());
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Uses the Dixon's Q critical value for statistical anomaly classification based on sample size.
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> X)
    {
        EnsureFitted();

        var scores = ScoreAnomalies(X);
        double qCritical = GetQCritical(_nSamples);

        // Points with Q > Q critical are anomalies (-1), otherwise inliers (1)
        return AnomalyTensorOps.LabelsFromThreshold(scores, NumOps.FromDouble(qCritical));
    }

    /// <summary>
    /// Gets the critical Q value for Dixon's test based on sample size and alpha level.
    /// </summary>
    /// <param name="n">Sample size.</param>
    /// <returns>Critical Q value.</returns>
    public double GetQCritical(int n)
    {
        // For n > 25, use approximation (must check before dictionary lookup)
        if (n > 25)
        {
            double approxValue = 0.27 + 1.0 / n;
            if (_alpha != 0.05)
            {
                // Adjust for different alpha
                double adjustmentFactor = Math.Log(0.05) / Math.Log(_alpha);
                return Math.Min(1.0, approxValue * adjustmentFactor);
            }
            return approxValue;
        }

        // Critical values for Dixon's Q test at alpha = 0.05
        // Source: Dixon (1950)
        var criticalValues = new Dictionary<int, double>
        {
            { 3, 0.941 }, { 4, 0.765 }, { 5, 0.642 }, { 6, 0.560 }, { 7, 0.507 },
            { 8, 0.468 }, { 9, 0.437 }, { 10, 0.412 }, { 11, 0.392 }, { 12, 0.376 },
            { 13, 0.361 }, { 14, 0.349 }, { 15, 0.338 }, { 16, 0.329 }, { 17, 0.320 },
            { 18, 0.313 }, { 19, 0.306 }, { 20, 0.300 }, { 21, 0.295 }, { 22, 0.290 },
            { 23, 0.285 }, { 24, 0.281 }, { 25, 0.277 }
        };

        if (_alpha != 0.05)
        {
            // Adjust critical value for different alpha
            // This is an approximation
            double adjustmentFactor = Math.Log(0.05) / Math.Log(_alpha);
            if (criticalValues.TryGetValue(n, out double baseValue))
            {
                return Math.Min(1.0, baseValue * adjustmentFactor);
            }
        }

        if (criticalValues.TryGetValue(n, out double value))
        {
            return value;
        }

        // Fallback for n < 3 (shouldn't happen due to validation)
        return 1.0;
    }
}
