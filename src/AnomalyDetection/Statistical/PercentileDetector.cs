using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.Statistical;

/// <summary>
/// Detects anomalies using percentile-based thresholds.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This detector uses percentiles to identify extreme values. Points
/// that fall below the low percentile or above the high percentile are flagged as anomalies.
/// It's simple and effective for univariate or feature-wise anomaly detection.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Compute the specified percentiles for each feature
/// 2. For each point, check if values are outside the percentile bounds
/// 3. Score based on how far outside the bounds a point falls
/// </para>
/// <para>
/// <b>When to use:</b>
/// - Simple univariate anomaly detection
/// - When you don't want to assume a distribution
/// - As a baseline method
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Low percentile: 5 (5th percentile)
/// - High percentile: 95 (95th percentile)
/// - Contamination: 0.1 (10%)
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
    [ModelPaper("Exploratory Data Analysis", "https://doi.org/10.1002/bimj.4710230408")]
public class PercentileDetector<T> : AnomalyDetectorBase<T>
{
    private readonly double _lowPercentile;
    private readonly double _highPercentile;
    private Vector<T>? _lowThresholds;
    private Vector<T>? _highThresholds;
    private Vector<T>? _ranges;

    /// <summary>
    /// Gets the low percentile threshold.
    /// </summary>
    public double LowPercentile => _lowPercentile;

    /// <summary>
    /// Gets the high percentile threshold.
    /// </summary>
    public double HighPercentile => _highPercentile;

    /// <summary>
    /// Creates a new Percentile anomaly detector.
    /// </summary>
    /// <param name="lowPercentile">Low percentile threshold. Default is 5.</param>
    /// <param name="highPercentile">High percentile threshold. Default is 95.</param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public PercentileDetector(double lowPercentile = 5, double highPercentile = 95,
        double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (lowPercentile < 0 || lowPercentile > 100)
        {
            throw new ArgumentOutOfRangeException(nameof(lowPercentile),
                "LowPercentile must be between 0 and 100.");
        }

        if (highPercentile < 0 || highPercentile > 100)
        {
            throw new ArgumentOutOfRangeException(nameof(highPercentile),
                "HighPercentile must be between 0 and 100.");
        }

        if (lowPercentile >= highPercentile)
        {
            throw new ArgumentException(
                "LowPercentile must be less than HighPercentile.");
        }

        _lowPercentile = lowPercentile;
        _highPercentile = highPercentile;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        int n = X.Rows;
        int d = X.Columns;

        _lowThresholds = new Vector<T>(d);
        _highThresholds = new Vector<T>(d);
        _ranges = new Vector<T>(d);
        T eps = NumOps.FromDouble(1e-10);

        for (int j = 0; j < d; j++)
        {
            var values = new T[n];
            for (int i = 0; i < n; i++)
            {
                values[i] = X[i, j];
            }

            Array.Sort(values, (a, b) => NumOps.Compare(a, b));

            // Compute percentiles
            _lowThresholds[j] = Percentile(values, _lowPercentile);
            _highThresholds[j] = Percentile(values, _highPercentile);
            _ranges[j] = NumOps.Subtract(_highThresholds[j], _lowThresholds[j]);
            if (NumOps.LessThan(_ranges[j], eps)) _ranges[j] = eps;
        }

        // Calculate scores for training data to set threshold
        var trainingScores = ScoreAnomaliesInternal(X);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
    }

    private T Percentile(T[] sortedValues, double percentile)
    {
        int n = sortedValues.Length;
        double index = (percentile / 100.0) * (n - 1);
        int lower = (int)Math.Floor(index);
        int upper = (int)Math.Ceiling(index);

        if (lower == upper || upper >= n)
        {
            return sortedValues[Math.Min(lower, n - 1)];
        }

        T fraction = NumOps.FromDouble(index - lower);
        return NumOps.Add(sortedValues[lower],
            NumOps.Multiply(fraction, NumOps.Subtract(sortedValues[upper], sortedValues[lower])));
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

        var lowThresholds = _lowThresholds;
        var highThresholds = _highThresholds;
        var ranges = _ranges;

        if (lowThresholds == null || highThresholds == null || ranges == null)
        {
            throw new InvalidOperationException("Model not properly fitted.");
        }

        var scores = new Vector<T>(X.Rows);

        for (int i = 0; i < X.Rows; i++)
        {
            T maxScore = NumOps.Zero;

            for (int j = 0; j < X.Columns; j++)
            {
                T value = X[i, j];
                T score;

                if (NumOps.LessThan(value, lowThresholds[j]))
                {
                    // Below low percentile
                    score = NumOps.Divide(NumOps.Subtract(lowThresholds[j], value), ranges[j]);
                }
                else if (NumOps.GreaterThan(value, highThresholds[j]))
                {
                    // Above high percentile
                    score = NumOps.Divide(NumOps.Subtract(value, highThresholds[j]), ranges[j]);
                }
                else
                {
                    // Within normal range
                    score = NumOps.Zero;
                }

                if (NumOps.GreaterThan(score, maxScore))
                {
                    maxScore = score;
                }
            }

            scores[i] = maxScore;
        }

        return scores;
    }
}
