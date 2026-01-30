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
public class DixonQTestDetector<T> : AnomalyDetectorBase<T>
{
    private readonly double _alpha;
    private Vector<T>? _minValues;
    private Vector<T>? _maxValues;
    private Vector<T>? _ranges;
    private Vector<T>? _secondMin;
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
        _minValues = new Vector<T>(_nFeatures);
        _maxValues = new Vector<T>(_nFeatures);
        _ranges = new Vector<T>(_nFeatures);
        _secondMin = new Vector<T>(_nFeatures);
        _secondMax = new Vector<T>(_nFeatures);

        for (int j = 0; j < _nFeatures; j++)
        {
            var column = X.GetColumn(j);
            var sorted = column.ToArray();
            Array.Sort(sorted, (a, b) => NumOps.LessThan(a, b) ? -1 : (NumOps.GreaterThan(a, b) ? 1 : 0));

            _minValues[j] = sorted[0];
            _maxValues[j] = sorted[sorted.Length - 1];
            _secondMin[j] = sorted[1];
            _secondMax[j] = sorted[sorted.Length - 2];
            _ranges[j] = NumOps.Subtract(_maxValues[j], _minValues[j]);
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

        if (X.Columns != _nFeatures)
        {
            throw new ArgumentException(
                $"Input has {X.Columns} features, but model was fitted with {_nFeatures} features.",
                nameof(X));
        }

        var scores = new Vector<T>(X.Rows);

        for (int i = 0; i < X.Rows; i++)
        {
            T maxQStatistic = NumOps.Zero;

            for (int j = 0; j < X.Columns; j++)
            {
                // Skip features with zero range
                if (NumOps.Equals(_ranges![j], NumOps.Zero))
                {
                    continue;
                }

                // Classic Dixon's Q test:
                // For minimum: Q = (x2 - x1) / (xn - x1)
                // For maximum: Q = (xn - x(n-1)) / (xn - x1)
                // For general points, compute how "extreme" they are using the same logic:
                // If closer to min: Q = (x - min) / range compared to (x2 - min) / range
                // If closer to max: Q = (max - x) / range compared to (max - x(n-1)) / range

                T value = X[i, j];
                T distFromMin = NumOps.Subtract(value, _minValues![j]);
                T distFromMax = NumOps.Subtract(_maxValues![j], value);

                double distMinD = NumOps.ToDouble(distFromMin);
                double distMaxD = NumOps.ToDouble(distFromMax);
                double rangeD = NumOps.ToDouble(_ranges[j]);

                T qStatistic;
                double valueD = NumOps.ToDouble(value);
                double minD = NumOps.ToDouble(_minValues![j]);
                double maxD = NumOps.ToDouble(_maxValues![j]);

                if (valueD < minD)
                {
                    // Point is MORE extreme than training minimum (beyond min)
                    // Higher Q score indicates potential outlier
                    T gapToSecond = NumOps.Subtract(_secondMin![j], _minValues[j]);
                    double gapD = NumOps.ToDouble(gapToSecond);
                    double beyondMin = minD - valueD;
                    qStatistic = NumOps.FromDouble((gapD + beyondMin) / rangeD + 1.0);
                }
                else if (valueD > maxD)
                {
                    // Point is MORE extreme than training maximum (beyond max)
                    T gapFromSecond = NumOps.Subtract(_maxValues[j], _secondMax![j]);
                    double gapD = NumOps.ToDouble(gapFromSecond);
                    double beyondMax = valueD - maxD;
                    qStatistic = NumOps.FromDouble((gapD + beyondMax) / rangeD + 1.0);
                }
                else if (distMinD <= distMaxD)
                {
                    // Point is within range, closer to minimum
                    // Classic Dixon Q: ratio of gap to nearest neighbor vs total range
                    qStatistic = NumOps.FromDouble(distMinD / rangeD);
                }
                else
                {
                    // Point is within range, closer to maximum
                    qStatistic = NumOps.FromDouble(distMaxD / rangeD);
                }

                if (NumOps.GreaterThan(qStatistic, maxQStatistic))
                {
                    maxQStatistic = qStatistic;
                }
            }

            scores[i] = maxQStatistic;
        }

        return scores;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Uses the Dixon's Q critical value for statistical anomaly classification based on sample size.
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> X)
    {
        EnsureFitted();

        var scores = ScoreAnomalies(X);
        var predictions = new Vector<T>(scores.Length);
        double qCritical = GetQCritical(_nSamples);
        T criticalT = NumOps.FromDouble(qCritical);

        for (int i = 0; i < scores.Length; i++)
        {
            // Points with Q > Q critical are anomalies (-1), otherwise inliers (1)
            predictions[i] = NumOps.GreaterThan(scores[i], criticalT)
                ? NumOps.FromDouble(-1)
                : NumOps.FromDouble(1);
        }

        return predictions;
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
