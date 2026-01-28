using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.Statistical;

/// <summary>
/// Detects anomalies using Hampel identifier (median-based outlier detection).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The Hampel identifier is a robust outlier detection method that uses
/// a sliding window to compute local median and MAD. It's particularly effective for time series
/// data where local context matters.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. For each point, compute median and MAD in a local window
/// 2. Compute the deviation from local median in MAD units
/// 3. Points exceeding the threshold are anomalies
/// </para>
/// <para>
/// <b>When to use:</b>
/// - Time series with local patterns
/// - When global statistics don't capture local anomalies
/// - As a robust alternative to moving average methods
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Window size: 7 (3 on each side)
/// - Threshold: 3 MAD units
/// - Scale factor: 1.4826
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Hampel, F.R. (1974). "The Influence Curve and its Role in Robust Estimation."
/// </para>
/// </remarks>
public class HampelDetector<T> : AnomalyDetectorBase<T>
{
    private readonly int _windowSize;
    private readonly double _madThreshold;
    private readonly double _scaleFactor;

    /// <summary>
    /// Gets the window size.
    /// </summary>
    public int WindowSize => _windowSize;

    /// <summary>
    /// Gets the threshold in MAD units.
    /// </summary>
    public double ThresholdMAD => _madThreshold;

    /// <summary>
    /// Creates a new Hampel anomaly detector.
    /// </summary>
    /// <param name="windowSize">Size of the local window. Default is 7.</param>
    /// <param name="threshold">Threshold in MAD units. Default is 3.</param>
    /// <param name="scaleFactor">MAD scale factor. Default is 1.4826.</param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public HampelDetector(int windowSize = 7, double threshold = 3, double scaleFactor = 1.4826,
        double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (windowSize < 3)
        {
            throw new ArgumentOutOfRangeException(nameof(windowSize),
                "WindowSize must be at least 3. Recommended is 7.");
        }

        if (threshold <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(threshold),
                "Threshold must be positive. Recommended is 3.");
        }

        _windowSize = windowSize;
        _madThreshold = threshold;
        _scaleFactor = scaleFactor;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

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

        int n = X.Rows;
        int d = X.Columns;
        int halfWindow = _windowSize / 2;

        var scores = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            double maxScore = 0;

            for (int j = 0; j < d; j++)
            {
                // Extract local window
                int start = Math.Max(0, i - halfWindow);
                int end = Math.Min(n, i + halfWindow + 1);
                int windowLen = end - start;

                var windowValues = new double[windowLen];
                for (int k = start; k < end; k++)
                {
                    windowValues[k - start] = NumOps.ToDouble(X[k, j]);
                }

                // Compute local median
                Array.Sort(windowValues);
                double localMedian = windowLen % 2 == 0
                    ? (windowValues[windowLen / 2 - 1] + windowValues[windowLen / 2]) / 2
                    : windowValues[windowLen / 2];

                // Compute local MAD
                var absDeviations = new double[windowLen];
                for (int k = 0; k < windowLen; k++)
                {
                    absDeviations[k] = Math.Abs(windowValues[k] - localMedian);
                }

                Array.Sort(absDeviations);
                double localMad = windowLen % 2 == 0
                    ? (absDeviations[windowLen / 2 - 1] + absDeviations[windowLen / 2]) / 2
                    : absDeviations[windowLen / 2];

                localMad *= _scaleFactor;
                if (localMad < 1e-10) localMad = 1e-10;

                // Compute score for this feature
                double value = NumOps.ToDouble(X[i, j]);
                double deviation = Math.Abs(value - localMedian);
                double featureScore = deviation / localMad;

                if (featureScore > maxScore)
                {
                    maxScore = featureScore;
                }
            }

            scores[i] = NumOps.FromDouble(maxScore);
        }

        return scores;
    }
}
