using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.TimeSeries;

/// <summary>
/// Detects anomalies using moving average deviation in time series.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This detector computes a moving average and identifies points that
/// deviate significantly from their local average. It's one of the simplest and most intuitive
/// time series anomaly detection methods.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Compute moving average and moving standard deviation
/// 2. For each point, compute deviation from local mean in std units
/// 3. High deviations indicate anomalies
/// </para>
/// <para>
/// <b>When to use:</b>
/// - Simple time series without complex patterns
/// - As a baseline method
/// - Real-time anomaly detection (streaming data)
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Window size: 20
/// - Threshold: 3 standard deviations
/// - Contamination: 0.1 (10%)
/// </para>
/// </remarks>
public class MovingAverageDetector<T> : AnomalyDetectorBase<T>
{
    private readonly int _windowSize;
    private readonly double _stdThreshold;

    /// <summary>
    /// Gets the window size.
    /// </summary>
    public int WindowSize => _windowSize;

    /// <summary>
    /// Gets the standard deviation threshold.
    /// </summary>
    public double StdThreshold => _stdThreshold;

    /// <summary>
    /// Creates a new Moving Average anomaly detector.
    /// </summary>
    /// <param name="windowSize">Size of the moving window. Default is 20.</param>
    /// <param name="stdThreshold">Number of standard deviations for threshold. Default is 3.</param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public MovingAverageDetector(int windowSize = 20, double stdThreshold = 3,
        double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (windowSize < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(windowSize),
                "WindowSize must be at least 2. Recommended is 20.");
        }

        if (stdThreshold <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(stdThreshold),
                "StdThreshold must be positive. Recommended is 3.");
        }

        _windowSize = windowSize;
        _stdThreshold = stdThreshold;
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

        if (X.Columns != 1)
        {
            throw new ArgumentException(
                "Moving Average expects univariate time series (1 column).",
                nameof(X));
        }

        int n = X.Rows;
        var scores = new Vector<T>(n);

        // Extract values
        var values = new double[n];
        for (int i = 0; i < n; i++)
        {
            values[i] = NumOps.ToDouble(X[i, 0]);
        }

        // Compute moving average and std for each point
        for (int i = 0; i < n; i++)
        {
            int start = Math.Max(0, i - _windowSize + 1);
            int windowLen = i - start + 1;

            // Compute local mean
            double sum = 0;
            for (int j = start; j <= i; j++)
            {
                sum += values[j];
            }
            double localMean = sum / windowLen;

            // Compute local std
            double sqSum = 0;
            for (int j = start; j <= i; j++)
            {
                sqSum += Math.Pow(values[j] - localMean, 2);
            }
            double localStd = Math.Sqrt(sqSum / windowLen);
            if (localStd < 1e-10) localStd = 1e-10;

            // Compute score
            double deviation = Math.Abs(values[i] - localMean);
            double score = deviation / localStd;

            scores[i] = NumOps.FromDouble(score);
        }

        return scores;
    }
}
