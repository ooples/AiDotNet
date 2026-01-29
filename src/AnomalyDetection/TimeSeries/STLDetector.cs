using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.TimeSeries;

/// <summary>
/// Detects anomalies using STL (Seasonal and Trend decomposition using Loess).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> STL decomposes a time series into three components: trend, seasonal,
/// and residual. The residual component contains the irregular variations. Large residuals
/// indicate anomalies that don't fit the expected trend and seasonal patterns.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Apply STL decomposition to extract trend, seasonal, and residual
/// 2. Standardize the residual component
/// 3. Large standardized residuals indicate anomalies
/// </para>
/// <para>
/// <b>When to use:</b>
/// - Time series with clear trend and/or seasonal patterns
/// - When anomalies disrupt expected patterns
/// - Sales, weather, sensor data with periodicity
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Season length: 7 (weekly pattern)
/// - Trend smoothness: 15
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Cleveland, R.B., et al. (1990). "STL: A Seasonal-Trend Decomposition Procedure
/// Based on Loess." Journal of Official Statistics.
/// </para>
/// </remarks>
public class STLDetector<T> : AnomalyDetectorBase<T>
{
    private readonly int _seasonLength;
    private readonly int _trendSmoothness;
    private double[]? _trend;
    private double[]? _seasonal;
    private double _residualStd;

    /// <summary>
    /// Gets the season length.
    /// </summary>
    public int SeasonLength => _seasonLength;

    /// <summary>
    /// Gets the trend smoothness parameter.
    /// </summary>
    public int TrendSmoothness => _trendSmoothness;

    /// <summary>
    /// Creates a new STL anomaly detector.
    /// </summary>
    /// <param name="seasonLength">Length of seasonal period. Default is 7.</param>
    /// <param name="trendSmoothness">Smoothness of trend component. Default is 15.</param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public STLDetector(int seasonLength = 7, int trendSmoothness = 15,
        double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (seasonLength < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(seasonLength),
                "SeasonLength must be at least 2. Recommended is 7.");
        }

        if (trendSmoothness < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(trendSmoothness),
                "TrendSmoothness must be at least 1. Recommended is 15.");
        }

        _seasonLength = seasonLength;
        _trendSmoothness = trendSmoothness;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        if (X.Columns != 1)
        {
            throw new ArgumentException(
                "STL expects univariate time series (1 column).",
                nameof(X));
        }

        int n = X.Rows;

        // Validate that series length is at least the season length
        if (n < _seasonLength)
        {
            throw new ArgumentException(
                $"Time series length ({n}) must be at least the season length ({_seasonLength}). " +
                "Either provide more data or use a smaller season length.",
                nameof(X));
        }

        var values = new double[n];
        for (int i = 0; i < n; i++)
        {
            values[i] = NumOps.ToDouble(X[i, 0]);
        }

        // Perform STL decomposition
        DecomposeSTL(values);

        // Calculate scores for training data to set threshold
        var trainingScores = ScoreAnomaliesInternal(X);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
    }

    private void DecomposeSTL(double[] values)
    {
        int n = values.Length;

        // Initialize
        _trend = new double[n];
        _seasonal = new double[n];

        // Iterative STL (simplified)
        for (int iter = 0; iter < 3; iter++)
        {
            // Step 1: Detrend
            var detrended = new double[n];
            for (int i = 0; i < n; i++)
            {
                detrended[i] = values[i] - _trend[i];
            }

            // Step 2: Extract seasonal using subseries
            var seasonalTemp = new double[n];
            for (int s = 0; s < _seasonLength; s++)
            {
                // Extract subseries for this season position
                var subseries = new List<double>();
                for (int i = s; i < n; i += _seasonLength)
                {
                    subseries.Add(detrended[i]);
                }

                // Smooth subseries (simple average for now)
                double subMean = subseries.Average();

                // Assign to seasonal
                for (int i = s; i < n; i += _seasonLength)
                {
                    seasonalTemp[i] = subMean;
                }
            }

            // Normalize seasonal (mean = 0)
            double seasonalMean = seasonalTemp.Average();
            for (int i = 0; i < n; i++)
            {
                _seasonal[i] = seasonalTemp[i] - seasonalMean;
            }

            // Step 3: Deseasonalize
            var deseasoned = new double[n];
            for (int i = 0; i < n; i++)
            {
                deseasoned[i] = values[i] - _seasonal[i];
            }

            // Step 4: Smooth to get trend (moving average)
            _trend = SmoothMovingAverage(deseasoned, _trendSmoothness);
        }

        // Compute residual standard deviation
        var residuals = new double[n];
        for (int i = 0; i < n; i++)
        {
            residuals[i] = values[i] - _trend[i] - _seasonal[i];
        }

        _residualStd = Math.Sqrt(residuals.Select(r => r * r).Average());
        if (_residualStd < 1e-10) _residualStd = 1e-10;
    }

    private double[] SmoothMovingAverage(double[] values, int windowSize)
    {
        int n = values.Length;
        var smoothed = new double[n];
        int halfWindow = windowSize / 2;

        for (int i = 0; i < n; i++)
        {
            int start = Math.Max(0, i - halfWindow);
            int end = Math.Min(n, i + halfWindow + 1);
            double sum = 0;
            for (int j = start; j < end; j++)
            {
                sum += values[j];
            }
            smoothed[i] = sum / (end - start);
        }

        return smoothed;
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
                "STL expects univariate time series (1 column).",
                nameof(X));
        }

        var trend = _trend;
        var seasonal = _seasonal;

        if (trend == null || seasonal == null)
        {
            throw new InvalidOperationException("Model not properly fitted.");
        }

        var scores = new Vector<T>(X.Rows);

        for (int i = 0; i < X.Rows; i++)
        {
            double value = NumOps.ToDouble(X[i, 0]);

            // Use modular indexing for seasonal if test data is longer
            int trendIdx = Math.Min(i, trend.Length - 1);
            int seasonIdx = i % _seasonLength;

            double residual = value - trend[trendIdx] - seasonal[seasonIdx];
            double score = Math.Abs(residual) / _residualStd;

            scores[i] = NumOps.FromDouble(score);
        }

        return scores;
    }
}
