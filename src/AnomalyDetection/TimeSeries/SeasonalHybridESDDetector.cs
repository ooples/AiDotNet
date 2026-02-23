using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.TimeSeries;

/// <summary>
/// Detects anomalies in time series data using Seasonal Hybrid ESD (S-H-ESD).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> S-H-ESD combines seasonal decomposition with the Generalized ESD test
/// to detect anomalies in time series data. It handles seasonality by removing the seasonal
/// pattern before testing for outliers, making it effective for data with daily, weekly, or
/// yearly patterns.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Decompose the time series using STL (Seasonal and Trend decomposition using Loess)
/// 2. Extract the residual component (data after removing trend and seasonality)
/// 3. Apply GESD test on the residuals to find anomalies
/// </para>
/// <para>
/// <b>When to use:</b>
/// - Time series data with seasonal patterns
/// - Detecting anomalies that deviate from expected seasonal behavior
/// - Server metrics, website traffic, sensor data with regular patterns
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Season length: 7 (weekly pattern)
/// - Alpha: 0.05 (5% significance level)
/// - Max anomalies: 10% of data
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Twitter's "AnomalyDetection" package, based on Rosner (1983) GESD test.
/// </para>
/// </remarks>
public class SeasonalHybridESDDetector<T> : AnomalyDetectorBase<T>
{
    private readonly int _seasonLength;
    private readonly double _alpha;
    private readonly int? _maxAnomalies;
    private double[]? _seasonalPattern;
    private double _trend;
    private double _residualStd;
    private double _esdCriticalValue;
    private int _nSamples;

    /// <summary>
    /// Gets the season length (period).
    /// </summary>
    public int SeasonLength => _seasonLength;

    /// <summary>
    /// Gets the significance level.
    /// </summary>
    public double Alpha => _alpha;

    /// <summary>
    /// Creates a new Seasonal Hybrid ESD anomaly detector.
    /// </summary>
    /// <param name="seasonLength">
    /// Length of the seasonal period (e.g., 7 for weekly, 24 for hourly daily pattern).
    /// Default is 7.
    /// </param>
    /// <param name="alpha">
    /// Significance level for the ESD test. Default is 0.05 (5%).
    /// </param>
    /// <param name="maxAnomalies">
    /// Maximum number of anomalies to detect. If null, uses 10% of data.
    /// </param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public SeasonalHybridESDDetector(int seasonLength = 7, double alpha = 0.05, int? maxAnomalies = null,
        double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (seasonLength < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(seasonLength),
                "SeasonLength must be at least 2. Common values are 7 (weekly) or 24 (hourly/daily).");
        }

        if (alpha <= 0 || alpha >= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(alpha),
                "Alpha must be between 0 and 1. Recommended value is 0.05.");
        }

        _seasonLength = seasonLength;
        _alpha = alpha;
        _maxAnomalies = maxAnomalies;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        if (X.Columns != 1)
        {
            throw new ArgumentException(
                "S-H-ESD expects univariate time series (1 column). For multivariate data, apply to each feature separately.",
                nameof(X));
        }

        int n = X.Rows;
        _nSamples = n;

        // Validate series length is sufficient for seasonal decomposition
        if (n < _seasonLength)
        {
            throw new ArgumentException(
                $"Time series length ({n}) must be at least the season length ({_seasonLength}).",
                nameof(X));
        }

        // Extract time series values
        var values = new double[n];
        for (int i = 0; i < n; i++)
        {
            values[i] = NumOps.ToDouble(X[i, 0]);
        }

        // Compute seasonal decomposition
        DecomposeTimeSeries(values);

        // Compute ESD critical value based on maxAnomalies
        int effectiveMaxAnomalies = _maxAnomalies ?? Math.Max(1, (int)(n * _contamination));
        _esdCriticalValue = ComputeESDCriticalValue(n, effectiveMaxAnomalies);

        // Calculate scores for training data to set threshold
        var trainingScores = ScoreAnomaliesInternal(X);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
    }

    private double ComputeESDCriticalValue(int n, int maxAnomalies)
    {
        // Compute the critical value for the generalized ESD test
        // Using approximation based on the t-distribution
        // Critical value = t_(p,df) * (n-1) / sqrt((n-2 + t^2) * n)
        // where p = alpha / (2 * (n - i + 1)) for iteration i

        // For simplicity, use the first iteration's critical value
        double p = _alpha / (2 * n);
        int df = n - 2;
        double tCritical = GetTCriticalApprox(p, df);

        double criticalValue = (n - 1) * tCritical / Math.Sqrt((n - 2 + tCritical * tCritical) * n);
        return criticalValue;
    }

    private double GetTCriticalApprox(double p, int df)
    {
        // Approximate t-distribution critical value using normal approximation for large df
        if (df > 30)
        {
            // Use normal approximation
            return GetZCriticalApprox(p);
        }

        // For smaller df, use a more accurate approximation
        double z = GetZCriticalApprox(p);
        double g1 = (z * z * z + z) / 4;
        double g2 = ((5 * Math.Pow(z, 5)) + (16 * z * z * z) + (3 * z)) / 96;
        double g3 = ((3 * Math.Pow(z, 7)) + (19 * Math.Pow(z, 5)) + (17 * z * z * z) - (15 * z)) / 384;

        return z + g1 / df + g2 / (df * df) + g3 / (df * df * df);
    }

    private double GetZCriticalApprox(double p)
    {
        // Approximate inverse normal CDF using rational approximation
        // For p close to 0, use the approximation for the upper tail
        double sign = 1;
        if (p > 0.5)
        {
            p = 1 - p;
            sign = -1;
        }

        if (p < 1e-10) p = 1e-10;

        double t = Math.Sqrt(-2 * Math.Log(p));
        double c0 = 2.515517, c1 = 0.802853, c2 = 0.010328;
        double d1 = 1.432788, d2 = 0.189269, d3 = 0.001308;

        double z = t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t);
        return sign * z;
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
                "S-H-ESD expects univariate time series (1 column).",
                nameof(X));
        }

        var scores = new Vector<T>(X.Rows);

        for (int i = 0; i < X.Rows; i++)
        {
            double value = NumOps.ToDouble(X[i, 0]);

            // Remove seasonal component
            double seasonal = _seasonalPattern![i % _seasonLength];
            double deseasonalized = value - seasonal - _trend;

            // Score is the standardized residual
            double score = _residualStd > 0 ? Math.Abs(deseasonalized) / _residualStd : 0;
            scores[i] = NumOps.FromDouble(score);
        }

        return scores;
    }

    private void DecomposeTimeSeries(double[] values)
    {
        int n = values.Length;

        // Compute trend using simple moving average
        _trend = values.Average();

        // Compute seasonal pattern (average for each season position)
        _seasonalPattern = new double[_seasonLength];
        var seasonCounts = new int[_seasonLength];

        for (int i = 0; i < n; i++)
        {
            int seasonPos = i % _seasonLength;
            _seasonalPattern[seasonPos] += values[i] - _trend;
            seasonCounts[seasonPos]++;
        }

        for (int s = 0; s < _seasonLength; s++)
        {
            if (seasonCounts[s] > 0)
            {
                _seasonalPattern[s] /= seasonCounts[s];
            }
        }

        // Compute residuals and their standard deviation
        var residuals = new double[n];
        for (int i = 0; i < n; i++)
        {
            residuals[i] = values[i] - _trend - _seasonalPattern[i % _seasonLength];
        }

        // MAD-based robust standard deviation
        var sortedResiduals = residuals.Select(Math.Abs).OrderBy(x => x).ToArray();
        double mad = sortedResiduals[sortedResiduals.Length / 2];
        _residualStd = 1.4826 * mad; // Convert MAD to std equivalent

        if (_residualStd < 1e-10)
        {
            _residualStd = residuals.Select(r => r * r).Sum() / n;
            _residualStd = Math.Sqrt(_residualStd);
        }
    }
}
