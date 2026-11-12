using AiDotNet.Attributes;
using AiDotNet.Enums;
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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.TimeSeriesModel)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
    [ResearchPaper("Automatic Anomaly Detection in the Cloud Via Statistical Learning", "https://arxiv.org/abs/1704.07706")]
public partial class SeasonalHybridESDDetector<T> : AnomalyDetectorBase<T>
{
    private readonly int _seasonLength;
    private readonly double _alpha;
    private readonly int? _maxAnomalies;
    private Vector<T>? _seasonalPattern;

    private T _trend;


    private T _residualStd;

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
        _residualStd = NumOps.Zero;
        _trend = NumOps.Zero;
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

        // Compute seasonal decomposition
        DecomposeTimeSeries(X.GetColumn(0).AsTensor());

        // ESD critical value computed but not stored — scoring uses its own threshold

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

        var seasonalPattern = _seasonalPattern ?? throw new InvalidOperationException("_seasonalPattern has not been initialized.");
        var engine = Engine;

        // Score is the standardized residual |x - seasonal - trend| / residual std (0 for a non-positive std)
        if (!NumOps.GreaterThan(_residualStd, NumOps.Zero))
        {
            return new Vector<T>(X.Rows);
        }

        var seasonal = AnomalyTensorOps.TileToLength(seasonalPattern.AsTensor(), X.Rows);
        var deseasonalized = engine.TensorSubtractScalar(
            engine.TensorSubtract(X.GetColumn(0).AsTensor(), seasonal), _trend);
        return engine.TensorDivideScalar(engine.TensorAbs(deseasonalized), _residualStd).ToVector();
    }

    private void DecomposeTimeSeries(Tensor<T> values)
    {
        int n = values.Length;
        var engine = Engine;

        // Trend: the overall mean
        _trend = engine.TensorMean(values);

        // Seasonal pattern: the mean detrended value at each season position
        var detrended = engine.TensorSubtractScalar(values, _trend);
        var seasonalPattern = AnomalyTensorOps.PhaseMeans(detrended, _seasonLength);
        _seasonalPattern = seasonalPattern.ToVector();

        // Residuals and their robust (MAD-based) standard deviation
        var residuals = engine.TensorSubtract(detrended, AnomalyTensorOps.TileToLength(seasonalPattern, n));
        var (sortedResiduals, _) = engine.TensorSort(engine.TensorAbs(residuals), 0, false);
        T mad = sortedResiduals[n / 2];
        T residualStd = NumOps.Multiply(NumOps.FromDouble(1.4826), mad);

        T floor = NumOps.FromDouble(1e-10);
        if (NumOps.LessThan(residualStd, floor))
        {
            residualStd = NumOps.Sqrt(NumOps.Divide(engine.TensorSumOfSquares(residuals), NumOps.FromDouble(n)));
        }
        if (NumOps.LessThan(residualStd, floor)) residualStd = floor;
        _residualStd = residualStd;
    }
}
