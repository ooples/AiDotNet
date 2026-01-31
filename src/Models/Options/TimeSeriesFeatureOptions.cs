namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for time series feature extraction transformers.
/// </summary>
/// <remarks>
/// <para>
/// This unified options class configures all time series feature extractors including
/// rolling statistics, volatility measures, correlation calculations, and lag/lead features.
/// </para>
/// <para><b>For Beginners:</b> This class is like a settings panel for feature extraction.
/// You can configure:
/// - What statistics to calculate (mean, std, etc.)
/// - How large the rolling windows should be
/// - Whether to auto-detect optimal settings
/// - What lag and lead features to create
/// </para>
/// </remarks>
public class TimeSeriesFeatureOptions
{
    #region Window Configuration

    /// <summary>
    /// Gets or sets the window sizes for rolling calculations.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> These are the "lookback periods" for rolling calculations.
    /// Common choices are [7, 14, 30] for daily data (weekly, bi-weekly, monthly patterns).
    /// </para>
    /// </remarks>
    public int[] WindowSizes { get; set; } = [7, 14, 30];

    /// <summary>
    /// Gets or sets whether to automatically detect optimal window sizes.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When enabled, the system analyzes your data's patterns
    /// (like weekly or monthly cycles) and suggests the best window sizes.
    /// </para>
    /// </remarks>
    public bool AutoDetectWindowSizes { get; set; } = false;

    /// <summary>
    /// Gets or sets the auto-detection method for window sizes.
    /// </summary>
    public WindowAutoDetectionMethod AutoDetectionMethod { get; set; } = WindowAutoDetectionMethod.Autocorrelation;

    /// <summary>
    /// Gets or sets the maximum number of auto-detected window sizes.
    /// </summary>
    public int MaxAutoDetectedWindows { get; set; } = 5;

    /// <summary>
    /// Gets or sets the minimum window size for auto-detection.
    /// </summary>
    public int MinWindowSize { get; set; } = 2;

    /// <summary>
    /// Gets or sets the maximum window size for auto-detection.
    /// </summary>
    public int MaxWindowSize { get; set; } = 365;

    #endregion

    #region Rolling Statistics Configuration

    /// <summary>
    /// Gets or sets which rolling statistics to calculate.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Select which statistics you want calculated over the rolling window.
    /// More statistics = more features but also more computation.
    /// </para>
    /// </remarks>
    public RollingStatistics EnabledStatistics { get; set; } = RollingStatistics.All;

    /// <summary>
    /// Gets or sets custom percentiles to calculate (in addition to standard quartiles).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Percentiles show what value a certain percentage of data falls below.
    /// For example, the 95th percentile is the value where 95% of values are smaller.
    /// Common choices: [0.05, 0.10, 0.90, 0.95] for risk analysis.
    /// </para>
    /// </remarks>
    public double[] CustomPercentiles { get; set; } = [0.05, 0.25, 0.75, 0.95];

    #endregion

    #region Volatility Configuration

    /// <summary>
    /// Gets or sets whether to calculate rolling volatility measures.
    /// </summary>
    public bool EnableVolatility { get; set; } = false;

    /// <summary>
    /// Gets or sets which volatility measures to calculate.
    /// </summary>
    public VolatilityMeasures EnabledVolatilityMeasures { get; set; } = VolatilityMeasures.All;

    /// <summary>
    /// Gets or sets the annualization factor for volatility calculations.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This scales volatility to an annual basis.
    /// For daily data, use 252 (trading days). For hourly, use 252*24.
    /// </para>
    /// </remarks>
    public double AnnualizationFactor { get; set; } = 252.0;

    /// <summary>
    /// Gets or sets whether to calculate returns (simple and log).
    /// </summary>
    public bool CalculateReturns { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to calculate momentum indicators.
    /// </summary>
    public bool CalculateMomentum { get; set; } = true;

    #endregion

    #region Correlation Configuration

    /// <summary>
    /// Gets or sets whether to calculate rolling correlations.
    /// </summary>
    public bool EnableCorrelation { get; set; } = false;

    /// <summary>
    /// Gets or sets the window sizes specifically for correlation calculations.
    /// </summary>
    /// <remarks>
    /// <para>If null, uses the general WindowSizes setting.</para>
    /// </remarks>
    public int[]? CorrelationWindowSizes { get; set; }

    /// <summary>
    /// Gets or sets whether to calculate full correlation matrix or just upper triangle.
    /// </summary>
    public bool FullCorrelationMatrix { get; set; } = false;

    #endregion

    #region Lag/Lead Configuration

    /// <summary>
    /// Gets or sets the lag steps for lagged feature generation.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Lag steps create features from past values.
    /// [1, 2, 3] means create features for "1 step ago", "2 steps ago", "3 steps ago".
    /// </para>
    /// </remarks>
    public int[] LagSteps { get; set; } = [];

    /// <summary>
    /// Gets or sets the lead steps for leading feature generation.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Lead steps create features from future values.
    /// This is useful for training targets but should not be used for production features.
    /// </para>
    /// </remarks>
    public int[] LeadSteps { get; set; } = [];

    #endregion

    #region Processing Configuration

    /// <summary>
    /// Gets or sets whether to use parallel processing for large datasets.
    /// </summary>
    public bool UseParallelProcessing { get; set; } = true;

    /// <summary>
    /// Gets or sets the minimum data length to trigger parallel processing.
    /// </summary>
    public int ParallelThreshold { get; set; } = 1000;

    /// <summary>
    /// Gets or sets how edge cases (beginning of series) should be handled.
    /// </summary>
    public EdgeHandling EdgeHandling { get; set; } = EdgeHandling.NaN;

    #endregion

    #region Feature Naming Configuration

    /// <summary>
    /// Gets or sets whether to generate descriptive feature names.
    /// </summary>
    public bool GenerateFeatureNames { get; set; } = true;

    /// <summary>
    /// Gets or sets the separator for feature name components.
    /// </summary>
    public string FeatureNameSeparator { get; set; } = "_";

    /// <summary>
    /// Gets or sets the input feature names (column names).
    /// </summary>
    /// <remarks>
    /// <para>If null, generic names like "feature_0", "feature_1" will be used.</para>
    /// </remarks>
    public string[]? InputFeatureNames { get; set; }

    #endregion

    #region Validation

    /// <summary>
    /// Validates the options and returns any validation errors.
    /// </summary>
    /// <returns>List of validation error messages, empty if valid.</returns>
    public List<string> Validate()
    {
        var errors = new List<string>();

        if (WindowSizes.Length == 0 && !AutoDetectWindowSizes)
        {
            errors.Add("WindowSizes must contain at least one value, or enable AutoDetectWindowSizes.");
        }

        if (WindowSizes.Any(w => w < 2))
        {
            errors.Add("All window sizes must be at least 2.");
        }

        if (MinWindowSize < 2)
        {
            errors.Add("MinWindowSize must be at least 2.");
        }

        if (MaxWindowSize < MinWindowSize)
        {
            errors.Add("MaxWindowSize must be greater than or equal to MinWindowSize.");
        }

        if (CustomPercentiles.Any(p => p < 0 || p > 1))
        {
            errors.Add("CustomPercentiles must be between 0 and 1.");
        }

        if (AnnualizationFactor <= 0)
        {
            errors.Add("AnnualizationFactor must be positive.");
        }

        if (LagSteps.Any(l => l < 1))
        {
            errors.Add("All lag steps must be at least 1.");
        }

        if (LeadSteps.Any(l => l < 1))
        {
            errors.Add("All lead steps must be at least 1.");
        }

        if (ParallelThreshold < 1)
        {
            errors.Add("ParallelThreshold must be at least 1.");
        }

        if (EnableCorrelation)
        {
            if (CorrelationWindowSizes is { Length: 0 })
            {
                errors.Add("CorrelationWindowSizes must contain at least one value when correlation is enabled.");
            }

            if (CorrelationWindowSizes?.Any(w => w < 2) == true)
            {
                errors.Add("All correlation window sizes must be at least 2.");
            }
        }

        return errors;
    }

    /// <summary>
    /// Creates a new options instance with default settings optimized for financial data.
    /// </summary>
    public static TimeSeriesFeatureOptions CreateForFinance()
    {
        return new TimeSeriesFeatureOptions
        {
            WindowSizes = [5, 10, 20, 60, 120, 252],
            EnabledStatistics = RollingStatistics.Mean | RollingStatistics.StandardDeviation |
                                RollingStatistics.Min | RollingStatistics.Max,
            EnableVolatility = true,
            EnabledVolatilityMeasures = VolatilityMeasures.All,
            CalculateReturns = true,
            CalculateMomentum = true,
            CustomPercentiles = [0.01, 0.05, 0.10, 0.90, 0.95, 0.99],
            LagSteps = [1, 2, 3, 5, 10, 20],
            AnnualizationFactor = 252.0
        };
    }

    /// <summary>
    /// Creates a new options instance with minimal settings for fast processing.
    /// </summary>
    public static TimeSeriesFeatureOptions CreateMinimal()
    {
        return new TimeSeriesFeatureOptions
        {
            WindowSizes = [7],
            EnabledStatistics = RollingStatistics.Mean | RollingStatistics.StandardDeviation,
            EnableVolatility = false,
            EnableCorrelation = false,
            CustomPercentiles = [],
            LagSteps = [1]
        };
    }

    #endregion
}

/// <summary>
/// Flags for selecting which rolling statistics to calculate.
/// </summary>
[Flags]
public enum RollingStatistics
{
    /// <summary>No statistics.</summary>
    None = 0,

    /// <summary>Arithmetic mean (average).</summary>
    Mean = 1 << 0,

    /// <summary>Median (50th percentile).</summary>
    Median = 1 << 1,

    /// <summary>Standard deviation.</summary>
    StandardDeviation = 1 << 2,

    /// <summary>Variance.</summary>
    Variance = 1 << 3,

    /// <summary>Minimum value.</summary>
    Min = 1 << 4,

    /// <summary>Maximum value.</summary>
    Max = 1 << 5,

    /// <summary>Sum of values.</summary>
    Sum = 1 << 6,

    /// <summary>Count of values.</summary>
    Count = 1 << 7,

    /// <summary>Range (max - min).</summary>
    Range = 1 << 8,

    /// <summary>Skewness (asymmetry measure).</summary>
    Skewness = 1 << 9,

    /// <summary>Kurtosis (tail heaviness measure).</summary>
    Kurtosis = 1 << 10,

    /// <summary>Interquartile range (Q3 - Q1).</summary>
    IQR = 1 << 11,

    /// <summary>Median absolute deviation.</summary>
    MAD = 1 << 12,

    /// <summary>First quartile (25th percentile).</summary>
    FirstQuartile = 1 << 13,

    /// <summary>Third quartile (75th percentile).</summary>
    ThirdQuartile = 1 << 14,

    /// <summary>All central tendency measures.</summary>
    CentralTendency = Mean | Median,

    /// <summary>All dispersion measures.</summary>
    Dispersion = StandardDeviation | Variance | MAD | IQR,

    /// <summary>All range measures.</summary>
    RangeMeasures = Min | Max | Range | Sum | Count,

    /// <summary>All distribution shape measures.</summary>
    DistributionShape = Skewness | Kurtosis,

    /// <summary>All quartile measures.</summary>
    Quartiles = FirstQuartile | Median | ThirdQuartile | IQR,

    /// <summary>All available statistics.</summary>
    All = CentralTendency | Dispersion | RangeMeasures | DistributionShape | Quartiles
}

/// <summary>
/// Flags for selecting which volatility measures to calculate.
/// </summary>
[Flags]
public enum VolatilityMeasures
{
    /// <summary>No volatility measures.</summary>
    None = 0,

    /// <summary>Realized volatility (standard deviation of returns).</summary>
    RealizedVolatility = 1 << 0,

    /// <summary>Parkinson volatility (high-low range based).</summary>
    ParkinsonVolatility = 1 << 1,

    /// <summary>Garman-Klass volatility (OHLC based).</summary>
    GarmanKlassVolatility = 1 << 2,

    /// <summary>Simple returns (price change / previous price).</summary>
    SimpleReturns = 1 << 3,

    /// <summary>Log returns (ln(price / previous price)).</summary>
    LogReturns = 1 << 4,

    /// <summary>Price momentum (current price / past price - 1).</summary>
    Momentum = 1 << 5,

    /// <summary>All volatility measures.</summary>
    All = RealizedVolatility | ParkinsonVolatility | GarmanKlassVolatility |
          SimpleReturns | LogReturns | Momentum
}

/// <summary>
/// Methods for auto-detecting optimal window sizes.
/// </summary>
public enum WindowAutoDetectionMethod
{
    /// <summary>
    /// Use autocorrelation function (ACF) to detect seasonality.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> ACF measures how similar the data is to itself at different time lags.
    /// Peaks in ACF indicate seasonal patterns.
    /// </para>
    /// </remarks>
    Autocorrelation,

    /// <summary>
    /// Use spectral analysis (FFT) to detect dominant frequencies.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This finds repeating cycles in your data using frequency analysis.
    /// Good for detecting multiple overlapping patterns.
    /// </para>
    /// </remarks>
    SpectralAnalysis,

    /// <summary>
    /// Use grid search with cross-validation to find best windows.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Tries many different window sizes and picks the ones
    /// that give the best prediction results. Slower but more accurate.
    /// </para>
    /// </remarks>
    GridSearch,

    /// <summary>
    /// Use simple heuristic rules based on data characteristics.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Uses simple rules like "use sqrt(n)" or standard
    /// intervals. Fast but less tailored to your specific data.
    /// </para>
    /// </remarks>
    Heuristic
}

/// <summary>
/// How to handle edge cases where the full window is not available.
/// </summary>
public enum EdgeHandling
{
    /// <summary>
    /// Fill with NaN where window extends beyond data boundaries.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The first few values will be NaN because we don't have
    /// enough history yet. This is the safest approach - it clearly marks incomplete data.
    /// </para>
    /// </remarks>
    NaN,

    /// <summary>
    /// Use partial windows (calculate with available data).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Even if we don't have a full window, calculate with
    /// whatever data is available. This gives more values but they may be less reliable.
    /// </para>
    /// </remarks>
    Partial,

    /// <summary>
    /// Truncate output to only include complete windows.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Only output values where we had a complete window.
    /// The output will be shorter than the input.
    /// </para>
    /// </remarks>
    Truncate,

    /// <summary>
    /// Use the first available value to fill the beginning.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use the first complete calculation to fill in
    /// all the positions before it. The beginning values will all be the same.
    /// </para>
    /// </remarks>
    ForwardFill
}
