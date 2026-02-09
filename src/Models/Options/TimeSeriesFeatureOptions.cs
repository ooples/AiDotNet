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
public class TimeSeriesFeatureOptions : ModelOptions
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

    /// <summary>
    /// Gets or sets the OHLC column configuration for proper volatility calculations.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> OHLC stands for Open, High, Low, Close - the four key prices
    /// recorded for each time period (e.g., each day) in financial data.
    ///
    /// When you have OHLC data, the Parkinson and Garman-Klass volatility measures can use
    /// the actual high/low/open/close values for more accurate calculations instead of approximations.
    ///
    /// Example usage:
    /// <code>
    /// options.OhlcColumns = new OhlcColumnConfig
    /// {
    ///     OpenIndex = 0,    // First column is Open price
    ///     HighIndex = 1,    // Second column is High price
    ///     LowIndex = 2,     // Third column is Low price
    ///     CloseIndex = 3    // Fourth column is Close price
    /// };
    /// </code>
    /// </para>
    /// </remarks>
    public OhlcColumnConfig? OhlcColumns { get; set; }

    /// <summary>
    /// Gets or sets the decay factor (lambda) for EWMA volatility calculation.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The decay factor controls how quickly old observations lose their weight.
    /// A higher lambda (closer to 1) means older data stays relevant longer.
    /// A lower lambda means recent data dominates the calculation.
    ///
    /// Common values:
    /// - 0.94: RiskMetrics daily volatility standard
    /// - 0.97: RiskMetrics monthly volatility standard
    /// - 0.84-0.94: Typical range for financial applications
    /// </para>
    /// </remarks>
    public double EwmaDecayFactor { get; set; } = 0.94;

    /// <summary>
    /// Gets or sets the omega (constant) parameter for GARCH(1,1) model.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Omega is the long-run average variance weight.
    /// In GARCH(1,1): sigma²_t = omega + alpha * r²_{t-1} + beta * sigma²_{t-1}
    /// For stability, omega + alpha + beta should be less than 1.
    /// </para>
    /// </remarks>
    public double GarchOmega { get; set; } = 0.00001;

    /// <summary>
    /// Gets or sets the alpha parameter for GARCH(1,1) model.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Alpha determines how much recent squared returns affect volatility.
    /// Higher alpha means volatility reacts more quickly to market shocks.
    /// Typical range: 0.05 - 0.15
    /// </para>
    /// </remarks>
    public double GarchAlpha { get; set; } = 0.09;

    /// <summary>
    /// Gets or sets the beta parameter for GARCH(1,1) model.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Beta determines how persistent volatility is.
    /// Higher beta means volatility shocks take longer to decay.
    /// Typical range: 0.85 - 0.95
    /// alpha + beta should be close to but less than 1 for stationarity.
    /// </para>
    /// </remarks>
    public double GarchBeta { get; set; } = 0.90;

    #endregion

    #region Technical Indicators Configuration

    /// <summary>
    /// Gets or sets whether to calculate technical indicators.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Technical indicators are mathematical calculations based on price,
    /// volume, or other data that traders use to predict future price movements.
    /// These are widely used in finance and algorithmic trading.
    /// </para>
    /// </remarks>
    public bool EnableTechnicalIndicators { get; set; } = false;

    /// <summary>
    /// Gets or sets which technical indicators to calculate.
    /// </summary>
    public TechnicalIndicators EnabledIndicators { get; set; } = TechnicalIndicators.All;

    /// <summary>
    /// Gets or sets the short period for EMA/MACD calculations.
    /// </summary>
    /// <remarks>
    /// <para>Default is 12, commonly used for MACD.</para>
    /// </remarks>
    public int ShortPeriod { get; set; } = 12;

    /// <summary>
    /// Gets or sets the long period for EMA/MACD calculations.
    /// </summary>
    /// <remarks>
    /// <para>Default is 26, commonly used for MACD.</para>
    /// </remarks>
    public int LongPeriod { get; set; } = 26;

    /// <summary>
    /// Gets or sets the signal period for MACD signal line.
    /// </summary>
    /// <remarks>
    /// <para>Default is 9, commonly used for MACD signal line.</para>
    /// </remarks>
    public int SignalPeriod { get; set; } = 9;

    /// <summary>
    /// Gets or sets the RSI period.
    /// </summary>
    /// <remarks>
    /// <para>Default is 14, the most commonly used RSI period.</para>
    /// </remarks>
    public int RsiPeriod { get; set; } = 14;

    /// <summary>
    /// Gets or sets the number of standard deviations for Bollinger Bands.
    /// </summary>
    /// <remarks>
    /// <para>Default is 2.0, the standard setting.</para>
    /// </remarks>
    public double BollingerBandStdDev { get; set; } = 2.0;

    /// <summary>
    /// Gets or sets the stochastic oscillator K period.
    /// </summary>
    public int StochasticKPeriod { get; set; } = 14;

    /// <summary>
    /// Gets or sets the stochastic oscillator D period (smoothing).
    /// </summary>
    public int StochasticDPeriod { get; set; } = 3;

    /// <summary>
    /// Gets or sets the CCI (Commodity Channel Index) period.
    /// </summary>
    public int CciPeriod { get; set; } = 20;

    /// <summary>
    /// Gets or sets the ADX (Average Directional Index) period.
    /// </summary>
    public int AdxPeriod { get; set; } = 14;

    #endregion

    #region Seasonality Configuration

    /// <summary>
    /// Gets or sets whether to generate seasonality and calendar features.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Seasonality features capture repeating patterns in time series data.
    /// For example, sales might be higher on weekends, or energy usage peaks in summer.
    /// These features help models learn these cyclical patterns.
    /// </para>
    /// </remarks>
    public bool EnableSeasonality { get; set; } = false;

    /// <summary>
    /// Gets or sets which seasonality features to generate.
    /// </summary>
    public SeasonalityFeatures EnabledSeasonalityFeatures { get; set; } = SeasonalityFeatures.All;

    /// <summary>
    /// Gets or sets the seasonal periods for Fourier features.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> These are the cycle lengths you want to capture.
    /// Common values: 7 (weekly), 30/31 (monthly), 365/252 (yearly).
    /// For hourly data, include 24 (daily cycle).
    /// </para>
    /// </remarks>
    public int[] SeasonalPeriods { get; set; } = [7, 30, 365];

    /// <summary>
    /// Gets or sets the number of Fourier terms per seasonal period.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> More terms capture more complex seasonal patterns.
    /// Typically 1-3 terms per period is sufficient. Too many terms can lead to overfitting.
    /// </para>
    /// </remarks>
    public int FourierTerms { get; set; } = 2;

    /// <summary>
    /// Gets or sets the start date of the time series for calendar calculations.
    /// </summary>
    /// <remarks>
    /// <para>If null, the transformer will use index-based features instead of calendar features.</para>
    /// </remarks>
    public DateTime? TimeSeriesStartDate { get; set; }

    /// <summary>
    /// Gets or sets the time interval between data points for calendar feature calculation.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This specifies the frequency of your data:
    /// - Daily data: TimeSpan.FromDays(1)
    /// - Hourly data: TimeSpan.FromHours(1)
    /// - 5-minute data: TimeSpan.FromMinutes(5)
    /// </para>
    /// </remarks>
    public TimeSpan? TimeSeriesInterval { get; set; }

    /// <summary>
    /// Gets or sets custom holiday dates to generate holiday features.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Add dates for holidays relevant to your data.
    /// The transformer will create binary features indicating if a data point falls on a holiday.
    /// </para>
    /// </remarks>
    public DateTime[]? HolidayDates { get; set; }

    /// <summary>
    /// Gets or sets the number of days before/after holidays to flag as "near holiday".
    /// </summary>
    public int HolidayWindowDays { get; set; } = 2;

    /// <summary>
    /// Gets or sets whether data represents trading days (skips weekends/holidays).
    /// </summary>
    public bool IsTradingDayData { get; set; } = false;

    #endregion

    #region Differencing Configuration

    /// <summary>
    /// Gets or sets whether to apply differencing transformations.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Differencing makes time series data stationary by removing trends.
    /// Many forecasting models (like ARIMA) require stationary data to work well.
    /// Stationary data has constant mean and variance over time.
    /// </para>
    /// </remarks>
    public bool EnableDifferencing { get; set; } = false;

    /// <summary>
    /// Gets or sets which differencing features to compute.
    /// </summary>
    public DifferencingFeatures EnabledDifferencingFeatures { get; set; } = DifferencingFeatures.All;

    /// <summary>
    /// Gets or sets the differencing order (number of times to difference).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Order 1 = compute change from previous value.
    /// Order 2 = difference the differences (acceleration). Usually 1 or 2 is sufficient.
    /// </para>
    /// </remarks>
    public int DifferencingOrder { get; set; } = 1;

    /// <summary>
    /// Gets or sets the seasonal period for seasonal differencing.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> For weekly patterns use 7, for yearly patterns use 365 (or 252 for trading days).
    /// Seasonal differencing removes repeating patterns at specific intervals.
    /// </para>
    /// </remarks>
    public int SeasonalDifferencingPeriod { get; set; } = 7;

    /// <summary>
    /// Gets or sets the polynomial degree for detrending.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Degree 1 = linear trend (straight line).
    /// Degree 2 = quadratic trend (parabola). Higher degrees capture more complex trends.
    /// </para>
    /// </remarks>
    public int DetrendingPolynomialDegree { get; set; } = 1;

    /// <summary>
    /// Gets or sets the smoothing parameter (lambda) for Hodrick-Prescott filter.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Higher lambda = smoother trend extraction.
    /// Common values: 1600 for quarterly data, 6.25 for annual data, 129600 for monthly data.
    /// </para>
    /// </remarks>
    public double HodrickPrescottLambda { get; set; } = 1600;

    /// <summary>
    /// Gets or sets the seasonal period for STL decomposition.
    /// </summary>
    public int StlSeasonalPeriod { get; set; } = 7;

    /// <summary>
    /// Gets or sets the number of iterations for STL decomposition robustness.
    /// </summary>
    public int StlRobustIterations { get; set; } = 2;

    #endregion

    #region Rolling Regression Configuration

    /// <summary>
    /// Gets or sets whether to calculate rolling regression features.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Rolling regression features measure how an asset moves relative
    /// to a benchmark (like the market). These include:
    /// - Beta: How much the asset moves with the market (beta > 1 means more volatile than market)
    /// - Alpha: Excess return over what the market explains
    /// - R²: How well the market explains the asset's movements
    /// - Sharpe Ratio: Risk-adjusted return (higher is better)
    /// - Sortino Ratio: Downside risk-adjusted return (ignores upside volatility)
    /// </para>
    /// </remarks>
    public bool EnableRollingRegression { get; set; } = false;

    /// <summary>
    /// Gets or sets which rolling regression features to calculate.
    /// </summary>
    public RollingRegressionFeatures EnabledRegressionFeatures { get; set; } = RollingRegressionFeatures.All;

    /// <summary>
    /// Gets or sets the index of the benchmark column for regression calculations.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The benchmark is what you compare against (like the S&P 500).
    /// Set this to the column index where your benchmark data is located.
    /// If null, the first column will be used as the benchmark and other columns compared against it.
    /// </para>
    /// </remarks>
    public int? BenchmarkColumnIndex { get; set; }

    /// <summary>
    /// Gets or sets the risk-free rate for Sharpe and Sortino ratio calculations.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The risk-free rate is what you could earn with zero risk
    /// (like treasury bills). Typical values are 0.02-0.05 (2-5% annual).
    /// This should match the frequency of your data:
    /// - For daily data with 252 trading days: dailyRate = annualRate / 252
    /// - For monthly data: monthlyRate = annualRate / 12
    /// </para>
    /// </remarks>
    public double RiskFreeRate { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets whether the risk-free rate is already period-adjusted.
    /// </summary>
    /// <remarks>
    /// <para>If false, the risk-free rate will be divided by the annualization factor.</para>
    /// </remarks>
    public bool RiskFreeRateIsPeriodAdjusted { get; set; } = false;

    /// <summary>
    /// Gets or sets the minimum acceptable return (MAR) for Sortino ratio calculation.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The MAR is the minimum return you consider acceptable.
    /// Returns below this are considered "downside" risk. Typically set to 0 (no loss)
    /// or to the risk-free rate.
    /// </para>
    /// </remarks>
    public double MinimumAcceptableReturn { get; set; } = 0.0;

    #endregion

    #region Anomaly Detection Configuration

    /// <summary>
    /// Gets or sets whether to calculate anomaly detection features.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Anomaly detection features help identify unusual patterns
    /// in time series data. These can indicate:
    /// - Sudden spikes or drops in values
    /// - Values that are far from normal behavior
    /// - Shifts in the underlying mean or trend
    ///
    /// Common uses: fraud detection, equipment failure prediction, quality control.
    /// </para>
    /// </remarks>
    public bool EnableAnomalyDetection { get; set; } = false;

    /// <summary>
    /// Gets or sets which anomaly detection features to calculate.
    /// </summary>
    public AnomalyFeatures EnabledAnomalyFeatures { get; set; } = AnomalyFeatures.All;

    /// <summary>
    /// Gets or sets the Z-score threshold for flagging anomalies.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Z-score measures how many standard deviations a value is from the mean.
    /// A threshold of 3 means values more than 3 standard deviations away are flagged.
    /// Common thresholds: 2.0 (lenient), 3.0 (standard), 4.0 (strict).
    /// </para>
    /// </remarks>
    public double ZScoreThreshold { get; set; } = 3.0;

    /// <summary>
    /// Gets or sets the IQR multiplier for outlier detection.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> IQR (Interquartile Range) is Q3 - Q1.
    /// Values below Q1 - k*IQR or above Q3 + k*IQR are outliers.
    /// Standard value is 1.5 (mild outliers). Use 3.0 for extreme outliers.
    /// </para>
    /// </remarks>
    public double IqrMultiplier { get; set; } = 1.5;

    /// <summary>
    /// Gets or sets the CUSUM sensitivity parameter (k).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> CUSUM detects shifts in the mean. The k parameter
    /// controls sensitivity - smaller k catches smaller shifts but has more false alarms.
    /// Typical range: 0.25 to 1.0. Default is 0.5.
    /// </para>
    /// </remarks>
    public double CusumK { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the CUSUM decision threshold (h).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When cumulative sum exceeds h (in standard deviations),
    /// a mean shift is detected. Higher h means fewer false alarms but slower detection.
    /// Typical range: 4.0 to 5.0. Default is 4.0.
    /// </para>
    /// </remarks>
    public double CusumH { get; set; } = 4.0;

    /// <summary>
    /// Gets or sets the number of trees for isolation forest scoring.
    /// </summary>
    /// <remarks>
    /// <para>More trees give more stable scores but take longer to compute.
    /// Typical range: 50-200. Default is 100.</para>
    /// </remarks>
    public int IsolationForestTrees { get; set; } = 100;

    /// <summary>
    /// Gets or sets the subsample size for isolation forest.
    /// </summary>
    /// <remarks>
    /// <para>Smaller subsamples are faster but less accurate. Default is 256.</para>
    /// </remarks>
    public int IsolationForestSubsampleSize { get; set; } = 256;

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

    /// <summary>
    /// EWMA (Exponentially Weighted Moving Average) volatility.
    /// Gives more weight to recent observations using a decay factor.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> EWMA weights recent data more heavily than older data.
    /// This makes it more responsive to recent market changes.
    /// The decay factor (lambda) controls how fast old data loses importance.
    /// A typical value is 0.94 (RiskMetrics standard).
    /// </para>
    /// </remarks>
    EwmaVolatility = 1 << 6,

    /// <summary>
    /// GARCH(1,1) volatility estimator.
    /// Generalized AutoRegressive Conditional Heteroskedasticity model.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> GARCH models capture volatility clustering - the tendency
    /// for high volatility periods to follow high volatility, and low to follow low.
    /// GARCH(1,1) uses: sigma²_t = omega + alpha * r²_{t-1} + beta * sigma²_{t-1}
    /// where r is the return and sigma² is variance.
    /// </para>
    /// </remarks>
    GarchVolatility = 1 << 7,

    /// <summary>
    /// Yang-Zhang volatility estimator using OHLC data.
    /// Accounts for both overnight jumps and intraday volatility.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Yang-Zhang is the most accurate OHLC-based volatility estimator.
    /// It combines overnight volatility (close to open), open-to-close volatility,
    /// and Rogers-Satchell volatility for a comprehensive measure.
    /// </para>
    /// </remarks>
    YangZhangVolatility = 1 << 8,

    /// <summary>
    /// Rogers-Satchell volatility estimator.
    /// A drift-independent historical volatility measure using OHLC data.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Rogers-Satchell is unbiased even when the asset has a trend (drift).
    /// It uses all four OHLC prices to estimate volatility more accurately than close-only methods.
    /// </para>
    /// </remarks>
    RogersSatchellVolatility = 1 << 9,

    /// <summary>Basic volatility measures (returns, realized vol, momentum).</summary>
    Basic = SimpleReturns | LogReturns | RealizedVolatility | Momentum,

    /// <summary>OHLC-based volatility estimators.</summary>
    OhlcBased = ParkinsonVolatility | GarmanKlassVolatility | YangZhangVolatility | RogersSatchellVolatility,

    /// <summary>Advanced volatility models (EWMA, GARCH).</summary>
    Advanced = EwmaVolatility | GarchVolatility,

    /// <summary>All volatility measures.</summary>
    All = RealizedVolatility | ParkinsonVolatility | GarmanKlassVolatility |
          SimpleReturns | LogReturns | Momentum | EwmaVolatility | GarchVolatility |
          YangZhangVolatility | RogersSatchellVolatility
}

/// <summary>
/// Flags for selecting which rolling regression features to calculate.
/// </summary>
[Flags]
public enum RollingRegressionFeatures
{
    /// <summary>No regression features.</summary>
    None = 0,

    /// <summary>
    /// Rolling Beta - measures asset's sensitivity to benchmark movements.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Beta measures how much an asset moves when the market moves.
    /// Beta = 1: Moves with the market
    /// Beta > 1: More volatile than the market (amplifies movements)
    /// Beta &lt; 1: Less volatile than the market (dampens movements)
    /// Beta &lt; 0: Moves opposite to the market
    /// </para>
    /// </remarks>
    Beta = 1 << 0,

    /// <summary>
    /// Rolling Alpha - excess return over benchmark.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Alpha measures the "extra" return not explained by market movements.
    /// Alpha > 0: Asset outperforms what its beta would predict
    /// Alpha &lt; 0: Asset underperforms what its beta would predict
    /// A skilled fund manager aims for positive alpha.
    /// </para>
    /// </remarks>
    Alpha = 1 << 1,

    /// <summary>
    /// Rolling R-squared - coefficient of determination.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> R² measures how much of the asset's movement is explained by the benchmark.
    /// R² = 1.0: 100% explained by benchmark (perfect correlation)
    /// R² = 0.5: 50% explained by benchmark
    /// R² = 0.0: No relationship to benchmark
    /// </para>
    /// </remarks>
    RSquared = 1 << 2,

    /// <summary>
    /// Rolling Sharpe Ratio - risk-adjusted return measure.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Sharpe ratio measures return per unit of total risk.
    /// Sharpe > 1.0: Good risk-adjusted return
    /// Sharpe > 2.0: Very good
    /// Sharpe > 3.0: Excellent
    /// Negative Sharpe: Returns below risk-free rate
    /// Formula: (Return - RiskFreeRate) / StandardDeviation
    /// </para>
    /// </remarks>
    SharpeRatio = 1 << 3,

    /// <summary>
    /// Rolling Sortino Ratio - downside risk-adjusted return measure.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Sortino ratio is like Sharpe but only considers downside volatility.
    /// It doesn't penalize upside volatility (which is good for investors).
    /// Formula: (Return - MAR) / DownsideDeviation
    /// where MAR is the Minimum Acceptable Return.
    /// </para>
    /// </remarks>
    SortinoRatio = 1 << 4,

    /// <summary>
    /// Rolling Correlation with benchmark.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Correlation measures how closely two assets move together.
    /// +1: Perfect positive correlation (move together)
    /// 0: No correlation
    /// -1: Perfect negative correlation (move opposite)
    /// </para>
    /// </remarks>
    Correlation = 1 << 5,

    /// <summary>
    /// Rolling Tracking Error - standard deviation of return differences.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Tracking error measures how closely a portfolio follows its benchmark.
    /// Low tracking error: Portfolio closely tracks benchmark
    /// High tracking error: Portfolio diverges significantly from benchmark
    /// </para>
    /// </remarks>
    TrackingError = 1 << 6,

    /// <summary>
    /// Rolling Information Ratio - alpha per unit of tracking error.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Information ratio measures the consistency of outperformance.
    /// IR > 0.5: Good
    /// IR > 1.0: Excellent
    /// Formula: Alpha / TrackingError
    /// </para>
    /// </remarks>
    InformationRatio = 1 << 7,

    /// <summary>Standard CAPM regression features (Alpha, Beta, R²).</summary>
    CAPMFeatures = Alpha | Beta | RSquared,

    /// <summary>Risk-adjusted return measures.</summary>
    RiskAdjusted = SharpeRatio | SortinoRatio | InformationRatio,

    /// <summary>All regression features.</summary>
    All = Beta | Alpha | RSquared | SharpeRatio | SortinoRatio | Correlation | TrackingError | InformationRatio
}

/// <summary>
/// Flags for selecting which anomaly detection features to calculate.
/// </summary>
[Flags]
public enum AnomalyFeatures
{
    /// <summary>No anomaly features.</summary>
    None = 0,

    /// <summary>
    /// Rolling Z-score: measures how many standard deviations a value is from the rolling mean.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Z-score tells you how "unusual" a value is.
    /// Z = 0: Value equals the mean
    /// Z = ±1: Value is 1 standard deviation from mean (common)
    /// Z = ±2: Value is 2 standard deviations from mean (unusual)
    /// Z = ±3: Value is 3 standard deviations from mean (rare, ~0.3% probability)
    /// </para>
    /// </remarks>
    ZScore = 1 << 0,

    /// <summary>
    /// Z-score anomaly flag: 1 if |Z-score| exceeds threshold, 0 otherwise.
    /// </summary>
    ZScoreFlag = 1 << 1,

    /// <summary>
    /// Modified Z-score using median absolute deviation (more robust to outliers).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Modified Z-score uses median instead of mean,
    /// making it more robust when there are already outliers in your data.
    /// </para>
    /// </remarks>
    ModifiedZScore = 1 << 2,

    /// <summary>
    /// IQR outlier score: distance from the nearest quartile boundary.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> IQR-based methods use the interquartile range to identify outliers.
    /// Values below Q1 - k*IQR or above Q3 + k*IQR are considered outliers.
    /// This method is robust to extreme values.
    /// </para>
    /// </remarks>
    IqrOutlierScore = 1 << 3,

    /// <summary>
    /// IQR outlier flag: 1 if value is outside IQR bounds, 0 otherwise.
    /// </summary>
    IqrOutlierFlag = 1 << 4,

    /// <summary>
    /// CUSUM (Cumulative Sum) statistic for detecting mean shifts.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> CUSUM accumulates small deviations from the mean.
    /// When the sum exceeds a threshold, it indicates the mean has shifted.
    /// Good for detecting gradual drifts in process control.
    /// </para>
    /// </remarks>
    CusumStatistic = 1 << 5,

    /// <summary>
    /// CUSUM change point flag: 1 when CUSUM exceeds threshold, 0 otherwise.
    /// </summary>
    CusumFlag = 1 << 6,

    /// <summary>
    /// Isolation score: higher values indicate more anomalous points.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Isolation forest isolates observations by randomly selecting
    /// a feature and split value. Anomalies are isolated more quickly (fewer splits needed)
    /// because they are "few and different". Score of 1 = definite anomaly, 0.5 = normal.
    /// </para>
    /// </remarks>
    IsolationScore = 1 << 7,

    /// <summary>
    /// Percentile rank: where the current value falls in the rolling distribution (0-1).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Percentile rank tells you what percentage of values
    /// in the window are below the current value. A rank of 0.99 means the current value
    /// is higher than 99% of recent values.
    /// </para>
    /// </remarks>
    PercentileRank = 1 << 8,

    /// <summary>Z-score based anomaly detection features.</summary>
    ZScoreFeatures = ZScore | ZScoreFlag | ModifiedZScore,

    /// <summary>IQR-based anomaly detection features.</summary>
    IqrFeatures = IqrOutlierScore | IqrOutlierFlag,

    /// <summary>Control chart features (CUSUM).</summary>
    ControlChartFeatures = CusumStatistic | CusumFlag,

    /// <summary>All anomaly detection features.</summary>
    All = ZScore | ZScoreFlag | ModifiedZScore | IqrOutlierScore | IqrOutlierFlag |
          CusumStatistic | CusumFlag | IsolationScore | PercentileRank
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

/// <summary>
/// Configuration for OHLC (Open, High, Low, Close) column indices.
/// </summary>
/// <remarks>
/// <para>
/// This class specifies which columns in your data contain OHLC prices, enabling proper
/// calculation of volatility measures like Parkinson and Garman-Klass that require
/// high/low/open/close values.
/// </para>
/// <para><b>For Beginners:</b> OHLC is standard financial data format:
/// - Open: First price when trading started
/// - High: Highest price during the period
/// - Low: Lowest price during the period
/// - Close: Last price when trading ended
///
/// These four values together tell the full story of price movement during each period.
/// Volatility estimators like Parkinson and Garman-Klass use this information to more
/// accurately measure how volatile (risky) an asset is.
/// </para>
/// </remarks>
public class OhlcColumnConfig
{
    /// <summary>
    /// Gets or sets the column index for Open prices.
    /// </summary>
    /// <remarks>
    /// <para>Set to null if open prices are not available.</para>
    /// </remarks>
    public int? OpenIndex { get; set; }

    /// <summary>
    /// Gets or sets the column index for High prices.
    /// </summary>
    /// <remarks>
    /// <para>Required for Parkinson and Garman-Klass volatility.</para>
    /// </remarks>
    public int? HighIndex { get; set; }

    /// <summary>
    /// Gets or sets the column index for Low prices.
    /// </summary>
    /// <remarks>
    /// <para>Required for Parkinson and Garman-Klass volatility.</para>
    /// </remarks>
    public int? LowIndex { get; set; }

    /// <summary>
    /// Gets or sets the column index for Close prices.
    /// </summary>
    /// <remarks>
    /// <para>Required for Garman-Klass volatility and returns calculations.</para>
    /// </remarks>
    public int? CloseIndex { get; set; }

    /// <summary>
    /// Checks if this configuration has valid High/Low indices for Parkinson volatility.
    /// </summary>
    public bool HasHighLow => HighIndex.HasValue && LowIndex.HasValue;

    /// <summary>
    /// Checks if this configuration has valid OHLC indices for Garman-Klass volatility.
    /// </summary>
    public bool HasOhlc => OpenIndex.HasValue && HighIndex.HasValue &&
                          LowIndex.HasValue && CloseIndex.HasValue;

    /// <summary>
    /// Creates a standard OHLC configuration assuming columns are in order: Open, High, Low, Close.
    /// </summary>
    public static OhlcColumnConfig CreateStandard()
    {
        return new OhlcColumnConfig
        {
            OpenIndex = 0,
            HighIndex = 1,
            LowIndex = 2,
            CloseIndex = 3
        };
    }

    /// <summary>
    /// Creates a configuration for data with only High, Low, Close (no Open).
    /// </summary>
    public static OhlcColumnConfig CreateHlc(int highIndex = 0, int lowIndex = 1, int closeIndex = 2)
    {
        return new OhlcColumnConfig
        {
            HighIndex = highIndex,
            LowIndex = lowIndex,
            CloseIndex = closeIndex
        };
    }
}

/// <summary>
/// Flags for selecting which technical indicators to calculate.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Technical indicators are mathematical tools that analyze price patterns.
/// Traders use these to identify trends, momentum, overbought/oversold conditions, and potential
/// reversal points.
/// </para>
/// </remarks>
[Flags]
public enum TechnicalIndicators
{
    /// <summary>No indicators.</summary>
    None = 0,

    /// <summary>
    /// Simple Moving Average (SMA) - Equal-weighted average of past N prices.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> SMA smooths out price data by averaging the last N prices.
    /// It helps identify the overall trend direction.</para>
    /// </remarks>
    SMA = 1 << 0,

    /// <summary>
    /// Exponential Moving Average (EMA) - Weighted average giving more weight to recent prices.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> EMA responds faster to recent price changes than SMA because
    /// it weights recent prices more heavily.</para>
    /// </remarks>
    EMA = 1 << 1,

    /// <summary>
    /// Weighted Moving Average (WMA) - Linear-weighted average of past N prices.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> WMA also weights recent prices more heavily, but uses a linear
    /// weighting scheme instead of exponential.</para>
    /// </remarks>
    WMA = 1 << 2,

    /// <summary>
    /// Double Exponential Moving Average (DEMA) - Reduces lag of standard EMA.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> DEMA combines two EMAs to reduce the delay in signal detection.</para>
    /// </remarks>
    DEMA = 1 << 3,

    /// <summary>
    /// Triple Exponential Moving Average (TEMA) - Further reduces lag using three EMAs.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> TEMA is even more responsive than DEMA by combining three EMAs.</para>
    /// </remarks>
    TEMA = 1 << 4,

    /// <summary>
    /// Bollinger Bands - Volatility bands around a moving average.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Bollinger Bands show when prices are relatively high or low.
    /// Bands widen during volatile periods and narrow during calm periods.</para>
    /// </remarks>
    BollingerBands = 1 << 5,

    /// <summary>
    /// Relative Strength Index (RSI) - Momentum oscillator (0-100).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> RSI measures how fast and how much prices are rising/falling.
    /// Values above 70 suggest overbought conditions; below 30 suggest oversold.</para>
    /// </remarks>
    RSI = 1 << 6,

    /// <summary>
    /// Moving Average Convergence Divergence (MACD) - Trend-following momentum indicator.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> MACD shows the relationship between two EMAs.
    /// Crossovers of the MACD and signal line can indicate buy/sell opportunities.</para>
    /// </remarks>
    MACD = 1 << 7,

    /// <summary>
    /// Average True Range (ATR) - Volatility indicator.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> ATR measures how much an asset typically moves.
    /// Higher ATR = more volatile; useful for setting stop-losses.</para>
    /// </remarks>
    ATR = 1 << 8,

    /// <summary>
    /// Stochastic Oscillator - Momentum indicator (0-100).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Compares closing price to the price range over N periods.
    /// Values above 80 suggest overbought; below 20 suggest oversold.</para>
    /// </remarks>
    StochasticOscillator = 1 << 9,

    /// <summary>
    /// Commodity Channel Index (CCI) - Momentum indicator.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> CCI measures how far price has moved from its average.
    /// Readings above +100 suggest overbought; below -100 suggest oversold.</para>
    /// </remarks>
    CCI = 1 << 10,

    /// <summary>
    /// Williams %R - Momentum indicator (-100 to 0).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Shows where today's close is relative to the high-low range.
    /// Values near 0 indicate overbought; near -100 indicate oversold.</para>
    /// </remarks>
    WilliamsR = 1 << 11,

    /// <summary>
    /// Average Directional Index (ADX) - Trend strength indicator (0-100).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> ADX measures how strong a trend is, regardless of direction.
    /// Values above 25 suggest a strong trend; below 20 suggest a weak or no trend.</para>
    /// </remarks>
    ADX = 1 << 12,

    /// <summary>
    /// On-Balance Volume (OBV) - Volume-based momentum indicator.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> OBV adds volume on up days and subtracts on down days.
    /// Rising OBV confirms an uptrend; falling OBV confirms a downtrend.</para>
    /// </remarks>
    OBV = 1 << 13,

    /// <summary>
    /// All moving average indicators.
    /// </summary>
    MovingAverages = SMA | EMA | WMA | DEMA | TEMA,

    /// <summary>
    /// All momentum indicators.
    /// </summary>
    Momentum = RSI | MACD | StochasticOscillator | CCI | WilliamsR | ADX,

    /// <summary>
    /// All volatility indicators.
    /// </summary>
    Volatility = BollingerBands | ATR,

    /// <summary>
    /// All volume-based indicators.
    /// </summary>
    Volume = OBV,

    /// <summary>
    /// All available technical indicators.
    /// </summary>
    All = MovingAverages | Momentum | Volatility | Volume
}

/// <summary>
/// Flags for selecting which seasonality and calendar features to generate.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Seasonality features capture time-based patterns in your data.
/// Different features are useful for different scenarios:
/// - Fourier features: Best for smooth, cyclical patterns
/// - Time features: Useful when patterns vary by hour, day, month
/// - Holiday features: Important for retail, energy, travel data
/// - Trading day features: Specific to financial data
/// </para>
/// </remarks>
[Flags]
public enum SeasonalityFeatures
{
    /// <summary>No seasonality features.</summary>
    None = 0,

    /// <summary>
    /// Fourier features (sin/cos at seasonal frequencies).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Fourier features are smooth sine and cosine waves
    /// that help capture cyclical patterns. They're especially useful for:
    /// - Smooth periodic patterns (like daily temperature cycles)
    /// - When you want to avoid step-changes between time periods
    /// </para>
    /// </remarks>
    FourierFeatures = 1 << 0,

    /// <summary>
    /// Hour of day feature (0-23).
    /// </summary>
    HourOfDay = 1 << 1,

    /// <summary>
    /// Day of week feature (0-6, Sunday=0).
    /// </summary>
    DayOfWeek = 1 << 2,

    /// <summary>
    /// Day of month feature (1-31).
    /// </summary>
    DayOfMonth = 1 << 3,

    /// <summary>
    /// Day of year feature (1-366).
    /// </summary>
    DayOfYear = 1 << 4,

    /// <summary>
    /// Week of year feature (1-53).
    /// </summary>
    WeekOfYear = 1 << 5,

    /// <summary>
    /// Month of year feature (1-12).
    /// </summary>
    MonthOfYear = 1 << 6,

    /// <summary>
    /// Quarter of year feature (1-4).
    /// </summary>
    QuarterOfYear = 1 << 7,

    /// <summary>
    /// Year feature (actual year number).
    /// </summary>
    Year = 1 << 8,

    /// <summary>
    /// Is weekend binary feature.
    /// </summary>
    IsWeekend = 1 << 9,

    /// <summary>
    /// Is month start/end features.
    /// </summary>
    MonthStartEnd = 1 << 10,

    /// <summary>
    /// Is quarter start/end features.
    /// </summary>
    QuarterStartEnd = 1 << 11,

    /// <summary>
    /// Holiday indicator features.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates binary features indicating if a data point
    /// falls on or near a holiday. Useful for retail, travel, and energy forecasting.
    /// </para>
    /// </remarks>
    HolidayFeatures = 1 << 12,

    /// <summary>
    /// Trading day of month (skips weekends/holidays).
    /// </summary>
    TradingDayOfMonth = 1 << 13,

    /// <summary>
    /// Trading day of week (1-5).
    /// </summary>
    TradingDayOfWeek = 1 << 14,

    /// <summary>
    /// All time-based features.
    /// </summary>
    TimeFeatures = HourOfDay | DayOfWeek | DayOfMonth | DayOfYear | WeekOfYear |
                   MonthOfYear | QuarterOfYear | Year,

    /// <summary>
    /// All calendar event features.
    /// </summary>
    CalendarEvents = IsWeekend | MonthStartEnd | QuarterStartEnd | HolidayFeatures,

    /// <summary>
    /// All trading-specific features.
    /// </summary>
    TradingFeatures = TradingDayOfMonth | TradingDayOfWeek,

    /// <summary>
    /// All available seasonality features.
    /// </summary>
    All = FourierFeatures | TimeFeatures | CalendarEvents | TradingFeatures
}

/// <summary>
/// Flags for selecting which differencing and stationarity features to compute.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Differencing transforms help make time series stationary,
/// which is required by many forecasting models. A stationary series has:
/// - Constant mean over time (no trend)
/// - Constant variance over time (no changing volatility)
/// - No seasonal patterns
/// </para>
/// </remarks>
[Flags]
public enum DifferencingFeatures
{
    /// <summary>No differencing features.</summary>
    None = 0,

    /// <summary>
    /// First-order differencing: y[t] - y[t-1].
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Computes the change from one time step to the next.
    /// This removes linear trends and is the most common differencing operation.
    /// </para>
    /// </remarks>
    FirstDifference = 1 << 0,

    /// <summary>
    /// Second-order differencing: diff(diff(y)).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Computes the "change in change" (acceleration).
    /// Removes quadratic trends. Rarely need more than second-order differencing.
    /// </para>
    /// </remarks>
    SecondDifference = 1 << 1,

    /// <summary>
    /// Seasonal differencing: y[t] - y[t-period].
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Removes seasonal patterns by comparing values
    /// to the same point in the previous season. For weekly data with period=7,
    /// compares each Monday to the previous Monday, etc.
    /// </para>
    /// </remarks>
    SeasonalDifference = 1 << 2,

    /// <summary>
    /// Percent change: (y[t] - y[t-1]) / y[t-1].
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Shows percentage change between time steps.
    /// Useful when absolute changes depend on the level (e.g., stock prices).
    /// </para>
    /// </remarks>
    PercentChange = 1 << 3,

    /// <summary>
    /// Log difference: log(y[t]) - log(y[t-1]).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Similar to percent change but symmetric.
    /// Common in finance as it approximates percentage returns for small changes.
    /// </para>
    /// </remarks>
    LogDifference = 1 << 4,

    /// <summary>
    /// Linear detrending: removes best-fit straight line.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Fits a straight line to the data and subtracts it.
    /// Useful when there's an obvious upward or downward trend.
    /// </para>
    /// </remarks>
    LinearDetrend = 1 << 5,

    /// <summary>
    /// Polynomial detrending: removes best-fit polynomial.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Fits a curved line (polynomial) and subtracts it.
    /// Use when the trend is curved rather than straight.
    /// </para>
    /// </remarks>
    PolynomialDetrend = 1 << 6,

    /// <summary>
    /// Hodrick-Prescott filter: extracts trend and cycle components.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> A popular method in economics to separate
    /// long-term trend from short-term fluctuations. Returns both components.
    /// </para>
    /// </remarks>
    HodrickPrescottFilter = 1 << 7,

    /// <summary>
    /// STL decomposition: Seasonal-Trend decomposition using LOESS.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Separates data into three parts:
    /// - Seasonal: repeating patterns (e.g., weekly cycles)
    /// - Trend: long-term direction
    /// - Residual: random noise
    /// </para>
    /// </remarks>
    StlDecomposition = 1 << 8,

    /// <summary>
    /// All basic differencing methods.
    /// </summary>
    BasicDifferencing = FirstDifference | SecondDifference | SeasonalDifference,

    /// <summary>
    /// All return-based transforms.
    /// </summary>
    Returns = PercentChange | LogDifference,

    /// <summary>
    /// All detrending methods.
    /// </summary>
    Detrending = LinearDetrend | PolynomialDetrend | HodrickPrescottFilter,

    /// <summary>
    /// All decomposition methods.
    /// </summary>
    Decomposition = StlDecomposition,

    /// <summary>
    /// All available differencing features.
    /// </summary>
    All = BasicDifferencing | Returns | Detrending | Decomposition
}
