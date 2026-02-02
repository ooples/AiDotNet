namespace AiDotNet.Interfaces;

/// <summary>
/// Defines a specialized data transformer for extracting features from time series data.
/// </summary>
/// <remarks>
/// <para>
/// This interface extends the standard data transformer pattern with time series-specific
/// functionality, including auto-detection of optimal parameters, window-based feature
/// extraction, and support for both univariate and multivariate time series.
/// </para>
/// <para><b>For Beginners:</b> Time series data is data collected over time, like:
/// - Stock prices recorded every minute
/// - Temperature readings every hour
/// - Sales figures every day
///
/// This interface helps you extract useful features from such data, like:
/// - Rolling averages (what's the average of the last 7 days?)
/// - Lagged values (what was the value 3 days ago?)
/// - Volatility (how much does the value fluctuate?)
///
/// These features help machine learning models understand patterns in time series data.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public interface ITimeSeriesFeatureExtractor<T> : IDataTransformer<T, Tensor<T>, Tensor<T>>
{
    /// <summary>
    /// Gets the window sizes used for rolling calculations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Window sizes determine how many past observations are used for each rolling calculation.
    /// Multiple window sizes can be specified to capture patterns at different time scales.
    /// </para>
    /// <para><b>For Beginners:</b> A window is like a sliding frame that moves through your data.
    ///
    /// For example, with a window size of 7:
    /// - For day 7: calculate using days 1-7
    /// - For day 8: calculate using days 2-8
    /// - For day 9: calculate using days 3-9
    ///
    /// Using multiple window sizes (e.g., 7, 14, 30) captures both short-term and long-term patterns.
    /// </para>
    /// </remarks>
    int[] WindowSizes { get; }

    /// <summary>
    /// Gets whether auto-detection of optimal parameters is enabled.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When enabled, the transformer will analyze the data during fitting to automatically
    /// determine optimal window sizes based on detected patterns like seasonality.
    /// </para>
    /// <para><b>For Beginners:</b> Auto-detection is like having the system figure out
    /// the best settings for you based on your data's patterns.
    ///
    /// For example, if your data has a weekly pattern, it might suggest a window of 7.
    /// If it has a monthly pattern, it might suggest 30.
    /// </para>
    /// </remarks>
    bool AutoDetectEnabled { get; }

    /// <summary>
    /// Gets the names of features that will be generated.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Returns descriptive names for each output feature, useful for model interpretation
    /// and debugging. Names follow a pattern like "{column}_{statistic}_{window}".
    /// </para>
    /// <para><b>For Beginners:</b> Feature names help you understand what each output column means.
    ///
    /// For example:
    /// - "close_rolling_mean_7" = 7-day rolling average of the "close" column
    /// - "volume_lag_1" = volume from 1 time step ago
    /// - "price_volatility_14" = 14-day volatility of price
    /// </para>
    /// </remarks>
    string[] FeatureNames { get; }

    /// <summary>
    /// Detects optimal window sizes based on the data's characteristics.
    /// </summary>
    /// <param name="data">The time series data to analyze.</param>
    /// <returns>Suggested window sizes based on detected patterns.</returns>
    /// <remarks>
    /// <para>
    /// Analyzes the time series to detect seasonality, trends, and autocorrelation patterns,
    /// then suggests window sizes that would capture these patterns effectively.
    /// </para>
    /// <para><b>For Beginners:</b> This method looks at your data and suggests
    /// the best window sizes to use. It's like having an expert analyze your data
    /// and recommend settings based on what patterns they see.
    ///
    /// Detection methods include:
    /// - Autocorrelation analysis (ACF) - finds repeating patterns
    /// - Seasonality detection - finds daily, weekly, monthly cycles
    /// - Trend analysis - identifies long-term directions
    /// </para>
    /// </remarks>
    int[] DetectOptimalWindowSizes(Tensor<T> data);

    /// <summary>
    /// Gets the number of input features (columns) expected.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For multivariate time series, this indicates how many variables are being processed.
    /// Set during fitting based on the input data shape.
    /// </para>
    /// </remarks>
    int InputFeatureCount { get; }

    /// <summary>
    /// Gets the number of output features that will be generated.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The output feature count depends on which statistics are enabled, how many window sizes
    /// are used, and the number of input features. This is useful for pre-allocating output arrays.
    /// </para>
    /// <para><b>For Beginners:</b> If you have 3 input columns and calculate 5 statistics
    /// with 2 window sizes, you'll get 3 × 5 × 2 = 30 output features.
    /// </para>
    /// </remarks>
    int OutputFeatureCount { get; }

    /// <summary>
    /// Validates that the input data meets the requirements for this transformer.
    /// </summary>
    /// <param name="data">The data to validate.</param>
    /// <returns>True if the data is valid, false otherwise.</returns>
    /// <remarks>
    /// <para>
    /// Checks that the data has sufficient length for the configured window sizes,
    /// has the expected number of features, and contains no invalid values.
    /// </para>
    /// </remarks>
    bool ValidateInput(Tensor<T> data);

    /// <summary>
    /// Gets validation errors for the input data.
    /// </summary>
    /// <param name="data">The data to validate.</param>
    /// <returns>List of validation error messages, empty if valid.</returns>
    List<string> GetValidationErrors(Tensor<T> data);

    #region Incremental/Streaming Support

    /// <summary>
    /// Gets whether this transformer supports incremental (streaming) transformation.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Incremental transformation allows you to process new data points
    /// one at a time, without having to re-process all historical data. This is useful for:
    /// - Real-time applications where new data arrives continuously
    /// - Large datasets that don't fit in memory
    /// - Production systems that need low-latency updates
    /// </para>
    /// </remarks>
    bool SupportsIncrementalTransform { get; }

    /// <summary>
    /// Initializes the incremental state from historical data.
    /// </summary>
    /// <param name="historicalData">Initial historical data to seed the rolling window.</param>
    /// <remarks>
    /// <para>
    /// Call this method once with enough historical data to fill the largest window,
    /// then use <see cref="TransformIncremental"/> for each new data point.
    /// </para>
    /// <para><b>For Beginners:</b> Before processing data incrementally, we need some history.
    /// For example, if using a 30-day rolling average, we need at least 30 days of data
    /// to start. This method provides that initial history.
    /// </para>
    /// </remarks>
    void InitializeIncremental(Tensor<T> historicalData);

    /// <summary>
    /// Transforms a single new data point incrementally, maintaining internal state.
    /// </summary>
    /// <param name="newDataPoint">The new data point (single row with same number of features as fitted data).</param>
    /// <returns>The computed features for this data point.</returns>
    /// <remarks>
    /// <para>
    /// This method updates internal rolling window state and computes features for a single
    /// new observation. It's more efficient than re-processing all data when new points arrive.
    /// </para>
    /// <para><b>For Beginners:</b> When a new data point arrives (like today's stock price),
    /// this method:
    /// 1. Adds it to the rolling window
    /// 2. Computes all the rolling features (mean, std, etc.)
    /// 3. Returns those features
    /// 4. Keeps track of history for the next point
    ///
    /// This is much faster than re-processing all historical data each time.
    /// </para>
    /// </remarks>
    T[] TransformIncremental(T[] newDataPoint);

    /// <summary>
    /// Gets the current state of the incremental buffer for inspection or serialization.
    /// </summary>
    /// <returns>The current rolling window state, or null if not initialized.</returns>
    IncrementalState<T>? GetIncrementalState();

    /// <summary>
    /// Restores the incremental state from a previously saved state.
    /// </summary>
    /// <param name="state">The state to restore.</param>
    /// <remarks>
    /// <para>
    /// This allows saving and restoring the transformer's state between sessions,
    /// useful for production deployments where the service may restart.
    /// </para>
    /// </remarks>
    void SetIncrementalState(IncrementalState<T> state);

    #endregion
}

/// <summary>
/// Represents the internal state of a time series transformer for incremental processing.
/// </summary>
/// <typeparam name="T">The numeric type for data.</typeparam>
[Serializable]
public class IncrementalState<T>
{
    /// <summary>
    /// The rolling buffer of recent values for each input feature.
    /// </summary>
    public T[][] RollingBuffer { get; set; } = [];

    /// <summary>
    /// The current position (index) in the circular buffer.
    /// </summary>
    public int BufferPosition { get; set; }

    /// <summary>
    /// The number of data points that have been processed.
    /// </summary>
    public long PointsProcessed { get; set; }

    /// <summary>
    /// Whether the buffer has been fully filled at least once.
    /// </summary>
    public bool BufferFilled { get; set; }

    /// <summary>
    /// Additional state information specific to the transformer type.
    /// </summary>
    public Dictionary<string, object> ExtendedState { get; set; } = [];
}

/// <summary>
/// Represents the serializable state of a fitted time series transformer.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This class allows you to save a fitted transformer to a file
/// and reload it later without needing to re-fit. This is useful for:
/// - Production deployments where you train once and deploy the fitted model
/// - Sharing trained transformers between team members
/// - Versioning and reproducibility of your feature engineering pipeline
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for data.</typeparam>
[Serializable]
public class TransformerState<T>
{
    /// <summary>
    /// The type name of the transformer.
    /// </summary>
    public string TransformerType { get; set; } = "";

    /// <summary>
    /// The version of the serialization format.
    /// </summary>
    public int Version { get; set; } = 1;

    /// <summary>
    /// Whether the transformer has been fitted.
    /// </summary>
    public bool IsFitted { get; set; }

    /// <summary>
    /// The window sizes used by the transformer.
    /// </summary>
    public int[] WindowSizes { get; set; } = [];

    /// <summary>
    /// The generated feature names.
    /// </summary>
    public string[] FeatureNames { get; set; } = [];

    /// <summary>
    /// The input feature names.
    /// </summary>
    public string[] InputFeatureNames { get; set; } = [];

    /// <summary>
    /// The number of input features.
    /// </summary>
    public int InputFeatureCount { get; set; }

    /// <summary>
    /// The number of output features.
    /// </summary>
    public int OutputFeatureCount { get; set; }

    /// <summary>
    /// The incremental state, if initialized.
    /// </summary>
    public IncrementalState<T>? IncrementalState { get; set; }

    /// <summary>
    /// Transformer-specific parameters and learned values.
    /// </summary>
    public Dictionary<string, object> Parameters { get; set; } = [];

    /// <summary>
    /// The serialized options used to configure the transformer.
    /// </summary>
    public Dictionary<string, object> Options { get; set; } = [];
}
