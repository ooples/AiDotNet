using AiDotNet.Preprocessing.TimeSeries;
using AiDotNet.Tensors;

namespace AiDotNet;

/// <summary>
/// Time series feature extraction extensions for AiModelBuilder.
/// </summary>
/// <remarks>
/// <para>
/// This partial class adds time series feature engineering capabilities to the builder,
/// enabling easy configuration of rolling statistics, volatility measures, correlations,
/// and lag/lead features.
/// </para>
/// <para><b>For Beginners:</b> Time series feature extraction transforms raw sequential data
/// (like daily stock prices or hourly temperatures) into features that help ML models
/// understand patterns over time:
///
/// - Rolling statistics (average of last 7 days, volatility over last month)
/// - Lag features (what was the value yesterday, last week?)
/// - Correlation features (how do different variables move together?)
///
/// These features capture temporal patterns that raw data doesn't explicitly show.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input type.</typeparam>
/// <typeparam name="TOutput">The output type.</typeparam>
public partial class AiModelBuilder<T, TInput, TOutput>
{
    #region Time Series Fields

    private TimeSeriesFeatureOptions? _timeSeriesOptions;
    private readonly List<ITimeSeriesFeatureExtractor<T>> _timeSeriesExtractors = [];

    #endregion

    #region Time Series Configuration Methods

    /// <summary>
    /// Configures time series feature extraction with the specified options.
    /// </summary>
    /// <param name="options">The options for time series feature extraction.</param>
    /// <returns>The builder for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// This method configures automatic extraction of time series features based on the
    /// provided options. Features include rolling statistics, volatility measures,
    /// correlations, and lag/lead features.
    /// </para>
    /// <para><b>For Beginners:</b> This is the main way to add time series features.
    /// Just pass in your configuration and the builder handles the rest:
    ///
    /// <code>
    /// var builder = new AiModelBuilder&lt;double, Tensor&lt;double&gt;, double[]&gt;()
    ///     .ConfigureTimeSeriesFeatures(new TimeSeriesFeatureOptions
    ///     {
    ///         WindowSizes = new[] { 7, 14, 30 },
    ///         EnabledStatistics = RollingStatistics.Mean | RollingStatistics.StandardDeviation,
    ///         EnableVolatility = true,
    ///         LagSteps = new[] { 1, 2, 3 }
    ///     });
    /// </code>
    /// </para>
    /// </remarks>
    /// <example>
    /// <code>
    /// // Financial time series with common settings
    /// var builder = new AiModelBuilder&lt;double, Tensor&lt;double&gt;, double[]&gt;()
    ///     .ConfigureTimeSeriesFeatures(TimeSeriesFeatureOptions.CreateForFinance())
    ///     .Build();
    /// </code>
    /// </example>
    public AiModelBuilder<T, TInput, TOutput> ConfigureTimeSeriesFeatures(TimeSeriesFeatureOptions options)
    {
        if (options is null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        // Validate options
        var errors = options.Validate();
        if (errors.Count > 0)
        {
            throw new ArgumentException(
                $"Invalid time series options: {string.Join("; ", errors)}");
        }

        _timeSeriesOptions = options;
        CreateTimeSeriesExtractors(options);

        return this;
    }

    /// <summary>
    /// Configures time series feature extraction using a fluent configuration action.
    /// </summary>
    /// <param name="configure">An action to configure the options.</param>
    /// <returns>The builder for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// This overload provides a fluent API for configuring time series options inline.
    /// </para>
    /// <para><b>For Beginners:</b> This is a convenient way to configure options
    /// without creating a separate object:
    ///
    /// <code>
    /// var builder = new AiModelBuilder&lt;double, Tensor&lt;double&gt;, double[]&gt;()
    ///     .ConfigureTimeSeriesFeatures(opts =>
    ///     {
    ///         opts.WindowSizes = new[] { 5, 10, 20 };
    ///         opts.EnableVolatility = true;
    ///         opts.LagSteps = new[] { 1, 2, 5 };
    ///     });
    /// </code>
    /// </para>
    /// </remarks>
    public AiModelBuilder<T, TInput, TOutput> ConfigureTimeSeriesFeatures(
        Action<TimeSeriesFeatureOptions> configure)
    {
        if (configure is null)
        {
            throw new ArgumentNullException(nameof(configure));
        }

        var options = new TimeSeriesFeatureOptions();
        configure(options);

        return ConfigureTimeSeriesFeatures(options);
    }

    /// <summary>
    /// Configures time series feature extraction for financial data.
    /// </summary>
    /// <returns>The builder for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// This convenience method configures common financial time series features including:
    /// - Rolling statistics at trading-relevant window sizes (5, 10, 20, 60, 120, 252 days)
    /// - Volatility measures (realized, Parkinson, Garman-Klass)
    /// - Returns (simple and log)
    /// - Momentum indicators
    /// - Common lag features for autoregressive patterns
    /// </para>
    /// <para><b>For Beginners:</b> Use this for stock prices, forex, crypto, or any
    /// financial time series. It sets up industry-standard features used by quants
    /// and algorithmic traders.
    /// </para>
    /// </remarks>
    /// <example>
    /// <code>
    /// var builder = new AiModelBuilder&lt;double, Tensor&lt;double&gt;, double[]&gt;()
    ///     .ConfigureTimeSeriesFeaturesForFinance()
    ///     .Build();
    /// </code>
    /// </example>
    public AiModelBuilder<T, TInput, TOutput> ConfigureTimeSeriesFeaturesForFinance()
    {
        return ConfigureTimeSeriesFeatures(TimeSeriesFeatureOptions.CreateForFinance());
    }

    /// <summary>
    /// Configures minimal time series feature extraction for fast processing.
    /// </summary>
    /// <returns>The builder for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// This convenience method configures a minimal set of features for fast processing:
    /// - Single window size (7)
    /// - Only mean and standard deviation
    /// - No volatility, correlation, or lag features
    /// </para>
    /// <para><b>For Beginners:</b> Use this when you need quick results or want to
    /// start simple before adding more complex features.
    /// </para>
    /// </remarks>
    public AiModelBuilder<T, TInput, TOutput> ConfigureTimeSeriesFeaturesMinimal()
    {
        return ConfigureTimeSeriesFeatures(TimeSeriesFeatureOptions.CreateMinimal());
    }

    /// <summary>
    /// Adds a custom time series feature extractor.
    /// </summary>
    /// <param name="extractor">The extractor to add.</param>
    /// <returns>The builder for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// Use this to add custom or specialized extractors that aren't covered by the
    /// standard options. Custom extractors are applied after the built-in ones.
    /// </para>
    /// <para><b>For Beginners:</b> Advanced users can create their own extractors
    /// by implementing ITimeSeriesFeatureExtractor and adding them here.
    /// </para>
    /// </remarks>
    public AiModelBuilder<T, TInput, TOutput> AddTimeSeriesExtractor(
        ITimeSeriesFeatureExtractor<T> extractor)
    {
        if (extractor is null)
        {
            throw new ArgumentNullException(nameof(extractor));
        }

        _timeSeriesExtractors.Add(extractor);
        return this;
    }

    #endregion

    #region Time Series Accessors

    /// <summary>
    /// Gets the configured time series options, or null if not configured.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this to inspect the current configuration
    /// or verify settings after building.
    /// </para>
    /// </remarks>
    public TimeSeriesFeatureOptions? TimeSeriesOptions => _timeSeriesOptions;

    /// <summary>
    /// Gets the configured time series feature extractors.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This returns all configured extractors, useful
    /// for debugging or understanding what features will be generated.
    /// </para>
    /// </remarks>
    public IReadOnlyList<ITimeSeriesFeatureExtractor<T>> TimeSeriesExtractors =>
        _timeSeriesExtractors.AsReadOnly();

    /// <summary>
    /// Gets the total number of output features from all time series extractors.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is only accurate after fitting the extractors with data, as the output
    /// count depends on the input data dimensions.
    /// </para>
    /// </remarks>
    public int TotalTimeSeriesFeatureCount =>
        _timeSeriesExtractors.Sum(e => e.OutputFeatureCount);

    /// <summary>
    /// Gets all feature names from configured time series extractors.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Returns combined feature names from all extractors. Only available after
    /// fitting with data.
    /// </para>
    /// </remarks>
    public IEnumerable<string> AllTimeSeriesFeatureNames =>
        _timeSeriesExtractors.SelectMany(e => e.FeatureNames);

    #endregion

    #region Private Helpers

    /// <summary>
    /// Creates time series extractors based on the configured options.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates the actual processing components
    /// based on your configuration. Each type of feature (rolling stats, volatility, etc.)
    /// gets its own extractor that specializes in that calculation.
    /// </para>
    /// </remarks>
    private void CreateTimeSeriesExtractors(TimeSeriesFeatureOptions options)
    {
        _timeSeriesExtractors.Clear();

        // Add rolling statistics extractor if any statistics are enabled
        if (options.EnabledStatistics != RollingStatistics.None)
        {
            _timeSeriesExtractors.Add(new RollingStatsTransformer<T>(options));
        }

        // Add volatility extractor if enabled
        if (options.EnableVolatility)
        {
            _timeSeriesExtractors.Add(new RollingVolatilityTransformer<T>(options));
        }

        // Add lag/lead extractor if any steps are configured
        if (options.LagSteps.Length > 0 || options.LeadSteps.Length > 0)
        {
            _timeSeriesExtractors.Add(new LagLeadTransformer<T>(options));
        }

        // Add correlation extractor if enabled
        if (options.EnableCorrelation)
        {
            _timeSeriesExtractors.Add(new RollingCorrelationTransformer<T>(options));
        }
    }

    /// <summary>
    /// Applies all time series extractors to the input data.
    /// </summary>
    /// <param name="data">The input time series data.</param>
    /// <returns>Tensor containing all extracted features.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This combines the outputs of all configured extractors
    /// into a single feature tensor. Each extractor adds its own columns to the result.
    ///
    /// For example, if you have:
    /// - Rolling stats: 30 features
    /// - Volatility: 15 features
    /// - Lag features: 12 features
    ///
    /// The output will have 57 feature columns.
    /// </para>
    /// </remarks>
    public Tensor<T> ExtractTimeSeriesFeatures(Tensor<T> data)
    {
        if (_timeSeriesExtractors.Count == 0)
        {
            throw new InvalidOperationException(
                "No time series extractors configured. Call ConfigureTimeSeriesFeatures first.");
        }

        // Fit and transform each extractor
        var allFeatures = new List<Tensor<T>>();
        foreach (var extractor in _timeSeriesExtractors)
        {
            extractor.Fit(data);
            allFeatures.Add(extractor.Transform(data));
        }

        // Concatenate all features horizontally
        return ConcatenateTensorsHorizontally(allFeatures);
    }

    /// <summary>
    /// Concatenates multiple tensors horizontally (along the feature dimension).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This stacks feature columns side by side.
    /// If you have tensors with shapes [100, 10] and [100, 20], the result
    /// will have shape [100, 30] (100 time steps, 30 features).
    /// </para>
    /// </remarks>
    private static Tensor<T> ConcatenateTensorsHorizontally(List<Tensor<T>> tensors)
    {
        if (tensors.Count == 0)
        {
            throw new ArgumentException("At least one tensor is required.", nameof(tensors));
        }

        if (tensors.Count == 1)
        {
            return tensors[0];
        }

        int timeSteps = tensors[0].Shape[0];
        int totalFeatures = tensors.Sum(t => t.Shape[1]);

        var result = new Tensor<T>(new[] { timeSteps, totalFeatures });

        int featureOffset = 0;
        foreach (var tensor in tensors)
        {
            if (tensor.Shape[0] != timeSteps)
            {
                throw new ArgumentException(
                    $"All tensors must have the same number of time steps. Expected {timeSteps}, got {tensor.Shape[0]}.");
            }

            int numFeatures = tensor.Shape[1];
            for (int t = 0; t < timeSteps; t++)
            {
                for (int f = 0; f < numFeatures; f++)
                {
                    result[t, featureOffset + f] = tensor[t, f];
                }
            }
            featureOffset += numFeatures;
        }

        return result;
    }

    #endregion
}
