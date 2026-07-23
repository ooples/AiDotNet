global using AiDotNet.TimeSeries;

namespace AiDotNet.Factories;

/// <summary>
/// A factory class that creates time series models for forecasting and analysis.
/// </summary>
/// <typeparam name="T">The data type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Time series models are specialized algorithms that analyze data collected over time 
/// (like daily temperatures, monthly sales, or yearly population) to identify patterns and make predictions 
/// about future values.
/// </para>
/// <para>
/// This factory helps you create different types of time series models without needing to know their 
/// internal implementation details. Think of it like ordering a specific tool from a catalog - you just 
/// specify what you need, and the factory provides it.
/// </para>
/// </remarks>
public class TimeSeriesModelFactory<T, TInput, TOutput>
{
    /// <summary>
    /// Creates a time series model of the specified type with custom options.
    /// </summary>
    /// <param name="modelType">The type of time series model to create.</param>
    /// <param name="options">Configuration options for the model.</param>
    /// <returns>An implementation of ITimeSeriesModel<T> for the specified model type.</returns>
    /// <exception cref="ArgumentException">Thrown when an unsupported model type is specified or when the options don't match the model type.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method creates a specific type of time series model with custom settings. 
    /// Different models are better suited for different types of time series data.
    /// </para>
    /// <para>
    /// Choose the model that best matches the patterns in your data. For example:
    /// - Use ARIMA for data with trends but no seasonality
    /// - Use SARIMA for data with both trends and seasonal patterns
    /// - Use Exponential Smoothing when recent observations are more important than older ones
    /// - Use Prophet for data with multiple seasonal patterns and holiday effects
    /// </para>
    /// <para>
    /// Note that you must provide options that match the model type. For example, if you specify 
    /// TimeSeriesModelType.ARIMA, you must provide ARIMAOptions<T>.
    /// </para>
    /// </remarks>
    public static ITimeSeriesModel<T> CreateModel(TimeSeriesModelType modelType, TimeSeriesRegressionOptions<T> options)
    {
        // If the options are already the correct specific type, use them directly
        // Otherwise, convert the base options to the appropriate model-specific options
        var modelOptions = ConvertToModelSpecificOptions(modelType, options);

        return (modelType, modelOptions) switch
        {
            (TimeSeriesModelType.ARIMA, ARIMAOptions<T> arimaOptions) => new ARIMAModel<T>(arimaOptions),
            (TimeSeriesModelType.SARIMA, SARIMAOptions<T> sarimaOptions) => new SARIMAModel<T>(sarimaOptions),
            (TimeSeriesModelType.ARMA, ARMAOptions<T> armaOptions) => new ARMAModel<T>(armaOptions),
            (TimeSeriesModelType.AutoRegressive, ARModelOptions<T> arOptions) => new ARModel<T>(arOptions),
            (TimeSeriesModelType.MA, MAModelOptions<T> maOptions) => new MAModel<T>(maOptions),
            (TimeSeriesModelType.ExponentialSmoothing, ExponentialSmoothingOptions<T> esOptions) =>
                new ExponentialSmoothingModel<T>(esOptions),
            (TimeSeriesModelType.SimpleExponentialSmoothing, ExponentialSmoothingOptions<T> esOptions) =>
                new ExponentialSmoothingModel<T>(esOptions),
            (TimeSeriesModelType.DoubleExponentialSmoothing, ExponentialSmoothingOptions<T> esOptions) =>
                new ExponentialSmoothingModel<T>(esOptions),
            (TimeSeriesModelType.TripleExponentialSmoothing, ExponentialSmoothingOptions<T> esOptions) =>
                new ExponentialSmoothingModel<T>(esOptions),
            (TimeSeriesModelType.StateSpace, StateSpaceModelOptions<T> ssOptions) => new StateSpaceModel<T>(ssOptions),
            (TimeSeriesModelType.TBATS, TBATSModelOptions<T> tbatsOptions) => new TBATSModel<T>(tbatsOptions),
            (TimeSeriesModelType.DynamicRegressionWithARIMAErrors, DynamicRegressionWithARIMAErrorsOptions<T> drOptions) => new DynamicRegressionWithARIMAErrors<T>(drOptions),
            (TimeSeriesModelType.ARIMAX, ARIMAXModelOptions<T> arimaxOptions) => new ARIMAXModel<T>(arimaxOptions),
            (TimeSeriesModelType.GARCH, GARCHModelOptions<T> garchOptions) => new GARCHModel<T>(garchOptions),
            (TimeSeriesModelType.VAR, VARModelOptions<T> varOptions) => new VectorAutoRegressionModel<T>(varOptions),
            (TimeSeriesModelType.VARMA, VARMAModelOptions<T> varmaOptions) => new VARMAModel<T>(varmaOptions),
            (TimeSeriesModelType.ProphetModel, ProphetOptions<T, TInput, TOutput> prophetOptions) => new ProphetModel<T, TInput, TOutput>(prophetOptions),
            (TimeSeriesModelType.NeuralNetworkARIMA, NeuralNetworkARIMAOptions<T> nnarimaOptions) => new NeuralNetworkARIMAModel<T>(nnarimaOptions),
            (TimeSeriesModelType.BayesianStructuralTimeSeriesModel, BayesianStructuralTimeSeriesOptions<T> bstsOptions) => new BayesianStructuralTimeSeriesModel<T>(bstsOptions),
            (TimeSeriesModelType.SpectralAnalysis, SpectralAnalysisOptions<T> saOptions) => new SpectralAnalysisModel<T>(saOptions),
            (TimeSeriesModelType.STLDecomposition, STLDecompositionOptions<T> stlOptions) => new STLDecomposition<T>(stlOptions),
            (TimeSeriesModelType.InterventionAnalysis, InterventionAnalysisOptions<T, Matrix<T>, Vector<T>> iaOptions) => new InterventionAnalysisModel<T>(iaOptions),
            (TimeSeriesModelType.TransferFunctionModel, TransferFunctionOptions<T, Matrix<T>, Vector<T>> tfOptions) => new TransferFunctionModel<T>(tfOptions),
            (TimeSeriesModelType.UnobservedComponentsModel, UnobservedComponentsOptions<T, TInput, TOutput> ucOptions) => new UnobservedComponentsModel<T, TInput, TOutput>(ucOptions),
            _ => throw new ArgumentException($"Unsupported model type or invalid options: {modelType}")
        };
    }

    /// <summary>
    /// Converts base TimeSeriesRegressionOptions to the appropriate model-specific options type.
    /// </summary>
    /// <param name="modelType">The type of time series model.</param>
    /// <param name="options">The base options to convert.</param>
    /// <returns>Model-specific options if conversion is needed, or the original options if already correct type.</returns>
    private static TimeSeriesRegressionOptions<T> ConvertToModelSpecificOptions(TimeSeriesModelType modelType, TimeSeriesRegressionOptions<T> options)
    {
        // If options is already the correct specific type, return as-is
        return modelType switch
        {
            TimeSeriesModelType.ARIMA when options is ARIMAOptions<T> => options,
            TimeSeriesModelType.ARIMA => new ARIMAOptions<T>
            {
                LagOrder = options.LagOrder,
                IncludeTrend = options.IncludeTrend,
                SeasonalPeriod = options.SeasonalPeriod,
                AutocorrelationCorrection = options.AutocorrelationCorrection
            },

            TimeSeriesModelType.SARIMA when options is SARIMAOptions<T> => options,
            TimeSeriesModelType.SARIMA => new SARIMAOptions<T>
            {
                LagOrder = options.LagOrder,
                SeasonalPeriod = options.SeasonalPeriod > 0 ? options.SeasonalPeriod : 12
            },

            TimeSeriesModelType.ARMA when options is ARMAOptions<T> => options,
            TimeSeriesModelType.ARMA => new ARMAOptions<T>
            {
                LagOrder = options.LagOrder,
                IncludeTrend = options.IncludeTrend,
                SeasonalPeriod = options.SeasonalPeriod
            },

            TimeSeriesModelType.AutoRegressive when options is ARModelOptions<T> => options,
            TimeSeriesModelType.AutoRegressive => new ARModelOptions<T>
            {
                AROrder = options.LagOrder,
                LagOrder = options.LagOrder,
                IncludeTrend = options.IncludeTrend,
                SeasonalPeriod = options.SeasonalPeriod
            },

            TimeSeriesModelType.MA when options is MAModelOptions<T> => options,
            TimeSeriesModelType.MA => new MAModelOptions<T>
            {
                LagOrder = options.LagOrder,
                IncludeTrend = options.IncludeTrend,
                SeasonalPeriod = options.SeasonalPeriod
            },

            TimeSeriesModelType.ExponentialSmoothing when options is ExponentialSmoothingOptions<T> => options,
            TimeSeriesModelType.ExponentialSmoothing => new ExponentialSmoothingOptions<T>
            {
                SeasonalPeriod = options.SeasonalPeriod,
                UseTrend = options.IncludeTrend
            },

            TimeSeriesModelType.SimpleExponentialSmoothing when options is ExponentialSmoothingOptions<T> => options,
            TimeSeriesModelType.SimpleExponentialSmoothing => new ExponentialSmoothingOptions<T>
            {
                UseTrend = false,
                UseSeasonal = false
            },

            TimeSeriesModelType.DoubleExponentialSmoothing when options is ExponentialSmoothingOptions<T> => options,
            TimeSeriesModelType.DoubleExponentialSmoothing => new ExponentialSmoothingOptions<T>
            {
                UseTrend = true,
                UseSeasonal = false
            },

            TimeSeriesModelType.TripleExponentialSmoothing when options is ExponentialSmoothingOptions<T> => options,
            TimeSeriesModelType.TripleExponentialSmoothing => new ExponentialSmoothingOptions<T>
            {
                UseTrend = true,
                UseSeasonal = true,
                SeasonalPeriod = options.SeasonalPeriod > 0 ? options.SeasonalPeriod : 12
            },

            // For all other model types, return as-is (they should provide correct options)
            _ => options
        };
    }

    /// <summary>
    /// Creates a time series model of the specified type with default options.
    /// </summary>
    /// <param name="modelType">The type of time series model to create.</param>
    /// <returns>An implementation of ITimeSeriesModel<T> for the specified model type.</returns>
    /// <exception cref="ArgumentException">Thrown when an unsupported model type is specified.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method creates a time series model with default settings. It's a simpler 
    /// version of the other CreateModel method that doesn't require you to specify custom options.
    /// </para>
    /// <para>
    /// This is useful when you're just getting started and don't yet know which specific settings to use. 
    /// The default options provide a reasonable starting point for most common scenarios.
    /// </para>
    /// <para>
    /// For more control over the model's behavior, use the other CreateModel method that accepts custom options.
    /// </para>
    /// </remarks>
    public static ITimeSeriesModel<T> CreateModel(TimeSeriesModelType modelType)
    {
        TimeSeriesRegressionOptions<T> options = modelType switch
        {
            TimeSeriesModelType.ARIMA => new ARIMAOptions<T>(),
            TimeSeriesModelType.SARIMA => new SARIMAOptions<T>(),
            TimeSeriesModelType.ARMA => new ARMAOptions<T>(),
            TimeSeriesModelType.AutoRegressive => new ARModelOptions<T>(),
            TimeSeriesModelType.MA => new MAModelOptions<T>(),
            TimeSeriesModelType.ExponentialSmoothing => new ExponentialSmoothingOptions<T>(),
            TimeSeriesModelType.SimpleExponentialSmoothing => new ExponentialSmoothingOptions<T>
            {
                UseTrend = false,
                UseSeasonal = false
            },
            TimeSeriesModelType.DoubleExponentialSmoothing => new ExponentialSmoothingOptions<T>
            {
                UseTrend = true,
                UseSeasonal = false
            },
            TimeSeriesModelType.TripleExponentialSmoothing => new ExponentialSmoothingOptions<T>
            {
                UseTrend = true,
                UseSeasonal = true,
                SeasonalPeriod = 12  // Default to monthly seasonality
            },
            TimeSeriesModelType.StateSpace => new StateSpaceModelOptions<T>(),
            TimeSeriesModelType.TBATS => new TBATSModelOptions<T>(),
            TimeSeriesModelType.DynamicRegressionWithARIMAErrors => new DynamicRegressionWithARIMAErrorsOptions<T>(),
            TimeSeriesModelType.ARIMAX => new ARIMAXModelOptions<T>(),
            TimeSeriesModelType.GARCH => new GARCHModelOptions<T>(),
            TimeSeriesModelType.VAR => new VARModelOptions<T>(),
            TimeSeriesModelType.VARMA => new VARMAModelOptions<T>(),
            TimeSeriesModelType.ProphetModel => new ProphetOptions<T, TInput, TOutput>(),
            TimeSeriesModelType.NeuralNetworkARIMA => new NeuralNetworkARIMAOptions<T>(),
            TimeSeriesModelType.BayesianStructuralTimeSeriesModel => new BayesianStructuralTimeSeriesOptions<T>(),
            TimeSeriesModelType.SpectralAnalysis => new SpectralAnalysisOptions<T>(),
            TimeSeriesModelType.STLDecomposition => new STLDecompositionOptions<T>(),
            TimeSeriesModelType.InterventionAnalysis => new InterventionAnalysisOptions<T, TInput, TOutput>(),
            TimeSeriesModelType.TransferFunctionModel => new TransferFunctionOptions<T, TInput, TOutput>(),
            TimeSeriesModelType.UnobservedComponentsModel => new UnobservedComponentsOptions<T, TInput, TOutput>(),
            TimeSeriesModelType.Custom => throw new ArgumentException("Custom models require custom options to be provided explicitly"),
            _ => throw new ArgumentException($"Unsupported model type: {modelType}")
        };

        return CreateModel(modelType, options);
    }
}
