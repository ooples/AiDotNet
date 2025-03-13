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
public class TimeSeriesModelFactory<T>
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
    /// Available model types include:
    /// <list type="bullet">
    /// <item><description>ARIMA (AutoRegressive Integrated Moving Average): A flexible model that can handle data with trends.</description></item>
    /// <item><description>ExponentialSmoothing: Good for data with seasonal patterns that repeat at regular intervals.</description></item>
    /// <item><description>SARIMA (Seasonal ARIMA): An extension of ARIMA that can also handle seasonal effects in the data.</description></item>
    /// </list>
    /// </para>
    /// <para>
    /// Note that you must provide options that match the model type. For example, if you specify 
    /// TimeSeriesModelType.ARIMA, you must provide ARIMAOptions<T>.
    /// </para>
    /// </remarks>
    public static ITimeSeriesModel<T> CreateModel(TimeSeriesModelType modelType, TimeSeriesRegressionOptions<T> options)
    {
        return (modelType, options) switch
        {
            (TimeSeriesModelType.ARIMA, ARIMAOptions<T> arimaOptions) => new ARIMAModel<T>(arimaOptions),
            (TimeSeriesModelType.ExponentialSmoothing, ExponentialSmoothingOptions<T> esOptions) => new ExponentialSmoothingModel<T>(esOptions),
            (TimeSeriesModelType.SARIMA, SARIMAOptions<T> sarimaOptions) => new SARIMAModel<T>(sarimaOptions),
            _ => throw new ArgumentException($"Unsupported model type or invalid options: {modelType}")
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
            TimeSeriesModelType.ExponentialSmoothing => new ExponentialSmoothingOptions<T>(),
            TimeSeriesModelType.SARIMA => new SARIMAOptions<T>(),
            _ => throw new ArgumentException($"Unsupported model type: {modelType}")
        };

        return CreateModel(modelType, options);
    }
}