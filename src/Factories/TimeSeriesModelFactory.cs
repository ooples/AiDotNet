global using AiDotNet.TimeSeries;

namespace AiDotNet.Factories;

public class TimeSeriesModelFactory<T>
{
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