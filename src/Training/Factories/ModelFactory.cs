using System.Globalization;
using System.Reflection;
using AiDotNet.Enums;
using AiDotNet.Factories;
using AiDotNet.Models.Options;
using AiDotNet.Training.Configuration;

namespace AiDotNet.Training.Factories;

/// <summary>
/// Factory for creating time series models from <see cref="ModelConfig"/> objects.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This factory creates model instances from configuration objects.
/// You specify the model name (like "ARIMA" or "ExponentialSmoothing") and any parameters,
/// and the factory creates the right model with those settings.
/// </para>
/// </remarks>
internal static class ModelFactory<T, TInput, TOutput>
{
    /// <summary>
    /// Creates a time series model from a <see cref="ModelConfig"/> object.
    /// </summary>
    /// <param name="config">The model configuration with name and parameters.</param>
    /// <returns>An <see cref="ITimeSeriesModel{T}"/> instance configured with the specified parameters.</returns>
    /// <exception cref="ArgumentNullException">Thrown when config is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the model name is empty or does not match any known model type.</exception>
    public static ITimeSeriesModel<T> Create(ModelConfig config)
    {
        if (config is null)
        {
            throw new ArgumentNullException(nameof(config));
        }

        if (string.IsNullOrWhiteSpace(config.Name))
        {
            throw new ArgumentException("Model name cannot be null or empty.", nameof(config));
        }

        if (!Enum.TryParse<TimeSeriesModelType>(config.Name, ignoreCase: true, out var modelType))
        {
            throw new ArgumentException(
                $"Unknown model name: '{config.Name}'. Valid names are: {string.Join(", ", Enum.GetNames(typeof(TimeSeriesModelType)))}",
                nameof(config));
        }

        if (config.Params.Count == 0)
        {
            return TimeSeriesModelFactory<T, TInput, TOutput>.CreateModel(modelType);
        }

        // Create default options, then apply parameters from config
        var options = CreateDefaultOptions(modelType);
        ApplyParameters(options, config.Params);

        return TimeSeriesModelFactory<T, TInput, TOutput>.CreateModel(modelType, options);
    }

    /// <summary>
    /// Creates a time series model by name string with default parameters.
    /// </summary>
    /// <param name="modelName">The name of the model type (case-insensitive).</param>
    /// <returns>An <see cref="ITimeSeriesModel{T}"/> instance with default parameters.</returns>
    public static ITimeSeriesModel<T> Create(string modelName)
    {
        return Create(new ModelConfig { Name = modelName });
    }

    /// <summary>
    /// Creates default options for the specified model type.
    /// </summary>
    private static TimeSeriesRegressionOptions<T> CreateDefaultOptions(TimeSeriesModelType modelType)
    {
        return modelType switch
        {
            TimeSeriesModelType.ARIMA => new ARIMAOptions<T>(),
            TimeSeriesModelType.SARIMA => new SARIMAOptions<T>(),
            TimeSeriesModelType.ARMA => new ARMAOptions<T>(),
            TimeSeriesModelType.AutoRegressive => new ARModelOptions<T>(),
            TimeSeriesModelType.MA => new MAModelOptions<T>(),
            TimeSeriesModelType.ExponentialSmoothing or
            TimeSeriesModelType.SimpleExponentialSmoothing or
            TimeSeriesModelType.DoubleExponentialSmoothing or
            TimeSeriesModelType.TripleExponentialSmoothing => new ExponentialSmoothingOptions<T>(),
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
            _ => new TimeSeriesRegressionOptions<T>()
        };
    }

    /// <summary>
    /// Applies parameter dictionary values to the options object via reflection.
    /// </summary>
    /// <param name="options">The options object to populate.</param>
    /// <param name="parameters">The parameter dictionary from the configuration.</param>
    /// <param name="strict">When true, throws on unknown or unconvertible parameters. Default is false.</param>
    private static void ApplyParameters(TimeSeriesRegressionOptions<T> options, Dictionary<string, object> parameters, bool strict = false)
    {
        var optionsType = options.GetType();
        var warnings = new List<string>();

        foreach (var kvp in parameters)
        {
            // Find matching property (case-insensitive)
            var property = optionsType.GetProperty(
                kvp.Key,
                BindingFlags.Public | BindingFlags.Instance | BindingFlags.IgnoreCase);

            if (property is null || !property.CanWrite)
            {
                // Also check common ARIMA aliases: p -> LagOrder, d -> DifferencingOrder, q -> MovingAverageOrder
                property = ResolveAlias(optionsType, kvp.Key);
            }

            if (property is null || !property.CanWrite)
            {
                var availableProps = string.Join(", ",
                    optionsType.GetProperties(BindingFlags.Public | BindingFlags.Instance)
                        .Where(p => p.CanWrite)
                        .Select(p => p.Name));
                var message = $"Unknown parameter '{kvp.Key}' for {optionsType.Name}. Available properties: {availableProps}";
                warnings.Add(message);
                System.Diagnostics.Trace.TraceWarning($"[ModelFactory] {message}");
                continue;
            }

            try
            {
                object paramValue = kvp.Value;
                var convertedValue = ConvertValue(paramValue, property.PropertyType);
                property.SetValue(options, convertedValue);
            }
            catch (Exception ex) when (ex is InvalidCastException or FormatException or OverflowException)
            {
                var message = $"Cannot convert parameter '{kvp.Key}' value '{kvp.Value}' to {property.PropertyType.Name}: {ex.Message}";
                warnings.Add(message);
                System.Diagnostics.Trace.TraceWarning($"[ModelFactory] {message}");
            }
        }

        if (strict && warnings.Count > 0)
        {
            throw new ArgumentException(
                $"Model configuration has {warnings.Count} parameter error(s):\n" +
                string.Join("\n", warnings.Select(w => $"  - {w}")));
        }
    }

    /// <summary>
    /// Resolves common parameter name aliases to actual property names.
    /// </summary>
    private static PropertyInfo? ResolveAlias(Type optionsType, string alias)
    {
        var normalizedAlias = alias.ToLowerInvariant();
        string? propertyName = normalizedAlias switch
        {
            "p" => "LagOrder",
            "d" => "DifferencingOrder",
            "q" => "MovingAverageOrder",
            "seasonalperiod" or "seasonal_period" or "m" => "SeasonalPeriod",
            "includetrend" or "include_trend" or "trend" => "IncludeTrend",
            _ => null
        };

        if (propertyName is null)
        {
            return null;
        }

        return optionsType.GetProperty(propertyName, BindingFlags.Public | BindingFlags.Instance | BindingFlags.IgnoreCase);
    }

    /// <summary>
    /// Converts a value from the YAML dictionary to the target property type.
    /// </summary>
    private static object? ConvertValue(object value, Type targetType)
    {
        if (value is null)
        {
            return targetType.IsValueType ? Activator.CreateInstance(targetType) : null;
        }

        // Handle nullable types
        var underlyingType = Nullable.GetUnderlyingType(targetType);
        if (underlyingType is not null)
        {
            targetType = underlyingType;
        }

        if (targetType == typeof(int))
        {
            return Convert.ToInt32(value, CultureInfo.InvariantCulture);
        }

        if (targetType == typeof(double))
        {
            return Convert.ToDouble(value, CultureInfo.InvariantCulture);
        }

        if (targetType == typeof(float))
        {
            return Convert.ToSingle(value, CultureInfo.InvariantCulture);
        }

        if (targetType == typeof(bool))
        {
            return Convert.ToBoolean(value, CultureInfo.InvariantCulture);
        }

        if (targetType == typeof(string))
        {
            return value.ToString();
        }

        if (targetType == typeof(long))
        {
            return Convert.ToInt64(value, CultureInfo.InvariantCulture);
        }

        if (targetType.IsEnum && value is string stringValue)
        {
            return Enum.Parse(targetType, stringValue, ignoreCase: true);
        }

        // Fallback: try ChangeType
        return Convert.ChangeType(value, targetType, CultureInfo.InvariantCulture);
    }
}
