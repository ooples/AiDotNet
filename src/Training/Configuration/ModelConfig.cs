namespace AiDotNet.Training.Configuration;

/// <summary>
/// Configuration for the model section of a training recipe.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This defines which model to use and its parameters.
/// The name should match a <see cref="AiDotNet.Enums.TimeSeriesModelType"/> value
/// (e.g., "ARIMA", "ExponentialSmoothing", "SARIMA").
/// </para>
/// </remarks>
public class ModelConfig
{
    /// <summary>
    /// Gets or sets the name of the model type to create.
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the model-specific parameters as key-value pairs.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Parameter names are matched case-insensitively to properties on the model's options class.
    /// For example, for ARIMA: <c>p</c> maps to LagOrder, <c>d</c> to DifferencingOrder, <c>q</c> to MovingAverageOrder.
    /// </para>
    /// </remarks>
    public Dictionary<string, object> Params { get; set; } = new();
}
