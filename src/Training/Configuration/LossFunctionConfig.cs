namespace AiDotNet.Training.Configuration;

/// <summary>
/// Configuration for the loss function section of a training recipe.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The loss function measures how far the model's predictions are from
/// the correct answers. The name should match a <see cref="AiDotNet.Enums.LossType"/> value
/// (e.g., "MeanSquaredError", "CrossEntropy", "Huber").
/// </para>
/// </remarks>
public class LossFunctionConfig
{
    /// <summary>
    /// Gets or sets the name of the loss function type to create.
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets loss function-specific parameters as key-value pairs.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Common parameters include:
    /// <list type="bullet">
    /// <item><description>Huber: <c>delta</c> (double, default 1.0)</description></item>
    /// <item><description>Focal: <c>gamma</c> (double, default 2.0), <c>alpha</c> (double, default 0.25)</description></item>
    /// <item><description>Quantile: <c>quantile</c> (double, default 0.5)</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    public Dictionary<string, object> Params { get; set; } = new();
}
