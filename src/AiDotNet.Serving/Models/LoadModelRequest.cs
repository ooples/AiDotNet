using AiDotNet.Serving.Configuration;

namespace AiDotNet.Serving.Models;

/// <summary>
/// Request to load a new model.
/// </summary>
public class LoadModelRequest
{
    /// <summary>
    /// Gets or sets the unique name for the model.
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the file path to the serialized model.
    /// </summary>
    public string Path { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the numeric type used by the model.
    /// Default is Double.
    /// </summary>
    public NumericType NumericType { get; set; } = NumericType.Double;
}

