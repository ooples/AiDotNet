namespace AiDotNet.Serving.Configuration;

/// <summary>
/// Represents a model to be loaded when the server starts.
/// </summary>
public class StartupModel
{
    /// <summary>
    /// Gets or sets the name of the model.
    /// This will be used as the identifier for API requests.
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

    /// <summary>
    /// Gets or sets the expected SHA-256 hash (hex) of the model file.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When specified, AiDotNet.Serving validates the model file hash before loading.
    /// </para>
    /// </remarks>
    public string? Sha256 { get; set; }
}

