namespace AiDotNet.Serving.Models;

/// <summary>
/// Information about a loaded model.
/// </summary>
public class ModelInfo
{
    /// <summary>
    /// Gets or sets the unique name of the model.
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the numeric type used by the model.
    /// </summary>
    public string NumericType { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the expected number of input features.
    /// </summary>
    public int InputDimension { get; set; }

    /// <summary>
    /// Gets or sets the number of output dimensions.
    /// </summary>
    public int OutputDimension { get; set; }

    /// <summary>
    /// Gets or sets the timestamp when the model was loaded.
    /// </summary>
    public DateTime LoadedAt { get; set; }

    /// <summary>
    /// Gets or sets the file path from which the model was loaded (if applicable).
    /// </summary>
    public string? SourcePath { get; set; }
}

