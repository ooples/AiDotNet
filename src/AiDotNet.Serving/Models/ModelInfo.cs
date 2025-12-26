using AiDotNet.Serving.Configuration;

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
    public NumericType NumericType { get; set; } = NumericType.Double;

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

    /// <summary>
    /// Gets or sets the model version from the registry (if loaded from a registry).
    /// </summary>
    public int? RegistryVersion { get; set; }

    /// <summary>
    /// Gets or sets the registry model stage (if loaded from a registry).
    /// </summary>
    public string? RegistryStage { get; set; }

    /// <summary>
    /// Gets or sets whether this model was loaded from a model registry.
    /// </summary>
    public bool IsFromRegistry { get; set; }

    /// <summary>
    /// Gets or sets whether batching is enabled for this model.
    /// </summary>
    public bool EnableBatching { get; set; } = true;

    /// <summary>
    /// Gets or sets whether this model supports multimodal inputs (text, images, etc.).
    /// </summary>
    public bool IsMultimodal { get; set; }

    /// <summary>
    /// Gets or sets the embedding dimension for multimodal models.
    /// </summary>
    public int? EmbeddingDimension { get; set; }

    /// <summary>
    /// Gets or sets the maximum sequence length for text inputs in multimodal models.
    /// </summary>
    public int? MaxSequenceLength { get; set; }

    /// <summary>
    /// Gets or sets the expected image size for multimodal models.
    /// </summary>
    public int? ImageSize { get; set; }
}

