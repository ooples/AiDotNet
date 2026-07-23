using AiDotNet.Interfaces;

namespace AiDotNet.Onnx;

/// <summary>
/// Metadata about a loaded ONNX model.
/// </summary>
public class OnnxModelMetadata : IOnnxModelMetadata
{
    /// <inheritdoc/>
    public string ModelName { get; init; } = string.Empty;

    /// <inheritdoc/>
    public string? Description { get; init; }

    /// <inheritdoc/>
    public string? ProducerName { get; init; }

    /// <inheritdoc/>
    public string? ProducerVersion { get; init; }

    /// <inheritdoc/>
    public long OpsetVersion { get; init; }

    /// <inheritdoc/>
    public IReadOnlyList<IOnnxTensorInfo> Inputs { get; init; } = [];

    /// <inheritdoc/>
    public IReadOnlyList<IOnnxTensorInfo> Outputs { get; init; } = [];

    /// <summary>
    /// Gets the domain from the model.
    /// </summary>
    public string? Domain { get; init; }

    /// <summary>
    /// Gets the graph name from the model.
    /// </summary>
    public string? GraphName { get; init; }

    /// <summary>
    /// Gets the custom metadata as key-value pairs.
    /// </summary>
    public IReadOnlyDictionary<string, string> CustomMetadata { get; init; } =
        new Dictionary<string, string>();
}

/// <summary>
/// Information about an ONNX tensor (input or output).
/// </summary>
public class OnnxTensorInfo : IOnnxTensorInfo
{
    /// <inheritdoc/>
    public string Name { get; init; } = string.Empty;

    /// <inheritdoc/>
    public int[] Shape { get; init; } = [];

    /// <inheritdoc/>
    public string ElementType { get; init; } = "float";

    /// <summary>
    /// Gets whether this tensor has dynamic dimensions (shape contains -1).
    /// </summary>
    public bool HasDynamicShape => Shape.Any(d => d < 0);

    /// <summary>
    /// Gets the total number of elements if the shape is fully known, -1 otherwise.
    /// </summary>
    public long TotalElements => HasDynamicShape ? -1 : Shape.Aggregate(1L, (a, b) => a * b);

    /// <summary>
    /// Creates a string representation of the tensor info.
    /// </summary>
    public override string ToString()
    {
        var shapeStr = string.Join(", ", Shape.Select(d => d < 0 ? "?" : d.ToString()));
        return $"{Name}: {ElementType}[{shapeStr}]";
    }
}
