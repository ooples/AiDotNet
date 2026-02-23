namespace AiDotNet.Deployment.Export.Onnx;

/// <summary>
/// Represents an ONNX node (input or output).
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> OnnxNode provides AI safety functionality. Default values follow the original paper settings.</para>
/// </remarks>
public class OnnxNode
{
    /// <summary>
    /// Gets or sets the node name.
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the data type (e.g., "float", "double", "int32").
    /// </summary>
    public string DataType { get; set; } = "float";

    /// <summary>
    /// Gets or sets the shape dimensions. Null means shape will be inferred.
    /// </summary>
    public int[]? Shape { get; set; }

    /// <summary>
    /// Gets or sets the node documentation string.
    /// </summary>
    public string? DocString { get; set; }
}
