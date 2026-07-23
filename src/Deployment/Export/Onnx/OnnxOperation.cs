using System.Collections.Generic;

namespace AiDotNet.Deployment.Export.Onnx;

/// <summary>
/// Represents an ONNX operation (node in the computational graph).
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> OnnxOperation provides AI safety functionality. Default values follow the original paper settings.</para>
/// </remarks>
public class OnnxOperation
{
    /// <summary>
    /// Gets or sets the operation type (e.g., "Conv", "Relu", "MatMul").
    /// </summary>
    public string Type { get; set; } = string.Empty;

    /// <summary>
    /// Gets the list of input names.
    /// </summary>
    public List<string> Inputs { get; set; } = new();

    /// <summary>
    /// Gets the list of output names.
    /// </summary>
    public List<string> Outputs { get; set; } = new();

    /// <summary>
    /// Gets the operation attributes (parameters).
    /// </summary>
    public Dictionary<string, object> Attributes { get; set; } = new();

    /// <summary>
    /// Gets or sets the operation name.
    /// </summary>
    public string? Name { get; set; }

    /// <summary>
    /// Gets or sets the domain (for custom operators).
    /// </summary>
    public string Domain { get; set; } = "ai.onnx";
}
