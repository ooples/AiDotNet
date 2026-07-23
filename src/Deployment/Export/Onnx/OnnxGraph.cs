namespace AiDotNet.Deployment.Export.Onnx;

/// <summary>
/// Represents an ONNX computational graph.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> OnnxGraph provides AI safety functionality. Default values follow the original paper settings.</para>
/// </remarks>
public class OnnxGraph
{
    /// <summary>
    /// Gets or sets the name of the graph.
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the ONNX opset version.
    /// </summary>
    public int OpsetVersion { get; set; } = 13;

    /// <summary>
    /// Gets the list of input nodes.
    /// </summary>
    public List<OnnxNode> Inputs { get; } = new();

    /// <summary>
    /// Gets the list of output nodes.
    /// </summary>
    public List<OnnxNode> Outputs { get; } = new();

    /// <summary>
    /// Gets the list of operations in the graph.
    /// </summary>
    public List<OnnxOperation> Operations { get; } = new();

    /// <summary>
    /// Gets the initializers (weights and biases).
    /// </summary>
    public Dictionary<string, object> Initializers { get; } = new();

    /// <summary>
    /// Gets or sets the graph documentation string.
    /// </summary>
    public string? DocString { get; set; }
}
