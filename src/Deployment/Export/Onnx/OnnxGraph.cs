namespace AiDotNet.Deployment.Export.Onnx;

/// <summary>
/// Represents an ONNX computational graph.
/// </summary>
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

/// <summary>
/// Represents an ONNX node (input or output).
/// </summary>
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

/// <summary>
/// Represents an ONNX operation (node in the computational graph).
/// </summary>
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
