using AiDotNet.Onnx.Protobuf;
using Google.Protobuf;

namespace AiDotNet.Onnx;

/// <summary>
/// Thin facade over the vendored ONNX protobuf types that lets layer converters add
/// nodes, initializers, inputs, and outputs to a model graph without touching the
/// generated <see cref="GraphProto"/> directly.
///
/// One <see cref="OnnxGraphBuilder"/> is created per export. Layer converters call
/// <see cref="AddNode"/>, <see cref="AddInitializer"/>, <see cref="AddInput"/>,
/// <see cref="AddOutput"/>, and <see cref="NextTensorName"/> as needed; the final
/// <see cref="Build"/> assembles a <see cref="ModelProto"/> ready to write.
///
/// This class is intentionally minimal in this initial commit — layer converters
/// landing in subsequent commits drive the API additions they need.
/// </summary>
public sealed class OnnxGraphBuilder
{
    private readonly OnnxExportOptions _options;
    private readonly GraphProto _graph = new() { Name = "AiDotNetModel" };
    private int _tensorCounter;

    public OnnxGraphBuilder(OnnxExportOptions options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
    }

    /// <summary>Opset version the resulting model targets.</summary>
    public int OpsetVersion => _options.OpsetVersion;

    /// <summary>
    /// Reserves a unique tensor name with the given prefix. Layer converters call this
    /// when they need a name for an intermediate tensor (e.g., the output of a Gemm
    /// before its bias add).
    /// </summary>
    public string NextTensorName(string prefix)
    {
        if (string.IsNullOrWhiteSpace(prefix))
        {
            prefix = "tensor";
        }
        return $"{prefix}_{_tensorCounter++}";
    }

    /// <summary>Adds a node to the graph. (Wired up in a later commit with per-layer converters.)</summary>
    public void AddNode(NodeProto node)
    {
        if (node is null)
        {
            throw new ArgumentNullException(nameof(node));
        }
        _graph.Node.Add(node);
    }

    /// <summary>Adds a constant initializer (e.g., a layer's weight tensor).</summary>
    public void AddInitializer(TensorProto initializer)
    {
        if (initializer is null)
        {
            throw new ArgumentNullException(nameof(initializer));
        }
        _graph.Initializer.Add(initializer);
    }

    /// <summary>Adds a graph input (a tensor the model expects from the caller).</summary>
    public void AddInput(ValueInfoProto input)
    {
        if (input is null)
        {
            throw new ArgumentNullException(nameof(input));
        }
        _graph.Input.Add(input);
    }

    /// <summary>Adds a graph output (a tensor the model produces).</summary>
    public void AddOutput(ValueInfoProto output)
    {
        if (output is null)
        {
            throw new ArgumentNullException(nameof(output));
        }
        _graph.Output.Add(output);
    }

    /// <summary>
    /// Assembles the final ModelProto with producer metadata + opset declaration.
    /// Call once after every layer has emitted its nodes.
    /// </summary>
    public ModelProto Build()
    {
        var model = new ModelProto
        {
            IrVersion = 8, // ONNX IR v8 corresponds to opset 17. Bump when opset bumps.
            ProducerName = _options.ProducerName,
            ProducerVersion = _options.ProducerVersion,
            Graph = _graph,
        };
        if (!string.IsNullOrEmpty(_options.ModelDescription))
        {
            model.DocString = _options.ModelDescription;
        }
        model.OpsetImport.Add(new OperatorSetIdProto { Domain = "", Version = _options.OpsetVersion });
        return model;
    }

    /// <summary>
    /// Serializes the built ModelProto to a stream. Convenience for callers that don't
    /// want to touch Google.Protobuf directly.
    /// </summary>
    public void WriteTo(Stream stream)
    {
        if (stream is null)
        {
            throw new ArgumentNullException(nameof(stream));
        }
        Build().WriteTo(stream);
    }

    // Suppress unused-field warning until layer converters use this directly.
    internal GraphProto GraphForTesting => _graph;
}
