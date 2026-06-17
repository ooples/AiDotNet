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

    // ── Convenience helpers (float32 — v0.1 supports float only) ─────────────

    /// <summary>
    /// Adds a float32 constant initializer with the given shape. Used by layer
    /// converters to ship weights/biases into the graph.
    /// </summary>
    public string AddFloatInitializer(string namePrefix, float[] data, int[] shape)
    {
        if (data is null) throw new ArgumentNullException(nameof(data));
        if (shape is null) throw new ArgumentNullException(nameof(shape));
        long expected = 1;
        foreach (var d in shape) expected *= d;
        if (data.Length != expected)
        {
            throw new ArgumentException(
                $"Initializer '{namePrefix}' data length {data.Length} does not match shape product {expected}.",
                nameof(data));
        }

        var name = NextTensorName(namePrefix);
        var tensor = new TensorProto
        {
            Name = name,
            DataType = (int)TensorProto.Types.DataType.Float,
        };
        foreach (var d in shape) tensor.Dims.Add(d);
        tensor.FloatData.AddRange(data);
        _graph.Initializer.Add(tensor);
        return name;
    }

    /// <summary>
    /// Declares a float32 graph input with the given (fixed or symbolic) shape.
    /// Use -1 in <paramref name="shape"/> for a dimension that is dynamic at
    /// runtime (e.g., batch size); -1 becomes a symbolic dim named "batch_N".
    /// </summary>
    public void AddFloatInput(string name, int[] shape)
    {
        _graph.Input.Add(MakeFloatValueInfo(name, shape));
    }

    /// <summary>Declares a float32 graph output. See <see cref="AddFloatInput"/> for shape semantics.</summary>
    public void AddFloatOutput(string name, int[] shape)
    {
        _graph.Output.Add(MakeFloatValueInfo(name, shape));
    }

    private static ValueInfoProto MakeFloatValueInfo(string name, int[] shape)
    {
        if (string.IsNullOrWhiteSpace(name)) throw new ArgumentNullException(nameof(name));
        if (shape is null) throw new ArgumentNullException(nameof(shape));

        var tensorType = new TypeProto.Types.Tensor
        {
            ElemType = (int)TensorProto.Types.DataType.Float,
            Shape = new TensorShapeProto(),
        };
        for (int i = 0; i < shape.Length; i++)
        {
            var dim = new TensorShapeProto.Types.Dimension();
            if (shape[i] >= 0)
            {
                dim.DimValue = shape[i];
            }
            else
            {
                dim.DimParam = $"batch_{i}";
            }
            tensorType.Shape.Dim.Add(dim);
        }
        return new ValueInfoProto
        {
            Name = name,
            Type = new TypeProto { TensorType = tensorType },
        };
    }

    /// <summary>
    /// Adds an ONNX op node with named inputs and outputs and optional attributes.
    /// Returns the node so callers (rare) can attach extra attributes; most layer
    /// converters can ignore the return value.
    /// </summary>
    public NodeProto AddOp(string opType, string[] inputs, string[] outputs, string? name = null)
    {
        if (string.IsNullOrWhiteSpace(opType)) throw new ArgumentNullException(nameof(opType));
        if (inputs is null) throw new ArgumentNullException(nameof(inputs));
        if (outputs is null) throw new ArgumentNullException(nameof(outputs));

        var node = new NodeProto
        {
            OpType = opType,
            Name = name ?? NextTensorName(opType.ToLowerInvariant() + "_node"),
        };
        foreach (var i in inputs) node.Input.Add(i);
        foreach (var o in outputs) node.Output.Add(o);
        _graph.Node.Add(node);
        return node;
    }
}
