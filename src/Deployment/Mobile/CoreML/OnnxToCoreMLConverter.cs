using Google.Protobuf;
using System.Collections.Generic;

namespace AiDotNet.Deployment.Mobile.CoreML;

/// <summary>
/// Converts ONNX models to CoreML format.
/// Implements operator mapping and weight conversion for production deployment.
/// </summary>
internal static class OnnxToCoreMLConverter
{
    /// <summary>
    /// Converts ONNX model bytes to CoreML model structure.
    /// </summary>
    public static CoreMLModel ConvertOnnxToCoreML(byte[] onnxBytes, CoreMLConfiguration config)
    {
        // Parse ONNX protobuf
        var onnxGraph = ParseOnnxGraph(onnxBytes);

        var coreMLModel = new CoreMLModel
        {
            Description = CreateModelDescription(onnxGraph, config),
            Network = ConvertNeuralNetwork(onnxGraph, config)
        };

        return coreMLModel;
    }

    private static OnnxGraphInfo ParseOnnxGraph(byte[] onnxBytes)
    {
        // Parse ONNX protobuf to extract graph structure
        using var stream = new MemoryStream(onnxBytes);
        using var reader = new CodedInputStream(stream);

        var graphInfo = new OnnxGraphInfo();

        while (!reader.IsAtEnd)
        {
            var tag = reader.ReadTag();
            var fieldNumber = WireFormat.GetTagFieldNumber(tag);

            switch (fieldNumber)
            {
                case 7: // graph field
                    var graphBytes = reader.ReadBytes();
                    ParseGraph(graphBytes.ToByteArray(), graphInfo);
                    break;
                default:
                    reader.SkipLastField();
                    break;
            }
        }

        return graphInfo;
    }

    private static void ParseGraph(byte[] graphBytes, OnnxGraphInfo graphInfo)
    {
        using var stream = new MemoryStream(graphBytes);
        using var reader = new CodedInputStream(stream);

        while (!reader.IsAtEnd)
        {
            var tag = reader.ReadTag();
            var fieldNumber = WireFormat.GetTagFieldNumber(tag);

            switch (fieldNumber)
            {
                case 1: // node (operations)
                    var nodeBytes = reader.ReadBytes();
                    graphInfo.Operations.Add(ParseNode(nodeBytes.ToByteArray()));
                    break;
                case 2: // name
                    graphInfo.Name = reader.ReadString();
                    break;
                case 5: // initializer (weights)
                    var initBytes = reader.ReadBytes();
                    var (name, weights) = ParseTensor(initBytes.ToByteArray());
                    graphInfo.Initializers[name] = weights;
                    break;
                case 11: // input
                    var inputBytes = reader.ReadBytes();
                    graphInfo.Inputs.Add(ParseValueInfo(inputBytes.ToByteArray()));
                    break;
                case 12: // output
                    var outputBytes = reader.ReadBytes();
                    graphInfo.Outputs.Add(ParseValueInfo(outputBytes.ToByteArray()));
                    break;
                default:
                    reader.SkipLastField();
                    break;
            }
        }
    }

    private static OnnxNode ParseNode(byte[] nodeBytes)
    {
        using var stream = new MemoryStream(nodeBytes);
        using var reader = new CodedInputStream(stream);

        var node = new OnnxNode();

        while (!reader.IsAtEnd)
        {
            var tag = reader.ReadTag();
            var fieldNumber = WireFormat.GetTagFieldNumber(tag);

            switch (fieldNumber)
            {
                case 1: // input
                    node.Inputs.Add(reader.ReadString());
                    break;
                case 2: // output
                    node.Outputs.Add(reader.ReadString());
                    break;
                case 3: // name
                    node.Name = reader.ReadString();
                    break;
                case 4: // op_type
                    node.OpType = reader.ReadString();
                    break;
                default:
                    reader.SkipLastField();
                    break;
            }
        }

        return node;
    }

    private static (string name, float[] weights) ParseTensor(byte[] tensorBytes)
    {
        using var stream = new MemoryStream(tensorBytes);
        using var reader = new CodedInputStream(stream);

        string name = string.Empty;
        float[] weights = Array.Empty<float>();

        while (!reader.IsAtEnd)
        {
            var tag = reader.ReadTag();
            var fieldNumber = WireFormat.GetTagFieldNumber(tag);

            switch (fieldNumber)
            {
                case 3: // name
                    name = reader.ReadString();
                    break;
                case 9: // raw_data
                    var rawBytes = reader.ReadBytes().ToByteArray();
                    weights = BytesToFloatArray(rawBytes);
                    break;
                default:
                    reader.SkipLastField();
                    break;
            }
        }

        return (name, weights);
    }

    private static OnnxValueInfo ParseValueInfo(byte[] valueInfoBytes)
    {
        using var stream = new MemoryStream(valueInfoBytes);
        using var reader = new CodedInputStream(stream);

        var valueInfo = new OnnxValueInfo();

        while (!reader.IsAtEnd)
        {
            var tag = reader.ReadTag();
            var fieldNumber = WireFormat.GetTagFieldNumber(tag);

            switch (fieldNumber)
            {
                case 1: // name
                    valueInfo.Name = reader.ReadString();
                    break;
                case 2: // type
                    var typeBytes = reader.ReadBytes().ToByteArray();
                    valueInfo.Shape = ParseTypeProto(typeBytes);
                    break;
                default:
                    reader.SkipLastField();
                    break;
            }
        }

        return valueInfo;
    }

    private static int[] ParseTypeProto(byte[] typeBytes)
    {
        // Simplified: extract shape dimensions from tensor type
        var shape = new List<int>();

        using var stream = new MemoryStream(typeBytes);
        using var reader = new CodedInputStream(stream);

        while (!reader.IsAtEnd)
        {
            var tag = reader.ReadTag();
            if (WireFormat.GetTagWireType(tag) == WireFormat.WireType.Varint)
            {
                var dim = (int)reader.ReadInt64();
                if (dim > 0) shape.Add(dim);
            }
            else
            {
                reader.SkipLastField();
            }
        }

        return shape.ToArray();
    }

    private static float[] BytesToFloatArray(byte[] bytes)
    {
        var floats = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, floats, 0, bytes.Length);
        return floats;
    }

    private static CoreMLModelDescription CreateModelDescription(OnnxGraphInfo onnxGraph, CoreMLConfiguration config)
    {
        var description = new CoreMLModelDescription
        {
            Metadata = new CoreMLMetadata
            {
                Description = config.ModelDescription ?? "Converted from ONNX",
                Author = config.ModelAuthor ?? "AiDotNet",
                License = config.ModelLicense ?? "",
                Version = "1.0"
            }
        };

        // Map inputs
        foreach (var input in onnxGraph.Inputs)
        {
            description.Inputs.Add(new CoreMLFeature
            {
                Name = input.Name,
                Shape = input.Shape,
                DataType = config.QuantizationBits == 16
                    ? CoreMLProto.ArrayDataType.Float16
                    : CoreMLProto.ArrayDataType.Float32
            });
        }

        // Map outputs
        foreach (var output in onnxGraph.Outputs)
        {
            description.Outputs.Add(new CoreMLFeature
            {
                Name = output.Name,
                Shape = output.Shape,
                DataType = CoreMLProto.ArrayDataType.Float32
            });
        }

        return description;
    }

    private static CoreMLNeuralNetwork ConvertNeuralNetwork(OnnxGraphInfo onnxGraph, CoreMLConfiguration config)
    {
        var network = new CoreMLNeuralNetwork();

        // Add preprocessing for inputs
        foreach (var input in onnxGraph.Inputs)
        {
            network.Preprocessing.Add(input.Name);
        }

        // Convert operators
        var layerIndex = 0;
        foreach (var op in onnxGraph.Operations)
        {
            var layer = ConvertOperatorToLayer(op, onnxGraph.Initializers, layerIndex++);
            if (layer != null)
            {
                network.Layers.Add(layer);
            }
        }

        return network;
    }

    private static CoreMLLayer? ConvertOperatorToLayer(OnnxNode op, Dictionary<string, float[]> initializers, int layerIndex)
    {
        var layer = new CoreMLLayer
        {
            Name = string.IsNullOrEmpty(op.Name) ? $"layer_{layerIndex}" : op.Name,
            Inputs = new List<string>(op.Inputs),
            Outputs = new List<string>(op.Outputs)
        };

        switch (op.OpType)
        {
            case "MatMul":
            case "Gemm":
                // Map to InnerProduct (fully connected layer)
                layer.Type = "InnerProduct";

                // Extract weights from initializers
                var weightsKey = op.Inputs.Count > 1 ? op.Inputs[1] : null;
                if (weightsKey != null && initializers.TryGetValue(weightsKey, out var weights))
                {
                    layer.Weights = weights;
                    layer.InputSize = weights.Length / (weights.Length > 0 ? (int)Math.Sqrt(weights.Length) : 1);
                    layer.OutputSize = (int)Math.Sqrt(weights.Length);
                }

                // Extract bias if present
                var biasKey = op.Inputs.Count > 2 ? op.Inputs[2] : null;
                if (biasKey != null && initializers.TryGetValue(biasKey, out var bias))
                {
                    layer.Bias = bias;
                    layer.HasBias = true;
                }
                break;

            case "Relu":
                layer.Type = "Activation";
                break;

            case "Add":
                layer.Type = "Add";
                break;

            case "Identity":
                // Skip identity layers
                return null;

            default:
                throw new NotSupportedException(
                    $"ONNX operator '{op.OpType}' is not yet supported in ONNX→CoreML conversion. " +
                    $"Supported operators: MatMul, Gemm, Relu, Add. " +
                    $"For complex models, consider using ONNX Runtime CoreML execution provider or third-party conversion tools.");
        }

        return layer;
    }
}

/// <summary>
/// ONNX graph information extracted from protobuf.
/// </summary>
internal class OnnxGraphInfo
{
    public string Name { get; set; } = string.Empty;
    public List<OnnxNode> Operations { get; set; } = new();
    public Dictionary<string, float[]> Initializers { get; set; } = new();
    public List<OnnxValueInfo> Inputs { get; set; } = new();
    public List<OnnxValueInfo> Outputs { get; set; } = new();
}

/// <summary>
/// ONNX node (operator) information.
/// </summary>
internal class OnnxNode
{
    public string Name { get; set; } = string.Empty;
    public string OpType { get; set; } = string.Empty;
    public List<string> Inputs { get; set; } = new();
    public List<string> Outputs { get; set; } = new();
}

/// <summary>
/// ONNX value information (inputs/outputs).
/// </summary>
internal class OnnxValueInfo
{
    public string Name { get; set; } = string.Empty;
    public int[] Shape { get; set; } = Array.Empty<int>();
}
