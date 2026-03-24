using System.Collections.Generic;
using Google.Protobuf;

namespace AiDotNet.Deployment.Mobile.CoreML;

/// <summary>
/// CoreML protobuf message serialization for creating .mlmodel files.
/// Based on Apple's CoreML spec: https://apple.github.io/coremltools/mlmodel/Format/Model.html
/// </summary>
internal static class CoreMLProto
{
    public enum FeatureType
    {
        Double = 1,
        Int64 = 2,
        String = 3,
        Image = 4,
        MultiArray = 5,
        Dictionary = 6,
        Sequence = 7
    }

    public enum ArrayDataType
    {
        Float32 = 65568,
        Double = 65600,
        Int32 = 131104,
        Float16 = 65552
    }

    /// <summary>
    /// Creates a CoreML Model protobuf from an ONNX graph.
    /// </summary>
    public static byte[] CreateModelProto(CoreMLModel model)
    {
        using var stream = new MemoryStream();
        using var writer = new CodedOutputStream(stream);

        // Field 1: specificationVersion (required)
        writer.WriteTag(1, WireFormat.WireType.Varint);
        writer.WriteInt32(5); // CoreML spec version 5 (iOS 14+)

        // Field 2: description
        writer.WriteTag(2, WireFormat.WireType.LengthDelimited);
        var descBytes = CreateModelDescription(model.Description);
        writer.WriteBytes(ByteString.CopyFrom(descBytes));

        // Field 200: neuralNetwork (oneof Type)
        writer.WriteTag(200, WireFormat.WireType.LengthDelimited);
        var nnBytes = CreateNeuralNetwork(model.Network);
        writer.WriteBytes(ByteString.CopyFrom(nnBytes));

        writer.Flush();
        return stream.ToArray();
    }

    private static byte[] CreateModelDescription(CoreMLModelDescription desc)
    {
        using var stream = new MemoryStream();
        using var writer = new CodedOutputStream(stream);

        // Field 1: input (repeated)
        foreach (var input in desc.Inputs)
        {
            writer.WriteTag(1, WireFormat.WireType.LengthDelimited);
            writer.WriteBytes(ByteString.CopyFrom(CreateFeatureDescription(input)));
        }

        // Field 10: output (repeated)
        foreach (var output in desc.Outputs)
        {
            writer.WriteTag(10, WireFormat.WireType.LengthDelimited);
            writer.WriteBytes(ByteString.CopyFrom(CreateFeatureDescription(output)));
        }

        // Field 100: metadata
        if (!string.IsNullOrEmpty(desc.Metadata.Author))
        {
            writer.WriteTag(100, WireFormat.WireType.LengthDelimited);
            writer.WriteBytes(ByteString.CopyFrom(CreateMetadata(desc.Metadata)));
        }

        writer.Flush();
        return stream.ToArray();
    }

    private static byte[] CreateFeatureDescription(CoreMLFeature feature)
    {
        using var stream = new MemoryStream();
        using var writer = new CodedOutputStream(stream);

        // Field 1: name
        writer.WriteTag(1, WireFormat.WireType.LengthDelimited);
        writer.WriteString(feature.Name);

        // Field 2: shortDescription
        if (!string.IsNullOrEmpty(feature.Description))
        {
            writer.WriteTag(2, WireFormat.WireType.LengthDelimited);
            writer.WriteString(feature.Description);
        }

        // Field 3: type
        writer.WriteTag(3, WireFormat.WireType.LengthDelimited);
        writer.WriteBytes(ByteString.CopyFrom(CreateFeatureType(feature)));

        writer.Flush();
        return stream.ToArray();
    }

    private static byte[] CreateFeatureType(CoreMLFeature feature)
    {
        using var stream = new MemoryStream();
        using var writer = new CodedOutputStream(stream);

        // Field 5: multiArrayType (for tensors)
        writer.WriteTag(5, WireFormat.WireType.LengthDelimited);
        writer.WriteBytes(ByteString.CopyFrom(CreateMultiArrayType(feature.Shape, feature.DataType)));

        writer.Flush();
        return stream.ToArray();
    }

    private static byte[] CreateMultiArrayType(int[] shape, ArrayDataType dataType)
    {
        using var stream = new MemoryStream();
        using var writer = new CodedOutputStream(stream);

        // Field 1: shape (repeated)
        foreach (var dim in shape)
        {
            writer.WriteTag(1, WireFormat.WireType.Varint);
            writer.WriteInt64(dim);
        }

        // Field 2: dataType
        writer.WriteTag(2, WireFormat.WireType.Varint);
        writer.WriteInt32((int)dataType);

        writer.Flush();
        return stream.ToArray();
    }

    private static byte[] CreateMetadata(CoreMLMetadata metadata)
    {
        using var stream = new MemoryStream();
        using var writer = new CodedOutputStream(stream);

        // Field 1: shortDescription
        if (!string.IsNullOrEmpty(metadata.Description))
        {
            writer.WriteTag(1, WireFormat.WireType.LengthDelimited);
            writer.WriteString(metadata.Description);
        }

        // Field 3: author
        if (!string.IsNullOrEmpty(metadata.Author))
        {
            writer.WriteTag(3, WireFormat.WireType.LengthDelimited);
            writer.WriteString(metadata.Author);
        }

        // Field 4: license
        if (!string.IsNullOrEmpty(metadata.License))
        {
            writer.WriteTag(4, WireFormat.WireType.LengthDelimited);
            writer.WriteString(metadata.License);
        }

        // Field 6: versionString
        if (!string.IsNullOrEmpty(metadata.Version))
        {
            writer.WriteTag(6, WireFormat.WireType.LengthDelimited);
            writer.WriteString(metadata.Version);
        }

        writer.Flush();
        return stream.ToArray();
    }

    private static byte[] CreateNeuralNetwork(CoreMLNeuralNetwork network)
    {
        using var stream = new MemoryStream();
        using var writer = new CodedOutputStream(stream);

        // Field 1: layers (repeated)
        foreach (var layer in network.Layers)
        {
            writer.WriteTag(1, WireFormat.WireType.LengthDelimited);
            writer.WriteBytes(ByteString.CopyFrom(CreateNeuralNetworkLayer(layer)));
        }

        // Field 10: preprocessing (repeated)
        foreach (var input in network.Preprocessing)
        {
            writer.WriteTag(10, WireFormat.WireType.LengthDelimited);
            writer.WriteBytes(ByteString.CopyFrom(CreatePreprocessing(input)));
        }

        writer.Flush();
        return stream.ToArray();
    }

    private static byte[] CreateNeuralNetworkLayer(CoreMLLayer layer)
    {
        using var stream = new MemoryStream();
        using var writer = new CodedOutputStream(stream);

        // Field 1: name
        writer.WriteTag(1, WireFormat.WireType.LengthDelimited);
        writer.WriteString(layer.Name);

        // Field 2: input (repeated)
        foreach (var input in layer.Inputs)
        {
            writer.WriteTag(2, WireFormat.WireType.LengthDelimited);
            writer.WriteString(input);
        }

        // Field 3: output (repeated)
        foreach (var output in layer.Outputs)
        {
            writer.WriteTag(3, WireFormat.WireType.LengthDelimited);
            writer.WriteString(output);
        }

        // Field 100+: layer type (oneof)
        writer.WriteBytes(ByteString.CopyFrom(CreateLayerParams(layer)));

        writer.Flush();
        return stream.ToArray();
    }

    private static byte[] CreateLayerParams(CoreMLLayer layer)
    {
        using var stream = new MemoryStream();
        using var writer = new CodedOutputStream(stream);

        switch (layer.Type)
        {
            case "InnerProduct":
                writer.WriteTag(100, WireFormat.WireType.LengthDelimited);
                writer.WriteBytes(ByteString.CopyFrom(CreateInnerProductLayer(layer)));
                break;
            case "Activation":
                writer.WriteTag(105, WireFormat.WireType.LengthDelimited);
                writer.WriteBytes(ByteString.CopyFrom(CreateActivationLayer(layer)));
                break;
            case "Add":
                writer.WriteTag(102, WireFormat.WireType.LengthDelimited);
                writer.WriteBytes(ByteString.CopyFrom(CreateAddLayer()));
                break;
            default:
                throw new NotSupportedException($"Layer type {layer.Type} not supported in CoreML conversion");
        }

        writer.Flush();
        return stream.ToArray();
    }

    private static byte[] CreateInnerProductLayer(CoreMLLayer layer)
    {
        using var stream = new MemoryStream();
        using var writer = new CodedOutputStream(stream);

        // Validate layer sizes before casting to prevent negative values from wrapping to large unsigned values
        if (layer.InputSize < 0 || layer.OutputSize < 0)
            throw new ArgumentException(
                $"Layer '{layer.Name}' has invalid size: InputSize={layer.InputSize}, OutputSize={layer.OutputSize}. " +
                "Both must be non-negative for CoreML protobuf serialization.");

        // Field 1: inputChannels
        writer.WriteTag(1, WireFormat.WireType.Varint);
        writer.WriteUInt64((ulong)layer.InputSize);

        // Field 2: outputChannels
        writer.WriteTag(2, WireFormat.WireType.Varint);
        writer.WriteUInt64((ulong)layer.OutputSize);

        // Field 3: hasBias
        writer.WriteTag(3, WireFormat.WireType.Varint);
        writer.WriteBool(layer.HasBias);

        // Field 4: weights
        if (layer.Weights != null)
        {
            writer.WriteTag(4, WireFormat.WireType.LengthDelimited);
            writer.WriteBytes(ByteString.CopyFrom(CreateWeightParams(layer.Weights)));
        }

        // Field 5: bias
        if (layer.Bias != null)
        {
            writer.WriteTag(5, WireFormat.WireType.LengthDelimited);
            writer.WriteBytes(ByteString.CopyFrom(CreateWeightParams(layer.Bias)));
        }

        writer.Flush();
        return stream.ToArray();
    }

    private static byte[] CreateActivationLayer(CoreMLLayer layer)
    {
        using var stream = new MemoryStream();
        using var writer = new CodedOutputStream(stream);

        // Field corresponding to activation type (e.g., ReLU = 10)
        writer.WriteTag(10, WireFormat.WireType.LengthDelimited);
        // ReLU has no parameters, write empty message
        writer.WriteBytes(ByteString.Empty);

        writer.Flush();
        return stream.ToArray();
    }

    private static byte[] CreateAddLayer()
    {
        using var stream = new MemoryStream();
        using var writer = new CodedOutputStream(stream);

        // Add layer has no parameters
        writer.Flush();
        return stream.ToArray();
    }

    private static byte[] CreateWeightParams(float[] weights)
    {
        using var stream = new MemoryStream();
        using var writer = new CodedOutputStream(stream);

        // Field 1: floatValue (repeated packed)
        writer.WriteTag(1, WireFormat.WireType.LengthDelimited);
        using (var dataStream = new MemoryStream())
        using (var dataWriter = new BinaryWriter(dataStream))
        {
            foreach (var weight in weights)
            {
                dataWriter.Write(weight);
            }
            writer.WriteBytes(ByteString.CopyFrom(dataStream.ToArray()));
        }

        writer.Flush();
        return stream.ToArray();
    }

    private static byte[] CreatePreprocessing(string inputName)
    {
        using var stream = new MemoryStream();
        using var writer = new CodedOutputStream(stream);

        // Minimal preprocessing - just pass through
        writer.Flush();
        return stream.ToArray();
    }
}

/// <summary>
/// Represents a CoreML model structure.
/// </summary>
internal class CoreMLModel
{
    public CoreMLModelDescription Description { get; set; } = new();
    public CoreMLNeuralNetwork Network { get; set; } = new();
}

/// <summary>
/// Model description with inputs, outputs, and metadata.
/// </summary>
internal class CoreMLModelDescription
{
    public List<CoreMLFeature> Inputs { get; set; } = new();
    public List<CoreMLFeature> Outputs { get; set; } = new();
    public CoreMLMetadata Metadata { get; set; } = new();
}

/// <summary>
/// Feature description for inputs/outputs.
/// </summary>
internal class CoreMLFeature
{
    public string Name { get; set; } = string.Empty;
    public string Description { get; set; } = string.Empty;
    public int[] Shape { get; set; } = Array.Empty<int>();
    public CoreMLProto.ArrayDataType DataType { get; set; } = CoreMLProto.ArrayDataType.Float32;
}

/// <summary>
/// Model metadata.
/// </summary>
internal class CoreMLMetadata
{
    public string Description { get; set; } = string.Empty;
    public string Author { get; set; } = string.Empty;
    public string License { get; set; } = string.Empty;
    public string Version { get; set; } = string.Empty;
}

/// <summary>
/// Neural network structure.
/// </summary>
internal class CoreMLNeuralNetwork
{
    public List<CoreMLLayer> Layers { get; set; } = new();
    public List<string> Preprocessing { get; set; } = new();
}

/// <summary>
/// Neural network layer.
/// </summary>
internal class CoreMLLayer
{
    public string Name { get; set; } = string.Empty;
    public string Type { get; set; } = string.Empty;
    public List<string> Inputs { get; set; } = new();
    public List<string> Outputs { get; set; } = new();
    public int InputSize { get; set; }
    public int OutputSize { get; set; }
    public bool HasBias { get; set; }
    public float[]? Weights { get; set; }
    public float[]? Bias { get; set; }
}
