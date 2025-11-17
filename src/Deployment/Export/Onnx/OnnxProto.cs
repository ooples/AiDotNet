using Google.Protobuf;
using Google.Protobuf.Collections;

namespace AiDotNet.Deployment.Export.Onnx;

/// <summary>
/// ONNX protobuf message definitions for model serialization.
/// Based on ONNX IR spec: https://github.com/onnx/onnx/blob/main/onnx/onnx.proto
/// </summary>
internal static class OnnxProto
{
    public enum DataType
    {
        UNDEFINED = 0,
        FLOAT = 1,
        UINT8 = 2,
        INT8 = 3,
        UINT16 = 4,
        INT16 = 5,
        INT32 = 6,
        INT64 = 7,
        STRING = 8,
        BOOL = 9,
        FLOAT16 = 10,
        DOUBLE = 11,
        UINT32 = 12,
        UINT64 = 13,
        COMPLEX64 = 14,
        COMPLEX128 = 15
    }

    /// <summary>
    /// Creates an ONNX ModelProto structure
    /// </summary>
    public static byte[] CreateModelProto(OnnxGraph graph, ExportConfiguration config)
    {
        using var stream = new MemoryStream();
        using var writer = new CodedOutputStream(stream);

        // Field 1: ir_version (required)
        writer.WriteTag(1, WireFormat.WireType.Varint);
        writer.WriteInt64(8); // ONNX IR version 8

        // Field 2: opset_import (required)
        writer.WriteTag(2, WireFormat.WireType.LengthDelimited);
        var opsetBytes = CreateOpsetImport(graph.OpsetVersion);
        writer.WriteBytes(ByteString.CopyFrom(opsetBytes));

        // Field 3: producer_name
        writer.WriteTag(3, WireFormat.WireType.LengthDelimited);
        writer.WriteString("AiDotNet");

        // Field 4: producer_version
        writer.WriteTag(4, WireFormat.WireType.LengthDelimited);
        writer.WriteString("1.0");

        // Field 7: graph (required)
        writer.WriteTag(7, WireFormat.WireType.LengthDelimited);
        var graphBytes = CreateGraphProto(graph, config);
        writer.WriteBytes(ByteString.CopyFrom(graphBytes));

        // Field 8: model_version
        if (!string.IsNullOrEmpty(config.ModelVersion))
        {
            writer.WriteTag(8, WireFormat.WireType.Varint);
            if (long.TryParse(config.ModelVersion, out var version))
                writer.WriteInt64(version);
            else
                writer.WriteInt64(1);
        }

        writer.Flush();
        return stream.ToArray();
    }

    private static byte[] CreateOpsetImport(int opsetVersion)
    {
        using var stream = new MemoryStream();
        using var writer = new CodedOutputStream(stream);

        // Field 1: domain (empty for default domain)
        writer.WriteTag(1, WireFormat.WireType.LengthDelimited);
        writer.WriteString("");

        // Field 2: version
        writer.WriteTag(2, WireFormat.WireType.Varint);
        writer.WriteInt64(opsetVersion);

        writer.Flush();
        return stream.ToArray();
    }

    private static byte[] CreateGraphProto(OnnxGraph graph, ExportConfiguration config)
    {
        using var stream = new MemoryStream();
        using var writer = new CodedOutputStream(stream);

        // Field 1: node (repeated)
        foreach (var operation in graph.Operations)
        {
            writer.WriteTag(1, WireFormat.WireType.LengthDelimited);
            var nodeBytes = CreateNodeProto(operation);
            writer.WriteBytes(ByteString.CopyFrom(nodeBytes));
        }

        // Field 2: name
        writer.WriteTag(2, WireFormat.WireType.LengthDelimited);
        writer.WriteString(graph.Name);

        // Field 5: initializer (repeated) - for weights
        foreach (var kvp in graph.Initializers)
        {
            var name = kvp.Key;
            var data = kvp.Value;
            writer.WriteTag(5, WireFormat.WireType.LengthDelimited);
            var tensorBytes = CreateTensorProto(name, data);
            writer.WriteBytes(ByteString.CopyFrom(tensorBytes));
        }

        // Field 11: input (repeated)
        foreach (var input in graph.Inputs)
        {
            writer.WriteTag(11, WireFormat.WireType.LengthDelimited);
            var valueInfoBytes = CreateValueInfoProto(input);
            writer.WriteBytes(ByteString.CopyFrom(valueInfoBytes));
        }

        // Field 12: output (repeated)
        foreach (var output in graph.Outputs)
        {
            writer.WriteTag(12, WireFormat.WireType.LengthDelimited);
            var valueInfoBytes = CreateValueInfoProto(output);
            writer.WriteBytes(ByteString.CopyFrom(valueInfoBytes));
        }

        writer.Flush();
        return stream.ToArray();
    }

    private static byte[] CreateNodeProto(OnnxOperation operation)
    {
        using var stream = new MemoryStream();
        using var writer = new CodedOutputStream(stream);

        // Field 1: input (repeated)
        foreach (var input in operation.Inputs)
        {
            writer.WriteTag(1, WireFormat.WireType.LengthDelimited);
            writer.WriteString(input);
        }

        // Field 2: output (repeated)
        foreach (var output in operation.Outputs)
        {
            writer.WriteTag(2, WireFormat.WireType.LengthDelimited);
            writer.WriteString(output);
        }

        // Field 3: name (optional)
        if (!string.IsNullOrEmpty(operation.Name))
        {
            writer.WriteTag(3, WireFormat.WireType.LengthDelimited);
            writer.WriteString(operation.Name);
        }

        // Field 4: op_type (required)
        writer.WriteTag(4, WireFormat.WireType.LengthDelimited);
        writer.WriteString(operation.Type);

        // Field 5: domain (optional)
        if (!string.IsNullOrEmpty(operation.Domain) && operation.Domain != "ai.onnx")
        {
            writer.WriteTag(5, WireFormat.WireType.LengthDelimited);
            writer.WriteString(operation.Domain);
        }

        // Field 6: attribute (repeated)
        foreach (var kvp in operation.Attributes)
        {
            var name = kvp.Key;
            var value = kvp.Value;
            writer.WriteTag(6, WireFormat.WireType.LengthDelimited);
            var attrBytes = CreateAttributeProto(name, value);
            writer.WriteBytes(ByteString.CopyFrom(attrBytes));
        }

        writer.Flush();
        return stream.ToArray();
    }

    private static byte[] CreateAttributeProto(string name, object value)
    {
        using var stream = new MemoryStream();
        using var writer = new CodedOutputStream(stream);

        // Field 1: name (required)
        writer.WriteTag(1, WireFormat.WireType.LengthDelimited);
        writer.WriteString(name);

        // Determine type and write value
        switch (value)
        {
            case int intValue:
                writer.WriteTag(20, WireFormat.WireType.Varint); // type = INT
                writer.WriteInt32(2);
                writer.WriteTag(3, WireFormat.WireType.Varint); // i
                writer.WriteInt64(intValue);
                break;
            case long longValue:
                writer.WriteTag(20, WireFormat.WireType.Varint); // type = INT
                writer.WriteInt32(2);
                writer.WriteTag(3, WireFormat.WireType.Varint); // i
                writer.WriteInt64(longValue);
                break;
            case float floatValue:
                writer.WriteTag(20, WireFormat.WireType.Varint); // type = FLOAT
                writer.WriteInt32(1);
                writer.WriteTag(2, WireFormat.WireType.Fixed32); // f
                writer.WriteFloat(floatValue);
                break;
            case int[] intArray:
                writer.WriteTag(20, WireFormat.WireType.Varint); // type = INTS
                writer.WriteInt32(7);
                foreach (var i in intArray)
                {
                    writer.WriteTag(8, WireFormat.WireType.Varint); // ints
                    writer.WriteInt64(i);
                }
                break;
            case string strValue:
                writer.WriteTag(20, WireFormat.WireType.Varint); // type = STRING
                writer.WriteInt32(3);
                writer.WriteTag(4, WireFormat.WireType.LengthDelimited); // s
                writer.WriteBytes(ByteString.CopyFromUtf8(strValue));
                break;
        }

        writer.Flush();
        return stream.ToArray();
    }

    private static byte[] CreateValueInfoProto(OnnxNode node)
    {
        using var stream = new MemoryStream();
        using var writer = new CodedOutputStream(stream);

        // Field 1: name
        writer.WriteTag(1, WireFormat.WireType.LengthDelimited);
        writer.WriteString(node.Name);

        // Field 2: type
        writer.WriteTag(2, WireFormat.WireType.LengthDelimited);
        var typeBytes = CreateTypeProto(node);
        writer.WriteBytes(ByteString.CopyFrom(typeBytes));

        writer.Flush();
        return stream.ToArray();
    }

    private static byte[] CreateTypeProto(OnnxNode node)
    {
        using var stream = new MemoryStream();
        using var writer = new CodedOutputStream(stream);

        // Field 1: tensor_type
        writer.WriteTag(1, WireFormat.WireType.LengthDelimited);
        var tensorTypeBytes = CreateTensorTypeProto(node);
        writer.WriteBytes(ByteString.CopyFrom(tensorTypeBytes));

        writer.Flush();
        return stream.ToArray();
    }

    private static byte[] CreateTensorTypeProto(OnnxNode node)
    {
        using var stream = new MemoryStream();
        using var writer = new CodedOutputStream(stream);

        // Field 1: elem_type
        writer.WriteTag(1, WireFormat.WireType.Varint);
        writer.WriteInt32(GetDataTypeValue(node.DataType));

        // Field 2: shape
        if (node.Shape != null && node.Shape.Length > 0)
        {
            writer.WriteTag(2, WireFormat.WireType.LengthDelimited);
            var shapeBytes = CreateTensorShapeProto(node.Shape);
            writer.WriteBytes(ByteString.CopyFrom(shapeBytes));
        }

        writer.Flush();
        return stream.ToArray();
    }

    private static byte[] CreateTensorShapeProto(int[] shape)
    {
        using var stream = new MemoryStream();
        using var writer = new CodedOutputStream(stream);

        foreach (var dim in shape)
        {
            writer.WriteTag(1, WireFormat.WireType.LengthDelimited); // dim (repeated)
            var dimBytes = CreateDimensionProto(dim);
            writer.WriteBytes(ByteString.CopyFrom(dimBytes));
        }

        writer.Flush();
        return stream.ToArray();
    }

    private static byte[] CreateDimensionProto(int dimValue)
    {
        using var stream = new MemoryStream();
        using var writer = new CodedOutputStream(stream);

        // Field 1: dim_value
        writer.WriteTag(1, WireFormat.WireType.Varint);
        writer.WriteInt64(dimValue);

        writer.Flush();
        return stream.ToArray();
    }

    private static byte[] CreateTensorProto(string name, object data)
    {
        // Runtime type dispatch for object initializers
        return data switch
        {
            Vector<float> floatVec => CreateTensorProto(name, floatVec),
            Vector<double> doubleVec => CreateTensorProto(name, doubleVec),
            Vector<int> intVec => CreateTensorProto(name, intVec),
            Vector<long> longVec => CreateTensorProto(name, longVec),
            Vector<sbyte> sbyteVec => CreateTensorProto(name, sbyteVec),
            Vector<short> shortVec => CreateTensorProto(name, shortVec),
            Vector<byte> byteVec => CreateTensorProto(name, byteVec),
            Vector<ushort> ushortVec => CreateTensorProto(name, ushortVec),
            Vector<uint> uintVec => CreateTensorProto(name, uintVec),
            Vector<ulong> ulongVec => CreateTensorProto(name, ulongVec),
            _ => throw new NotSupportedException($"Unsupported tensor data type: {data.GetType().Name}")
        };
    }

    private static byte[] CreateTensorProto<T>(string name, Vector<T> data)
    {
        using var stream = new MemoryStream();
        using var writer = new CodedOutputStream(stream);

        // Field 1: dims (repeated)
        writer.WriteTag(1, WireFormat.WireType.Varint);
        writer.WriteInt64(data.Length);

        // Field 2: data_type
        writer.WriteTag(2, WireFormat.WireType.Varint);
        writer.WriteInt32(GetDataTypeValue(typeof(T)));

        // Field 8: name (per ONNX TensorProto specification)
        writer.WriteTag(8, WireFormat.WireType.LengthDelimited);
        writer.WriteString(name);

        // Field 9: raw_data
        writer.WriteTag(9, WireFormat.WireType.LengthDelimited);
        var rawBytes = VectorToBytes(data);
        writer.WriteBytes(ByteString.CopyFrom(rawBytes));

        writer.Flush();
        return stream.ToArray();
    }

    private static int GetDataTypeValue(string dataTypeName)
    {
        return dataTypeName.ToLower() switch
        {
            "float" => (int)DataType.FLOAT,
            "double" => (int)DataType.DOUBLE,
            "int8" => (int)DataType.INT8,
            "int16" => (int)DataType.INT16,
            "int32" => (int)DataType.INT32,
            "int64" => (int)DataType.INT64,
            "uint8" => (int)DataType.UINT8,
            "uint16" => (int)DataType.UINT16,
            "uint32" => (int)DataType.UINT32,
            "uint64" => (int)DataType.UINT64,
            "bool" => (int)DataType.BOOL,
            _ => (int)DataType.FLOAT
        };
    }

    private static int GetDataTypeValue(Type type)
    {
        if (type == typeof(float)) return (int)DataType.FLOAT;
        if (type == typeof(double)) return (int)DataType.DOUBLE;
        if (type == typeof(sbyte)) return (int)DataType.INT8;
        if (type == typeof(short)) return (int)DataType.INT16;
        if (type == typeof(int)) return (int)DataType.INT32;
        if (type == typeof(long)) return (int)DataType.INT64;
        if (type == typeof(byte)) return (int)DataType.UINT8;
        if (type == typeof(ushort)) return (int)DataType.UINT16;
        if (type == typeof(uint)) return (int)DataType.UINT32;
        if (type == typeof(ulong)) return (int)DataType.UINT64;
        if (type == typeof(bool)) return (int)DataType.BOOL;
        return (int)DataType.FLOAT;
    }

    private static byte[] VectorToBytes<T>(Vector<T> vector)
    {
        var elementSize = System.Runtime.InteropServices.Marshal.SizeOf<T>();
        var bytes = new byte[vector.Length * elementSize];
        Buffer.BlockCopy(vector.ToArray(), 0, bytes, 0, bytes.Length);
        return bytes;
    }
}
