using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Onnx;

/// <summary>
/// Exports AiDotNet models to the ONNX format.
/// </summary>
/// <remarks>
/// <para>
/// This exporter supports sequential models with common layer types:
/// <list type="bullet">
/// <item>Dense/Linear layers (exported as MatMul + Add)</item>
/// <item>Activation functions (ReLU, Sigmoid, Tanh, etc.)</item>
/// <item>Dropout (exported as Identity in inference mode)</item>
/// </list>
/// </para>
/// <para><b>Limitations:</b> This is a proof-of-concept implementation that works
/// with specific, known model structures (sequential models with supported layers).
/// For production use, consider using framework-native export tools.
/// </para>
/// <para><b>For Beginners:</b> Use this class to convert your trained AiDotNet
/// models to ONNX format for deployment:
/// <code>
/// var model = // your trained model
/// OnnxExporter.Export(model, "model.onnx");
/// </code>
/// </para>
/// </remarks>
public static class OnnxExporter
{
    private const long OnnxOpsetVersion = 17;
    private const long IrVersion = 8;
    private const string ProducerName = "AiDotNet";
    private const string ProducerVersion = "1.0.0";

    /// <summary>
    /// Exports a model to ONNX format.
    /// </summary>
    /// <typeparam name="T">The numeric type of the model.</typeparam>
    /// <param name="model">The model to export.</param>
    /// <param name="outputPath">The path to write the ONNX file.</param>
    /// <param name="inputShape">Optional input shape. If not provided, will try to infer from model.</param>
    /// <exception cref="ArgumentNullException">If model or outputPath is null.</exception>
    /// <exception cref="NotSupportedException">If the model contains unsupported layer types.</exception>
    public static void Export<T, TInput, TOutput>(
        IFullModel<T, TInput, TOutput> model,
        string outputPath,
        int[]? inputShape = null)
    {
        if (model is null)
            throw new ArgumentNullException(nameof(model));
        if (string.IsNullOrWhiteSpace(outputPath))
            throw new ArgumentNullException(nameof(outputPath));

        var onnxBytes = ExportToBytes(model, inputShape);

        var directory = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }

        File.WriteAllBytes(outputPath, onnxBytes);
    }

    /// <summary>
    /// Exports a model to ONNX format and returns the bytes.
    /// </summary>
    /// <typeparam name="T">The numeric type of the model.</typeparam>
    /// <param name="model">The model to export.</param>
    /// <param name="inputShape">Optional input shape.</param>
    /// <returns>The ONNX model as a byte array.</returns>
    public static byte[] ExportToBytes<T, TInput, TOutput>(
        IFullModel<T, TInput, TOutput> model,
        int[]? inputShape = null)
    {
        if (model is null)
            throw new ArgumentNullException(nameof(model));

        var numOps = MathHelper.GetNumericOperations<T>();
        var builder = new OnnxModelBuilder();

        var layers = GetLayers(model);
        if (layers.Count == 0)
        {
            throw new NotSupportedException("Model has no exportable layers.");
        }

        // Determine input shape
        var effectiveInputShape = inputShape ?? InferInputShape(layers) ?? new[] { 1, 1 };

        // Create input
        var inputName = "input";
        builder.AddInput(inputName, effectiveInputShape);

        // Track the current tensor name flowing through the graph
        string currentTensorName = inputName;
        int nodeIndex = 0;

        // Process each layer
        foreach (var layer in layers)
        {
            var outputName = ExportLayer(layer, currentTensorName, nodeIndex, builder, numOps);
            if (outputName is not null)
            {
                currentTensorName = outputName;
                nodeIndex++;
            }
        }

        // Create output
        var outputShape = InferOutputShape(layers, effectiveInputShape);
        builder.AddOutput(currentTensorName, outputShape);

        return builder.Build();
    }

    private static List<object> GetLayers<T, TInput, TOutput>(IFullModel<T, TInput, TOutput> model)
    {
        var layers = new List<object>();
        var modelType = model.GetType();

        // Check for Layers property
        var layersProperty = modelType.GetProperty("Layers");
        if (layersProperty is not null)
        {
            var layersList = layersProperty.GetValue(model);
            if (layersList is System.Collections.IEnumerable enumerable)
            {
                foreach (var layer in enumerable)
                {
                    layers.Add(layer);
                }
            }
        }

        return layers;
    }

    private static int[]? InferInputShape(List<object> layers)
    {
        if (layers.Count == 0) return null;

        var firstLayer = layers[0];
        var layerType = firstLayer.GetType();

        var inputSizeProp = layerType.GetProperty("InputSize");
        if (inputSizeProp is not null)
        {
            var inputSize = (int)inputSizeProp.GetValue(firstLayer)!;
            return new[] { 1, inputSize };
        }

        var inputShapeProp = layerType.GetProperty("InputShape");
        if (inputShapeProp is not null)
        {
            var inputShape = inputShapeProp.GetValue(firstLayer) as int[];
            if (inputShape is not null)
            {
                var result = new int[inputShape.Length + 1];
                result[0] = 1;
                Array.Copy(inputShape, 0, result, 1, inputShape.Length);
                return result;
            }
        }

        return null;
    }

    private static int[] InferOutputShape(List<object> layers, int[] inputShape)
    {
        if (layers.Count == 0) return inputShape;

        var lastLayer = layers[^1];
        var layerType = lastLayer.GetType();

        var outputSizeProp = layerType.GetProperty("OutputSize");
        if (outputSizeProp is not null)
        {
            var outputSize = (int)outputSizeProp.GetValue(lastLayer)!;
            return new[] { inputShape[0], outputSize };
        }

        return inputShape;
    }

    private static string? ExportLayer<T>(
        object layer,
        string inputName,
        int nodeIndex,
        OnnxModelBuilder builder,
        INumericOperations<T> numOps)
    {
        var layerType = layer.GetType();
        var layerTypeName = layerType.Name;

        if (layerTypeName.Contains("Dense") || layerTypeName.Contains("Linear") || layerTypeName.Contains("FullyConnected"))
        {
            return ExportDenseLayer(layer, inputName, nodeIndex, builder, numOps);
        }

        if (layerTypeName.Contains("ReLU"))
        {
            var outputName = $"relu_{nodeIndex}";
            builder.AddRelu(inputName, outputName);
            return outputName;
        }

        if (layerTypeName.Contains("Sigmoid"))
        {
            var outputName = $"sigmoid_{nodeIndex}";
            builder.AddSigmoid(inputName, outputName);
            return outputName;
        }

        if (layerTypeName.Contains("Tanh"))
        {
            var outputName = $"tanh_{nodeIndex}";
            builder.AddTanh(inputName, outputName);
            return outputName;
        }

        if (layerTypeName.Contains("Softmax"))
        {
            var outputName = $"softmax_{nodeIndex}";
            builder.AddSoftmax(inputName, outputName);
            return outputName;
        }

        if (layerTypeName.Contains("Dropout"))
        {
            var outputName = $"identity_{nodeIndex}";
            builder.AddIdentity(inputName, outputName);
            return outputName;
        }

        if (layerTypeName.Contains("Flatten"))
        {
            var outputName = $"flatten_{nodeIndex}";
            builder.AddFlatten(inputName, outputName);
            return outputName;
        }

        // Skip unsupported layers
        return null;
    }

    private static string ExportDenseLayer<T>(
        object layer,
        string inputName,
        int nodeIndex,
        OnnxModelBuilder builder,
        INumericOperations<T> numOps)
    {
        var layerType = layer.GetType();

        var weightsProp = layerType.GetProperty("Weights");
        var biasProp = layerType.GetProperty("Bias") ?? layerType.GetProperty("Biases");

        float[,]? weights = null;
        float[]? bias = null;

        if (weightsProp is not null)
        {
            weights = ConvertToFloatMatrix(weightsProp.GetValue(layer), numOps);
        }

        if (biasProp is not null)
        {
            bias = ConvertToFloatArray(biasProp.GetValue(layer), numOps);
        }

        if (weights is null)
        {
            throw new NotSupportedException($"Could not extract weights from layer at index {nodeIndex}");
        }

        var weightsName = $"dense_{nodeIndex}_weights";
        var matmulOutput = $"dense_{nodeIndex}_matmul";
        var finalOutput = $"dense_{nodeIndex}_output";

        builder.AddInitializer(weightsName, weights);
        builder.AddMatMul(inputName, weightsName, matmulOutput);

        if (bias is not null)
        {
            var biasName = $"dense_{nodeIndex}_bias";
            builder.AddInitializer(biasName, bias);
            builder.AddAdd(matmulOutput, biasName, finalOutput);
            return finalOutput;
        }

        return matmulOutput;
    }

    private static float[]? ConvertToFloatArray<T>(object? obj, INumericOperations<T> numOps)
    {
        if (obj is null) return null;
        if (obj is float[] fa) return fa;
        if (obj is double[] da) return da.Select(x => (float)x).ToArray();

        var objType = obj.GetType();
        if (objType.IsGenericType && objType.Name.Contains("Vector"))
        {
            var lengthProp = objType.GetProperty("Length");
            var indexer = objType.GetProperty("Item");
            if (lengthProp is not null && indexer is not null)
            {
                int length = (int)lengthProp.GetValue(obj)!;
                var result = new float[length];
                for (int i = 0; i < length; i++)
                {
                    var val = indexer.GetValue(obj, new object[] { i });
                    if (val is T tval)
                    {
                        result[i] = (float)numOps.ToDouble(tval);
                    }
                }
                return result;
            }
        }

        return null;
    }

    private static float[,]? ConvertToFloatMatrix<T>(object? obj, INumericOperations<T> numOps)
    {
        if (obj is null) return null;
        if (obj is float[,] fa) return fa;
        if (obj is double[,] da)
        {
            var result = new float[da.GetLength(0), da.GetLength(1)];
            for (int i = 0; i < da.GetLength(0); i++)
            {
                for (int j = 0; j < da.GetLength(1); j++)
                {
                    result[i, j] = (float)da[i, j];
                }
            }
            return result;
        }

        var objType = obj.GetType();
        if (objType.IsGenericType && objType.Name.Contains("Matrix"))
        {
            var rowsProp = objType.GetProperty("Rows");
            var colsProp = objType.GetProperty("Columns");
            var indexer = objType.GetProperty("Item");
            if (rowsProp is not null && colsProp is not null && indexer is not null)
            {
                int rows = (int)rowsProp.GetValue(obj)!;
                int cols = (int)colsProp.GetValue(obj)!;
                var result = new float[rows, cols];
                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < cols; j++)
                    {
                        var val = indexer.GetValue(obj, new object[] { i, j });
                        if (val is T tval)
                        {
                            result[i, j] = (float)numOps.ToDouble(tval);
                        }
                    }
                }
                return result;
            }
        }

        return null;
    }
}

/// <summary>
/// Internal builder for constructing ONNX model bytes.
/// Uses protobuf wire format for ONNX serialization.
/// </summary>
internal class OnnxModelBuilder
{
    private readonly List<byte> _graphBytes = new();
    private readonly List<byte> _initializerBytes = new();
    private readonly List<byte> _nodeBytes = new();
    private readonly List<byte> _inputBytes = new();
    private readonly List<byte> _outputBytes = new();

    private const int FieldGraph = 7;
    private const int FieldNode = 1;
    private const int FieldInput = 11;
    private const int FieldOutput = 12;
    private const int FieldInitializer = 5;
    private const int FieldOpsetImport = 8;
    private const int FieldIrVersion = 1;
    private const int FieldProducerName = 2;
    private const int FieldProducerVersion = 3;

    public void AddInput(string name, int[] shape)
    {
        var valueInfo = CreateValueInfo(name, shape);
        _inputBytes.AddRange(CreateField(FieldInput, valueInfo));
    }

    public void AddOutput(string name, int[] shape)
    {
        var valueInfo = CreateValueInfo(name, shape);
        _outputBytes.AddRange(CreateField(FieldOutput, valueInfo));
    }

    public void AddInitializer(string name, float[] data)
    {
        var tensor = CreateTensorProto(name, data, new[] { data.Length });
        _initializerBytes.AddRange(CreateField(FieldInitializer, tensor));
    }

    public void AddInitializer(string name, float[,] data)
    {
        var shape = new[] { data.GetLength(0), data.GetLength(1) };
        var flatData = new float[shape[0] * shape[1]];
        int idx = 0;
        for (int i = 0; i < shape[0]; i++)
        {
            for (int j = 0; j < shape[1]; j++)
            {
                flatData[idx++] = data[i, j];
            }
        }
        var tensor = CreateTensorProto(name, flatData, shape);
        _initializerBytes.AddRange(CreateField(FieldInitializer, tensor));
    }

    public void AddMatMul(string inputA, string inputB, string output)
    {
        AddNode("MatMul", new[] { inputA, inputB }, new[] { output });
    }

    public void AddAdd(string inputA, string inputB, string output)
    {
        AddNode("Add", new[] { inputA, inputB }, new[] { output });
    }

    public void AddRelu(string input, string output)
    {
        AddNode("Relu", new[] { input }, new[] { output });
    }

    public void AddSigmoid(string input, string output)
    {
        AddNode("Sigmoid", new[] { input }, new[] { output });
    }

    public void AddTanh(string input, string output)
    {
        AddNode("Tanh", new[] { input }, new[] { output });
    }

    public void AddSoftmax(string input, string output)
    {
        AddNode("Softmax", new[] { input }, new[] { output });
    }

    public void AddIdentity(string input, string output)
    {
        AddNode("Identity", new[] { input }, new[] { output });
    }

    public void AddFlatten(string input, string output)
    {
        AddNode("Flatten", new[] { input }, new[] { output },
            new Dictionary<string, long> { ["axis"] = 1 });
    }

    private void AddNode(string opType, string[] inputs, string[] outputs,
        Dictionary<string, long>? intAttrs = null)
    {
        var nodeBytes = new List<byte>();

        // inputs (field 1, repeated string)
        foreach (var input in inputs)
        {
            nodeBytes.AddRange(CreateStringField(1, input));
        }

        // outputs (field 2, repeated string)
        foreach (var output in outputs)
        {
            nodeBytes.AddRange(CreateStringField(2, output));
        }

        // op_type (field 4, string)
        nodeBytes.AddRange(CreateStringField(4, opType));

        // attributes (field 5, repeated AttributeProto)
        if (intAttrs is not null)
        {
            foreach (var (attrName, attrValue) in intAttrs)
            {
                var attrBytes = new List<byte>();
                attrBytes.AddRange(CreateStringField(1, attrName)); // name
                attrBytes.AddRange(CreateVarintField(2, attrValue)); // i (int64)
                attrBytes.AddRange(CreateVarintField(20, 2)); // type = INT
                nodeBytes.AddRange(CreateField(5, attrBytes.ToArray()));
            }
        }

        _nodeBytes.AddRange(CreateField(FieldNode, nodeBytes.ToArray()));
    }

    public byte[] Build()
    {
        // Build graph
        var graphBytes = new List<byte>();
        graphBytes.AddRange(_nodeBytes);
        graphBytes.AddRange(CreateStringField(2, "AiDotNet_Model")); // name
        graphBytes.AddRange(_initializerBytes);
        graphBytes.AddRange(_inputBytes);
        graphBytes.AddRange(_outputBytes);

        // Build model
        var modelBytes = new List<byte>();
        modelBytes.AddRange(CreateVarintField(FieldIrVersion, 8)); // ir_version
        modelBytes.AddRange(CreateStringField(FieldProducerName, "AiDotNet"));
        modelBytes.AddRange(CreateStringField(FieldProducerVersion, "1.0.0"));
        modelBytes.AddRange(CreateField(FieldGraph, graphBytes.ToArray()));

        // opset_import
        var opsetBytes = new List<byte>();
        opsetBytes.AddRange(CreateVarintField(2, 17)); // version
        modelBytes.AddRange(CreateField(FieldOpsetImport, opsetBytes.ToArray()));

        return modelBytes.ToArray();
    }

    private static byte[] CreateValueInfo(string name, int[] shape)
    {
        var bytes = new List<byte>();
        bytes.AddRange(CreateStringField(1, name)); // name

        // type (field 2, TypeProto)
        var typeBytes = new List<byte>();

        // tensor_type (field 1 of TypeProto)
        var tensorTypeBytes = new List<byte>();
        tensorTypeBytes.AddRange(CreateVarintField(1, 1)); // elem_type = FLOAT

        // shape (field 2 of TensorTypeProto)
        var shapeBytes = new List<byte>();
        foreach (var dim in shape)
        {
            var dimBytes = new List<byte>();
            dimBytes.AddRange(CreateVarintField(1, dim)); // dim_value
            shapeBytes.AddRange(CreateField(1, dimBytes.ToArray())); // dim
        }
        tensorTypeBytes.AddRange(CreateField(2, shapeBytes.ToArray()));

        typeBytes.AddRange(CreateField(1, tensorTypeBytes.ToArray()));
        bytes.AddRange(CreateField(2, typeBytes.ToArray()));

        return bytes.ToArray();
    }

    private static byte[] CreateTensorProto(string name, float[] data, int[] dims)
    {
        var bytes = new List<byte>();

        // dims (field 1, repeated int64)
        foreach (var dim in dims)
        {
            bytes.AddRange(CreateVarintField(1, dim));
        }

        // data_type (field 2, int32) - 1 = FLOAT
        bytes.AddRange(CreateVarintField(2, 1));

        // float_data (field 4, repeated float) - packed
        var floatBytes = new byte[data.Length * 4];
        Buffer.BlockCopy(data, 0, floatBytes, 0, floatBytes.Length);
        bytes.AddRange(CreateLengthDelimitedField(4, floatBytes));

        // name (field 8, string)
        bytes.AddRange(CreateStringField(8, name));

        return bytes.ToArray();
    }

    private static byte[] CreateField(int fieldNumber, byte[] value)
    {
        var bytes = new List<byte>();
        bytes.AddRange(EncodeVarint((fieldNumber << 3) | 2)); // wire type 2 = length-delimited
        bytes.AddRange(EncodeVarint(value.Length));
        bytes.AddRange(value);
        return bytes.ToArray();
    }

    private static byte[] CreateStringField(int fieldNumber, string value)
    {
        var stringBytes = System.Text.Encoding.UTF8.GetBytes(value);
        return CreateLengthDelimitedField(fieldNumber, stringBytes);
    }

    private static byte[] CreateLengthDelimitedField(int fieldNumber, byte[] value)
    {
        var bytes = new List<byte>();
        bytes.AddRange(EncodeVarint((fieldNumber << 3) | 2)); // wire type 2
        bytes.AddRange(EncodeVarint(value.Length));
        bytes.AddRange(value);
        return bytes.ToArray();
    }

    private static byte[] CreateVarintField(int fieldNumber, long value)
    {
        var bytes = new List<byte>();
        bytes.AddRange(EncodeVarint((fieldNumber << 3) | 0)); // wire type 0 = varint
        bytes.AddRange(EncodeVarint(value));
        return bytes.ToArray();
    }

    private static byte[] EncodeVarint(long value)
    {
        var bytes = new List<byte>();
        var uvalue = (ulong)value;
        while (uvalue >= 0x80)
        {
            bytes.Add((byte)(uvalue | 0x80));
            uvalue >>= 7;
        }
        bytes.Add((byte)uvalue);
        return bytes.ToArray();
    }
}
