using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.Onnx;

/// <summary>
/// Converts between AiDotNet Tensor types and ONNX Runtime tensor types.
/// </summary>
/// <remarks>
/// <para>
/// This class provides static methods for converting tensors between AiDotNet's
/// generic Tensor&lt;T&gt; format and ONNX Runtime's DenseTensor format.
/// </para>
/// <para><b>For Beginners:</b> When running ONNX models, we need to convert our data
/// to a format that ONNX Runtime understands. This converter handles that translation:
/// <list type="bullet">
/// <item>ToOnnx: Converts your AiDotNet tensor to ONNX format for model input</item>
/// <item>FromOnnx: Converts ONNX model output back to AiDotNet tensor</item>
/// </list>
/// </para>
/// </remarks>
public static class OnnxTensorConverter
{
    /// <summary>
    /// Converts an AiDotNet Tensor to an ONNX DenseTensor of floats.
    /// </summary>
    /// <typeparam name="T">The source tensor's element type.</typeparam>
    /// <param name="tensor">The AiDotNet tensor to convert.</param>
    /// <returns>An ONNX DenseTensor with the same shape and data.</returns>
    public static OnnxTensors.DenseTensor<float> ToOnnxFloat<T>(Tensor<T> tensor)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var shape = tensor.Shape.Select(d => (long)d).ToArray();
        var onnxTensor = new OnnxTensors.DenseTensor<float>(tensor.Shape);

        var sourceData = tensor.ToArray();
        var buffer = onnxTensor.Buffer.Span;

        for (int i = 0; i < sourceData.Length; i++)
        {
            buffer[i] = (float)numOps.ToDouble(sourceData[i]);
        }

        return onnxTensor;
    }

    /// <summary>
    /// Converts an AiDotNet Tensor to an ONNX DenseTensor of doubles.
    /// </summary>
    /// <typeparam name="T">The source tensor's element type.</typeparam>
    /// <param name="tensor">The AiDotNet tensor to convert.</param>
    /// <returns>An ONNX DenseTensor with the same shape and data.</returns>
    public static OnnxTensors.DenseTensor<double> ToOnnxDouble<T>(Tensor<T> tensor)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var onnxTensor = new OnnxTensors.DenseTensor<double>(tensor.Shape);

        var sourceData = tensor.ToArray();
        var buffer = onnxTensor.Buffer.Span;

        for (int i = 0; i < sourceData.Length; i++)
        {
            buffer[i] = numOps.ToDouble(sourceData[i]);
        }

        return onnxTensor;
    }

    /// <summary>
    /// Converts an AiDotNet Tensor to an ONNX DenseTensor of long integers.
    /// </summary>
    /// <typeparam name="T">The source tensor's element type.</typeparam>
    /// <param name="tensor">The AiDotNet tensor to convert.</param>
    /// <returns>An ONNX DenseTensor with the same shape and data.</returns>
    public static OnnxTensors.DenseTensor<long> ToOnnxLong<T>(Tensor<T> tensor)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var onnxTensor = new OnnxTensors.DenseTensor<long>(tensor.Shape);

        var sourceData = tensor.ToArray();
        var buffer = onnxTensor.Buffer.Span;

        for (int i = 0; i < sourceData.Length; i++)
        {
            buffer[i] = (long)numOps.ToDouble(sourceData[i]);
        }

        return onnxTensor;
    }

    /// <summary>
    /// Converts an AiDotNet Tensor to an ONNX DenseTensor of integers.
    /// </summary>
    /// <typeparam name="T">The source tensor's element type.</typeparam>
    /// <param name="tensor">The AiDotNet tensor to convert.</param>
    /// <returns>An ONNX DenseTensor with the same shape and data.</returns>
    public static OnnxTensors.DenseTensor<int> ToOnnxInt<T>(Tensor<T> tensor)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var onnxTensor = new OnnxTensors.DenseTensor<int>(tensor.Shape);

        var sourceData = tensor.ToArray();
        var buffer = onnxTensor.Buffer.Span;

        for (int i = 0; i < sourceData.Length; i++)
        {
            buffer[i] = (int)numOps.ToDouble(sourceData[i]);
        }

        return onnxTensor;
    }

    /// <summary>
    /// Converts an ONNX DenseTensor of floats to an AiDotNet Tensor.
    /// </summary>
    /// <typeparam name="T">The target tensor's element type.</typeparam>
    /// <param name="onnxTensor">The ONNX tensor to convert.</param>
    /// <returns>An AiDotNet Tensor with the same shape and data.</returns>
    public static Tensor<T> FromOnnxFloat<T>(OnnxTensors.Tensor<float> onnxTensor)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var shape = onnxTensor.Dimensions.ToArray();
        var length = shape.Aggregate(1, (a, b) => a * b);

        var data = new T[length];
        int index = 0;

        foreach (var value in onnxTensor)
        {
            data[index++] = numOps.FromDouble(value);
        }

        return new Tensor<T>(data, shape);
    }

    /// <summary>
    /// Converts an ONNX DenseTensor of doubles to an AiDotNet Tensor.
    /// </summary>
    /// <typeparam name="T">The target tensor's element type.</typeparam>
    /// <param name="onnxTensor">The ONNX tensor to convert.</param>
    /// <returns>An AiDotNet Tensor with the same shape and data.</returns>
    public static Tensor<T> FromOnnxDouble<T>(OnnxTensors.Tensor<double> onnxTensor)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var shape = onnxTensor.Dimensions.ToArray();
        var length = shape.Aggregate(1, (a, b) => a * b);

        var data = new T[length];
        int index = 0;

        foreach (var value in onnxTensor)
        {
            data[index++] = numOps.FromDouble(value);
        }

        return new Tensor<T>(data, shape);
    }

    /// <summary>
    /// Converts an ONNX DenseTensor of long integers to an AiDotNet Tensor.
    /// </summary>
    /// <typeparam name="T">The target tensor's element type.</typeparam>
    /// <param name="onnxTensor">The ONNX tensor to convert.</param>
    /// <returns>An AiDotNet Tensor with the same shape and data.</returns>
    public static Tensor<T> FromOnnxLong<T>(OnnxTensors.Tensor<long> onnxTensor)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var shape = onnxTensor.Dimensions.ToArray();
        var length = shape.Aggregate(1, (a, b) => a * b);

        var data = new T[length];
        int index = 0;

        foreach (var value in onnxTensor)
        {
            data[index++] = numOps.FromDouble(value);
        }

        return new Tensor<T>(data, shape);
    }

    /// <summary>
    /// Converts an ONNX DenseTensor of integers to an AiDotNet Tensor.
    /// </summary>
    /// <typeparam name="T">The target tensor's element type.</typeparam>
    /// <param name="onnxTensor">The ONNX tensor to convert.</param>
    /// <returns>An AiDotNet Tensor with the same shape and data.</returns>
    public static Tensor<T> FromOnnxInt<T>(OnnxTensors.Tensor<int> onnxTensor)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var shape = onnxTensor.Dimensions.ToArray();
        var length = shape.Aggregate(1, (a, b) => a * b);

        var data = new T[length];
        int index = 0;

        foreach (var value in onnxTensor)
        {
            data[index++] = numOps.FromDouble(value);
        }

        return new Tensor<T>(data, shape);
    }

    /// <summary>
    /// Creates an ONNX DenseTensor with a prepended batch dimension.
    /// </summary>
    /// <typeparam name="T">The source tensor's element type.</typeparam>
    /// <param name="tensor">The AiDotNet tensor to convert.</param>
    /// <returns>An ONNX DenseTensor with shape [1, ...original_shape].</returns>
    public static OnnxTensors.DenseTensor<float> ToOnnxFloatWithBatch<T>(Tensor<T> tensor)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var batchShape = new int[tensor.Shape.Length + 1];
        batchShape[0] = 1;
        tensor.Shape.CopyTo(batchShape, 1);

        var onnxTensor = new OnnxTensors.DenseTensor<float>(batchShape);
        var sourceData = tensor.ToArray();
        var buffer = onnxTensor.Buffer.Span;

        for (int i = 0; i < sourceData.Length; i++)
        {
            buffer[i] = (float)numOps.ToDouble(sourceData[i]);
        }

        return onnxTensor;
    }

    /// <summary>
    /// Converts an ONNX tensor and removes the batch dimension if it's 1.
    /// </summary>
    /// <typeparam name="T">The target tensor's element type.</typeparam>
    /// <param name="onnxTensor">The ONNX tensor to convert.</param>
    /// <returns>An AiDotNet Tensor, with batch dimension removed if batch size is 1.</returns>
    public static Tensor<T> FromOnnxFloatRemoveBatch<T>(OnnxTensors.Tensor<float> onnxTensor)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var dims = onnxTensor.Dimensions.ToArray();

        // Remove batch dimension if it's 1
        int[] shape;
        if (dims.Length > 1 && dims[0] == 1)
        {
            shape = dims.Skip(1).ToArray();
        }
        else
        {
            shape = dims;
        }

        var length = shape.Aggregate(1, (a, b) => a * b);
        var data = new T[length];
        int index = 0;

        foreach (var value in onnxTensor)
        {
            if (index >= length) break;
            data[index++] = numOps.FromDouble(value);
        }

        return new Tensor<T>(data, shape);
    }

    /// <summary>
    /// Gets the ONNX element type name for a .NET type.
    /// </summary>
    /// <param name="type">The .NET type.</param>
    /// <returns>The ONNX element type name.</returns>
    public static string GetOnnxTypeName(Type type)
    {
        return type.Name switch
        {
            "Single" => "float",
            "Double" => "double",
            "Int32" => "int32",
            "Int64" => "int64",
            "Int16" => "int16",
            "Byte" => "uint8",
            "SByte" => "int8",
            "UInt16" => "uint16",
            "UInt32" => "uint32",
            "UInt64" => "uint64",
            "Boolean" => "bool",
            "String" => "string",
            _ => "unknown"
        };
    }

    /// <summary>
    /// Converts an AiDotNet Tensor to an ONNX NamedOnnxValue based on the target element type.
    /// </summary>
    /// <typeparam name="T">The source tensor's element type.</typeparam>
    /// <param name="name">The input name for the ONNX model.</param>
    /// <param name="tensor">The AiDotNet tensor to convert.</param>
    /// <param name="elementType">The target ONNX element type (e.g., "float", "double", "int64").</param>
    /// <returns>A NamedOnnxValue with the converted tensor.</returns>
    public static Microsoft.ML.OnnxRuntime.NamedOnnxValue ToOnnxValue<T>(string name, Tensor<T> tensor, string elementType)
    {
        return elementType.ToLowerInvariant() switch
        {
            "double" or "float64" => Microsoft.ML.OnnxRuntime.NamedOnnxValue.CreateFromTensor(name, ToOnnxDouble(tensor)),
            "int64" or "long" => Microsoft.ML.OnnxRuntime.NamedOnnxValue.CreateFromTensor(name, ToOnnxLong(tensor)),
            "int32" or "int" => Microsoft.ML.OnnxRuntime.NamedOnnxValue.CreateFromTensor(name, ToOnnxInt(tensor)),
            // float is the most common and default for neural networks
            _ => Microsoft.ML.OnnxRuntime.NamedOnnxValue.CreateFromTensor(name, ToOnnxFloat(tensor))
        };
    }

    /// <summary>
    /// Converts an ONNX DisposableNamedOnnxValue to an AiDotNet Tensor based on its actual element type.
    /// </summary>
    /// <typeparam name="T">The target tensor's element type.</typeparam>
    /// <param name="result">The ONNX result value to convert.</param>
    /// <returns>An AiDotNet Tensor with the converted data, or null if conversion failed.</returns>
    public static Tensor<T>? FromOnnxValue<T>(Microsoft.ML.OnnxRuntime.DisposableNamedOnnxValue result)
    {
        // Try float first (most common)
        var floatTensor = result.AsTensor<float>();
        if (floatTensor is not null)
        {
            return FromOnnxFloat<T>(floatTensor);
        }

        // Try double
        var doubleTensor = result.AsTensor<double>();
        if (doubleTensor is not null)
        {
            return FromOnnxDouble<T>(doubleTensor);
        }

        // Try int64
        var longTensor = result.AsTensor<long>();
        if (longTensor is not null)
        {
            return FromOnnxLong<T>(longTensor);
        }

        // Try int32
        var intTensor = result.AsTensor<int>();
        if (intTensor is not null)
        {
            return FromOnnxInt<T>(intTensor);
        }

        return null;
    }
}
