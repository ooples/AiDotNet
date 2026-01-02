using System;
using System.Collections.Concurrent;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Cached GPU buffer entry for persistent tensor management.
/// </summary>
internal sealed class GpuBufferCacheEntry : IDisposable
{
    public IGpuBuffer Buffer { get; }
    public PersistentTensorRole Role { get; }
    public int Version { get; set; }

    public GpuBufferCacheEntry(IGpuBuffer buffer, PersistentTensorRole role)
    {
        Buffer = buffer;
        Role = role;
        Version = 0;
    }

    public void Dispose()
    {
        Buffer.Dispose();
    }
}

/// <summary>
/// IEngine implementation that routes supported ops to DirectGpuEngine and falls back to CPU.
/// </summary>
public class DirectGpuTensorEngine : CpuEngine, IEngine, IDisposable
{
    private readonly DirectGpuEngine? _directGpu;
    private readonly bool _ownsDirectGpu;

    // GPU buffer cache for persistent tensors - keyed by tensor data array reference
    private readonly ConcurrentDictionary<object, GpuBufferCacheEntry> _persistentBufferCache = new();

    // Version tracking for invalidation
    private readonly ConcurrentDictionary<object, int> _tensorVersions = new();

    public DirectGpuTensorEngine()
    {
        _directGpu = new DirectGpuEngine();
        _ownsDirectGpu = true;
    }

    public DirectGpuTensorEngine(DirectGpuEngine directGpu)
    {
        _directGpu = directGpu;
        _ownsDirectGpu = false;
    }

    public bool IsGpuAvailable => _directGpu?.IsAvailable == true;

    public new string Name => IsGpuAvailable
        ? $"Direct GPU Engine ({_directGpu!.BackendName} {_directGpu.DeviceName})"
        : "CPU Engine (DirectGpu unavailable)";

    public new bool SupportsGpu => IsGpuAvailable;

    DirectGpuEngine? IEngine.DirectGpu => _directGpu;

    string IEngine.Name => Name;

    bool IEngine.SupportsGpu => SupportsGpu;

    private bool TryGetBackend(out IDirectGpuBackend backend)
    {
        backend = _directGpu?.Backend!;
        return IsGpuAvailable && backend != null;
    }

    private static float ToFloatScalar<T>(T value)
    {
        if (typeof(T) == typeof(float))
            return (float)(object)value!;

        return DirectGpuEngine.ToFloatArray(new[] { value })[0];
    }

    private static T FromFloatScalar<T>(float value)
    {
        return DirectGpuEngine.FromFloatArray<T>(new[] { value })[0];
    }

    private T[]? TryRunUnary<T>(T[] input, Action<IDirectGpuBackend, IGpuBuffer, IGpuBuffer, int> op)
    {
        if (!TryGetBackend(out var backend))
            return null;

        float[] inputFloat = DirectGpuEngine.ToFloatArray(input);
        using var bufferA = backend.AllocateBuffer(inputFloat);
        using var bufferB = backend.AllocateBuffer(inputFloat.Length);
        op(backend, bufferA, bufferB, inputFloat.Length);
        backend.Synchronize();
        float[] resultFloat = backend.DownloadBuffer(bufferB);
        return DirectGpuEngine.FromFloatArray<T>(resultFloat);
    }

    private T[]? TryRunBinary<T>(T[] left, T[] right, Action<IDirectGpuBackend, IGpuBuffer, IGpuBuffer, IGpuBuffer, int> op)
    {
        if (!TryGetBackend(out var backend))
            return null;
        if (left.Length != right.Length)
            return null;

        float[] leftFloat = DirectGpuEngine.ToFloatArray(left);
        float[] rightFloat = DirectGpuEngine.ToFloatArray(right);
        using var bufferA = backend.AllocateBuffer(leftFloat);
        using var bufferB = backend.AllocateBuffer(rightFloat);
        using var bufferC = backend.AllocateBuffer(leftFloat.Length);
        op(backend, bufferA, bufferB, bufferC, leftFloat.Length);
        backend.Synchronize();
        float[] resultFloat = backend.DownloadBuffer(bufferC);
        return DirectGpuEngine.FromFloatArray<T>(resultFloat);
    }

    private T[]? TryRunScalar<T>(T[] input, T scalar, Action<IDirectGpuBackend, IGpuBuffer, IGpuBuffer, float, int> op)
    {
        if (!TryGetBackend(out var backend))
            return null;

        float[] inputFloat = DirectGpuEngine.ToFloatArray(input);
        using var bufferA = backend.AllocateBuffer(inputFloat);
        using var bufferB = backend.AllocateBuffer(inputFloat.Length);
        op(backend, bufferA, bufferB, ToFloatScalar(scalar), inputFloat.Length);
        backend.Synchronize();
        float[] resultFloat = backend.DownloadBuffer(bufferB);
        return DirectGpuEngine.FromFloatArray<T>(resultFloat);
    }

    private static bool ShapesMatch(int[] left, int[] right)
    {
        return left.Length == right.Length && left.SequenceEqual(right);
    }

    Vector<T> IEngine.Add<T>(Vector<T> a, Vector<T> b)
    {
        var result = TryRunBinary(a.Data, b.Data, static (backend, left, right, output, size) => backend.Add(left, right, output, size));
        return result != null ? new Vector<T>(result) : base.Add(a, b);
    }

    Vector<T> IEngine.Subtract<T>(Vector<T> a, Vector<T> b)
    {
        var result = TryRunBinary(a.Data, b.Data, static (backend, left, right, output, size) => backend.Subtract(left, right, output, size));
        return result != null ? new Vector<T>(result) : base.Subtract(a, b);
    }

    Vector<T> IEngine.Multiply<T>(Vector<T> a, Vector<T> b)
    {
        var result = TryRunBinary(a.Data, b.Data, static (backend, left, right, output, size) => backend.Multiply(left, right, output, size));
        return result != null ? new Vector<T>(result) : base.Multiply(a, b);
    }

    Vector<T> IEngine.Multiply<T>(Vector<T> vector, T scalar)
    {
        var result = TryRunScalar(vector.Data, scalar, static (backend, input, output, value, size) => backend.Scale(input, output, value, size));
        return result != null ? new Vector<T>(result) : base.Multiply(vector, scalar);
    }

    Vector<T> IEngine.Divide<T>(Vector<T> a, Vector<T> b)
    {
        var result = TryRunBinary(a.Data, b.Data, static (backend, left, right, output, size) => backend.Divide(left, right, output, size));
        return result != null ? new Vector<T>(result) : base.Divide(a, b);
    }

    Vector<T> IEngine.Divide<T>(Vector<T> vector, T scalar)
    {
        var scalarValue = ToFloatScalar(scalar);
        if (scalarValue == 0)
            return base.Divide(vector, scalar);

        var result = TryRunScalar(vector.Data, scalar, static (backend, input, output, value, size) => backend.Scale(input, output, 1.0f / value, size));
        return result != null ? new Vector<T>(result) : base.Divide(vector, scalar);
    }

    Vector<T> IEngine.Max<T>(Vector<T> a, Vector<T> b)
    {
        var result = TryRunBinary(a.Data, b.Data, static (backend, left, right, output, size) => backend.Max(left, right, output, size));
        return result != null ? new Vector<T>(result) : base.Max(a, b);
    }

    Vector<T> IEngine.Min<T>(Vector<T> a, Vector<T> b)
    {
        var result = TryRunBinary(a.Data, b.Data, static (backend, left, right, output, size) => backend.Min(left, right, output, size));
        return result != null ? new Vector<T>(result) : base.Min(a, b);
    }

    Vector<T> IEngine.Abs<T>(Vector<T> vector)
    {
        var result = TryRunUnary(vector.Data, static (backend, input, output, size) => backend.Abs(input, output, size));
        return result != null ? new Vector<T>(result) : base.Abs(vector);
    }

    Vector<T> IEngine.Exp<T>(Vector<T> vector)
    {
        var result = TryRunUnary(vector.Data, static (backend, input, output, size) => backend.Exp(input, output, size));
        return result != null ? new Vector<T>(result) : base.Exp(vector);
    }

    Vector<T> IEngine.Exp2<T>(Vector<T> vector)
    {
        var result = TryRunUnary(vector.Data, static (backend, input, output, size) => backend.Exp2(input, output, size));
        return result != null ? new Vector<T>(result) : base.Exp2(vector);
    }

    Vector<T> IEngine.Exp10<T>(Vector<T> vector)
    {
        var result = TryRunUnary(vector.Data, static (backend, input, output, size) => backend.Exp10(input, output, size));
        return result != null ? new Vector<T>(result) : base.Exp10(vector);
    }

    Vector<T> IEngine.Log<T>(Vector<T> vector)
    {
        var result = TryRunUnary(vector.Data, static (backend, input, output, size) => backend.Log(input, output, size));
        return result != null ? new Vector<T>(result) : base.Log(vector);
    }

    Vector<T> IEngine.Log2<T>(Vector<T> vector)
    {
        var result = TryRunUnary(vector.Data, static (backend, input, output, size) => backend.Log2(input, output, size));
        return result != null ? new Vector<T>(result) : base.Log2(vector);
    }

    Vector<T> IEngine.Sqrt<T>(Vector<T> vector)
    {
        var result = TryRunUnary(vector.Data, static (backend, input, output, size) => backend.Sqrt(input, output, size));
        return result != null ? new Vector<T>(result) : base.Sqrt(vector);
    }

    Vector<T> IEngine.Power<T>(Vector<T> vector, T exponent)
    {
        var result = TryRunScalar(vector.Data, exponent, static (backend, input, output, value, size) => backend.Power(input, output, value, size));
        return result != null ? new Vector<T>(result) : base.Power(vector, exponent);
    }

    Vector<T> IEngine.Tanh<T>(Vector<T> vector)
    {
        var result = TryRunUnary(vector.Data, static (backend, input, output, size) => backend.Tanh(input, output, size));
        return result != null ? new Vector<T>(result) : base.Tanh(vector);
    }

    Vector<T> IEngine.Sigmoid<T>(Vector<T> vector)
    {
        var result = TryRunUnary(vector.Data, static (backend, input, output, size) => backend.Sigmoid(input, output, size));
        return result != null ? new Vector<T>(result) : base.Sigmoid(vector);
    }

    Vector<T> IEngine.ReLU<T>(Vector<T> vector)
    {
        var result = TryRunUnary(vector.Data, static (backend, input, output, size) => backend.Relu(input, output, size));
        return result != null ? new Vector<T>(result) : base.ReLU(vector);
    }

    Vector<T> IEngine.GELU<T>(Vector<T> vector)
    {
        var result = TryRunUnary(vector.Data, static (backend, input, output, size) => backend.Gelu(input, output, size));
        return result != null ? new Vector<T>(result) : base.GELU(vector);
    }

    Matrix<T> IEngine.MatrixMultiply<T>(Matrix<T> a, Matrix<T> b)
    {
        if (!IsGpuAvailable || _directGpu == null)
            return base.MatrixMultiply(a, b);

        if (a.Columns != b.Rows)
            return base.MatrixMultiply(a, b);

        try
        {
            var resultData = _directGpu.MatMul(a.AsSpan().ToArray(), b.AsSpan().ToArray(), a.Rows, a.Columns, b.Columns);
            if (resultData == null)
                return base.MatrixMultiply(a, b);

            var result = new Matrix<T>(a.Rows, b.Columns);
            resultData.AsSpan().CopyTo(result.AsWritableSpan());
            return result;
        }
        catch
        {
            return base.MatrixMultiply(a, b);
        }
    }

    Matrix<T> IEngine.MatrixAdd<T>(Matrix<T> a, Matrix<T> b)
    {
        if (a.Rows != b.Rows || a.Columns != b.Columns)
            return base.MatrixAdd(a, b);

        var result = TryRunBinary(a.AsSpan().ToArray(), b.AsSpan().ToArray(), static (backend, left, right, output, size) => backend.Add(left, right, output, size));
        if (result == null)
            return base.MatrixAdd(a, b);

        var matrix = new Matrix<T>(a.Rows, a.Columns);
        result.AsSpan().CopyTo(matrix.AsWritableSpan());
        return matrix;
    }

    Matrix<T> IEngine.MatrixSubtract<T>(Matrix<T> a, Matrix<T> b)
    {
        if (a.Rows != b.Rows || a.Columns != b.Columns)
            return base.MatrixSubtract(a, b);

        var result = TryRunBinary(a.AsSpan().ToArray(), b.AsSpan().ToArray(), static (backend, left, right, output, size) => backend.Subtract(left, right, output, size));
        if (result == null)
            return base.MatrixSubtract(a, b);

        var matrix = new Matrix<T>(a.Rows, a.Columns);
        result.AsSpan().CopyTo(matrix.AsWritableSpan());
        return matrix;
    }

    Matrix<T> IEngine.MatrixMultiplyScalar<T>(Matrix<T> matrix, T scalar)
    {
        var result = TryRunScalar(matrix.AsSpan().ToArray(), scalar, static (backend, input, output, value, size) => backend.Scale(input, output, value, size));
        if (result == null)
            return base.MatrixMultiplyScalar(matrix, scalar);

        var output = new Matrix<T>(matrix.Rows, matrix.Columns);
        result.AsSpan().CopyTo(output.AsWritableSpan());
        return output;
    }

    Tensor<T> IEngine.TensorAdd<T>(Tensor<T> a, Tensor<T> b)
    {
        if (!ShapesMatch(a.Shape, b.Shape))
            return base.TensorAdd(a, b);

        var result = TryRunBinary(a.Data, b.Data, static (backend, left, right, output, size) => backend.Add(left, right, output, size));
        return result != null ? new Tensor<T>(result, a.Shape) : base.TensorAdd(a, b);
    }

    Tensor<T> IEngine.TensorSubtract<T>(Tensor<T> a, Tensor<T> b)
    {
        if (!ShapesMatch(a.Shape, b.Shape))
            return base.TensorSubtract(a, b);

        var result = TryRunBinary(a.Data, b.Data, static (backend, left, right, output, size) => backend.Subtract(left, right, output, size));
        return result != null ? new Tensor<T>(result, a.Shape) : base.TensorSubtract(a, b);
    }

    Tensor<T> IEngine.TensorMultiply<T>(Tensor<T> a, Tensor<T> b)
    {
        if (!ShapesMatch(a.Shape, b.Shape))
            return base.TensorMultiply(a, b);

        var result = TryRunBinary(a.Data, b.Data, static (backend, left, right, output, size) => backend.Multiply(left, right, output, size));
        return result != null ? new Tensor<T>(result, a.Shape) : base.TensorMultiply(a, b);
    }

    Tensor<T> IEngine.TensorDivide<T>(Tensor<T> a, Tensor<T> b)
    {
        if (!ShapesMatch(a.Shape, b.Shape))
            return base.TensorDivide(a, b);

        var result = TryRunBinary(a.Data, b.Data, static (backend, left, right, output, size) => backend.Divide(left, right, output, size));
        return result != null ? new Tensor<T>(result, a.Shape) : base.TensorDivide(a, b);
    }

    Tensor<T> IEngine.TensorMultiplyScalar<T>(Tensor<T> tensor, T scalar)
    {
        var result = TryRunScalar(tensor.Data, scalar, static (backend, input, output, value, size) => backend.Scale(input, output, value, size));
        return result != null ? new Tensor<T>(result, tensor.Shape) : base.TensorMultiplyScalar(tensor, scalar);
    }

    Tensor<T> IEngine.TensorDivideScalar<T>(Tensor<T> tensor, T scalar)
    {
        var scalarValue = ToFloatScalar(scalar);
        if (scalarValue == 0)
            return base.TensorDivideScalar(tensor, scalar);

        var result = TryRunScalar(tensor.Data, scalar, static (backend, input, output, value, size) => backend.Scale(input, output, 1.0f / value, size));
        return result != null ? new Tensor<T>(result, tensor.Shape) : base.TensorDivideScalar(tensor, scalar);
    }

    Tensor<T> IEngine.TensorAbs<T>(Tensor<T> tensor)
    {
        var result = TryRunUnary(tensor.Data, static (backend, input, output, size) => backend.Abs(input, output, size));
        return result != null ? new Tensor<T>(result, tensor.Shape) : base.TensorAbs(tensor);
    }

    Tensor<T> IEngine.TensorExp<T>(Tensor<T> tensor)
    {
        var result = TryRunUnary(tensor.Data, static (backend, input, output, size) => backend.Exp(input, output, size));
        return result != null ? new Tensor<T>(result, tensor.Shape) : base.TensorExp(tensor);
    }

    Tensor<T> IEngine.TensorLog<T>(Tensor<T> tensor)
    {
        var result = TryRunUnary(tensor.Data, static (backend, input, output, size) => backend.Log(input, output, size));
        return result != null ? new Tensor<T>(result, tensor.Shape) : base.TensorLog(tensor);
    }

    Tensor<T> IEngine.TensorSqrt<T>(Tensor<T> tensor)
    {
        var result = TryRunUnary(tensor.Data, static (backend, input, output, size) => backend.Sqrt(input, output, size));
        return result != null ? new Tensor<T>(result, tensor.Shape) : base.TensorSqrt(tensor);
    }

    Tensor<T> IEngine.TensorNegate<T>(Tensor<T> tensor)
    {
        var result = TryRunScalar(tensor.Data, FromFloatScalar<T>(-1.0f), static (backend, input, output, value, size) => backend.Scale(input, output, value, size));
        return result != null ? new Tensor<T>(result, tensor.Shape) : base.TensorNegate(tensor);
    }

    Tensor<T> IEngine.TensorPower<T>(Tensor<T> tensor, T exponent)
    {
        var result = TryRunScalar(tensor.Data, exponent, static (backend, input, output, value, size) => backend.Power(input, output, value, size));
        return result != null ? new Tensor<T>(result, tensor.Shape) : base.TensorPower(tensor, exponent);
    }

    Tensor<T> IEngine.TensorMax<T>(Tensor<T> a, Tensor<T> b)
    {
        if (!ShapesMatch(a.Shape, b.Shape))
            return base.TensorMax(a, b);

        var result = TryRunBinary(a.Data, b.Data, static (backend, left, right, output, size) => backend.Max(left, right, output, size));
        return result != null ? new Tensor<T>(result, a.Shape) : base.TensorMax(a, b);
    }

    Tensor<T> IEngine.TensorMin<T>(Tensor<T> a, Tensor<T> b)
    {
        if (!ShapesMatch(a.Shape, b.Shape))
            return base.TensorMin(a, b);

        var result = TryRunBinary(a.Data, b.Data, static (backend, left, right, output, size) => backend.Min(left, right, output, size));
        return result != null ? new Tensor<T>(result, a.Shape) : base.TensorMin(a, b);
    }

    Tensor<T> IEngine.Tanh<T>(Tensor<T> tensor)
    {
        var result = TryRunUnary(tensor.Data, static (backend, input, output, size) => backend.Tanh(input, output, size));
        return result != null ? new Tensor<T>(result, tensor.Shape) : base.Tanh(tensor);
    }

    Tensor<T> IEngine.Sigmoid<T>(Tensor<T> tensor)
    {
        var result = TryRunUnary(tensor.Data, static (backend, input, output, size) => backend.Sigmoid(input, output, size));
        return result != null ? new Tensor<T>(result, tensor.Shape) : base.Sigmoid(tensor);
    }

    Tensor<T> IEngine.ReLU<T>(Tensor<T> tensor)
    {
        var result = TryRunUnary(tensor.Data, static (backend, input, output, size) => backend.Relu(input, output, size));
        return result != null ? new Tensor<T>(result, tensor.Shape) : base.ReLU(tensor);
    }

    Tensor<T> IEngine.GELU<T>(Tensor<T> tensor)
    {
        var result = TryRunUnary(tensor.Data, static (backend, input, output, size) => backend.Gelu(input, output, size));
        return result != null ? new Tensor<T>(result, tensor.Shape) : base.GELU(tensor);
    }

    T IEngine.TensorSum<T>(Tensor<T> tensor)
    {
        if (!TryGetBackend(out var backend))
            return base.TensorSum(tensor);

        float[] inputFloat = DirectGpuEngine.ToFloatArray(tensor.Data);
        using var bufferA = backend.AllocateBuffer(inputFloat);
        backend.Synchronize();
        float sum = backend.Sum(bufferA, inputFloat.Length);
        return FromFloatScalar<T>(sum);
    }

    T IEngine.TensorMaxValue<T>(Tensor<T> tensor)
    {
        if (!TryGetBackend(out var backend))
            return base.TensorMaxValue(tensor);

        float[] inputFloat = DirectGpuEngine.ToFloatArray(tensor.Data);
        using var bufferA = backend.AllocateBuffer(inputFloat);
        backend.Synchronize();
        float max = backend.Max(bufferA, inputFloat.Length);
        return FromFloatScalar<T>(max);
    }

    #region Fused Operations

    /// <summary>
    /// GPU-accelerated fused linear transformation: output = activation(input @ weights + bias).
    /// </summary>
    public new Tensor<T> FusedLinear<T>(Tensor<T> input, Tensor<T> weights, Tensor<T>? bias, FusedActivationType activation)
    {
        if (!TryGetBackend(out var backend))
            return base.FusedLinear(input, weights, bias, activation);

        if (input.Rank < 1 || weights.Rank != 2)
            return base.FusedLinear(input, weights, bias, activation);

        int batchSize = input.Shape[0];
        int inputFeatures = weights.Shape[0];
        int outputFeatures = weights.Shape[1];

        // Convert to float arrays
        float[] inputFloat = DirectGpuEngine.ToFloatArray(input.Data);
        float[] weightsFloat = DirectGpuEngine.ToFloatArray(weights.Data);
        float[]? biasFloat = bias != null ? DirectGpuEngine.ToFloatArray(bias.Data) : null;

        // Allocate GPU buffers
        using var inputBuffer = backend.AllocateBuffer(inputFloat);
        using var weightsBuffer = backend.AllocateBuffer(weightsFloat);
        using var biasBuffer = biasFloat != null ? backend.AllocateBuffer(biasFloat) : null;

        IGpuBuffer resultBuffer;
        try
        {
            // Use fused GPU kernels when available
            // Only use GPU path for natively supported fused ops (with bias)
            // For other cases, fall back to CPU which handles all combinations
            if (biasBuffer != null && (activation == FusedActivationType.ReLU ||
                                        activation == FusedActivationType.GELU ||
                                        activation == FusedActivationType.Sigmoid ||
                                        activation == FusedActivationType.Tanh))
            {
                resultBuffer = activation switch
                {
                    FusedActivationType.ReLU => backend.GemmBiasRelu(inputBuffer, weightsBuffer, biasBuffer, batchSize, outputFeatures, inputFeatures),
                    FusedActivationType.GELU => backend.GemmBiasGelu(inputBuffer, weightsBuffer, biasBuffer, batchSize, outputFeatures, inputFeatures),
                    FusedActivationType.Sigmoid => backend.GemmBiasSigmoid(inputBuffer, weightsBuffer, biasBuffer, batchSize, outputFeatures, inputFeatures),
                    _ => backend.GemmBiasTanh(inputBuffer, weightsBuffer, biasBuffer, batchSize, outputFeatures, inputFeatures),
                };
            }
            else if (biasBuffer == null && activation == FusedActivationType.None)
            {
                // Simple MatMul only - use GPU
                resultBuffer = backend.MatMul(inputBuffer, weightsBuffer, batchSize, outputFeatures, inputFeatures);
            }
            else
            {
                // Fall back to CPU for other combinations
                return base.FusedLinear(input, weights, bias, activation);
            }

            // Download result
            int resultSize = batchSize * outputFeatures;
            float[] resultFloat = new float[resultSize];
            backend.DownloadBuffer(resultBuffer, resultFloat);
            resultBuffer.Dispose();
            backend.Synchronize();

            // Convert back to T
            T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
            return new Tensor<T>(resultData, new[] { batchSize, outputFeatures });
        }
        catch
        {
            // Fall back to CPU on any GPU error
            return base.FusedLinear(input, weights, bias, activation);
        }
    }

    private static IGpuBuffer GemmBiasNoActivation(IDirectGpuBackend backend, IGpuBuffer input, IGpuBuffer weights, IGpuBuffer bias, int M, int N, int K)
    {
        // Use GemmBiasRelu with a subsequent inverse to get just GEMM + Bias
        // This is a workaround since there's no direct GemmBias function
        // Fall back to return just MatMul result and let caller handle bias on CPU
        return backend.MatMul(input, weights, M, N, K);
    }

    private static IGpuBuffer GemmBiasWithActivation(IDirectGpuBackend backend, IGpuBuffer input, IGpuBuffer weights, IGpuBuffer bias, int M, int N, int K, FusedActivationType activation)
    {
        // For activations without native fused support, use MatMul + activation
        var result = backend.MatMul(input, weights, M, N, K);
        int size = M * N;
        ApplyGpuActivation(backend, result, size, activation);
        return result;
    }

    private static void ApplyGpuActivation(IDirectGpuBackend backend, IGpuBuffer buffer, int size, FusedActivationType activation)
    {
        switch (activation)
        {
            case FusedActivationType.ReLU:
                backend.Relu(buffer, buffer, size);
                break;
            case FusedActivationType.LeakyReLU:
                backend.LeakyRelu(buffer, buffer, 0.01f, size);
                break;
            case FusedActivationType.Sigmoid:
                backend.Sigmoid(buffer, buffer, size);
                break;
            case FusedActivationType.Tanh:
                backend.Tanh(buffer, buffer, size);
                break;
            case FusedActivationType.GELU:
                backend.Gelu(buffer, buffer, size);
                break;
            case FusedActivationType.Swish:
                backend.Swish(buffer, buffer, size);
                break;
            case FusedActivationType.None:
                break;
        }
    }

    /// <summary>
    /// GPU-accelerated fused 2D convolution with activation.
    /// </summary>
    public new Tensor<T> FusedConv2D<T>(
        Tensor<T> input,
        Tensor<T> kernel,
        Tensor<T>? bias,
        int strideH, int strideW,
        int padH, int padW,
        int dilationH, int dilationW,
        FusedActivationType activation)
    {
        if (!TryGetBackend(out var backend))
            return base.FusedConv2D(input, kernel, bias, strideH, strideW, padH, padW, dilationH, dilationW, activation);

        // Expected input shape: [batch, inChannels, height, width]
        // Expected kernel shape: [outChannels, inChannels, kernelH, kernelW]
        if (input.Rank != 4 || kernel.Rank != 4)
            return base.FusedConv2D(input, kernel, bias, strideH, strideW, padH, padW, dilationH, dilationW, activation);

        int batch = input.Shape[0];
        int inChannels = input.Shape[1];
        int inHeight = input.Shape[2];
        int inWidth = input.Shape[3];

        int outChannels = kernel.Shape[0];
        int kernelH = kernel.Shape[2];
        int kernelW = kernel.Shape[3];

        // Calculate output dimensions with dilation
        int effectiveKernelH = kernelH + (kernelH - 1) * (dilationH - 1);
        int effectiveKernelW = kernelW + (kernelW - 1) * (dilationW - 1);
        int outHeight = (inHeight + 2 * padH - effectiveKernelH) / strideH + 1;
        int outWidth = (inWidth + 2 * padW - effectiveKernelW) / strideW + 1;

        if (outHeight <= 0 || outWidth <= 0)
            return base.FusedConv2D(input, kernel, bias, strideH, strideW, padH, padW, dilationH, dilationW, activation);

        try
        {
            // Convert to float arrays
            float[] inputFloat = DirectGpuEngine.ToFloatArray(input.Data);
            float[] kernelFloat = DirectGpuEngine.ToFloatArray(kernel.Data);
            float[]? biasFloat = bias != null ? DirectGpuEngine.ToFloatArray(bias.Data) : null;

            // Allocate GPU buffers
            using var inputBuffer = backend.AllocateBuffer(inputFloat);
            using var kernelBuffer = backend.AllocateBuffer(kernelFloat);
            using var outputBuffer = backend.AllocateBuffer(batch * outChannels * outHeight * outWidth);

            // Execute GPU convolution
            backend.Conv2D(inputBuffer, kernelBuffer, outputBuffer,
                batch, inChannels, inHeight, inWidth,
                outChannels, outHeight, outWidth,
                kernelH, kernelW,
                strideH, strideW, padH, padW,
                dilationH, dilationW);

            // Add bias if present
            if (biasFloat != null)
            {
                // Bias is added per output channel, broadcast across batch and spatial dimensions
                int outputSize = batch * outChannels * outHeight * outWidth;
                int spatialSize = outHeight * outWidth;

                // Download, add bias, re-upload (GPU bias broadcast kernel would be more efficient)
                float[] outputFloat = new float[outputSize];
                backend.DownloadBuffer(outputBuffer, outputFloat);

                for (int b = 0; b < batch; b++)
                {
                    for (int c = 0; c < outChannels; c++)
                    {
                        float biasVal = biasFloat[c];
                        int baseIdx = (b * outChannels + c) * spatialSize;
                        for (int s = 0; s < spatialSize; s++)
                        {
                            outputFloat[baseIdx + s] += biasVal;
                        }
                    }
                }

                // Re-upload for activation
                using var biasedBuffer = backend.AllocateBuffer(outputFloat);

                // Apply activation on GPU
                if (activation != FusedActivationType.None)
                {
                    ApplyGpuActivation(backend, biasedBuffer, outputSize, activation);
                }

                backend.Synchronize();

                // Download final result
                float[] resultFloat = new float[outputSize];
                backend.DownloadBuffer(biasedBuffer, resultFloat);

                T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
                return new Tensor<T>(resultData, new[] { batch, outChannels, outHeight, outWidth });
            }
            else
            {
                // No bias - apply activation directly
                int outputSize = batch * outChannels * outHeight * outWidth;

                if (activation != FusedActivationType.None)
                {
                    ApplyGpuActivation(backend, outputBuffer, outputSize, activation);
                }

                backend.Synchronize();

                float[] resultFloat = new float[outputSize];
                backend.DownloadBuffer(outputBuffer, resultFloat);

                T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
                return new Tensor<T>(resultData, new[] { batch, outChannels, outHeight, outWidth });
            }
        }
        catch
        {
            // Fall back to CPU on any GPU error
            return base.FusedConv2D(input, kernel, bias, strideH, strideW, padH, padW, dilationH, dilationW, activation);
        }
    }

    /// <summary>
    /// GPU-accelerated fused batch normalization with activation.
    /// </summary>
    public new Tensor<T> FusedBatchNorm<T>(
        Tensor<T> input,
        Tensor<T> gamma,
        Tensor<T> beta,
        Tensor<T> runningMean,
        Tensor<T> runningVar,
        double epsilon,
        double momentum,
        bool training,
        FusedActivationType activation,
        out Tensor<T> saveMean,
        out Tensor<T> saveVar)
    {
        if (!TryGetBackend(out var backend))
            return base.FusedBatchNorm(input, gamma, beta, runningMean, runningVar, epsilon, momentum, training, activation, out saveMean, out saveVar);

        if (input.Rank != 2)
            return base.FusedBatchNorm(input, gamma, beta, runningMean, runningVar, epsilon, momentum, training, activation, out saveMean, out saveVar);

        int batchSize = input.Shape[0];
        int features = input.Shape[1];

        try
        {
            // Convert to float arrays
            float[] inputFloat = DirectGpuEngine.ToFloatArray(input.Data);
            float[] gammaFloat = DirectGpuEngine.ToFloatArray(gamma.Data);
            float[] betaFloat = DirectGpuEngine.ToFloatArray(beta.Data);
            float[] runningMeanFloat = DirectGpuEngine.ToFloatArray(runningMean.Data);
            float[] runningVarFloat = DirectGpuEngine.ToFloatArray(runningVar.Data);

            // Allocate GPU buffers
            using var inputBuffer = backend.AllocateBuffer(inputFloat);
            using var outputBuffer = backend.AllocateBuffer(batchSize * features);
            using var gammaBuffer = backend.AllocateBuffer(gammaFloat);
            using var betaBuffer = backend.AllocateBuffer(betaFloat);
            using var runningMeanBuffer = backend.AllocateBuffer(runningMeanFloat);
            using var runningVarBuffer = backend.AllocateBuffer(runningVarFloat);
            using var saveMeanBuffer = backend.AllocateBuffer(features);
            using var saveVarBuffer = backend.AllocateBuffer(features);

            // Execute batch norm (spatialSize=1 for 2D tensors)
            backend.BatchNorm(inputBuffer, outputBuffer, gammaBuffer, betaBuffer,
                runningMeanBuffer, runningVarBuffer, saveMeanBuffer, saveVarBuffer,
                batchSize, features, 1, (float)epsilon, (float)momentum, training);

            // Apply activation if needed
            if (activation != FusedActivationType.None)
            {
                ApplyGpuActivation(backend, outputBuffer, batchSize * features, activation);
            }

            backend.Synchronize();

            // Download results
            float[] resultFloat = new float[batchSize * features];
            float[] saveMeanFloat = new float[features];
            float[] saveVarFloat = new float[features];

            backend.DownloadBuffer(outputBuffer, resultFloat);
            backend.DownloadBuffer(saveMeanBuffer, saveMeanFloat);
            backend.DownloadBuffer(saveVarBuffer, saveVarFloat);

            // Convert back to T
            T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
            T[] saveMeanData = DirectGpuEngine.FromFloatArray<T>(saveMeanFloat);
            T[] saveVarData = DirectGpuEngine.FromFloatArray<T>(saveVarFloat);

            saveMean = new Tensor<T>(saveMeanData, new[] { features });
            saveVar = new Tensor<T>(saveVarData, new[] { features });
            return new Tensor<T>(resultData, input.Shape.ToArray());
        }
        catch
        {
            return base.FusedBatchNorm(input, gamma, beta, runningMean, runningVar, epsilon, momentum, training, activation, out saveMean, out saveVar);
        }
    }

    #endregion

    #region Attention Operations (GPU-accelerated)

    /// <summary>
    /// GPU-accelerated FlashAttention - memory-efficient O(N) attention algorithm.
    /// Falls back to CPU implementation when GPU is unavailable.
    /// </summary>
    public new Tensor<T> FlashAttention<T>(
        Tensor<T> query,
        Tensor<T> key,
        Tensor<T> value,
        double? scale,
        bool isCausal,
        out Tensor<T> softmaxStats)
    {
        if (!TryGetBackend(out var backend))
            return base.FlashAttention(query, key, value, scale, isCausal, out softmaxStats);

        // Validate tensor shapes [batch, heads, seq, head_dim]
        if (query.Rank != 4 || key.Rank != 4 || value.Rank != 4)
            return base.FlashAttention(query, key, value, scale, isCausal, out softmaxStats);

        int batch = query.Shape[0];
        int heads = query.Shape[1];
        int seqQ = query.Shape[2];
        int headDim = query.Shape[3];
        int seqK = key.Shape[2];

        // Compute scale if not provided
        float scaleFloat = (float)(scale ?? (1.0 / Math.Sqrt(headDim)));

        try
        {
            // Convert to float arrays
            float[] queryFloat = DirectGpuEngine.ToFloatArray(query.Data);
            float[] keyFloat = DirectGpuEngine.ToFloatArray(key.Data);
            float[] valueFloat = DirectGpuEngine.ToFloatArray(value.Data);

            // Allocate GPU buffers
            using var queryBuffer = backend.AllocateBuffer(queryFloat);
            using var keyBuffer = backend.AllocateBuffer(keyFloat);
            using var valueBuffer = backend.AllocateBuffer(valueFloat);
            using var outputBuffer = backend.AllocateBuffer(batch * heads * seqQ * headDim);
            using var statsBuffer = backend.AllocateBuffer(batch * heads * seqQ);

            // Execute GPU FlashAttention
            backend.FlashAttentionV2(queryBuffer, keyBuffer, valueBuffer, outputBuffer, statsBuffer,
                batch, heads, seqQ, seqK, headDim, scaleFloat, isCausal);

            backend.Synchronize();

            // Download results
            float[] outputFloat = new float[batch * heads * seqQ * headDim];
            float[] statsFloat = new float[batch * heads * seqQ];
            backend.DownloadBuffer(outputBuffer, outputFloat);
            backend.DownloadBuffer(statsBuffer, statsFloat);

            // Convert back to T
            T[] outputData = DirectGpuEngine.FromFloatArray<T>(outputFloat);
            T[] statsData = DirectGpuEngine.FromFloatArray<T>(statsFloat);

            softmaxStats = new Tensor<T>(statsData, new[] { batch, heads, seqQ });
            return new Tensor<T>(outputData, new[] { batch, heads, seqQ, headDim });
        }
        catch
        {
            // Fall back to CPU on any GPU error
            return base.FlashAttention(query, key, value, scale, isCausal, out softmaxStats);
        }
    }

    /// <summary>
    /// GPU-accelerated backward pass for FlashAttention.
    /// </summary>
    public new Tensor<T> FlashAttentionBackward<T>(
        Tensor<T> gradOutput,
        Tensor<T> query,
        Tensor<T> key,
        Tensor<T> value,
        Tensor<T> output,
        Tensor<T> softmaxStats,
        double scale,
        bool isCausal,
        out Tensor<T> gradQuery,
        out Tensor<T> gradKey,
        out Tensor<T> gradValue)
    {
        if (!TryGetBackend(out var backend))
            return base.FlashAttentionBackward(gradOutput, query, key, value, output, softmaxStats, scale, isCausal,
                out gradQuery, out gradKey, out gradValue);

        if (query.Rank != 4)
            return base.FlashAttentionBackward(gradOutput, query, key, value, output, softmaxStats, scale, isCausal,
                out gradQuery, out gradKey, out gradValue);

        int batch = query.Shape[0];
        int heads = query.Shape[1];
        int seqQ = query.Shape[2];
        int headDim = query.Shape[3];
        int seqK = key.Shape[2];

        try
        {
            // Convert to float arrays
            float[] gradOutFloat = DirectGpuEngine.ToFloatArray(gradOutput.Data);
            float[] queryFloat = DirectGpuEngine.ToFloatArray(query.Data);
            float[] keyFloat = DirectGpuEngine.ToFloatArray(key.Data);
            float[] valueFloat = DirectGpuEngine.ToFloatArray(value.Data);
            float[] outputFloat = DirectGpuEngine.ToFloatArray(output.Data);
            float[] statsFloat = DirectGpuEngine.ToFloatArray(softmaxStats.Data);

            // Allocate GPU buffers
            using var gradOutBuffer = backend.AllocateBuffer(gradOutFloat);
            using var queryBuffer = backend.AllocateBuffer(queryFloat);
            using var keyBuffer = backend.AllocateBuffer(keyFloat);
            using var valueBuffer = backend.AllocateBuffer(valueFloat);
            using var outputBuffer = backend.AllocateBuffer(outputFloat);
            using var statsBuffer = backend.AllocateBuffer(statsFloat);
            using var gradQBuffer = backend.AllocateBuffer(batch * heads * seqQ * headDim);
            using var gradKBuffer = backend.AllocateBuffer(batch * heads * seqK * headDim);
            using var gradVBuffer = backend.AllocateBuffer(batch * heads * seqK * headDim);

            // Execute GPU backward
            backend.FlashAttentionBackward(gradOutBuffer, queryBuffer, keyBuffer, valueBuffer,
                outputBuffer, statsBuffer, gradQBuffer, gradKBuffer, gradVBuffer,
                batch, heads, seqQ, seqK, headDim, (float)scale, isCausal);

            backend.Synchronize();

            // Download results
            float[] gradQFloat = new float[batch * heads * seqQ * headDim];
            float[] gradKFloat = new float[batch * heads * seqK * headDim];
            float[] gradVFloat = new float[batch * heads * seqK * headDim];
            backend.DownloadBuffer(gradQBuffer, gradQFloat);
            backend.DownloadBuffer(gradKBuffer, gradKFloat);
            backend.DownloadBuffer(gradVBuffer, gradVFloat);

            // Convert back to T
            gradQuery = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(gradQFloat), query.Shape.ToArray());
            gradKey = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(gradKFloat), key.Shape.ToArray());
            gradValue = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(gradVFloat), value.Shape.ToArray());

            return gradOutput;
        }
        catch
        {
            return base.FlashAttentionBackward(gradOutput, query, key, value, output, softmaxStats, scale, isCausal,
                out gradQuery, out gradKey, out gradValue);
        }
    }

    /// <summary>
    /// GPU-accelerated Grouped Query Attention for efficient inference.
    /// Falls back to CPU implementation when GPU is unavailable.
    /// </summary>
    public new Tensor<T> GroupedQueryAttention<T>(
        Tensor<T> query,
        Tensor<T> key,
        Tensor<T> value,
        int numQueriesPerKV,
        double? scale,
        bool isCausal,
        out Tensor<T> attentionWeights)
    {
        if (!TryGetBackend(out var backend))
            return base.GroupedQueryAttention(query, key, value, numQueriesPerKV, scale, isCausal, out attentionWeights);

        // Validate tensor shapes
        if (query.Rank != 4 || key.Rank != 4 || value.Rank != 4)
            return base.GroupedQueryAttention(query, key, value, numQueriesPerKV, scale, isCausal, out attentionWeights);

        int batch = query.Shape[0];
        int numQHeads = query.Shape[1];
        int seqQ = query.Shape[2];
        int headDim = query.Shape[3];
        int numKVHeads = key.Shape[1];
        int seqK = key.Shape[2];

        float scaleFloat = (float)(scale ?? (1.0 / Math.Sqrt(headDim)));

        try
        {
            // Convert to float arrays
            float[] queryFloat = DirectGpuEngine.ToFloatArray(query.Data);
            float[] keyFloat = DirectGpuEngine.ToFloatArray(key.Data);
            float[] valueFloat = DirectGpuEngine.ToFloatArray(value.Data);

            // Allocate GPU buffers
            using var queryBuffer = backend.AllocateBuffer(queryFloat);
            using var keyBuffer = backend.AllocateBuffer(keyFloat);
            using var valueBuffer = backend.AllocateBuffer(valueFloat);
            using var outputBuffer = backend.AllocateBuffer(batch * numQHeads * seqQ * headDim);
            using var attnWeightsBuffer = backend.AllocateBuffer(batch * numQHeads * seqQ * seqK);

            // Execute GPU GQA
            backend.GroupedQueryAttention(queryBuffer, keyBuffer, valueBuffer, outputBuffer, attnWeightsBuffer,
                batch, numQHeads, numKVHeads, seqQ, seqK, headDim, scaleFloat, isCausal);

            backend.Synchronize();

            // Download results
            float[] outputFloat = new float[batch * numQHeads * seqQ * headDim];
            float[] attnWeightsFloat = new float[batch * numQHeads * seqQ * seqK];
            backend.DownloadBuffer(outputBuffer, outputFloat);
            backend.DownloadBuffer(attnWeightsBuffer, attnWeightsFloat);

            // Convert back to T
            T[] outputData = DirectGpuEngine.FromFloatArray<T>(outputFloat);
            T[] attnWeightsData = DirectGpuEngine.FromFloatArray<T>(attnWeightsFloat);

            attentionWeights = new Tensor<T>(attnWeightsData, new[] { batch, numQHeads, seqQ, seqK });
            return new Tensor<T>(outputData, new[] { batch, numQHeads, seqQ, headDim });
        }
        catch
        {
            return base.GroupedQueryAttention(query, key, value, numQueriesPerKV, scale, isCausal, out attentionWeights);
        }
    }

    /// <summary>
    /// GPU-accelerated backward pass for Grouped Query Attention.
    /// </summary>
    public new Tensor<T> GroupedQueryAttentionBackward<T>(
        Tensor<T> gradOutput,
        Tensor<T> query,
        Tensor<T> key,
        Tensor<T> value,
        Tensor<T> attentionWeights,
        int numQueriesPerKV,
        double scale,
        out Tensor<T> gradQuery,
        out Tensor<T> gradKey,
        out Tensor<T> gradValue)
    {
        if (!TryGetBackend(out var backend))
            return base.GroupedQueryAttentionBackward(gradOutput, query, key, value, attentionWeights, numQueriesPerKV, scale,
                out gradQuery, out gradKey, out gradValue);

        if (query.Rank != 4)
            return base.GroupedQueryAttentionBackward(gradOutput, query, key, value, attentionWeights, numQueriesPerKV, scale,
                out gradQuery, out gradKey, out gradValue);

        int batch = query.Shape[0];
        int numQHeads = query.Shape[1];
        int seqQ = query.Shape[2];
        int headDim = query.Shape[3];
        int numKVHeads = key.Shape[1];
        int seqK = key.Shape[2];

        try
        {
            // Convert to float arrays
            float[] gradOutFloat = DirectGpuEngine.ToFloatArray(gradOutput.Data);
            float[] queryFloat = DirectGpuEngine.ToFloatArray(query.Data);
            float[] keyFloat = DirectGpuEngine.ToFloatArray(key.Data);
            float[] valueFloat = DirectGpuEngine.ToFloatArray(value.Data);
            float[] attnWeightsFloat = DirectGpuEngine.ToFloatArray(attentionWeights.Data);

            // Allocate GPU buffers
            using var gradOutBuffer = backend.AllocateBuffer(gradOutFloat);
            using var queryBuffer = backend.AllocateBuffer(queryFloat);
            using var keyBuffer = backend.AllocateBuffer(keyFloat);
            using var valueBuffer = backend.AllocateBuffer(valueFloat);
            using var attnWeightsBuffer = backend.AllocateBuffer(attnWeightsFloat);
            using var gradQBuffer = backend.AllocateBuffer(batch * numQHeads * seqQ * headDim);
            using var gradKBuffer = backend.AllocateBuffer(batch * numKVHeads * seqK * headDim);
            using var gradVBuffer = backend.AllocateBuffer(batch * numKVHeads * seqK * headDim);

            // Execute GPU backward
            backend.GroupedQueryAttentionBackward(gradOutBuffer, queryBuffer, keyBuffer, valueBuffer,
                attnWeightsBuffer, gradQBuffer, gradKBuffer, gradVBuffer,
                batch, numQHeads, numKVHeads, seqQ, seqK, headDim, (float)scale);

            backend.Synchronize();

            // Download results
            float[] gradQFloat = new float[batch * numQHeads * seqQ * headDim];
            float[] gradKFloat = new float[batch * numKVHeads * seqK * headDim];
            float[] gradVFloat = new float[batch * numKVHeads * seqK * headDim];
            backend.DownloadBuffer(gradQBuffer, gradQFloat);
            backend.DownloadBuffer(gradKBuffer, gradKFloat);
            backend.DownloadBuffer(gradVBuffer, gradVFloat);

            // Convert back to T
            gradQuery = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(gradQFloat), query.Shape.ToArray());
            gradKey = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(gradKFloat), key.Shape.ToArray());
            gradValue = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(gradVFloat), value.Shape.ToArray());

            return gradOutput;
        }
        catch
        {
            return base.GroupedQueryAttentionBackward(gradOutput, query, key, value, attentionWeights, numQueriesPerKV, scale,
                out gradQuery, out gradKey, out gradValue);
        }
    }

    #endregion

    #region Persistent Tensor Management

    /// <summary>
    /// Registers a tensor for GPU memory optimization by pre-allocating and uploading
    /// its data to GPU memory. This eliminates repeated CPU-GPU transfers for tensors
    /// that are reused across multiple operations (e.g., layer weights, biases).
    /// </summary>
    public new void RegisterPersistentTensor<T>(Tensor<T> tensor, PersistentTensorRole role)
    {
        base.RegisterPersistentTensor(tensor, role);

        if (!TryGetBackend(out var backend))
            return;

        // Use the tensor's data array as the cache key
        object key = tensor.Data;

        // Check if already registered
        if (_persistentBufferCache.ContainsKey(key))
            return;

        try
        {
            // Convert tensor data to float and upload to GPU
            float[] floatData = DirectGpuEngine.ToFloatArray(tensor.Data);
            IGpuBuffer gpuBuffer = backend.AllocateBuffer(floatData);
            backend.Synchronize();

            var entry = new GpuBufferCacheEntry(gpuBuffer, role);
            _persistentBufferCache.TryAdd(key, entry);
            _tensorVersions.TryAdd(key, 0);
        }
        catch
        {
            // Silently ignore GPU allocation failures - operations will fall back to CPU
        }
    }

    /// <summary>
    /// Unregisters a persistent tensor and releases its associated GPU memory.
    /// </summary>
    public new void UnregisterPersistentTensor<T>(Tensor<T> tensor)
    {
        base.UnregisterPersistentTensor(tensor);

        object key = tensor.Data;

        if (_persistentBufferCache.TryRemove(key, out var entry))
        {
            entry.Dispose();
        }
        _tensorVersions.TryRemove(key, out _);
    }

    /// <summary>
    /// Invalidates a persistent tensor's GPU buffer, triggering re-upload of its
    /// data to GPU memory. Call this after modifying the tensor's data on CPU.
    /// </summary>
    public new void InvalidatePersistentTensor<T>(Tensor<T> tensor)
    {
        base.InvalidatePersistentTensor(tensor);

        if (!TryGetBackend(out var backend))
            return;

        object key = tensor.Data;

        if (!_persistentBufferCache.TryGetValue(key, out var entry))
            return;

        try
        {
            // Dispose old buffer
            entry.Buffer.Dispose();

            // Upload new data
            float[] floatData = DirectGpuEngine.ToFloatArray(tensor.Data);
            IGpuBuffer newBuffer = backend.AllocateBuffer(floatData);
            backend.Synchronize();

            // Update cache entry with new buffer
            var newEntry = new GpuBufferCacheEntry(newBuffer, entry.Role);
            newEntry.Version = entry.Version + 1;

            _persistentBufferCache[key] = newEntry;
            _tensorVersions[key] = newEntry.Version;
        }
        catch
        {
            // On failure, remove from cache - operations will fall back to CPU
            _persistentBufferCache.TryRemove(key, out _);
            _tensorVersions.TryRemove(key, out _);
        }
    }

    /// <summary>
    /// Attempts to get a cached GPU buffer for a tensor.
    /// Returns null if the tensor is not registered as persistent.
    /// </summary>
    internal IGpuBuffer? TryGetCachedBuffer<T>(T[] tensorData)
    {
        if (_persistentBufferCache.TryGetValue(tensorData, out var entry))
        {
            return entry.Buffer;
        }
        return null;
    }

    /// <summary>
    /// Gets the number of tensors currently cached on GPU.
    /// </summary>
    public int CachedTensorCount => _persistentBufferCache.Count;

    #endregion

    public void Dispose()
    {
        // Dispose all cached GPU buffers
        foreach (var entry in _persistentBufferCache.Values)
        {
            entry.Dispose();
        }
        _persistentBufferCache.Clear();
        _tensorVersions.Clear();

        if (_ownsDirectGpu)
            _directGpu?.Dispose();
    }
}
