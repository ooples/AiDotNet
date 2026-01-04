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

    /// <summary>
    /// Helper struct for tracking GPU buffer ownership. Implements IDisposable
    /// to only dispose buffers we own (not cached ones).
    /// </summary>
    private readonly struct OwnedBuffer : IDisposable
    {
        private readonly IGpuBuffer _buffer;
        private readonly bool _ownsBuffer;

        /// <summary>
        /// Gets the underlying GPU buffer.
        /// </summary>
        public IGpuBuffer Buffer => _buffer;

        public OwnedBuffer(IGpuBuffer buffer, bool ownsBuffer)
        {
            _buffer = buffer;
            _ownsBuffer = ownsBuffer;
        }

        public void Dispose()
        {
            if (_ownsBuffer)
                _buffer.Dispose();
        }
    }

    /// <summary>
    /// Gets a GPU buffer for the tensor data, using cache if available.
    /// Returns an OwnedBuffer that only disposes if we allocated it (not cached).
    /// </summary>
    private OwnedBuffer GetOrAllocateBuffer<T>(IDirectGpuBackend backend, T[] data)
    {
        var cached = TryGetCachedBuffer(data);
        if (cached != null)
            return new OwnedBuffer(cached, ownsBuffer: false);

        float[] floatData = DirectGpuEngine.ToFloatArray(data);
        return new OwnedBuffer(backend.AllocateBuffer(floatData), ownsBuffer: true);
    }

    /// <summary>
    /// Allocates a new output buffer (always owned, never cached).
    /// </summary>
    private static OwnedBuffer AllocateOutputBuffer(IDirectGpuBackend backend, int size)
    {
        return new OwnedBuffer(backend.AllocateBuffer(size), ownsBuffer: true);
    }

    private T[]? TryRunUnary<T>(T[] input, Action<IDirectGpuBackend, IGpuBuffer, IGpuBuffer, int> op)
    {
        if (!TryGetBackend(out var backend))
            return null;

        using var bufferA = GetOrAllocateBuffer(backend, input);
        using var bufferB = AllocateOutputBuffer(backend, input.Length);
        op(backend, bufferA.Buffer, bufferB.Buffer, input.Length);
        backend.Synchronize();
        float[] resultFloat = backend.DownloadBuffer(bufferB.Buffer);
        return DirectGpuEngine.FromFloatArray<T>(resultFloat);
    }

    private T[]? TryRunBinary<T>(T[] left, T[] right, Action<IDirectGpuBackend, IGpuBuffer, IGpuBuffer, IGpuBuffer, int> op)
    {
        if (!TryGetBackend(out var backend))
            return null;
        if (left.Length != right.Length)
            return null;

        using var bufferA = GetOrAllocateBuffer(backend, left);
        using var bufferB = GetOrAllocateBuffer(backend, right);
        using var bufferC = AllocateOutputBuffer(backend, left.Length);
        op(backend, bufferA.Buffer, bufferB.Buffer, bufferC.Buffer, left.Length);
        backend.Synchronize();
        float[] resultFloat = backend.DownloadBuffer(bufferC.Buffer);
        return DirectGpuEngine.FromFloatArray<T>(resultFloat);
    }

    private T[]? TryRunScalar<T>(T[] input, T scalar, Action<IDirectGpuBackend, IGpuBuffer, IGpuBuffer, float, int> op)
    {
        if (!TryGetBackend(out var backend))
            return null;

        using var bufferA = GetOrAllocateBuffer(backend, input);
        using var bufferB = AllocateOutputBuffer(backend, input.Length);
        op(backend, bufferA.Buffer, bufferB.Buffer, ToFloatScalar(scalar), input.Length);
        backend.Synchronize();
        float[] resultFloat = backend.DownloadBuffer(bufferB.Buffer);
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

        using var bufferA = GetOrAllocateBuffer(backend, tensor.Data);
        backend.Synchronize();
        float sum = backend.Sum(bufferA.Buffer, tensor.Length);
        return FromFloatScalar<T>(sum);
    }

    T IEngine.TensorMaxValue<T>(Tensor<T> tensor)
    {
        if (!TryGetBackend(out var backend))
            return base.TensorMaxValue(tensor);

        using var bufferA = GetOrAllocateBuffer(backend, tensor.Data);
        backend.Synchronize();
        float max = backend.Max(bufferA.Buffer, tensor.Length);
        return FromFloatScalar<T>(max);
    }

    #region Fused Operations

    /// <summary>
    /// GPU-accelerated fused linear transformation: output = activation(input @ weights + bias).
    /// Uses cached GPU buffers for registered persistent tensors (weights/bias) to avoid
    /// redundant CPU→GPU transfers on every forward pass.
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

        // Use cache-aware buffer allocation (OwnedBuffer auto-disposes only if we allocated)
        using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
        using var weightsBuffer = GetOrAllocateBuffer(backend, weights.Data);
        using var biasBuffer = bias != null ? GetOrAllocateBuffer(backend, bias.Data) : default;

        try
        {
            IGpuBuffer resultBuffer;

            // Use fused GPU kernels when available
            // Only use GPU path for natively supported fused ops (with bias)
            // For other cases, fall back to CPU which handles all combinations
            if (bias != null && (activation == FusedActivationType.ReLU ||
                                 activation == FusedActivationType.GELU ||
                                 activation == FusedActivationType.Sigmoid ||
                                 activation == FusedActivationType.Tanh))
            {
                resultBuffer = activation switch
                {
                    FusedActivationType.ReLU => backend.GemmBiasRelu(inputBuffer.Buffer, weightsBuffer.Buffer, biasBuffer.Buffer, batchSize, outputFeatures, inputFeatures),
                    FusedActivationType.GELU => backend.GemmBiasGelu(inputBuffer.Buffer, weightsBuffer.Buffer, biasBuffer.Buffer, batchSize, outputFeatures, inputFeatures),
                    FusedActivationType.Sigmoid => backend.GemmBiasSigmoid(inputBuffer.Buffer, weightsBuffer.Buffer, biasBuffer.Buffer, batchSize, outputFeatures, inputFeatures),
                    _ => backend.GemmBiasTanh(inputBuffer.Buffer, weightsBuffer.Buffer, biasBuffer.Buffer, batchSize, outputFeatures, inputFeatures),
                };
            }
            else if (bias == null && activation == FusedActivationType.None)
            {
                // Simple MatMul only - use GPU
                resultBuffer = backend.MatMul(inputBuffer.Buffer, weightsBuffer.Buffer, batchSize, outputFeatures, inputFeatures);
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
    /// Uses cached GPU buffers for registered persistent tensors (kernel/bias) to avoid
    /// redundant CPU→GPU transfers on every forward pass.
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

        // Use cache-aware buffer allocation (OwnedBuffer auto-disposes only if we allocated)
        using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
        using var kernelBuffer = GetOrAllocateBuffer(backend, kernel.Data);
        using var outputBuffer = AllocateOutputBuffer(backend, batch * outChannels * outHeight * outWidth);

        try
        {
            // Execute GPU convolution
            backend.Conv2D(inputBuffer.Buffer, kernelBuffer.Buffer, outputBuffer.Buffer,
                batch, inChannels, inHeight, inWidth,
                outChannels, outHeight, outWidth,
                kernelH, kernelW,
                strideH, strideW, padH, padW,
                dilationH, dilationW);

            // Add bias if present
            if (bias != null)
            {
                // Bias is added per output channel, broadcast across batch and spatial dimensions
                int outputSize = batch * outChannels * outHeight * outWidth;
                int spatialSize = outHeight * outWidth;

                // Download, add bias, re-upload (GPU bias broadcast kernel would be more efficient)
                float[] outputFloat = new float[outputSize];
                backend.DownloadBuffer(outputBuffer.Buffer, outputFloat);

                // Get bias data (check cache first)
                using var biasBuffer = GetOrAllocateBuffer(backend, bias.Data);
                float[] biasFloat = new float[bias.Length];
                backend.DownloadBuffer(biasBuffer.Buffer, biasFloat);

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
                    ApplyGpuActivation(backend, outputBuffer.Buffer, outputSize, activation);
                }

                backend.Synchronize();

                float[] resultFloat = new float[outputSize];
                backend.DownloadBuffer(outputBuffer.Buffer, resultFloat);

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
    /// GPU-accelerated fused 3D convolution with activation.
    /// Currently delegates to CPU implementation as GPU Conv3D kernel is not yet available.
    /// </summary>
    public new Tensor<T> FusedConv3D<T>(
        Tensor<T> input,
        Tensor<T> kernel,
        Tensor<T>? bias,
        int strideD, int strideH, int strideW,
        int padD, int padH, int padW,
        int dilationD, int dilationH, int dilationW,
        FusedActivationType activation)
    {
        // TODO: Implement GPU-accelerated Conv3D when backend support is available
        // For now, delegate to CPU implementation which handles the sequential operations
        return base.FusedConv3D(input, kernel, bias, strideD, strideH, strideW, padD, padH, padW, dilationD, dilationH, dilationW, activation);
    }

    /// <summary>
    /// GPU-accelerated fused transposed 2D convolution with activation.
    /// Currently delegates to CPU implementation as GPU ConvTranspose2D kernel is not yet available.
    /// </summary>
    public new Tensor<T> FusedConvTranspose2D<T>(
        Tensor<T> input,
        Tensor<T> kernel,
        Tensor<T>? bias,
        int strideH, int strideW,
        int padH, int padW,
        int outputPadH, int outputPadW,
        FusedActivationType activation)
    {
        // TODO: Implement GPU-accelerated ConvTranspose2D when backend support is available
        // For now, delegate to CPU implementation which handles the sequential operations
        return base.FusedConvTranspose2D(input, kernel, bias, strideH, strideW, padH, padW, outputPadH, outputPadW, activation);
    }

    /// <summary>
    /// GPU-accelerated fused batch normalization with activation.
    /// Uses cached GPU buffers for registered persistent tensors (gamma/beta/running stats)
    /// to avoid redundant CPU→GPU transfers on every forward pass.
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

        // Use cache-aware buffer allocation (OwnedBuffer auto-disposes only if we allocated)
        using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
        using var outputBuffer = AllocateOutputBuffer(backend, batchSize * features);
        using var saveMeanBuffer = AllocateOutputBuffer(backend, features);
        using var saveVarBuffer = AllocateOutputBuffer(backend, features);
        using var gammaBuffer = GetOrAllocateBuffer(backend, gamma.Data);
        using var betaBuffer = GetOrAllocateBuffer(backend, beta.Data);
        using var runningMeanBuffer = GetOrAllocateBuffer(backend, runningMean.Data);
        using var runningVarBuffer = GetOrAllocateBuffer(backend, runningVar.Data);

        try
        {
            // Execute batch norm (spatialSize=1 for 2D tensors)
            backend.BatchNorm(inputBuffer.Buffer, outputBuffer.Buffer, gammaBuffer.Buffer, betaBuffer.Buffer,
                runningMeanBuffer.Buffer, runningVarBuffer.Buffer, saveMeanBuffer.Buffer, saveVarBuffer.Buffer,
                batchSize, features, 1, (float)epsilon, (float)momentum, training);

            // Apply activation if needed
            if (activation != FusedActivationType.None)
            {
                ApplyGpuActivation(backend, outputBuffer.Buffer, batchSize * features, activation);
            }

            backend.Synchronize();

            // Download results
            float[] resultFloat = new float[batchSize * features];
            float[] saveMeanFloat = new float[features];
            float[] saveVarFloat = new float[features];

            backend.DownloadBuffer(outputBuffer.Buffer, resultFloat);
            backend.DownloadBuffer(saveMeanBuffer.Buffer, saveMeanFloat);
            backend.DownloadBuffer(saveVarBuffer.Buffer, saveVarFloat);

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
    /// Uses cached GPU buffers for registered persistent tensors (e.g., KV cache) to avoid
    /// redundant CPU→GPU transfers on every forward pass.
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

        // Use cache-aware buffer allocation (especially important for KV cache)
        using var queryBuffer = GetOrAllocateBuffer(backend, query.Data);
        using var keyBuffer = GetOrAllocateBuffer(backend, key.Data);
        using var valueBuffer = GetOrAllocateBuffer(backend, value.Data);
        using var outputBuffer = AllocateOutputBuffer(backend, batch * heads * seqQ * headDim);
        using var statsBuffer = AllocateOutputBuffer(backend, batch * heads * seqQ);

        try
        {
            // Execute GPU FlashAttention
            backend.FlashAttentionV2(queryBuffer.Buffer, keyBuffer.Buffer, valueBuffer.Buffer, outputBuffer.Buffer, statsBuffer.Buffer,
                batch, heads, seqQ, seqK, headDim, scaleFloat, isCausal);

            backend.Synchronize();

            // Download results
            float[] outputFloat = new float[batch * heads * seqQ * headDim];
            float[] statsFloat = new float[batch * heads * seqQ];
            backend.DownloadBuffer(outputBuffer.Buffer, outputFloat);
            backend.DownloadBuffer(statsBuffer.Buffer, statsFloat);

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
    /// Uses cached GPU buffers for registered persistent tensors to avoid redundant transfers.
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

        // Use cache-aware buffer allocation
        using var gradOutBuffer = GetOrAllocateBuffer(backend, gradOutput.Data);
        using var queryBuffer = GetOrAllocateBuffer(backend, query.Data);
        using var keyBuffer = GetOrAllocateBuffer(backend, key.Data);
        using var valueBuffer = GetOrAllocateBuffer(backend, value.Data);
        using var outputBuffer = GetOrAllocateBuffer(backend, output.Data);
        using var statsBuffer = GetOrAllocateBuffer(backend, softmaxStats.Data);
        using var gradQBuffer = AllocateOutputBuffer(backend, batch * heads * seqQ * headDim);
        using var gradKBuffer = AllocateOutputBuffer(backend, batch * heads * seqK * headDim);
        using var gradVBuffer = AllocateOutputBuffer(backend, batch * heads * seqK * headDim);

        try
        {
            // Execute GPU backward
            backend.FlashAttentionBackward(gradOutBuffer.Buffer, queryBuffer.Buffer, keyBuffer.Buffer, valueBuffer.Buffer,
                outputBuffer.Buffer, statsBuffer.Buffer, gradQBuffer.Buffer, gradKBuffer.Buffer, gradVBuffer.Buffer,
                batch, heads, seqQ, seqK, headDim, (float)scale, isCausal);

            backend.Synchronize();

            // Download results
            float[] gradQFloat = new float[batch * heads * seqQ * headDim];
            float[] gradKFloat = new float[batch * heads * seqK * headDim];
            float[] gradVFloat = new float[batch * heads * seqK * headDim];
            backend.DownloadBuffer(gradQBuffer.Buffer, gradQFloat);
            backend.DownloadBuffer(gradKBuffer.Buffer, gradKFloat);
            backend.DownloadBuffer(gradVBuffer.Buffer, gradVFloat);

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
    /// Uses cached GPU buffers for registered persistent tensors (e.g., KV cache) to avoid
    /// redundant CPU→GPU transfers on every forward pass.
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

        // Use cache-aware buffer allocation (especially important for KV cache)
        using var queryBuffer = GetOrAllocateBuffer(backend, query.Data);
        using var keyBuffer = GetOrAllocateBuffer(backend, key.Data);
        using var valueBuffer = GetOrAllocateBuffer(backend, value.Data);
        using var outputBuffer = AllocateOutputBuffer(backend, batch * numQHeads * seqQ * headDim);
        using var attnWeightsBuffer = AllocateOutputBuffer(backend, batch * numQHeads * seqQ * seqK);

        try
        {
            // Execute GPU GQA
            backend.GroupedQueryAttention(queryBuffer.Buffer, keyBuffer.Buffer, valueBuffer.Buffer, outputBuffer.Buffer, attnWeightsBuffer.Buffer,
                batch, numQHeads, numKVHeads, seqQ, seqK, headDim, scaleFloat, isCausal);

            backend.Synchronize();

            // Download results
            float[] outputFloat = new float[batch * numQHeads * seqQ * headDim];
            float[] attnWeightsFloat = new float[batch * numQHeads * seqQ * seqK];
            backend.DownloadBuffer(outputBuffer.Buffer, outputFloat);
            backend.DownloadBuffer(attnWeightsBuffer.Buffer, attnWeightsFloat);

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
    /// Uses cached GPU buffers for registered persistent tensors to avoid redundant transfers.
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

        // Use cache-aware buffer allocation
        using var gradOutBuffer = GetOrAllocateBuffer(backend, gradOutput.Data);
        using var queryBuffer = GetOrAllocateBuffer(backend, query.Data);
        using var keyBuffer = GetOrAllocateBuffer(backend, key.Data);
        using var valueBuffer = GetOrAllocateBuffer(backend, value.Data);
        using var attnWeightsBuffer = GetOrAllocateBuffer(backend, attentionWeights.Data);
        using var gradQBuffer = AllocateOutputBuffer(backend, batch * numQHeads * seqQ * headDim);
        using var gradKBuffer = AllocateOutputBuffer(backend, batch * numKVHeads * seqK * headDim);
        using var gradVBuffer = AllocateOutputBuffer(backend, batch * numKVHeads * seqK * headDim);

        try
        {
            // Execute GPU backward
            backend.GroupedQueryAttentionBackward(gradOutBuffer.Buffer, queryBuffer.Buffer, keyBuffer.Buffer, valueBuffer.Buffer,
                attnWeightsBuffer.Buffer, gradQBuffer.Buffer, gradKBuffer.Buffer, gradVBuffer.Buffer,
                batch, numQHeads, numKVHeads, seqQ, seqK, headDim, (float)scale);

            backend.Synchronize();

            // Download results
            float[] gradQFloat = new float[batch * numQHeads * seqQ * headDim];
            float[] gradKFloat = new float[batch * numKVHeads * seqK * headDim];
            float[] gradVFloat = new float[batch * numKVHeads * seqK * headDim];
            backend.DownloadBuffer(gradQBuffer.Buffer, gradQFloat);
            backend.DownloadBuffer(gradKBuffer.Buffer, gradKFloat);
            backend.DownloadBuffer(gradVBuffer.Buffer, gradVFloat);

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

    #region FFT Operations (GPU-accelerated)

    /// <summary>
    /// GPU-accelerated 1D complex-to-complex FFT.
    /// </summary>
    void IEngine.FFT<T>(Tensor<T> inputReal, Tensor<T> inputImag, out Tensor<T> outputReal, out Tensor<T> outputImag)
    {
        if (!TryGetBackend(out var backend) || inputReal.Length != inputImag.Length)
        {
            base.FFT(inputReal, inputImag, out outputReal, out outputImag);
            return;
        }

        int n = inputReal.Shape[^1];
        if ((n & (n - 1)) != 0) // Not power of 2
        {
            base.FFT(inputReal, inputImag, out outputReal, out outputImag);
            return;
        }

        try
        {
            // Use cache-aware buffer allocation for inputs
            using var inputRealBuffer = GetOrAllocateBuffer(backend, inputReal.Data);
            using var inputImagBuffer = GetOrAllocateBuffer(backend, inputImag.Data);
            using var outputRealBuffer = AllocateOutputBuffer(backend, inputReal.Length);
            using var outputImagBuffer = AllocateOutputBuffer(backend, inputImag.Length);

            backend.FFT(inputRealBuffer.Buffer, inputImagBuffer.Buffer, outputRealBuffer.Buffer, outputImagBuffer.Buffer, n, inverse: false);
            backend.Synchronize();

            float[] outputRealFloat = new float[inputReal.Length];
            float[] outputImagFloat = new float[inputImag.Length];
            backend.DownloadBuffer(outputRealBuffer.Buffer, outputRealFloat);
            backend.DownloadBuffer(outputImagBuffer.Buffer, outputImagFloat);

            outputReal = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(outputRealFloat), inputReal.Shape.ToArray());
            outputImag = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(outputImagFloat), inputImag.Shape.ToArray());
        }
        catch
        {
            base.FFT(inputReal, inputImag, out outputReal, out outputImag);
        }
    }

    /// <summary>
    /// GPU-accelerated 1D complex-to-complex inverse FFT.
    /// </summary>
    void IEngine.IFFT<T>(Tensor<T> inputReal, Tensor<T> inputImag, out Tensor<T> outputReal, out Tensor<T> outputImag)
    {
        if (!TryGetBackend(out var backend) || inputReal.Length != inputImag.Length)
        {
            base.IFFT(inputReal, inputImag, out outputReal, out outputImag);
            return;
        }

        int n = inputReal.Shape[^1];
        if ((n & (n - 1)) != 0)
        {
            base.IFFT(inputReal, inputImag, out outputReal, out outputImag);
            return;
        }

        try
        {
            // Use cache-aware buffer allocation for inputs
            using var inputRealBuffer = GetOrAllocateBuffer(backend, inputReal.Data);
            using var inputImagBuffer = GetOrAllocateBuffer(backend, inputImag.Data);
            using var outputRealBuffer = AllocateOutputBuffer(backend, inputReal.Length);
            using var outputImagBuffer = AllocateOutputBuffer(backend, inputImag.Length);

            backend.FFT(inputRealBuffer.Buffer, inputImagBuffer.Buffer, outputRealBuffer.Buffer, outputImagBuffer.Buffer, n, inverse: true);
            backend.Synchronize();

            float[] outputRealFloat = new float[inputReal.Length];
            float[] outputImagFloat = new float[inputImag.Length];
            backend.DownloadBuffer(outputRealBuffer.Buffer, outputRealFloat);
            backend.DownloadBuffer(outputImagBuffer.Buffer, outputImagFloat);

            outputReal = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(outputRealFloat), inputReal.Shape.ToArray());
            outputImag = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(outputImagFloat), inputImag.Shape.ToArray());
        }
        catch
        {
            base.IFFT(inputReal, inputImag, out outputReal, out outputImag);
        }
    }

    /// <summary>
    /// GPU-accelerated 2D FFT.
    /// </summary>
    void IEngine.FFT2D<T>(Tensor<T> inputReal, Tensor<T> inputImag, out Tensor<T> outputReal, out Tensor<T> outputImag)
    {
        if (!TryGetBackend(out var backend) || inputReal.Rank < 2 || inputReal.Length != inputImag.Length)
        {
            base.FFT2D(inputReal, inputImag, out outputReal, out outputImag);
            return;
        }

        int height = inputReal.Shape[^2];
        int width = inputReal.Shape[^1];

        if ((height & (height - 1)) != 0 || (width & (width - 1)) != 0)
        {
            base.FFT2D(inputReal, inputImag, out outputReal, out outputImag);
            return;
        }

        try
        {
            // Use cache-aware buffer allocation for inputs
            using var inputRealBuffer = GetOrAllocateBuffer(backend, inputReal.Data);
            using var inputImagBuffer = GetOrAllocateBuffer(backend, inputImag.Data);
            using var outputRealBuffer = AllocateOutputBuffer(backend, inputReal.Length);
            using var outputImagBuffer = AllocateOutputBuffer(backend, inputImag.Length);

            backend.FFT2D(inputRealBuffer.Buffer, inputImagBuffer.Buffer, outputRealBuffer.Buffer, outputImagBuffer.Buffer, height, width, inverse: false);
            backend.Synchronize();

            float[] outputRealFloat = new float[inputReal.Length];
            float[] outputImagFloat = new float[inputImag.Length];
            backend.DownloadBuffer(outputRealBuffer.Buffer, outputRealFloat);
            backend.DownloadBuffer(outputImagBuffer.Buffer, outputImagFloat);

            outputReal = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(outputRealFloat), inputReal.Shape.ToArray());
            outputImag = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(outputImagFloat), inputImag.Shape.ToArray());
        }
        catch
        {
            base.FFT2D(inputReal, inputImag, out outputReal, out outputImag);
        }
    }

    /// <summary>
    /// GPU-accelerated 2D inverse FFT.
    /// </summary>
    void IEngine.IFFT2D<T>(Tensor<T> inputReal, Tensor<T> inputImag, out Tensor<T> outputReal, out Tensor<T> outputImag)
    {
        if (!TryGetBackend(out var backend) || inputReal.Rank < 2 || inputReal.Length != inputImag.Length)
        {
            base.IFFT2D(inputReal, inputImag, out outputReal, out outputImag);
            return;
        }

        int height = inputReal.Shape[^2];
        int width = inputReal.Shape[^1];

        if ((height & (height - 1)) != 0 || (width & (width - 1)) != 0)
        {
            base.IFFT2D(inputReal, inputImag, out outputReal, out outputImag);
            return;
        }

        try
        {
            // Use cache-aware buffer allocation for inputs
            using var inputRealBuffer = GetOrAllocateBuffer(backend, inputReal.Data);
            using var inputImagBuffer = GetOrAllocateBuffer(backend, inputImag.Data);
            using var outputRealBuffer = AllocateOutputBuffer(backend, inputReal.Length);
            using var outputImagBuffer = AllocateOutputBuffer(backend, inputImag.Length);

            backend.FFT2D(inputRealBuffer.Buffer, inputImagBuffer.Buffer, outputRealBuffer.Buffer, outputImagBuffer.Buffer, height, width, inverse: true);
            backend.Synchronize();

            float[] outputRealFloat = new float[inputReal.Length];
            float[] outputImagFloat = new float[inputImag.Length];
            backend.DownloadBuffer(outputRealBuffer.Buffer, outputRealFloat);
            backend.DownloadBuffer(outputImagBuffer.Buffer, outputImagFloat);

            outputReal = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(outputRealFloat), inputReal.Shape.ToArray());
            outputImag = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(outputImagFloat), inputImag.Shape.ToArray());
        }
        catch
        {
            base.IFFT2D(inputReal, inputImag, out outputReal, out outputImag);
        }
    }

    /// <summary>
    /// GPU-accelerated Short-Time Fourier Transform.
    /// </summary>
    void IEngine.STFT<T>(
        Tensor<T> input,
        int nFft,
        int hopLength,
        Tensor<T> window,
        bool center,
        out Tensor<T> magnitudeOut,
        out Tensor<T> phaseOut)
    {
        if (!TryGetBackend(out var backend) || (nFft & (nFft - 1)) != 0)
        {
            base.STFT(input, nFft, hopLength, window, center, out magnitudeOut, out phaseOut);
            return;
        }

        try
        {
            // For STFT, we need to process frame by frame
            // First, handle centering by padding the input
            T[] inputData = input.Data;
            if (center)
            {
                int padAmount = nFft / 2;
                T[] paddedData = new T[inputData.Length + 2 * padAmount];
                Array.Copy(inputData, 0, paddedData, padAmount, inputData.Length);
                inputData = paddedData;
            }

            int numSamples = inputData.Length;
            int numFrames = (numSamples - nFft) / hopLength + 1;
            int numFreqs = nFft / 2 + 1;

            if (numFrames <= 0)
            {
                base.STFT(input, nFft, hopLength, window, center, out magnitudeOut, out phaseOut);
                return;
            }

            float[] inputFloat = DirectGpuEngine.ToFloatArray(inputData);

            // Use cache-aware allocation for window (likely persistent)
            using var windowBuffer = GetOrAllocateBuffer(backend, window.Data);
            // Allocate working buffers
            using var frameBuffer = AllocateOutputBuffer(backend, nFft);
            using var windowedBuffer = AllocateOutputBuffer(backend, nFft);
            using var fftRealBuffer = AllocateOutputBuffer(backend, nFft);
            using var fftImagBuffer = AllocateOutputBuffer(backend, nFft);
            using var zeroBuffer = AllocateOutputBuffer(backend, nFft);

            float[] magnitudeData = new float[numFrames * numFreqs];
            float[] phaseData = new float[numFrames * numFreqs];

            for (int frame = 0; frame < numFrames; frame++)
            {
                int frameStart = frame * hopLength;

                // Extract frame from input
                float[] frameData = new float[nFft];
                Array.Copy(inputFloat, frameStart, frameData, 0, Math.Min(nFft, inputFloat.Length - frameStart));

                // Upload frame data
                using var currentFrameBuffer = backend.AllocateBuffer(frameData);

                // Apply window
                backend.ApplyWindow(currentFrameBuffer, windowBuffer.Buffer, windowedBuffer.Buffer, nFft);

                // Perform FFT (windowed signal as real input, zeros as imaginary)
                backend.FFT(windowedBuffer.Buffer, zeroBuffer.Buffer, fftRealBuffer.Buffer, fftImagBuffer.Buffer, nFft, inverse: false);

                // Download FFT results
                float[] fftReal = new float[nFft];
                float[] fftImag = new float[nFft];
                backend.DownloadBuffer(fftRealBuffer.Buffer, fftReal);
                backend.DownloadBuffer(fftImagBuffer.Buffer, fftImag);

                // Compute magnitude and phase for positive frequencies only
                for (int k = 0; k < numFreqs; k++)
                {
                    float real = fftReal[k];
                    float imag = fftImag[k];
                    magnitudeData[frame * numFreqs + k] = (float)Math.Sqrt(real * real + imag * imag);
                    phaseData[frame * numFreqs + k] = (float)Math.Atan2(imag, real);
                }
            }
            backend.Synchronize();

            int[] outputShape = input.Rank == 1
                ? new[] { numFrames, numFreqs }
                : new[] { input.Shape[0], numFrames, numFreqs };

            magnitudeOut = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(magnitudeData), outputShape);
            phaseOut = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(phaseData), outputShape);
        }
        catch
        {
            base.STFT(input, nFft, hopLength, window, center, out magnitudeOut, out phaseOut);
        }
    }

    /// <summary>
    /// GPU-accelerated inverse Short-Time Fourier Transform.
    /// </summary>
    Tensor<T> IEngine.ISTFT<T>(
        Tensor<T> magnitude,
        Tensor<T> phase,
        int nFft,
        int hopLength,
        Tensor<T> window,
        bool center,
        int? length)
    {
        if (!TryGetBackend(out var backend) || (nFft & (nFft - 1)) != 0)
        {
            return base.ISTFT(magnitude, phase, nFft, hopLength, window, center, length);
        }

        try
        {
            int numFrames = magnitude.Shape[^2];
            int numFreqs = magnitude.Shape[^1];

            float[] magnitudeFloat = DirectGpuEngine.ToFloatArray(magnitude.Data);
            float[] phaseFloat = DirectGpuEngine.ToFloatArray(phase.Data);
            float[] windowFloat = DirectGpuEngine.ToFloatArray(window.Data);

            // Reconstruct full spectrum (mirror for negative frequencies)
            int outputSamples = (numFrames - 1) * hopLength + nFft;
            float[] output = new float[outputSamples];
            float[] windowSum = new float[outputSamples];

            // Use cache-aware allocation for window (likely persistent)
            using var windowBuffer = GetOrAllocateBuffer(backend, window.Data);
            // Allocate working buffers
            using var outputRealBuffer = AllocateOutputBuffer(backend, nFft);
            using var outputImagBuffer = AllocateOutputBuffer(backend, nFft);

            for (int frame = 0; frame < numFrames; frame++)
            {
                // Convert polar to complex for full spectrum
                float[] frameReal = new float[nFft];
                float[] frameImag = new float[nFft];

                // Fill positive frequencies
                for (int k = 0; k < numFreqs; k++)
                {
                    float mag = magnitudeFloat[frame * numFreqs + k];
                    float ph = phaseFloat[frame * numFreqs + k];
                    frameReal[k] = mag * (float)Math.Cos(ph);
                    frameImag[k] = mag * (float)Math.Sin(ph);
                }

                // Mirror for negative frequencies (conjugate symmetry)
                for (int k = 1; k < nFft - numFreqs + 1; k++)
                {
                    int srcIdx = numFreqs - 1 - k;
                    if (srcIdx > 0 && srcIdx < numFreqs)
                    {
                        frameReal[nFft - k] = frameReal[srcIdx];
                        frameImag[nFft - k] = -frameImag[srcIdx];
                    }
                }

                using var frameRealBuffer = backend.AllocateBuffer(frameReal);
                using var frameImagBuffer = backend.AllocateBuffer(frameImag);

                // Perform inverse FFT
                backend.FFT(frameRealBuffer, frameImagBuffer, outputRealBuffer.Buffer, outputImagBuffer.Buffer, nFft, inverse: true);

                // Download result
                float[] ifftResult = new float[nFft];
                backend.DownloadBuffer(outputRealBuffer.Buffer, ifftResult);

                // Overlap-add with window
                int frameStart = frame * hopLength;
                for (int i = 0; i < nFft && frameStart + i < outputSamples; i++)
                {
                    float w = windowFloat[i];
                    output[frameStart + i] += ifftResult[i] * w;
                    windowSum[frameStart + i] += w * w;
                }
            }

            backend.Synchronize();

            // Normalize by window sum
            for (int i = 0; i < outputSamples; i++)
            {
                if (windowSum[i] > 1e-8f)
                {
                    output[i] /= windowSum[i];
                }
            }

            // Remove centering padding if needed
            if (center)
            {
                int padAmount = nFft / 2;
                int actualLength = length ?? (outputSamples - 2 * padAmount);
                float[] trimmed = new float[actualLength];
                Array.Copy(output, padAmount, trimmed, 0, Math.Min(actualLength, outputSamples - padAmount));
                output = trimmed;
            }
            else if (length.HasValue)
            {
                float[] trimmed = new float[length.Value];
                Array.Copy(output, 0, trimmed, 0, Math.Min(length.Value, output.Length));
                output = trimmed;
            }

            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(output), new[] { output.Length });
        }
        catch
        {
            return base.ISTFT(magnitude, phase, nFft, hopLength, window, center, length);
        }
    }

    /// <summary>
    /// GPU-accelerated Mel spectrogram computation.
    /// </summary>
    Tensor<T> IEngine.MelSpectrogram<T>(
        Tensor<T> input,
        int sampleRate,
        int nFft,
        int hopLength,
        int nMels,
        T fMin,
        T fMax,
        Tensor<T> window,
        bool powerToDb)
    {
        if (!TryGetBackend(out var backend) || (nFft & (nFft - 1)) != 0)
        {
            return base.MelSpectrogram(input, sampleRate, nFft, hopLength, nMels, fMin, fMax, window, powerToDb);
        }

        try
        {
            // First compute STFT
            ((IEngine)this).STFT(input, nFft, hopLength, window, center: true, out var magnitude, out var _);

            int numFrames = magnitude.Shape[^2];
            int numFreqs = magnitude.Shape[^1];

            // Create Mel filterbank
            var filterbank = ((IEngine)this).CreateMelFilterbank<T>(nMels, nFft, sampleRate, fMin, fMax);

            float[] magnitudeFloat = DirectGpuEngine.ToFloatArray(magnitude.Data);
            float[] filterbankFloat = DirectGpuEngine.ToFloatArray(filterbank.Data);

            // Compute power spectrum (magnitude squared)
            float[] powerSpec = new float[magnitudeFloat.Length];
            for (int i = 0; i < magnitudeFloat.Length; i++)
            {
                powerSpec[i] = magnitudeFloat[i] * magnitudeFloat[i];
            }

            // Use cache-aware allocation for filterbank (likely persistent)
            using var filterbankBuffer = GetOrAllocateBuffer(backend, filterbank.Data);
            // Allocate working buffers
            using var powerBuffer = backend.AllocateBuffer(powerSpec);
            using var melBuffer = AllocateOutputBuffer(backend, numFrames * nMels);

            // Apply Mel filterbank
            backend.ApplyMelFilterbank(powerBuffer, filterbankBuffer.Buffer, melBuffer.Buffer, numFrames, numFreqs, nMels);

            if (powerToDb)
            {
                using var dbBuffer = AllocateOutputBuffer(backend, numFrames * nMels);
                backend.PowerToDb(melBuffer.Buffer, dbBuffer.Buffer, numFrames * nMels, 1.0f, -80.0f);
                backend.Synchronize();

                float[] dbResult = new float[numFrames * nMels];
                backend.DownloadBuffer(dbBuffer.Buffer, dbResult);

                int[] outputShape = input.Rank == 1
                    ? new[] { numFrames, nMels }
                    : new[] { input.Shape[0], numFrames, nMels };

                return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(dbResult), outputShape);
            }
            else
            {
                backend.Synchronize();

                float[] melResult = new float[numFrames * nMels];
                backend.DownloadBuffer(melBuffer.Buffer, melResult);

                int[] outputShape = input.Rank == 1
                    ? new[] { numFrames, nMels }
                    : new[] { input.Shape[0], numFrames, nMels };

                return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(melResult), outputShape);
            }
        }
        catch
        {
            return base.MelSpectrogram(input, sampleRate, nFft, hopLength, nMels, fMin, fMax, window, powerToDb);
        }
    }

    /// <summary>
    /// GPU-accelerated Griffin-Lim algorithm for audio reconstruction from magnitude spectrogram.
    /// </summary>
    Tensor<T> IEngine.GriffinLim<T>(
        Tensor<T> magnitude,
        int nFft,
        int hopLength,
        Tensor<T> window,
        int iterations,
        double momentum,
        int? length)
    {
        if (!TryGetBackend(out var backend) || (nFft & (nFft - 1)) != 0)
        {
            return base.GriffinLim(magnitude, nFft, hopLength, window, iterations, momentum, length);
        }

        try
        {
            int numFrames = magnitude.Shape[^2];
            int numFreqs = magnitude.Shape[^1];

            float[] magnitudeFloat = DirectGpuEngine.ToFloatArray(magnitude.Data);

            // Initialize with random phase
            var random = new Random(42);
            float[] phase = new float[magnitudeFloat.Length];
            for (int i = 0; i < phase.Length; i++)
            {
                phase[i] = (float)(random.NextDouble() * 2 * Math.PI - Math.PI);
            }

            float[] prevPhase = new float[phase.Length];
            float momentumF = (float)momentum;

            for (int iter = 0; iter < iterations; iter++)
            {
                // Reconstruct signal using current phase estimate
                var phaseTensor = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(phase), magnitude.Shape.ToArray());
                var reconstructed = ((IEngine)this).ISTFT(magnitude, phaseTensor, nFft, hopLength, window, center: true, length);

                // Re-analyze to get new phase
                ((IEngine)this).STFT(reconstructed, nFft, hopLength, window, center: true, out var _, out var newPhaseTensor);

                float[] newPhase = DirectGpuEngine.ToFloatArray(newPhaseTensor.Data);

                // Apply momentum
                if (iter > 0 && momentumF > 0)
                {
                    for (int i = 0; i < phase.Length; i++)
                    {
                        // Unwrap phase difference for momentum
                        float diff = newPhase[i] - prevPhase[i];
                        while (diff > Math.PI) diff -= (float)(2 * Math.PI);
                        while (diff < -Math.PI) diff += (float)(2 * Math.PI);

                        float accelerated = prevPhase[i] + diff * (1 + momentumF);
                        phase[i] = accelerated;
                    }
                }
                else
                {
                    Array.Copy(newPhase, phase, phase.Length);
                }

                Array.Copy(newPhase, prevPhase, prevPhase.Length);
            }

            // Final reconstruction
            var finalPhaseTensor = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(phase), magnitude.Shape.ToArray());
            return ((IEngine)this).ISTFT(magnitude, finalPhaseTensor, nFft, hopLength, window, center: true, length);
        }
        catch
        {
            return base.GriffinLim(magnitude, nFft, hopLength, window, iterations, momentum, length);
        }
    }

    /// <summary>
    /// Creates a Mel filterbank matrix (CPU implementation, can be cached).
    /// </summary>
    Tensor<T> IEngine.CreateMelFilterbank<T>(int nMels, int nFft, int sampleRate, T fMin, T fMax)
    {
        // Filterbank creation is a one-time operation, use CPU base implementation
        return base.CreateMelFilterbank<T>(nMels, nFft, sampleRate, fMin, fMax);
    }

    /// <summary>
    /// Creates a window function (CPU implementation, can be cached).
    /// </summary>
    Tensor<T> IEngine.CreateWindow<T>(string windowType, int windowLength)
    {
        // Window creation is a one-time operation, use CPU base implementation
        return base.CreateWindow<T>(windowType, windowLength);
    }

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
