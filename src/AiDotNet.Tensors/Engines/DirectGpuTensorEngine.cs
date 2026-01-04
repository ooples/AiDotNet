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
            // For cases with bias and activation
            if (bias != null && activation != FusedActivationType.None)
            {
                // Use fused kernels for common activations (most efficient)
                switch (activation)
                {
                    case FusedActivationType.ReLU:
                        resultBuffer = backend.GemmBiasRelu(inputBuffer.Buffer, weightsBuffer.Buffer, biasBuffer.Buffer, batchSize, outputFeatures, inputFeatures);
                        break;
                    case FusedActivationType.GELU:
                        resultBuffer = backend.GemmBiasGelu(inputBuffer.Buffer, weightsBuffer.Buffer, biasBuffer.Buffer, batchSize, outputFeatures, inputFeatures);
                        break;
                    case FusedActivationType.Sigmoid:
                        resultBuffer = backend.GemmBiasSigmoid(inputBuffer.Buffer, weightsBuffer.Buffer, biasBuffer.Buffer, batchSize, outputFeatures, inputFeatures);
                        break;
                    case FusedActivationType.Tanh:
                        resultBuffer = backend.GemmBiasTanh(inputBuffer.Buffer, weightsBuffer.Buffer, biasBuffer.Buffer, batchSize, outputFeatures, inputFeatures);
                        break;
                    default:
                        // For other activations (LeakyReLU, Swish, etc.), use GemmBias + separate activation kernel
                        resultBuffer = backend.GemmBias(inputBuffer.Buffer, weightsBuffer.Buffer, biasBuffer.Buffer, batchSize, outputFeatures, inputFeatures);
                        int size = batchSize * outputFeatures;
                        ApplyGpuActivation(backend, resultBuffer, size, activation);
                        break;
                }
            }
            else if (bias != null && activation == FusedActivationType.None)
            {
                // GEMM + Bias only (no activation) - use GPU GemmBias kernel
                resultBuffer = backend.GemmBias(inputBuffer.Buffer, weightsBuffer.Buffer, biasBuffer.Buffer, batchSize, outputFeatures, inputFeatures);
            }
            else if (bias == null && activation == FusedActivationType.None)
            {
                // Simple MatMul only - use GPU
                resultBuffer = backend.MatMul(inputBuffer.Buffer, weightsBuffer.Buffer, batchSize, outputFeatures, inputFeatures);
            }
            else if (bias == null && activation != FusedActivationType.None)
            {
                // MatMul + activation (no bias) - use GPU MatMul followed by activation
                resultBuffer = backend.MatMul(inputBuffer.Buffer, weightsBuffer.Buffer, batchSize, outputFeatures, inputFeatures);
                int size = batchSize * outputFeatures;
                ApplyGpuActivation(backend, resultBuffer, size, activation);
            }
            else
            {
                // Fall back to CPU for other combinations (should not reach here now)
                return base.FusedLinear(input, weights, bias, activation);
            }

            // Download result - DownloadBuffer is blocking, no need for Synchronize after
            int resultSize = batchSize * outputFeatures;
            float[] resultFloat = new float[resultSize];
            backend.Synchronize(); // Ensure GPU compute is complete before download
            backend.DownloadBuffer(resultBuffer, resultFloat);
            resultBuffer.Dispose();

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
    /// Uses cached GPU buffers for registered persistent tensors (kernel/bias) to avoid
    /// redundant CPU→GPU transfers on every forward pass.
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
        if (!TryGetBackend(out var backend))
            return base.FusedConv3D(input, kernel, bias, strideD, strideH, strideW, padD, padH, padW, dilationD, dilationH, dilationW, activation);

        // Expected input shape: [batch, inChannels, depth, height, width]
        // Expected kernel shape: [outChannels, inChannels, kernelD, kernelH, kernelW]
        if (input.Rank != 5 || kernel.Rank != 5)
            return base.FusedConv3D(input, kernel, bias, strideD, strideH, strideW, padD, padH, padW, dilationD, dilationH, dilationW, activation);

        int batch = input.Shape[0];
        int inChannels = input.Shape[1];
        int inDepth = input.Shape[2];
        int inHeight = input.Shape[3];
        int inWidth = input.Shape[4];

        int outChannels = kernel.Shape[0];
        int kernelD = kernel.Shape[2];
        int kernelH = kernel.Shape[3];
        int kernelW = kernel.Shape[4];

        // Calculate output dimensions with dilation
        int effectiveKernelD = kernelD + (kernelD - 1) * (dilationD - 1);
        int effectiveKernelH = kernelH + (kernelH - 1) * (dilationH - 1);
        int effectiveKernelW = kernelW + (kernelW - 1) * (dilationW - 1);
        int outDepth = (inDepth + 2 * padD - effectiveKernelD) / strideD + 1;
        int outHeight = (inHeight + 2 * padH - effectiveKernelH) / strideH + 1;
        int outWidth = (inWidth + 2 * padW - effectiveKernelW) / strideW + 1;

        if (outDepth <= 0 || outHeight <= 0 || outWidth <= 0)
            return base.FusedConv3D(input, kernel, bias, strideD, strideH, strideW, padD, padH, padW, dilationD, dilationH, dilationW, activation);

        // Use cache-aware buffer allocation (OwnedBuffer auto-disposes only if we allocated)
        using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
        using var kernelBuffer = GetOrAllocateBuffer(backend, kernel.Data);
        using var outputBuffer = AllocateOutputBuffer(backend, batch * outChannels * outDepth * outHeight * outWidth);

        try
        {
            // Execute GPU 3D convolution
            backend.Conv3D(inputBuffer.Buffer, kernelBuffer.Buffer, outputBuffer.Buffer,
                batch, inChannels, inDepth, inHeight, inWidth,
                outChannels, outDepth, outHeight, outWidth,
                kernelD, kernelH, kernelW,
                strideD, strideH, strideW,
                padD, padH, padW,
                dilationD, dilationH, dilationW);

            // Add bias if present
            if (bias != null)
            {
                int outputSize = batch * outChannels * outDepth * outHeight * outWidth;
                int spatialSize = outDepth * outHeight * outWidth;

                // Download, add bias, re-upload
                float[] outputFloat = new float[outputSize];
                backend.DownloadBuffer(outputBuffer.Buffer, outputFloat);

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

                using var biasedBuffer = backend.AllocateBuffer(outputFloat);

                if (activation != FusedActivationType.None)
                {
                    ApplyGpuActivation(backend, biasedBuffer, outputSize, activation);
                }

                backend.Synchronize();

                float[] resultFloat = new float[outputSize];
                backend.DownloadBuffer(biasedBuffer, resultFloat);

                T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
                return new Tensor<T>(resultData, new[] { batch, outChannels, outDepth, outHeight, outWidth });
            }
            else
            {
                int outputSize = batch * outChannels * outDepth * outHeight * outWidth;

                if (activation != FusedActivationType.None)
                {
                    ApplyGpuActivation(backend, outputBuffer.Buffer, outputSize, activation);
                }

                backend.Synchronize();

                float[] resultFloat = new float[outputSize];
                backend.DownloadBuffer(outputBuffer.Buffer, resultFloat);

                T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
                return new Tensor<T>(resultData, new[] { batch, outChannels, outDepth, outHeight, outWidth });
            }
        }
        catch
        {
            return base.FusedConv3D(input, kernel, bias, strideD, strideH, strideW, padD, padH, padW, dilationD, dilationH, dilationW, activation);
        }
    }

    /// <summary>
    /// GPU-accelerated fused transposed 2D convolution with activation.
    /// Uses cached GPU buffers for registered persistent tensors (kernel/bias) to avoid
    /// redundant CPU→GPU transfers on every forward pass.
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
        if (!TryGetBackend(out var backend))
            return base.FusedConvTranspose2D(input, kernel, bias, strideH, strideW, padH, padW, outputPadH, outputPadW, activation);

        // Expected input shape: [batch, inChannels, height, width]
        // Expected kernel shape: [inChannels, outChannels, kernelH, kernelW]
        if (input.Rank != 4 || kernel.Rank != 4)
            return base.FusedConvTranspose2D(input, kernel, bias, strideH, strideW, padH, padW, outputPadH, outputPadW, activation);

        int batch = input.Shape[0];
        int inChannels = input.Shape[1];
        int inHeight = input.Shape[2];
        int inWidth = input.Shape[3];

        int outChannels = kernel.Shape[1];
        int kernelH = kernel.Shape[2];
        int kernelW = kernel.Shape[3];

        // Calculate output dimensions for transposed convolution
        int outHeight = (inHeight - 1) * strideH - 2 * padH + kernelH + outputPadH;
        int outWidth = (inWidth - 1) * strideW - 2 * padW + kernelW + outputPadW;

        if (outHeight <= 0 || outWidth <= 0)
            return base.FusedConvTranspose2D(input, kernel, bias, strideH, strideW, padH, padW, outputPadH, outputPadW, activation);

        // Use cache-aware buffer allocation (OwnedBuffer auto-disposes only if we allocated)
        using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
        using var kernelBuffer = GetOrAllocateBuffer(backend, kernel.Data);
        using var outputBuffer = AllocateOutputBuffer(backend, batch * outChannels * outHeight * outWidth);

        try
        {
            // Execute GPU transposed convolution
            backend.ConvTranspose2D(inputBuffer.Buffer, kernelBuffer.Buffer, outputBuffer.Buffer,
                batch, inChannels, inHeight, inWidth,
                outChannels, outHeight, outWidth,
                kernelH, kernelW,
                strideH, strideW, padH, padW,
                outputPadH, outputPadW);

            // Add bias if present
            if (bias != null)
            {
                int outputSize = batch * outChannels * outHeight * outWidth;
                int spatialSize = outHeight * outWidth;

                // Download, add bias, re-upload
                float[] outputFloat = new float[outputSize];
                backend.DownloadBuffer(outputBuffer.Buffer, outputFloat);

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

                using var biasedBuffer = backend.AllocateBuffer(outputFloat);

                if (activation != FusedActivationType.None)
                {
                    ApplyGpuActivation(backend, biasedBuffer, outputSize, activation);
                }

                backend.Synchronize();

                float[] resultFloat = new float[outputSize];
                backend.DownloadBuffer(biasedBuffer, resultFloat);

                T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
                return new Tensor<T>(resultData, new[] { batch, outChannels, outHeight, outWidth });
            }
            else
            {
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
            return base.FusedConvTranspose2D(input, kernel, bias, strideH, strideW, padH, padW, outputPadH, outputPadW, activation);
        }
    }

    /// <summary>
    /// GPU-accelerated 2D max pooling operation.
    /// Uses GPU kernels for efficient parallel computation of maximum values within pooling windows.
    /// </summary>
    public new Tensor<T> MaxPool2D<T>(Tensor<T> input, int poolSize, int stride = 0, int padding = 0)
    {
        if (stride == 0) stride = poolSize;

        if (!TryGetBackend(out var backend))
            return base.MaxPool2D(input, poolSize, stride, padding);

        // Expected input shape: [batch, channels, height, width]
        if (input.Rank != 4)
            return base.MaxPool2D(input, poolSize, stride, padding);

        int batch = input.Shape[0];
        int channels = input.Shape[1];
        int inHeight = input.Shape[2];
        int inWidth = input.Shape[3];

        // Calculate output dimensions
        int outHeight = (inHeight + 2 * padding - poolSize) / stride + 1;
        int outWidth = (inWidth + 2 * padding - poolSize) / stride + 1;

        if (outHeight <= 0 || outWidth <= 0)
            return base.MaxPool2D(input, poolSize, stride, padding);

        using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
        using var outputBuffer = AllocateOutputBuffer(backend, batch * channels * outHeight * outWidth);

        try
        {
            // Execute GPU max pooling (indices buffer is null for forward-only)
            backend.MaxPool2D(inputBuffer.Buffer, outputBuffer.Buffer, null,
                batch, channels, inHeight, inWidth,
                outHeight, outWidth,
                poolSize, poolSize,
                stride, stride, padding, padding);

            backend.Synchronize();

            int outputSize = batch * channels * outHeight * outWidth;
            float[] resultFloat = new float[outputSize];
            backend.DownloadBuffer(outputBuffer.Buffer, resultFloat);

            T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
            return new Tensor<T>(resultData, new[] { batch, channels, outHeight, outWidth });
        }
        catch
        {
            return base.MaxPool2D(input, poolSize, stride, padding);
        }
    }

    /// <summary>
    /// GPU-accelerated 2D max pooling with indices for backward pass.
    /// Returns both pooled output and indices of maximum values for gradient computation.
    /// </summary>
    public new Tensor<T> MaxPool2DWithIndices<T>(Tensor<T> input, int[] poolSize, int[] stride, out int[,,,,] maxIndices)
    {
        if (!TryGetBackend(out var backend))
            return base.MaxPool2DWithIndices(input, poolSize, stride, out maxIndices);

        if (input.Rank != 4 || poolSize.Length != 2 || stride.Length != 2)
            return base.MaxPool2DWithIndices(input, poolSize, stride, out maxIndices);

        int batch = input.Shape[0];
        int channels = input.Shape[1];
        int inHeight = input.Shape[2];
        int inWidth = input.Shape[3];

        int outHeight = (inHeight - poolSize[0]) / stride[0] + 1;
        int outWidth = (inWidth - poolSize[1]) / stride[1] + 1;

        if (outHeight <= 0 || outWidth <= 0)
            return base.MaxPool2DWithIndices(input, poolSize, stride, out maxIndices);

        using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
        using var outputBuffer = AllocateOutputBuffer(backend, batch * channels * outHeight * outWidth);
        using var indicesBuffer = AllocateOutputBuffer(backend, batch * channels * outHeight * outWidth);

        try
        {
            backend.MaxPool2D(inputBuffer.Buffer, outputBuffer.Buffer, indicesBuffer.Buffer,
                batch, channels, inHeight, inWidth,
                outHeight, outWidth,
                poolSize[0], poolSize[1],
                stride[0], stride[1], 0, 0);

            backend.Synchronize();

            int outputSize = batch * channels * outHeight * outWidth;
            float[] resultFloat = new float[outputSize];
            float[] indicesFloat = new float[outputSize];
            backend.DownloadBuffer(outputBuffer.Buffer, resultFloat);
            backend.DownloadBuffer(indicesBuffer.Buffer, indicesFloat);

            // Convert indices to int array
            maxIndices = new int[batch, channels, outHeight, outWidth, 2];
            for (int b = 0; b < batch; b++)
            {
                for (int c = 0; c < channels; c++)
                {
                    for (int oh = 0; oh < outHeight; oh++)
                    {
                        for (int ow = 0; ow < outWidth; ow++)
                        {
                            int flatIdx = ((b * channels + c) * outHeight + oh) * outWidth + ow;
                            int idx = (int)indicesFloat[flatIdx];
                            maxIndices[b, c, oh, ow, 0] = idx / inWidth;
                            maxIndices[b, c, oh, ow, 1] = idx % inWidth;
                        }
                    }
                }
            }

            T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
            return new Tensor<T>(resultData, new[] { batch, channels, outHeight, outWidth });
        }
        catch
        {
            return base.MaxPool2DWithIndices(input, poolSize, stride, out maxIndices);
        }
    }

    /// <summary>
    /// GPU-accelerated backward pass for 2D max pooling.
    /// Propagates gradients back through the max pooling operation using stored indices.
    /// </summary>
    public new Tensor<T> MaxPool2DBackward<T>(Tensor<T> gradOutput, int[,,,,] maxIndices, int[] inputShape, int[] poolSize, int[] stride)
    {
        if (!TryGetBackend(out var backend))
            return base.MaxPool2DBackward(gradOutput, maxIndices, inputShape, poolSize, stride);

        if (gradOutput.Rank != 4 || inputShape.Length != 4)
            return base.MaxPool2DBackward(gradOutput, maxIndices, inputShape, poolSize, stride);

        int batch = inputShape[0];
        int channels = inputShape[1];
        int inHeight = inputShape[2];
        int inWidth = inputShape[3];
        int outHeight = gradOutput.Shape[2];
        int outWidth = gradOutput.Shape[3];

        // Convert indices to flat GPU buffer
        int indexCount = batch * channels * outHeight * outWidth;
        float[] indicesFlat = new float[indexCount];
        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int oh = 0; oh < outHeight; oh++)
                {
                    for (int ow = 0; ow < outWidth; ow++)
                    {
                        int flatIdx = ((b * channels + c) * outHeight + oh) * outWidth + ow;
                        int h = maxIndices[b, c, oh, ow, 0];
                        int w = maxIndices[b, c, oh, ow, 1];
                        indicesFlat[flatIdx] = h * inWidth + w;
                    }
                }
            }
        }

        using var gradOutputBuffer = GetOrAllocateBuffer(backend, gradOutput.Data);
        using var indicesBuffer = backend.AllocateBuffer(indicesFlat);
        using var gradInputBuffer = AllocateOutputBuffer(backend, batch * channels * inHeight * inWidth);

        try
        {
            backend.MaxPool2DBackward(gradOutputBuffer.Buffer, indicesBuffer, gradInputBuffer.Buffer,
                batch, channels, inHeight, inWidth,
                outHeight, outWidth,
                poolSize[0], poolSize[1],
                stride[0], stride[1], 0, 0);

            backend.Synchronize();

            int inputSize = batch * channels * inHeight * inWidth;
            float[] resultFloat = new float[inputSize];
            backend.DownloadBuffer(gradInputBuffer.Buffer, resultFloat);

            T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
            return new Tensor<T>(resultData, inputShape);
        }
        catch
        {
            return base.MaxPool2DBackward(gradOutput, maxIndices, inputShape, poolSize, stride);
        }
    }

    /// <summary>
    /// GPU-accelerated 2D average pooling operation.
    /// Uses GPU kernels for efficient parallel computation of average values within pooling windows.
    /// </summary>
    public new Tensor<T> AvgPool2D<T>(Tensor<T> input, int poolSize, int stride = 0, int padding = 0)
    {
        if (stride == 0) stride = poolSize;

        if (!TryGetBackend(out var backend))
            return base.AvgPool2D(input, poolSize, stride, padding);

        // Expected input shape: [batch, channels, height, width]
        if (input.Rank != 4)
            return base.AvgPool2D(input, poolSize, stride, padding);

        int batch = input.Shape[0];
        int channels = input.Shape[1];
        int inHeight = input.Shape[2];
        int inWidth = input.Shape[3];

        // Calculate output dimensions
        int outHeight = (inHeight + 2 * padding - poolSize) / stride + 1;
        int outWidth = (inWidth + 2 * padding - poolSize) / stride + 1;

        if (outHeight <= 0 || outWidth <= 0)
            return base.AvgPool2D(input, poolSize, stride, padding);

        using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
        using var outputBuffer = AllocateOutputBuffer(backend, batch * channels * outHeight * outWidth);

        try
        {
            // Execute GPU average pooling
            backend.AvgPool2D(inputBuffer.Buffer, outputBuffer.Buffer,
                batch, channels, inHeight, inWidth,
                outHeight, outWidth,
                poolSize, poolSize,
                stride, stride, padding, padding,
                countIncludePad: true);

            backend.Synchronize();

            int outputSize = batch * channels * outHeight * outWidth;
            float[] resultFloat = new float[outputSize];
            backend.DownloadBuffer(outputBuffer.Buffer, resultFloat);

            T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
            return new Tensor<T>(resultData, new[] { batch, channels, outHeight, outWidth });
        }
        catch
        {
            return base.AvgPool2D(input, poolSize, stride, padding);
        }
    }

    /// <summary>
    /// GPU-accelerated 2D average pooling with array parameters.
    /// </summary>
    public new Tensor<T> AvgPool2D<T>(Tensor<T> input, int[] poolSize, int[] stride)
    {
        if (!TryGetBackend(out var backend))
            return base.AvgPool2D(input, poolSize, stride);

        if (input.Rank != 4 || poolSize.Length != 2 || stride.Length != 2)
            return base.AvgPool2D(input, poolSize, stride);

        int batch = input.Shape[0];
        int channels = input.Shape[1];
        int inHeight = input.Shape[2];
        int inWidth = input.Shape[3];

        int outHeight = (inHeight - poolSize[0]) / stride[0] + 1;
        int outWidth = (inWidth - poolSize[1]) / stride[1] + 1;

        if (outHeight <= 0 || outWidth <= 0)
            return base.AvgPool2D(input, poolSize, stride);

        using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
        using var outputBuffer = AllocateOutputBuffer(backend, batch * channels * outHeight * outWidth);

        try
        {
            backend.AvgPool2D(inputBuffer.Buffer, outputBuffer.Buffer,
                batch, channels, inHeight, inWidth,
                outHeight, outWidth,
                poolSize[0], poolSize[1],
                stride[0], stride[1], 0, 0,
                countIncludePad: true);

            backend.Synchronize();

            int outputSize = batch * channels * outHeight * outWidth;
            float[] resultFloat = new float[outputSize];
            backend.DownloadBuffer(outputBuffer.Buffer, resultFloat);

            T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
            return new Tensor<T>(resultData, new[] { batch, channels, outHeight, outWidth });
        }
        catch
        {
            return base.AvgPool2D(input, poolSize, stride);
        }
    }

    /// <summary>
    /// GPU-accelerated backward pass for 2D average pooling.
    /// Distributes gradients evenly across the input elements that contributed to each output.
    /// </summary>
    public new Tensor<T> AvgPool2DBackward<T>(Tensor<T> gradOutput, int[] inputShape, int[] poolSize, int[] stride)
    {
        if (!TryGetBackend(out var backend))
            return base.AvgPool2DBackward(gradOutput, inputShape, poolSize, stride);

        if (gradOutput.Rank != 4 || inputShape.Length != 4)
            return base.AvgPool2DBackward(gradOutput, inputShape, poolSize, stride);

        int batch = inputShape[0];
        int channels = inputShape[1];
        int inHeight = inputShape[2];
        int inWidth = inputShape[3];
        int outHeight = gradOutput.Shape[2];
        int outWidth = gradOutput.Shape[3];

        using var gradOutputBuffer = GetOrAllocateBuffer(backend, gradOutput.Data);
        using var gradInputBuffer = AllocateOutputBuffer(backend, batch * channels * inHeight * inWidth);

        try
        {
            backend.AvgPool2DBackward(gradOutputBuffer.Buffer, gradInputBuffer.Buffer,
                batch, channels, inHeight, inWidth,
                outHeight, outWidth,
                poolSize[0], poolSize[1],
                stride[0], stride[1], 0, 0,
                countIncludePad: true);

            backend.Synchronize();

            int inputSize = batch * channels * inHeight * inWidth;
            float[] resultFloat = new float[inputSize];
            backend.DownloadBuffer(gradInputBuffer.Buffer, resultFloat);

            T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
            return new Tensor<T>(resultData, inputShape);
        }
        catch
        {
            return base.AvgPool2DBackward(gradOutput, inputShape, poolSize, stride);
        }
    }

    /// <summary>
    /// GPU-accelerated depthwise 2D convolution.
    /// Each input channel is convolved with its own filter, commonly used in MobileNets.
    /// </summary>
    public new Tensor<T> DepthwiseConv2D<T>(Tensor<T> input, Tensor<T> kernel, int[] stride, int[] padding)
    {
        if (!TryGetBackend(out var backend))
            return base.DepthwiseConv2D(input, kernel, stride, padding);

        // Expected input shape: [batch, channels, height, width]
        // Expected kernel shape: [channels, 1, kernelH, kernelW] or [channels, kernelH, kernelW]
        if (input.Rank != 4)
            return base.DepthwiseConv2D(input, kernel, stride, padding);

        int batch = input.Shape[0];
        int channels = input.Shape[1];
        int inHeight = input.Shape[2];
        int inWidth = input.Shape[3];

        int kernelH = kernel.Rank == 4 ? kernel.Shape[2] : kernel.Shape[1];
        int kernelW = kernel.Rank == 4 ? kernel.Shape[3] : kernel.Shape[2];

        int strideH = stride.Length >= 1 ? stride[0] : 1;
        int strideW = stride.Length >= 2 ? stride[1] : strideH;
        int padH = padding.Length >= 1 ? padding[0] : 0;
        int padW = padding.Length >= 2 ? padding[1] : padH;

        int outHeight = (inHeight + 2 * padH - kernelH) / strideH + 1;
        int outWidth = (inWidth + 2 * padW - kernelW) / strideW + 1;

        if (outHeight <= 0 || outWidth <= 0)
            return base.DepthwiseConv2D(input, kernel, stride, padding);

        using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
        using var kernelBuffer = GetOrAllocateBuffer(backend, kernel.Data);
        using var outputBuffer = AllocateOutputBuffer(backend, batch * channels * outHeight * outWidth);

        try
        {
            backend.DepthwiseConv2D(inputBuffer.Buffer, kernelBuffer.Buffer, outputBuffer.Buffer,
                batch, channels, inHeight, inWidth,
                outHeight, outWidth,
                kernelH, kernelW,
                strideH, strideW, padH, padW);

            backend.Synchronize();

            int outputSize = batch * channels * outHeight * outWidth;
            float[] resultFloat = new float[outputSize];
            backend.DownloadBuffer(outputBuffer.Buffer, resultFloat);

            T[] resultData = DirectGpuEngine.FromFloatArray<T>(resultFloat);
            return new Tensor<T>(resultData, new[] { batch, channels, outHeight, outWidth });
        }
        catch
        {
            return base.DepthwiseConv2D(input, kernel, stride, padding);
        }
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
            // Note: DownloadBuffer calls inside the loop are blocking, no need for Synchronize after

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
            // Note: DownloadBuffer calls inside the loop are blocking, no need for Synchronize after

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

    #region Normalization Operations (GPU Accelerated)

    /// <summary>
    /// GPU-accelerated Softmax operation.
    /// </summary>
    Tensor<T> IEngine.Softmax<T>(Tensor<T> input, int axis)
    {
        if (!TryGetBackend(out var backend))
            return base.Softmax(input, axis);

        // Handle negative axis
        int rank = input.Rank;
        if (axis < 0) axis = rank + axis;
        if (axis < 0 || axis >= rank)
            return base.Softmax(input, axis);

        try
        {
            // For the common case where softmax is over the last dimension
            // and input is 2D [batch, features], we can use GPU directly
            if (axis == rank - 1 && rank == 2)
            {
                int batchSize = input.Shape[0];
                int features = input.Shape[1];

                float[] inputFloat = DirectGpuEngine.ToFloatArray(input.Data);
                using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
                using var outputBuffer = AllocateOutputBuffer(backend, input.Length);

                backend.Softmax(inputBuffer.Buffer, outputBuffer.Buffer, batchSize, features);
                backend.Synchronize();

                float[] resultFloat = backend.DownloadBuffer(outputBuffer.Buffer);
                return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), input.Shape.ToArray());
            }

            // For other cases, fall back to CPU
            return base.Softmax(input, axis);
        }
        catch
        {
            return base.Softmax(input, axis);
        }
    }

    /// <summary>
    /// GPU-accelerated Softmax backward operation.
    /// </summary>
    Tensor<T> IEngine.SoftmaxBackward<T>(Tensor<T> gradOutput, Tensor<T> output, int axis)
    {
        if (!TryGetBackend(out var backend))
            return base.SoftmaxBackward(gradOutput, output, axis);

        int rank = output.Rank;
        if (axis < 0) axis = rank + axis;
        if (axis < 0 || axis >= rank)
            return base.SoftmaxBackward(gradOutput, output, axis);

        try
        {
            // For 2D tensors with softmax over last dimension
            if (axis == rank - 1 && rank == 2)
            {
                int batchSize = output.Shape[0];
                int features = output.Shape[1];

                using var gradOutBuffer = GetOrAllocateBuffer(backend, gradOutput.Data);
                using var outputBuffer = GetOrAllocateBuffer(backend, output.Data);
                using var gradInputBuffer = AllocateOutputBuffer(backend, output.Length);

                backend.SoftmaxBackward(gradOutBuffer.Buffer, outputBuffer.Buffer, gradInputBuffer.Buffer, batchSize, features);
                backend.Synchronize();

                float[] resultFloat = backend.DownloadBuffer(gradInputBuffer.Buffer);
                return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), output.Shape.ToArray());
            }

            return base.SoftmaxBackward(gradOutput, output, axis);
        }
        catch
        {
            return base.SoftmaxBackward(gradOutput, output, axis);
        }
    }

    /// <summary>
    /// GPU-accelerated LayerNorm operation.
    /// </summary>
    Tensor<T> IEngine.LayerNorm<T>(Tensor<T> input, Tensor<T> gamma, Tensor<T> beta, double epsilon, out Tensor<T> mean, out Tensor<T> variance)
    {
        if (!TryGetBackend(out var backend))
            return base.LayerNorm(input, gamma, beta, epsilon, out mean, out variance);

        try
        {
            // Determine batch size and normalized size from gamma shape
            int normalizedSize = gamma.Length;
            int batchSize = input.Length / normalizedSize;

            if (batchSize * normalizedSize != input.Length)
                return base.LayerNorm(input, gamma, beta, epsilon, out mean, out variance);

            using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
            using var gammaBuffer = GetOrAllocateBuffer(backend, gamma.Data);
            using var betaBuffer = GetOrAllocateBuffer(backend, beta.Data);
            using var outputBuffer = AllocateOutputBuffer(backend, input.Length);
            using var saveMeanBuffer = AllocateOutputBuffer(backend, batchSize);
            using var saveVarBuffer = AllocateOutputBuffer(backend, batchSize);

            backend.LayerNorm(inputBuffer.Buffer, outputBuffer.Buffer, gammaBuffer.Buffer, betaBuffer.Buffer,
                saveMeanBuffer.Buffer, saveVarBuffer.Buffer, batchSize, normalizedSize, (float)epsilon);
            backend.Synchronize();

            float[] outputFloat = backend.DownloadBuffer(outputBuffer.Buffer);
            float[] meanFloat = backend.DownloadBuffer(saveMeanBuffer.Buffer);
            float[] varFloat = backend.DownloadBuffer(saveVarBuffer.Buffer);

            mean = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(meanFloat), new[] { batchSize });
            variance = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(varFloat), new[] { batchSize });
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(outputFloat), input.Shape.ToArray());
        }
        catch
        {
            return base.LayerNorm(input, gamma, beta, epsilon, out mean, out variance);
        }
    }

    /// <summary>
    /// GPU-accelerated LayerNorm backward operation.
    /// </summary>
    Tensor<T> IEngine.LayerNormBackward<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> gamma, Tensor<T> mean, Tensor<T> variance, double epsilon, out Tensor<T> gradGamma, out Tensor<T> gradBeta)
    {
        if (!TryGetBackend(out var backend))
            return base.LayerNormBackward(gradOutput, input, gamma, mean, variance, epsilon, out gradGamma, out gradBeta);

        try
        {
            int normalizedSize = gamma.Length;
            int batchSize = input.Length / normalizedSize;

            if (batchSize * normalizedSize != input.Length)
                return base.LayerNormBackward(gradOutput, input, gamma, mean, variance, epsilon, out gradGamma, out gradBeta);

            using var gradOutBuffer = GetOrAllocateBuffer(backend, gradOutput.Data);
            using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
            using var gammaBuffer = GetOrAllocateBuffer(backend, gamma.Data);
            using var saveMeanBuffer = GetOrAllocateBuffer(backend, mean.Data);
            using var saveVarBuffer = GetOrAllocateBuffer(backend, variance.Data);
            using var gradInputBuffer = AllocateOutputBuffer(backend, input.Length);
            using var gradGammaBuffer = AllocateOutputBuffer(backend, normalizedSize);
            using var gradBetaBuffer = AllocateOutputBuffer(backend, normalizedSize);

            backend.LayerNormBackward(gradOutBuffer.Buffer, inputBuffer.Buffer, gammaBuffer.Buffer,
                saveMeanBuffer.Buffer, saveVarBuffer.Buffer, gradInputBuffer.Buffer, gradGammaBuffer.Buffer, gradBetaBuffer.Buffer,
                batchSize, normalizedSize, (float)epsilon);
            backend.Synchronize();

            float[] gradInputFloat = backend.DownloadBuffer(gradInputBuffer.Buffer);
            float[] gradGammaFloat = backend.DownloadBuffer(gradGammaBuffer.Buffer);
            float[] gradBetaFloat = backend.DownloadBuffer(gradBetaBuffer.Buffer);

            gradGamma = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(gradGammaFloat), gamma.Shape.ToArray());
            gradBeta = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(gradBetaFloat), gamma.Shape.ToArray());
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(gradInputFloat), input.Shape.ToArray());
        }
        catch
        {
            return base.LayerNormBackward(gradOutput, input, gamma, mean, variance, epsilon, out gradGamma, out gradBeta);
        }
    }

    /// <summary>
    /// GPU-accelerated RmsNorm operation.
    /// </summary>
    Tensor<T> IEngine.RmsNorm<T>(Tensor<T> input, Tensor<T> gamma, double epsilon, out Tensor<T> rms)
    {
        if (!TryGetBackend(out var backend))
            return base.RmsNorm(input, gamma, epsilon, out rms);

        try
        {
            int normalizedSize = gamma.Length;
            int batchSize = input.Length / normalizedSize;

            if (batchSize * normalizedSize != input.Length)
                return base.RmsNorm(input, gamma, epsilon, out rms);

            using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
            using var gammaBuffer = GetOrAllocateBuffer(backend, gamma.Data);
            using var outputBuffer = AllocateOutputBuffer(backend, input.Length);
            using var saveRmsBuffer = AllocateOutputBuffer(backend, batchSize);

            backend.RmsNorm(inputBuffer.Buffer, outputBuffer.Buffer, gammaBuffer.Buffer, saveRmsBuffer.Buffer,
                batchSize, normalizedSize, (float)epsilon);
            backend.Synchronize();

            float[] outputFloat = backend.DownloadBuffer(outputBuffer.Buffer);
            float[] rmsFloat = backend.DownloadBuffer(saveRmsBuffer.Buffer);

            rms = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(rmsFloat), new[] { batchSize });
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(outputFloat), input.Shape.ToArray());
        }
        catch
        {
            return base.RmsNorm(input, gamma, epsilon, out rms);
        }
    }

    /// <summary>
    /// GPU-accelerated RmsNorm backward operation.
    /// </summary>
    Tensor<T> IEngine.RmsNormBackward<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> gamma, Tensor<T> rms, double epsilon, out Tensor<T> gradGamma)
    {
        if (!TryGetBackend(out var backend))
            return base.RmsNormBackward(gradOutput, input, gamma, rms, epsilon, out gradGamma);

        try
        {
            int normalizedSize = gamma.Length;
            int batchSize = input.Length / normalizedSize;

            if (batchSize * normalizedSize != input.Length)
                return base.RmsNormBackward(gradOutput, input, gamma, rms, epsilon, out gradGamma);

            using var gradOutBuffer = GetOrAllocateBuffer(backend, gradOutput.Data);
            using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
            using var gammaBuffer = GetOrAllocateBuffer(backend, gamma.Data);
            using var saveRmsBuffer = GetOrAllocateBuffer(backend, rms.Data);
            using var gradInputBuffer = AllocateOutputBuffer(backend, input.Length);
            using var gradGammaBuffer = AllocateOutputBuffer(backend, normalizedSize);

            backend.RmsNormBackward(gradOutBuffer.Buffer, inputBuffer.Buffer, gammaBuffer.Buffer, saveRmsBuffer.Buffer,
                gradInputBuffer.Buffer, gradGammaBuffer.Buffer, batchSize, normalizedSize, (float)epsilon);
            backend.Synchronize();

            float[] gradInputFloat = backend.DownloadBuffer(gradInputBuffer.Buffer);
            float[] gradGammaFloat = backend.DownloadBuffer(gradGammaBuffer.Buffer);

            gradGamma = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(gradGammaFloat), gamma.Shape.ToArray());
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(gradInputFloat), input.Shape.ToArray());
        }
        catch
        {
            return base.RmsNormBackward(gradOutput, input, gamma, rms, epsilon, out gradGamma);
        }
    }

    /// <summary>
    /// GPU-accelerated GroupNorm operation.
    /// </summary>
    Tensor<T> IEngine.GroupNorm<T>(Tensor<T> input, int numGroups, Tensor<T> gamma, Tensor<T> beta, double epsilon, out Tensor<T> mean, out Tensor<T> variance)
    {
        if (!TryGetBackend(out var backend))
            return base.GroupNorm(input, numGroups, gamma, beta, epsilon, out mean, out variance);

        try
        {
            // Input shape: [batch, channels, spatial...]
            if (input.Rank < 2)
                return base.GroupNorm(input, numGroups, gamma, beta, epsilon, out mean, out variance);

            int batch = input.Shape[0];
            int channels = input.Shape[1];
            int spatialSize = 1;
            for (int i = 2; i < input.Rank; i++)
                spatialSize *= input.Shape[i];

            if (channels % numGroups != 0)
                return base.GroupNorm(input, numGroups, gamma, beta, epsilon, out mean, out variance);

            using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
            using var gammaBuffer = GetOrAllocateBuffer(backend, gamma.Data);
            using var betaBuffer = GetOrAllocateBuffer(backend, beta.Data);
            using var outputBuffer = AllocateOutputBuffer(backend, input.Length);
            using var saveMeanBuffer = AllocateOutputBuffer(backend, batch * numGroups);
            using var saveVarBuffer = AllocateOutputBuffer(backend, batch * numGroups);

            backend.GroupNorm(inputBuffer.Buffer, outputBuffer.Buffer, gammaBuffer.Buffer, betaBuffer.Buffer,
                saveMeanBuffer.Buffer, saveVarBuffer.Buffer, batch, numGroups, channels, spatialSize, (float)epsilon);
            backend.Synchronize();

            float[] outputFloat = backend.DownloadBuffer(outputBuffer.Buffer);
            float[] meanFloat = backend.DownloadBuffer(saveMeanBuffer.Buffer);
            float[] varFloat = backend.DownloadBuffer(saveVarBuffer.Buffer);

            mean = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(meanFloat), new[] { batch, numGroups });
            variance = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(varFloat), new[] { batch, numGroups });
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(outputFloat), input.Shape.ToArray());
        }
        catch
        {
            return base.GroupNorm(input, numGroups, gamma, beta, epsilon, out mean, out variance);
        }
    }

    /// <summary>
    /// GPU-accelerated InstanceNorm operation.
    /// </summary>
    Tensor<T> IEngine.InstanceNorm<T>(Tensor<T> input, Tensor<T> gamma, Tensor<T> beta, double epsilon, out Tensor<T> mean, out Tensor<T> variance)
    {
        if (!TryGetBackend(out var backend))
            return base.InstanceNorm(input, gamma, beta, epsilon, out mean, out variance);

        try
        {
            // Input shape: [batch, channels, spatial...]
            if (input.Rank < 2)
                return base.InstanceNorm(input, gamma, beta, epsilon, out mean, out variance);

            int batch = input.Shape[0];
            int channels = input.Shape[1];
            int spatialSize = 1;
            for (int i = 2; i < input.Rank; i++)
                spatialSize *= input.Shape[i];

            using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
            using var gammaBuffer = GetOrAllocateBuffer(backend, gamma.Data);
            using var betaBuffer = GetOrAllocateBuffer(backend, beta.Data);
            using var outputBuffer = AllocateOutputBuffer(backend, input.Length);
            using var saveMeanBuffer = AllocateOutputBuffer(backend, batch * channels);
            using var saveVarBuffer = AllocateOutputBuffer(backend, batch * channels);

            backend.InstanceNorm(inputBuffer.Buffer, outputBuffer.Buffer, gammaBuffer.Buffer, betaBuffer.Buffer,
                saveMeanBuffer.Buffer, saveVarBuffer.Buffer, batch, channels, spatialSize, (float)epsilon);
            backend.Synchronize();

            float[] outputFloat = backend.DownloadBuffer(outputBuffer.Buffer);
            float[] meanFloat = backend.DownloadBuffer(saveMeanBuffer.Buffer);
            float[] varFloat = backend.DownloadBuffer(saveVarBuffer.Buffer);

            mean = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(meanFloat), new[] { batch, channels });
            variance = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(varFloat), new[] { batch, channels });
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(outputFloat), input.Shape.ToArray());
        }
        catch
        {
            return base.InstanceNorm(input, gamma, beta, epsilon, out mean, out variance);
        }
    }

    #endregion

    #region Dropout Operations (GPU Accelerated)

    /// <summary>
    /// GPU-accelerated Dropout operation.
    /// </summary>
    Tensor<T> IEngine.Dropout<T>(Tensor<T> input, double dropoutRate, bool training, out Tensor<T> mask)
    {
        if (!TryGetBackend(out var backend) || !training)
            return base.Dropout(input, dropoutRate, training, out mask);

        try
        {
            int size = input.Length;
            ulong seed = (ulong)DateTime.UtcNow.Ticks;

            using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
            using var outputBuffer = AllocateOutputBuffer(backend, size);
            using var maskBuffer = AllocateOutputBuffer(backend, size);

            backend.Dropout(inputBuffer.Buffer, outputBuffer.Buffer, maskBuffer.Buffer, size, (float)dropoutRate, seed, training);
            backend.Synchronize();

            float[] outputFloat = backend.DownloadBuffer(outputBuffer.Buffer);
            float[] maskFloat = backend.DownloadBuffer(maskBuffer.Buffer);

            mask = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(maskFloat), input.Shape.ToArray());
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(outputFloat), input.Shape.ToArray());
        }
        catch
        {
            return base.Dropout(input, dropoutRate, training, out mask);
        }
    }

    /// <summary>
    /// GPU-accelerated Dropout backward operation.
    /// </summary>
    Tensor<T> IEngine.DropoutBackward<T>(Tensor<T> gradOutput, Tensor<T> mask, double dropoutRate)
    {
        if (!TryGetBackend(out var backend))
            return base.DropoutBackward(gradOutput, mask, dropoutRate);

        try
        {
            int size = gradOutput.Length;

            using var gradOutBuffer = GetOrAllocateBuffer(backend, gradOutput.Data);
            using var maskBuffer = GetOrAllocateBuffer(backend, mask.Data);
            using var gradInputBuffer = AllocateOutputBuffer(backend, size);

            backend.DropoutBackward(gradOutBuffer.Buffer, maskBuffer.Buffer, gradInputBuffer.Buffer, size, (float)dropoutRate);
            backend.Synchronize();

            float[] resultFloat = backend.DownloadBuffer(gradInputBuffer.Buffer);
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), gradOutput.Shape.ToArray());
        }
        catch
        {
            return base.DropoutBackward(gradOutput, mask, dropoutRate);
        }
    }

    #endregion

    #region Embedding Operations (GPU Accelerated)

    /// <summary>
    /// GPU-accelerated Embedding lookup operation.
    /// </summary>
    Tensor<T> IEngine.Embedding<T>(Tensor<int> indices, Tensor<T> embeddingTable)
    {
        if (!TryGetBackend(out var backend))
            return base.Embedding(indices, embeddingTable);

        try
        {
            int numIndices = indices.Length;
            int embeddingDim = embeddingTable.Shape[^1];

            using var indicesBuffer = backend.AllocateIntBuffer(indices.Data);
            using var tableBuffer = GetOrAllocateBuffer(backend, embeddingTable.Data);
            using var outputBuffer = AllocateOutputBuffer(backend, numIndices * embeddingDim);

            backend.Embedding(indicesBuffer, tableBuffer.Buffer, outputBuffer.Buffer, numIndices, embeddingDim);
            backend.Synchronize();

            float[] outputFloat = backend.DownloadBuffer(outputBuffer.Buffer);

            // Output shape: indices.Shape + [embeddingDim]
            int[] outputShape = new int[indices.Shape.Length + 1];
            for (int i = 0; i < indices.Shape.Length; i++)
                outputShape[i] = indices.Shape[i];
            outputShape[^1] = embeddingDim;

            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(outputFloat), outputShape);
        }
        catch
        {
            return base.Embedding(indices, embeddingTable);
        }
    }

    /// <summary>
    /// GPU-accelerated Embedding backward operation.
    /// </summary>
    Tensor<T> IEngine.EmbeddingBackward<T>(Tensor<T> gradOutput, Tensor<int> indices, int vocabSize, int embeddingDim)
    {
        if (!TryGetBackend(out var backend))
            return base.EmbeddingBackward(gradOutput, indices, vocabSize, embeddingDim);

        try
        {
            int numIndices = indices.Length;

            using var gradOutBuffer = GetOrAllocateBuffer(backend, gradOutput.Data);
            using var indicesBuffer = backend.AllocateIntBuffer(indices.Data);
            using var gradEmbeddingBuffer = AllocateOutputBuffer(backend, vocabSize * embeddingDim);

            // Initialize to zero
            backend.Fill(gradEmbeddingBuffer.Buffer, 0f, vocabSize * embeddingDim);

            backend.EmbeddingBackward(gradOutBuffer.Buffer, indicesBuffer, gradEmbeddingBuffer.Buffer, numIndices, embeddingDim, vocabSize);
            backend.Synchronize();

            float[] resultFloat = backend.DownloadBuffer(gradEmbeddingBuffer.Buffer);
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), new[] { vocabSize, embeddingDim });
        }
        catch
        {
            return base.EmbeddingBackward(gradOutput, indices, vocabSize, embeddingDim);
        }
    }

    #endregion

    #region Loss Functions (GPU Accelerated)

    /// <summary>
    /// GPU-accelerated CrossEntropy loss computation.
    /// </summary>
    T IEngine.CrossEntropyLoss<T>(Tensor<T> predictions, Tensor<T> targets)
    {
        if (!TryGetBackend(out var backend))
            return base.CrossEntropyLoss(predictions, targets);

        try
        {
            // Assume predictions: [batch, numClasses], targets: [batch] or [batch, numClasses]
            if (predictions.Rank != 2)
                return base.CrossEntropyLoss(predictions, targets);

            int batchSize = predictions.Shape[0];
            int numClasses = predictions.Shape[1];

            using var predBuffer = GetOrAllocateBuffer(backend, predictions.Data);
            using var targetBuffer = GetOrAllocateBuffer(backend, targets.Data);

            float loss = backend.CrossEntropyLoss(predBuffer.Buffer, targetBuffer.Buffer, batchSize, numClasses);
            return DirectGpuEngine.FromFloatArray<T>(new[] { loss })[0];
        }
        catch
        {
            return base.CrossEntropyLoss(predictions, targets);
        }
    }

    /// <summary>
    /// GPU-accelerated CrossEntropy backward computation.
    /// </summary>
    Tensor<T> IEngine.CrossEntropyBackward<T>(Tensor<T> predictions, Tensor<T> targets)
    {
        if (!TryGetBackend(out var backend))
            return base.CrossEntropyBackward(predictions, targets);

        try
        {
            if (predictions.Rank != 2)
                return base.CrossEntropyBackward(predictions, targets);

            int batchSize = predictions.Shape[0];
            int numClasses = predictions.Shape[1];

            using var predBuffer = GetOrAllocateBuffer(backend, predictions.Data);
            using var targetBuffer = GetOrAllocateBuffer(backend, targets.Data);
            using var gradInputBuffer = AllocateOutputBuffer(backend, predictions.Length);

            backend.CrossEntropyBackward(predBuffer.Buffer, targetBuffer.Buffer, gradInputBuffer.Buffer, batchSize, numClasses);
            backend.Synchronize();

            float[] resultFloat = backend.DownloadBuffer(gradInputBuffer.Buffer);
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), predictions.Shape.ToArray());
        }
        catch
        {
            return base.CrossEntropyBackward(predictions, targets);
        }
    }

    /// <summary>
    /// GPU-accelerated MSE loss computation.
    /// </summary>
    T IEngine.MseLoss<T>(Tensor<T> predictions, Tensor<T> targets)
    {
        if (!TryGetBackend(out var backend))
            return base.MseLoss(predictions, targets);

        try
        {
            int size = predictions.Length;

            using var predBuffer = GetOrAllocateBuffer(backend, predictions.Data);
            using var targetBuffer = GetOrAllocateBuffer(backend, targets.Data);

            float loss = backend.MseLoss(predBuffer.Buffer, targetBuffer.Buffer, size);
            return DirectGpuEngine.FromFloatArray<T>(new[] { loss })[0];
        }
        catch
        {
            return base.MseLoss(predictions, targets);
        }
    }

    /// <summary>
    /// GPU-accelerated MSE backward computation.
    /// </summary>
    Tensor<T> IEngine.MseBackward<T>(Tensor<T> predictions, Tensor<T> targets)
    {
        if (!TryGetBackend(out var backend))
            return base.MseBackward(predictions, targets);

        try
        {
            int size = predictions.Length;

            using var predBuffer = GetOrAllocateBuffer(backend, predictions.Data);
            using var targetBuffer = GetOrAllocateBuffer(backend, targets.Data);
            using var gradInputBuffer = AllocateOutputBuffer(backend, size);

            backend.MseBackward(predBuffer.Buffer, targetBuffer.Buffer, gradInputBuffer.Buffer, size);
            backend.Synchronize();

            float[] resultFloat = backend.DownloadBuffer(gradInputBuffer.Buffer);
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), predictions.Shape.ToArray());
        }
        catch
        {
            return base.MseBackward(predictions, targets);
        }
    }

    #endregion

    #region Activation Backward Operations (GPU Accelerated)

    /// <summary>
    /// GPU-accelerated ReLU backward operation.
    /// </summary>
    Tensor<T> IEngine.ReluBackward<T>(Tensor<T> gradOutput, Tensor<T> input)
    {
        if (!TryGetBackend(out var backend))
            return base.ReluBackward(gradOutput, input);

        try
        {
            int size = gradOutput.Length;

            using var gradOutBuffer = GetOrAllocateBuffer(backend, gradOutput.Data);
            using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
            using var gradInputBuffer = AllocateOutputBuffer(backend, size);

            backend.ReluBackward(gradOutBuffer.Buffer, inputBuffer.Buffer, gradInputBuffer.Buffer, size);
            backend.Synchronize();

            float[] resultFloat = backend.DownloadBuffer(gradInputBuffer.Buffer);
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), gradOutput.Shape.ToArray());
        }
        catch
        {
            return base.ReluBackward(gradOutput, input);
        }
    }

    /// <summary>
    /// GPU-accelerated Sigmoid backward operation.
    /// </summary>
    Tensor<T> IEngine.SigmoidBackward<T>(Tensor<T> gradOutput, Tensor<T> output)
    {
        if (!TryGetBackend(out var backend))
            return base.SigmoidBackward(gradOutput, output);

        try
        {
            int size = gradOutput.Length;

            using var gradOutBuffer = GetOrAllocateBuffer(backend, gradOutput.Data);
            using var outputBuffer = GetOrAllocateBuffer(backend, output.Data);
            using var gradInputBuffer = AllocateOutputBuffer(backend, size);

            backend.SigmoidBackward(gradOutBuffer.Buffer, outputBuffer.Buffer, gradInputBuffer.Buffer, size);
            backend.Synchronize();

            float[] resultFloat = backend.DownloadBuffer(gradInputBuffer.Buffer);
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), gradOutput.Shape.ToArray());
        }
        catch
        {
            return base.SigmoidBackward(gradOutput, output);
        }
    }

    /// <summary>
    /// GPU-accelerated Tanh backward operation.
    /// </summary>
    Tensor<T> IEngine.TanhBackward<T>(Tensor<T> gradOutput, Tensor<T> output)
    {
        if (!TryGetBackend(out var backend))
            return base.TanhBackward(gradOutput, output);

        try
        {
            int size = gradOutput.Length;

            using var gradOutBuffer = GetOrAllocateBuffer(backend, gradOutput.Data);
            using var outputBuffer = GetOrAllocateBuffer(backend, output.Data);
            using var gradInputBuffer = AllocateOutputBuffer(backend, size);

            backend.TanhBackward(gradOutBuffer.Buffer, outputBuffer.Buffer, gradInputBuffer.Buffer, size);
            backend.Synchronize();

            float[] resultFloat = backend.DownloadBuffer(gradInputBuffer.Buffer);
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), gradOutput.Shape.ToArray());
        }
        catch
        {
            return base.TanhBackward(gradOutput, output);
        }
    }

    /// <summary>
    /// GPU-accelerated GELU backward operation.
    /// </summary>
    Tensor<T> IEngine.GeluBackward<T>(Tensor<T> gradOutput, Tensor<T> input)
    {
        if (!TryGetBackend(out var backend))
            return base.GeluBackward(gradOutput, input);

        try
        {
            int size = gradOutput.Length;

            using var gradOutBuffer = GetOrAllocateBuffer(backend, gradOutput.Data);
            using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
            using var gradInputBuffer = AllocateOutputBuffer(backend, size);

            backend.GeluBackward(gradOutBuffer.Buffer, inputBuffer.Buffer, gradInputBuffer.Buffer, size);
            backend.Synchronize();

            float[] resultFloat = backend.DownloadBuffer(gradInputBuffer.Buffer);
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), gradOutput.Shape.ToArray());
        }
        catch
        {
            return base.GeluBackward(gradOutput, input);
        }
    }

    /// <summary>
    /// GPU-accelerated LeakyReLU activation.
    /// </summary>
    Tensor<T> IEngine.LeakyReLU<T>(Tensor<T> input, T alpha)
    {
        if (!TryGetBackend(out var backend))
            return base.LeakyReLU(input, alpha);

        try
        {
            int size = input.Length;
            var numOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
            float negativeSlope = (float)numOps.ToDouble(alpha);

            using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
            using var outputBuffer = AllocateOutputBuffer(backend, size);

            backend.LeakyRelu(inputBuffer.Buffer, outputBuffer.Buffer, negativeSlope, size);
            backend.Synchronize();

            float[] resultFloat = backend.DownloadBuffer(outputBuffer.Buffer);
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), input.Shape.ToArray());
        }
        catch
        {
            return base.LeakyReLU(input, alpha);
        }
    }

    /// <summary>
    /// GPU-accelerated LeakyReLU backward operation.
    /// </summary>
    Tensor<T> IEngine.LeakyReluBackward<T>(Tensor<T> gradOutput, Tensor<T> input, double negativeSlope)
    {
        if (!TryGetBackend(out var backend))
            return base.LeakyReluBackward(gradOutput, input, negativeSlope);

        try
        {
            int size = gradOutput.Length;

            using var gradOutBuffer = GetOrAllocateBuffer(backend, gradOutput.Data);
            using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
            using var gradInputBuffer = AllocateOutputBuffer(backend, size);

            backend.LeakyReluBackward(gradOutBuffer.Buffer, inputBuffer.Buffer, gradInputBuffer.Buffer, (float)negativeSlope, size);
            backend.Synchronize();

            float[] resultFloat = backend.DownloadBuffer(gradInputBuffer.Buffer);
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), gradOutput.Shape.ToArray());
        }
        catch
        {
            return base.LeakyReluBackward(gradOutput, input, negativeSlope);
        }
    }

    /// <summary>
    /// GPU-accelerated ELU activation.
    /// </summary>
    Tensor<T> IEngine.ELU<T>(Tensor<T> input, double alpha)
    {
        if (!TryGetBackend(out var backend))
            return base.ELU(input, alpha);

        try
        {
            int size = input.Length;

            using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
            using var outputBuffer = AllocateOutputBuffer(backend, size);

            backend.Elu(inputBuffer.Buffer, outputBuffer.Buffer, (float)alpha, size);
            backend.Synchronize();

            float[] resultFloat = backend.DownloadBuffer(outputBuffer.Buffer);
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), input.Shape.ToArray());
        }
        catch
        {
            return base.ELU(input, alpha);
        }
    }

    /// <summary>
    /// GPU-accelerated Swish activation.
    /// </summary>
    Tensor<T> IEngine.Swish<T>(Tensor<T> input)
    {
        if (!TryGetBackend(out var backend))
            return base.Swish(input);

        try
        {
            int size = input.Length;

            using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
            using var outputBuffer = AllocateOutputBuffer(backend, size);

            backend.Swish(inputBuffer.Buffer, outputBuffer.Buffer, size);
            backend.Synchronize();

            float[] resultFloat = backend.DownloadBuffer(outputBuffer.Buffer);
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), input.Shape.ToArray());
        }
        catch
        {
            return base.Swish(input);
        }
    }

    /// <summary>
    /// GPU-accelerated Mish activation.
    /// </summary>
    Tensor<T> IEngine.Mish<T>(Tensor<T> input)
    {
        if (!TryGetBackend(out var backend))
            return base.Mish(input);

        try
        {
            int size = input.Length;

            using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
            using var outputBuffer = AllocateOutputBuffer(backend, size);

            backend.Mish(inputBuffer.Buffer, outputBuffer.Buffer, size);
            backend.Synchronize();

            float[] resultFloat = backend.DownloadBuffer(outputBuffer.Buffer);
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), input.Shape.ToArray());
        }
        catch
        {
            return base.Mish(input);
        }
    }

    /// <summary>
    /// GPU-accelerated Softplus activation.
    /// </summary>
    Tensor<T> IEngine.Softplus<T>(Tensor<T> input)
    {
        if (!TryGetBackend(out var backend))
            return base.Softplus(input);

        try
        {
            int size = input.Length;

            using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
            using var outputBuffer = AllocateOutputBuffer(backend, size);

            backend.Softplus(inputBuffer.Buffer, outputBuffer.Buffer, size);
            backend.Synchronize();

            float[] resultFloat = backend.DownloadBuffer(outputBuffer.Buffer);
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), input.Shape.ToArray());
        }
        catch
        {
            return base.Softplus(input);
        }
    }

    /// <summary>
    /// GPU-accelerated HardSwish activation.
    /// </summary>
    Tensor<T> IEngine.HardSwish<T>(Tensor<T> input)
    {
        if (!TryGetBackend(out var backend))
            return base.HardSwish(input);

        try
        {
            int size = input.Length;

            using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
            using var outputBuffer = AllocateOutputBuffer(backend, size);

            backend.Hardswish(inputBuffer.Buffer, outputBuffer.Buffer, size);
            backend.Synchronize();

            float[] resultFloat = backend.DownloadBuffer(outputBuffer.Buffer);
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), input.Shape.ToArray());
        }
        catch
        {
            return base.HardSwish(input);
        }
    }

    #endregion

    #region Convolution Backward Operations (GPU Accelerated)

    /// <summary>
    /// GPU-accelerated Conv2D backward for input gradients.
    /// </summary>
    Tensor<T> IEngine.Conv2DBackwardInput<T>(Tensor<T> gradOutput, Tensor<T> kernel, int[] inputShape,
        int[] stride, int[] padding, int[] dilation)
    {
        if (!TryGetBackend(out var backend))
            return base.Conv2DBackwardInput(gradOutput, kernel, inputShape, stride, padding, dilation);

        try
        {
            if (gradOutput.Rank != 4 || kernel.Rank != 4)
                return base.Conv2DBackwardInput(gradOutput, kernel, inputShape, stride, padding, dilation);

            int strideH = stride.Length > 0 ? stride[0] : 1;
            int strideW = stride.Length > 1 ? stride[1] : strideH;
            int padH = padding.Length > 0 ? padding[0] : 0;
            int padW = padding.Length > 1 ? padding[1] : padH;
            int dilationH = dilation.Length > 0 ? dilation[0] : 1;
            int dilationW = dilation.Length > 1 ? dilation[1] : dilationH;

            int batch = gradOutput.Shape[0];
            int outChannels = gradOutput.Shape[1];
            int outHeight = gradOutput.Shape[2];
            int outWidth = gradOutput.Shape[3];

            int inChannels = inputShape[1];
            int inHeight = inputShape[2];
            int inWidth = inputShape[3];

            int kernelH = kernel.Shape[2];
            int kernelW = kernel.Shape[3];

            using var gradOutBuffer = GetOrAllocateBuffer(backend, gradOutput.Data);
            using var kernelBuffer = GetOrAllocateBuffer(backend, kernel.Data);
            using var gradInputBuffer = AllocateOutputBuffer(backend, batch * inChannels * inHeight * inWidth);

            backend.Conv2DBackwardInput(gradOutBuffer.Buffer, kernelBuffer.Buffer, gradInputBuffer.Buffer,
                batch, inChannels, inHeight, inWidth, outChannels, outHeight, outWidth,
                kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW);
            backend.Synchronize();

            float[] resultFloat = backend.DownloadBuffer(gradInputBuffer.Buffer);
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), inputShape);
        }
        catch
        {
            return base.Conv2DBackwardInput(gradOutput, kernel, inputShape, stride, padding, dilation);
        }
    }

    /// <summary>
    /// GPU-accelerated Conv2D backward for kernel gradients.
    /// </summary>
    Tensor<T> IEngine.Conv2DBackwardKernel<T>(Tensor<T> gradOutput, Tensor<T> input, int[] kernelShape,
        int[] stride, int[] padding, int[] dilation)
    {
        if (!TryGetBackend(out var backend))
            return base.Conv2DBackwardKernel(gradOutput, input, kernelShape, stride, padding, dilation);

        try
        {
            if (input.Rank != 4 || gradOutput.Rank != 4)
                return base.Conv2DBackwardKernel(gradOutput, input, kernelShape, stride, padding, dilation);

            int strideH = stride.Length > 0 ? stride[0] : 1;
            int strideW = stride.Length > 1 ? stride[1] : strideH;
            int padH = padding.Length > 0 ? padding[0] : 0;
            int padW = padding.Length > 1 ? padding[1] : padH;
            int dilationH = dilation.Length > 0 ? dilation[0] : 1;
            int dilationW = dilation.Length > 1 ? dilation[1] : dilationH;

            int batch = input.Shape[0];
            int inChannels = input.Shape[1];
            int inHeight = input.Shape[2];
            int inWidth = input.Shape[3];

            int outChannels = gradOutput.Shape[1];
            int outHeight = gradOutput.Shape[2];
            int outWidth = gradOutput.Shape[3];

            int kernelH = kernelShape[2];
            int kernelW = kernelShape[3];

            using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
            using var gradOutBuffer = GetOrAllocateBuffer(backend, gradOutput.Data);
            using var gradKernelBuffer = AllocateOutputBuffer(backend, outChannels * inChannels * kernelH * kernelW);

            backend.Conv2DBackwardKernel(inputBuffer.Buffer, gradOutBuffer.Buffer, gradKernelBuffer.Buffer,
                batch, inChannels, inHeight, inWidth, outChannels, outHeight, outWidth,
                kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW);
            backend.Synchronize();

            float[] resultFloat = backend.DownloadBuffer(gradKernelBuffer.Buffer);
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), kernelShape);
        }
        catch
        {
            return base.Conv2DBackwardKernel(gradOutput, input, kernelShape, stride, padding, dilation);
        }
    }

    #endregion

    #region Global Pooling Operations (GPU Accelerated)

    /// <summary>
    /// GPU-accelerated Global Average Pooling.
    /// </summary>
    Tensor<T> IEngine.GlobalAvgPool2D<T>(Tensor<T> input)
    {
        if (!TryGetBackend(out var backend))
            return base.GlobalAvgPool2D(input);

        try
        {
            if (input.Rank != 4)
                return base.GlobalAvgPool2D(input);

            int batch = input.Shape[0];
            int channels = input.Shape[1];
            int height = input.Shape[2];
            int width = input.Shape[3];

            using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
            using var outputBuffer = AllocateOutputBuffer(backend, batch * channels);

            backend.GlobalAvgPool2D(inputBuffer.Buffer, outputBuffer.Buffer, batch, channels, height, width);
            backend.Synchronize();

            float[] resultFloat = backend.DownloadBuffer(outputBuffer.Buffer);
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), new[] { batch, channels, 1, 1 });
        }
        catch
        {
            return base.GlobalAvgPool2D(input);
        }
    }

    /// <summary>
    /// GPU-accelerated Global Max Pooling.
    /// </summary>
    Tensor<T> IEngine.GlobalMaxPool2D<T>(Tensor<T> input)
    {
        if (!TryGetBackend(out var backend))
            return base.GlobalMaxPool2D(input);

        try
        {
            if (input.Rank != 4)
                return base.GlobalMaxPool2D(input);

            int batch = input.Shape[0];
            int channels = input.Shape[1];
            int height = input.Shape[2];
            int width = input.Shape[3];

            using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
            using var outputBuffer = AllocateOutputBuffer(backend, batch * channels);

            backend.GlobalMaxPool2D(inputBuffer.Buffer, outputBuffer.Buffer, batch, channels, height, width);
            backend.Synchronize();

            float[] resultFloat = backend.DownloadBuffer(outputBuffer.Buffer);
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), new[] { batch, channels, 1, 1 });
        }
        catch
        {
            return base.GlobalMaxPool2D(input);
        }
    }

    /// <summary>
    /// GPU-accelerated Adaptive Average Pooling.
    /// </summary>
    Tensor<T> IEngine.AdaptiveAvgPool2D<T>(Tensor<T> input, int outputHeight, int outputWidth)
    {
        if (!TryGetBackend(out var backend))
            return base.AdaptiveAvgPool2D(input, outputHeight, outputWidth);

        try
        {
            if (input.Rank != 4)
                return base.AdaptiveAvgPool2D(input, outputHeight, outputWidth);

            int batch = input.Shape[0];
            int channels = input.Shape[1];
            int inHeight = input.Shape[2];
            int inWidth = input.Shape[3];

            using var inputBuffer = GetOrAllocateBuffer(backend, input.Data);
            using var outputBuffer = AllocateOutputBuffer(backend, batch * channels * outputHeight * outputWidth);

            backend.AdaptiveAvgPool2D(inputBuffer.Buffer, outputBuffer.Buffer, batch, channels, inHeight, inWidth, outputHeight, outputWidth);
            backend.Synchronize();

            float[] resultFloat = backend.DownloadBuffer(outputBuffer.Buffer);
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(resultFloat), new[] { batch, channels, outputHeight, outputWidth });
        }
        catch
        {
            return base.AdaptiveAvgPool2D(input, outputHeight, outputWidth);
        }
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
