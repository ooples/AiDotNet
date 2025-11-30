using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics.Arm;
using AiDotNet.Autodiff;
using AiDotNet.JitCompiler.CodeGen;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.JitCompiler.Runtime;

/// <summary>
/// Runtime support for vectorized SIMD operations.
/// </summary>
/// <remarks>
/// <para>
/// This class provides runtime implementations for vectorized operations that use
/// SIMD (Single Instruction Multiple Data) instructions. Modern CPUs can process
/// multiple data elements in parallel using vector registers (SSE, AVX, AVX-512).
/// </para>
/// <para><b>For Beginners:</b> These operations use special CPU instructions for speed.
///
/// Your CPU has vector registers that can hold multiple numbers:
/// - SSE: 4 floats at once (128-bit registers)
/// - AVX: 8 floats at once (256-bit registers)
/// - AVX-512: 16 floats at once (512-bit registers)
///
/// Instead of adding numbers one at a time:
///   a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]
///
/// SIMD does all 4 additions with one instruction!
/// This can make operations 4-16x faster for large arrays.
/// </para>
/// </remarks>
public static class VectorizedOps
{
    /// <summary>
    /// Binary operation types supported by vectorized operations.
    /// </summary>
    public enum BinaryOperation
    {
        Add,
        Subtract,
        Multiply,
        Divide
    }

    /// <summary>
    /// Unary operation types supported by vectorized operations.
    /// </summary>
    public enum UnaryOperation
    {
        Negate,
        Exp,
        Log,
        Sqrt,
        ReLU,
        Sigmoid,
        Tanh
    }

    /// <summary>
    /// Reduction operation types supported by vectorized operations.
    /// </summary>
    public enum ReductionOperation
    {
        Sum,
        Mean,
        Max,
        Min
    }

    /// <summary>
    /// Executes a vectorized binary operation (Add, Subtract, Multiply, Divide).
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="left">Left operand tensor.</param>
    /// <param name="right">Right operand tensor.</param>
    /// <param name="operation">The operation to perform.</param>
    /// <param name="vectorWidth">The SIMD vector width (unused, kept for API compatibility).</param>
    /// <param name="numVectors">Number of full vectors to process (unused, kept for API compatibility).</param>
    /// <param name="remainder">Number of remaining scalar elements (unused, kept for API compatibility).</param>
    /// <returns>Result tensor.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Tensor<T> ExecuteVectorizedBinary<T>(
        Tensor<T> left,
        Tensor<T> right,
        BinaryOperation operation,
        int vectorWidth,
        int numVectors,
        int remainder)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var leftSpan = left.AsSpan();
        var rightSpan = right.AsSpan();
        var result = new T[leftSpan.Length];
        var resultSpan = result.AsSpan();

        // Use INumericOperations<T> vectorized operations (follows OCP)
        switch (operation)
        {
            case BinaryOperation.Add:
                numOps.Add(leftSpan, rightSpan, resultSpan);
                break;
            case BinaryOperation.Subtract:
                numOps.Subtract(leftSpan, rightSpan, resultSpan);
                break;
            case BinaryOperation.Multiply:
                numOps.Multiply(leftSpan, rightSpan, resultSpan);
                break;
            case BinaryOperation.Divide:
                numOps.Divide(leftSpan, rightSpan, resultSpan);
                break;
        }

        return new Tensor<T>(result, left.Shape);
    }

    /// <summary>
    /// Executes a vectorized binary operation using string-based dispatch.
    /// </summary>
    /// <remarks>
    /// This overload is provided for backward compatibility. Prefer using the enum-based overload
    /// for better type safety and performance.
    /// </remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Tensor<T> ExecuteVectorizedBinary<T>(
        Tensor<T> left,
        Tensor<T> right,
        string operation,
        int vectorWidth,
        int numVectors,
        int remainder)
    {
        var op = ParseBinaryOperation(operation);
        return ExecuteVectorizedBinary(left, right, op, vectorWidth, numVectors, remainder);
    }

    /// <summary>
    /// Executes a vectorized unary operation (Negate, Exp, Log, ReLU, etc.).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Tensor<T> ExecuteVectorizedUnary<T>(
        Tensor<T> input,
        UnaryOperation operation,
        int vectorWidth,
        int numVectors,
        int remainder)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var inputSpan = input.AsSpan();
        var result = new T[inputSpan.Length];
        var resultSpan = result.AsSpan();

        // Use INumericOperations<T> vectorized operations (follows OCP)
        switch (operation)
        {
            case UnaryOperation.Negate:
                for (int i = 0; i < inputSpan.Length; i++)
                {
                    result[i] = numOps.Negate(inputSpan[i]);
                }
                break;
            case UnaryOperation.Exp:
                numOps.Exp(inputSpan, resultSpan);
                break;
            case UnaryOperation.Log:
                numOps.Log(inputSpan, resultSpan);
                break;
            case UnaryOperation.Sqrt:
                for (int i = 0; i < inputSpan.Length; i++)
                {
                    result[i] = numOps.Sqrt(inputSpan[i]);
                }
                break;
            case UnaryOperation.ReLU:
                ExecuteReLU(inputSpan, resultSpan, numOps);
                break;
            case UnaryOperation.Sigmoid:
                numOps.Sigmoid(inputSpan, resultSpan);
                break;
            case UnaryOperation.Tanh:
                numOps.Tanh(inputSpan, resultSpan);
                break;
        }

        return new Tensor<T>(result, input.Shape);
    }

    /// <summary>
    /// Executes a vectorized unary operation using string-based dispatch.
    /// </summary>
    /// <remarks>
    /// This overload is provided for backward compatibility. Prefer using the enum-based overload
    /// for better type safety and performance.
    /// </remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Tensor<T> ExecuteVectorizedUnary<T>(
        Tensor<T> input,
        string operation,
        int vectorWidth,
        int numVectors,
        int remainder)
    {
        var op = ParseUnaryOperation(operation);
        return ExecuteVectorizedUnary(input, op, vectorWidth, numVectors, remainder);
    }

    /// <summary>
    /// Executes ReLU (Rectified Linear Unit) activation function.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ExecuteReLU<T>(ReadOnlySpan<T> input, Span<T> result, INumericOperations<T> numOps)
    {
        var zero = numOps.Zero;
        for (int i = 0; i < input.Length; i++)
        {
            result[i] = numOps.GreaterThan(input[i], zero) ? input[i] : zero;
        }
    }

    /// <summary>
    /// Executes a vectorized reduction operation (Sum, Mean, Max, Min).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Tensor<T> ExecuteVectorizedReduction<T>(
        Tensor<T> input,
        ReductionOperation reductionType,
        int vectorWidth,
        int[]? axes,
        bool keepDims)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var inputSpan = input.AsSpan();

        // Simple case: reduce all elements
        if (axes == null || axes.Length == 0 || axes.Length == input.Shape.Length)
        {
            T result = ComputeFullReduction(inputSpan, reductionType, numOps);
            var resultArray = new T[1];
            resultArray[0] = result;
            var resultShape = keepDims ? CreateKeepDimsShape(input.Shape.Length) : new[] { 1 };
            return new Tensor<T>(resultArray, resultShape);
        }

        // Axis-specific reduction
        return ReduceAlongAxes(input, axes, reductionType, keepDims, numOps);
    }

    /// <summary>
    /// Executes a vectorized reduction operation using string-based dispatch.
    /// </summary>
    /// <remarks>
    /// This overload is provided for backward compatibility. Prefer using the enum-based overload
    /// for better type safety and performance.
    /// </remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Tensor<T> ExecuteVectorizedReduction<T>(
        Tensor<T> input,
        string reductionType,
        int vectorWidth,
        int[]? axes,
        bool keepDims)
    {
        var op = ParseReductionOperation(reductionType);
        return ExecuteVectorizedReduction(input, op, vectorWidth, axes, keepDims);
    }

    /// <summary>
    /// Computes a full reduction over all elements.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static T ComputeFullReduction<T>(ReadOnlySpan<T> data, ReductionOperation operation, INumericOperations<T> numOps)
    {
        switch (operation)
        {
            case ReductionOperation.Sum:
                return numOps.Sum(data);
            case ReductionOperation.Mean:
                var sum = numOps.Sum(data);
                return numOps.Divide(sum, numOps.FromDouble(data.Length));
            case ReductionOperation.Max:
                return numOps.Max(data);
            case ReductionOperation.Min:
                return numOps.Min(data);
            default:
                return numOps.Sum(data);
        }
    }

    /// <summary>
    /// Creates a shape array with all ones for keepDims.
    /// </summary>
    private static int[] CreateKeepDimsShape(int rank)
    {
        var shape = new int[rank];
        for (int i = 0; i < rank; i++)
            shape[i] = 1;
        return shape;
    }

    /// <summary>
    /// Executes a vectorized matrix multiplication with tiling.
    /// </summary>
#if NETCOREAPP3_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
#else
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
#endif
    public static Tensor<T> ExecuteVectorizedMatMul<T>(
        Tensor<T> left,
        Tensor<T> right,
        int vectorWidth,
        int tileSize)
    {
        // Validate shapes
        if (left.Shape.Length != 2 || right.Shape.Length != 2)
            throw new ArgumentException("MatMul requires 2D tensors");

        int M = left.Shape[0];
        int K = left.Shape[1];
        int N = right.Shape[1];

        if (K != right.Shape[0])
            throw new ArgumentException("Inner dimensions must match for matrix multiplication");

        var result = new T[M * N];
        var leftSpan = left.AsSpan();
        var rightSpan = right.AsSpan();
        var resultSpan = result.AsSpan();

        // Use type-specific implementations for float/double for better SIMD utilization
        if (typeof(T) == typeof(float))
        {
            ExecuteVectorizedMatMulFloat(
                MemoryMarshal.Cast<T, float>(leftSpan),
                MemoryMarshal.Cast<T, float>(rightSpan),
                MemoryMarshal.Cast<T, float>(resultSpan),
                M, K, N, tileSize);
        }
        else if (typeof(T) == typeof(double))
        {
            ExecuteVectorizedMatMulDouble(
                MemoryMarshal.Cast<T, double>(leftSpan),
                MemoryMarshal.Cast<T, double>(rightSpan),
                MemoryMarshal.Cast<T, double>(resultSpan),
                M, K, N, tileSize);
        }
        else
        {
            // Generic fallback using INumericOperations<T>
            ExecuteGenericMatMul(leftSpan, rightSpan, resultSpan, M, K, N);
        }

        return new Tensor<T>(result, new[] { M, N });
    }

#if NETCOREAPP3_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
#else
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
#endif
    private static void ExecuteVectorizedMatMulFloat(
        ReadOnlySpan<float> A,
        ReadOnlySpan<float> B,
        Span<float> C,
        int M, int K, int N,
        int tileSize)
    {
        // Initialize result to zero
        C.Clear();

        // Tiled matrix multiplication with SIMD
        var caps = SIMDCapabilities.Detect();
        int vectorCount = caps.GetVectorCount(sizeof(float));

        for (int i0 = 0; i0 < M; i0 += tileSize)
        {
            int iEnd = Math.Min(i0 + tileSize, M);

            for (int k0 = 0; k0 < K; k0 += tileSize)
            {
                int kEnd = Math.Min(k0 + tileSize, K);

                for (int j0 = 0; j0 < N; j0 += tileSize)
                {
                    int jEnd = Math.Min(j0 + tileSize, N);

                    // Inner loop - vectorized
                    for (int i = i0; i < iEnd; i++)
                    {
                        for (int k = k0; k < kEnd; k++)
                        {
                            float aik = A[i * K + k];
                            var aVec = new Vector<float>(aik);

                            int j = j0;
                            int jVecEnd = j0 + ((jEnd - j0) / vectorCount) * vectorCount;

                            // Vectorized inner loop
                            for (; j < jVecEnd; j += vectorCount)
                            {
                                int cIdx = i * N + j;
                                int bIdx = k * N + j;

                                var bVec = new Vector<float>(B.Slice(bIdx, vectorCount));
                                var cVec = new Vector<float>(C.Slice(cIdx, vectorCount));

                                (cVec + aVec * bVec).CopyTo(C.Slice(cIdx, vectorCount));
                            }

                            // Scalar remainder
                            for (; j < jEnd; j++)
                            {
                                C[i * N + j] += aik * B[k * N + j];
                            }
                        }
                    }
                }
            }
        }
    }

#if NETCOREAPP3_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
#else
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
#endif
    private static void ExecuteVectorizedMatMulDouble(
        ReadOnlySpan<double> A,
        ReadOnlySpan<double> B,
        Span<double> C,
        int M, int K, int N,
        int tileSize)
    {
        C.Clear();
        var caps = SIMDCapabilities.Detect();
        int vectorCount = caps.GetVectorCount(sizeof(double));

        for (int i0 = 0; i0 < M; i0 += tileSize)
        {
            int iEnd = Math.Min(i0 + tileSize, M);

            for (int k0 = 0; k0 < K; k0 += tileSize)
            {
                int kEnd = Math.Min(k0 + tileSize, K);

                for (int j0 = 0; j0 < N; j0 += tileSize)
                {
                    int jEnd = Math.Min(j0 + tileSize, N);

                    for (int i = i0; i < iEnd; i++)
                    {
                        for (int k = k0; k < kEnd; k++)
                        {
                            double aik = A[i * K + k];
                            var aVec = new Vector<double>(aik);

                            int j = j0;
                            int jVecEnd = j0 + ((jEnd - j0) / vectorCount) * vectorCount;

                            for (; j < jVecEnd; j += vectorCount)
                            {
                                int cIdx = i * N + j;
                                int bIdx = k * N + j;

                                var bVec = new Vector<double>(B.Slice(bIdx, vectorCount));
                                var cVec = new Vector<double>(C.Slice(cIdx, vectorCount));

                                (cVec + aVec * bVec).CopyTo(C.Slice(cIdx, vectorCount));
                            }

                            for (; j < jEnd; j++)
                            {
                                C[i * N + j] += aik * B[k * N + j];
                            }
                        }
                    }
                }
            }
        }
    }

    /// <summary>
    /// Generic matrix multiplication using INumericOperations for any numeric type.
    /// </summary>
    private static void ExecuteGenericMatMul<T>(
        ReadOnlySpan<T> A,
        ReadOnlySpan<T> B,
        Span<T> C,
        int M, int K, int N)
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                T sum = numOps.Zero;
                for (int k = 0; k < K; k++)
                {
                    sum = numOps.Add(sum, numOps.Multiply(A[i * K + k], B[k * N + j]));
                }
                C[i * N + j] = sum;
            }
        }
    }

    /// <summary>
    /// Reduces a tensor along specific axes.
    /// </summary>
    private static Tensor<T> ReduceAlongAxes<T>(
        Tensor<T> input,
        int[] axes,
        ReductionOperation reductionType,
        bool keepDims,
        INumericOperations<T> numOps)
    {
        var inputShape = input.Shape;
        var inputSpan = input.AsSpan();

        // Normalize negative axes
        var normalizedAxes = NormalizeAxes(axes, inputShape.Length);

        // Calculate output shape
        var outputShape = CalculateOutputShape(inputShape, normalizedAxes, keepDims);
        var outputSize = outputShape.Aggregate(1, (a, b) => a * b);

        // Initialize result array and accumulator counts
        var result = new T[outputSize];
        var counts = new int[outputSize];

        // Initialize based on reduction type
        T initValue = reductionType switch
        {
            ReductionOperation.Sum or ReductionOperation.Mean => numOps.Zero,
            ReductionOperation.Max => numOps.MinValue,
            ReductionOperation.Min => numOps.MaxValue,
            _ => numOps.Zero
        };

        for (int i = 0; i < outputSize; i++)
        {
            result[i] = initValue;
        }

        // Calculate strides for input tensor
        var inputStrides = CalculateStrides(inputShape);
        var outputStrides = CalculateStrides(outputShape);
        var outputShapeWithKeptDims = CalculateOutputShape(inputShape, normalizedAxes, true);

        // Iterate through all input elements
        var inputCoords = new int[inputShape.Length];
        for (int flatIdx = 0; flatIdx < inputSpan.Length; flatIdx++)
        {
            // Convert flat index to coordinates
            FlatIndexToCoords(flatIdx, inputShape, inputStrides, inputCoords);

            // Calculate output index by projecting out the reduced axes
            int outputIdx = CoordsToOutputIndex(inputCoords, normalizedAxes, outputShapeWithKeptDims, outputStrides, keepDims);

            // Apply reduction
            T inputValue = inputSpan[flatIdx];
            result[outputIdx] = ApplyReduction(result[outputIdx], inputValue, reductionType, numOps);
            counts[outputIdx]++;
        }

        // Finalize for mean reduction
        if (reductionType == ReductionOperation.Mean)
        {
            for (int i = 0; i < outputSize; i++)
            {
                if (counts[i] > 0)
                {
                    result[i] = numOps.Divide(result[i], numOps.FromDouble(counts[i]));
                }
            }
        }

        return new Tensor<T>(result, outputShape);
    }

    /// <summary>
    /// Normalizes axes, converting negative indices to positive.
    /// </summary>
    private static int[] NormalizeAxes(int[] axes, int rank)
    {
        var normalized = new int[axes.Length];
        for (int i = 0; i < axes.Length; i++)
        {
            normalized[i] = axes[i] < 0 ? rank + axes[i] : axes[i];
            if (normalized[i] < 0 || normalized[i] >= rank)
            {
                throw new ArgumentOutOfRangeException(nameof(axes), $"Axis {axes[i]} is out of bounds for tensor with rank {rank}");
            }
        }
        return normalized;
    }

    /// <summary>
    /// Calculates the output shape after reduction.
    /// </summary>
    private static int[] CalculateOutputShape(int[] inputShape, int[] axes, bool keepDims)
    {
        var axesSet = new HashSet<int>(axes);
        var outputShape = new List<int>();

        for (int i = 0; i < inputShape.Length; i++)
        {
            if (!axesSet.Contains(i))
            {
                outputShape.Add(inputShape[i]);
            }
            else if (keepDims)
            {
                outputShape.Add(1);
            }
        }

        if (outputShape.Count == 0)
        {
            outputShape.Add(1);
        }

        return outputShape.ToArray();
    }

    /// <summary>
    /// Calculates strides for a given shape.
    /// </summary>
    private static int[] CalculateStrides(int[] shape)
    {
        var strides = new int[shape.Length];
        int stride = 1;
        for (int i = shape.Length - 1; i >= 0; i--)
        {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }

    /// <summary>
    /// Converts a flat index to multi-dimensional coordinates.
    /// </summary>
    private static void FlatIndexToCoords(int flatIndex, int[] shape, int[] strides, int[] coords)
    {
        int remaining = flatIndex;
        for (int i = 0; i < shape.Length; i++)
        {
            coords[i] = remaining / strides[i];
            remaining %= strides[i];
        }
    }

    /// <summary>
    /// Converts input coordinates to output flat index, projecting out reduced axes.
    /// </summary>
    private static int CoordsToOutputIndex(int[] inputCoords, int[] reducedAxes, int[] outputShape, int[] outputStrides, bool keepDims)
    {
        var axesSet = new HashSet<int>(reducedAxes);
        int outputIdx = 0;
        int outputDim = 0;

        for (int i = 0; i < inputCoords.Length; i++)
        {
            if (!axesSet.Contains(i))
            {
                outputIdx += inputCoords[i] * outputStrides[outputDim];
                outputDim++;
            }
            else if (keepDims)
            {
                // For keepDims, the reduced dimension contributes 0 (since shape[dim] = 1)
                outputDim++;
            }
        }

        return outputIdx;
    }

    /// <summary>
    /// Applies a reduction operation between accumulator and value.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static T ApplyReduction<T>(T accumulator, T value, ReductionOperation operation, INumericOperations<T> numOps)
    {
        return operation switch
        {
            ReductionOperation.Sum or ReductionOperation.Mean => numOps.Add(accumulator, value),
            ReductionOperation.Max => numOps.GreaterThan(value, accumulator) ? value : accumulator,
            ReductionOperation.Min => numOps.LessThan(value, accumulator) ? value : accumulator,
            _ => numOps.Add(accumulator, value)
        };
    }

    /// <summary>
    /// Parses a string operation name to BinaryOperation enum.
    /// </summary>
    private static BinaryOperation ParseBinaryOperation(string operation)
    {
        return operation switch
        {
            "Add" => BinaryOperation.Add,
            "Subtract" => BinaryOperation.Subtract,
            "Multiply" => BinaryOperation.Multiply,
            "Divide" => BinaryOperation.Divide,
            _ => throw new ArgumentException($"Unknown binary operation: {operation}", nameof(operation))
        };
    }

    /// <summary>
    /// Parses a string operation name to UnaryOperation enum.
    /// </summary>
    private static UnaryOperation ParseUnaryOperation(string operation)
    {
        return operation switch
        {
            "Negate" => UnaryOperation.Negate,
            "Exp" => UnaryOperation.Exp,
            "Log" => UnaryOperation.Log,
            "Sqrt" => UnaryOperation.Sqrt,
            "ReLU" => UnaryOperation.ReLU,
            "Sigmoid" => UnaryOperation.Sigmoid,
            "Tanh" => UnaryOperation.Tanh,
            _ => throw new ArgumentException($"Unknown unary operation: {operation}", nameof(operation))
        };
    }

    /// <summary>
    /// Parses a string operation name to ReductionOperation enum.
    /// </summary>
    private static ReductionOperation ParseReductionOperation(string operation)
    {
        return operation switch
        {
            "Sum" => ReductionOperation.Sum,
            "Mean" => ReductionOperation.Mean,
            "Max" => ReductionOperation.Max,
            "Min" => ReductionOperation.Min,
            _ => throw new ArgumentException($"Unknown reduction operation: {operation}", nameof(operation))
        };
    }
}
