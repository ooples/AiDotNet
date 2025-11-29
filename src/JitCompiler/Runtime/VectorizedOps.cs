using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using AiDotNet.Autodiff;
using AiDotNet.Tensors.Helpers;

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
    /// Executes a vectorized binary operation (Add, Subtract, Multiply, Divide).
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="left">Left operand tensor.</param>
    /// <param name="right">Right operand tensor.</param>
    /// <param name="operation">The operation to perform.</param>
    /// <param name="vectorWidth">The SIMD vector width.</param>
    /// <param name="numVectors">Number of full vectors to process.</param>
    /// <param name="remainder">Number of remaining scalar elements.</param>
    /// <returns>Result tensor.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Tensor<T> ExecuteVectorizedBinary<T>(
        Tensor<T> left,
        Tensor<T> right,
        string operation,
        int vectorWidth,
        int numVectors,
        int remainder)
    {
        var leftData = left.Data;
        var rightData = right.Data;
        var result = new T[leftData.Length];

        if (typeof(T) == typeof(float))
        {
            ExecuteVectorizedBinaryFloat(
                MemoryMarshal.Cast<T, float>(leftData),
                MemoryMarshal.Cast<T, float>(rightData),
                MemoryMarshal.Cast<T, float>(result),
                operation, vectorWidth, numVectors, remainder);
        }
        else if (typeof(T) == typeof(double))
        {
            ExecuteVectorizedBinaryDouble(
                MemoryMarshal.Cast<T, double>(leftData),
                MemoryMarshal.Cast<T, double>(rightData),
                MemoryMarshal.Cast<T, double>(result),
                operation, vectorWidth, numVectors, remainder);
        }
        else
        {
            // Fallback for non-supported types
            for (int i = 0; i < leftData.Length; i++)
            {
                result[i] = ApplyBinaryScalar(leftData[i], rightData[i], operation);
            }
        }

        return new Tensor<T>(result, left.Shape);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ExecuteVectorizedBinaryFloat(
        ReadOnlySpan<float> left,
        ReadOnlySpan<float> right,
        Span<float> result,
        string operation,
        int vectorWidth,
        int numVectors,
        int remainder)
    {
        int i = 0;

        if (Vector.IsHardwareAccelerated && left.Length >= Vector<float>.Count)
        {
            var count = Vector<float>.Count;
            int vectorizedEnd = (left.Length / count) * count;

            for (; i < vectorizedEnd; i += count)
            {
                var vLeft = new Vector<float>(left.Slice(i, count));
                var vRight = new Vector<float>(right.Slice(i, count));

                Vector<float> vResult = operation switch
                {
                    "Add" => vLeft + vRight,
                    "Subtract" => vLeft - vRight,
                    "Multiply" => vLeft * vRight,
                    "Divide" => vLeft / vRight,
                    _ => vLeft + vRight
                };

                vResult.CopyTo(result.Slice(i, count));
            }
        }

        // Handle remainder with scalar operations
        for (; i < left.Length; i++)
        {
            result[i] = operation switch
            {
                "Add" => left[i] + right[i],
                "Subtract" => left[i] - right[i],
                "Multiply" => left[i] * right[i],
                "Divide" => left[i] / right[i],
                _ => left[i] + right[i]
            };
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ExecuteVectorizedBinaryDouble(
        ReadOnlySpan<double> left,
        ReadOnlySpan<double> right,
        Span<double> result,
        string operation,
        int vectorWidth,
        int numVectors,
        int remainder)
    {
        int i = 0;

        if (Vector.IsHardwareAccelerated && left.Length >= Vector<double>.Count)
        {
            var count = Vector<double>.Count;
            int vectorizedEnd = (left.Length / count) * count;

            for (; i < vectorizedEnd; i += count)
            {
                var vLeft = new Vector<double>(left.Slice(i, count));
                var vRight = new Vector<double>(right.Slice(i, count));

                Vector<double> vResult = operation switch
                {
                    "Add" => vLeft + vRight,
                    "Subtract" => vLeft - vRight,
                    "Multiply" => vLeft * vRight,
                    "Divide" => vLeft / vRight,
                    _ => vLeft + vRight
                };

                vResult.CopyTo(result.Slice(i, count));
            }
        }

        // Handle remainder
        for (; i < left.Length; i++)
        {
            result[i] = operation switch
            {
                "Add" => left[i] + right[i],
                "Subtract" => left[i] - right[i],
                "Multiply" => left[i] * right[i],
                "Divide" => left[i] / right[i],
                _ => left[i] + right[i]
            };
        }
    }

    /// <summary>
    /// Executes a vectorized unary operation (Negate, Exp, Log, ReLU, etc.).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Tensor<T> ExecuteVectorizedUnary<T>(
        Tensor<T> input,
        string operation,
        int vectorWidth,
        int numVectors,
        int remainder)
    {
        var data = input.Data;
        var result = new T[data.Length];

        if (typeof(T) == typeof(float))
        {
            ExecuteVectorizedUnaryFloat(
                MemoryMarshal.Cast<T, float>(data),
                MemoryMarshal.Cast<T, float>(result),
                operation, vectorWidth);
        }
        else if (typeof(T) == typeof(double))
        {
            ExecuteVectorizedUnaryDouble(
                MemoryMarshal.Cast<T, double>(data),
                MemoryMarshal.Cast<T, double>(result),
                operation, vectorWidth);
        }
        else
        {
            // Fallback
            for (int i = 0; i < data.Length; i++)
            {
                result[i] = ApplyUnaryScalar(data[i], operation);
            }
        }

        return new Tensor<T>(result, input.Shape);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ExecuteVectorizedUnaryFloat(
        ReadOnlySpan<float> input,
        Span<float> result,
        string operation,
        int vectorWidth)
    {
        int i = 0;

        if (Vector.IsHardwareAccelerated && input.Length >= Vector<float>.Count)
        {
            var count = Vector<float>.Count;
            var zero = Vector<float>.Zero;
            int vectorizedEnd = (input.Length / count) * count;

            for (; i < vectorizedEnd; i += count)
            {
                var v = new Vector<float>(input.Slice(i, count));

                Vector<float> vResult = operation switch
                {
                    "Negate" => -v,
                    "ReLU" => Vector.Max(zero, v),
                    // For Exp, Log, Sqrt, Sigmoid, Tanh - fall through to scalar
                    // because Vector<T> doesn't have these directly
                    _ => v
                };

                if (operation is "Exp" or "Log" or "Sqrt" or "Sigmoid" or "Tanh")
                {
                    // Use scalar fallback for complex operations
                    for (int j = 0; j < count; j++)
                    {
                        result[i + j] = ApplyUnaryOperation(input[i + j], operation);
                    }
                }
                else
                {
                    vResult.CopyTo(result.Slice(i, count));
                }
            }
        }

        // Handle remainder
        for (; i < input.Length; i++)
        {
            result[i] = ApplyUnaryOperation(input[i], operation);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ExecuteVectorizedUnaryDouble(
        ReadOnlySpan<double> input,
        Span<double> result,
        string operation,
        int vectorWidth)
    {
        int i = 0;

        if (Vector.IsHardwareAccelerated && input.Length >= Vector<double>.Count)
        {
            var count = Vector<double>.Count;
            var zero = Vector<double>.Zero;
            int vectorizedEnd = (input.Length / count) * count;

            for (; i < vectorizedEnd; i += count)
            {
                var v = new Vector<double>(input.Slice(i, count));

                Vector<double> vResult = operation switch
                {
                    "Negate" => -v,
                    "ReLU" => Vector.Max(zero, v),
                    _ => v
                };

                if (operation is "Exp" or "Log" or "Sqrt" or "Sigmoid" or "Tanh")
                {
                    for (int j = 0; j < count; j++)
                    {
                        result[i + j] = ApplyUnaryOperation(input[i + j], operation);
                    }
                }
                else
                {
                    vResult.CopyTo(result.Slice(i, count));
                }
            }
        }

        // Handle remainder
        for (; i < input.Length; i++)
        {
            result[i] = ApplyUnaryOperation(input[i], operation);
        }
    }

    /// <summary>
    /// Executes a vectorized reduction operation (Sum, Mean, Max).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Tensor<T> ExecuteVectorizedReduction<T>(
        Tensor<T> input,
        string reductionType,
        int vectorWidth,
        int[]? axes,
        bool keepDims)
    {
        var data = input.Data;

        // Simple case: reduce all elements
        if (axes == null || axes.Length == 0 || (axes.Length == input.Shape.Length))
        {
            double result;

            if (typeof(T) == typeof(float))
            {
                result = ExecuteVectorizedReductionFloat(
                    MemoryMarshal.Cast<T, float>(data),
                    reductionType);
            }
            else if (typeof(T) == typeof(double))
            {
                result = ExecuteVectorizedReductionDouble(
                    MemoryMarshal.Cast<T, double>(data),
                    reductionType);
            }
            else
            {
                result = 0;
                var numOps = MathHelper.GetNumericOperations<T>();
                for (int i = 0; i < data.Length; i++)
                {
                    result = ApplyReduction(result, numOps.ToDouble(data[i]), reductionType);
                }
                if (reductionType == "Mean") result /= data.Length;
            }

            var resultArray = new T[1];
            resultArray[0] = MathHelper.GetNumericOperations<T>().FromDouble(result);
            return new Tensor<T>(resultArray, keepDims ? new int[input.Shape.Length] : new[] { 1 });
        }

        // For axis-specific reduction, fall back to non-vectorized implementation
        return ReduceAlongAxes(input, axes, reductionType, keepDims);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double ExecuteVectorizedReductionFloat(ReadOnlySpan<float> input, string reductionType)
    {
        double result = reductionType switch
        {
            "Sum" or "Mean" => 0.0,
            "Max" => double.MinValue,
            "Min" => double.MaxValue,
            _ => 0.0
        };

        int i = 0;

        if (Vector.IsHardwareAccelerated && input.Length >= Vector<float>.Count)
        {
            var count = Vector<float>.Count;
            int vectorizedEnd = (input.Length / count) * count;

            if (reductionType is "Sum" or "Mean")
            {
                var vSum = Vector<float>.Zero;
                for (; i < vectorizedEnd; i += count)
                {
                    vSum += new Vector<float>(input.Slice(i, count));
                }
                // Horizontal sum
                for (int j = 0; j < count; j++)
                {
                    result += vSum[j];
                }
            }
            else if (reductionType == "Max")
            {
                var vMax = new Vector<float>(float.MinValue);
                for (; i < vectorizedEnd; i += count)
                {
                    vMax = Vector.Max(vMax, new Vector<float>(input.Slice(i, count)));
                }
                // Horizontal max
                for (int j = 0; j < count; j++)
                {
                    result = Math.Max(result, vMax[j]);
                }
            }
            else if (reductionType == "Min")
            {
                var vMin = new Vector<float>(float.MaxValue);
                for (; i < vectorizedEnd; i += count)
                {
                    vMin = Vector.Min(vMin, new Vector<float>(input.Slice(i, count)));
                }
                // Horizontal min
                for (int j = 0; j < count; j++)
                {
                    result = Math.Min(result, vMin[j]);
                }
            }
        }

        // Handle remainder
        for (; i < input.Length; i++)
        {
            result = ApplyReduction(result, input[i], reductionType);
        }

        if (reductionType == "Mean") result /= input.Length;

        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double ExecuteVectorizedReductionDouble(ReadOnlySpan<double> input, string reductionType)
    {
        double result = reductionType switch
        {
            "Sum" or "Mean" => 0.0,
            "Max" => double.MinValue,
            "Min" => double.MaxValue,
            _ => 0.0
        };

        int i = 0;

        if (Vector.IsHardwareAccelerated && input.Length >= Vector<double>.Count)
        {
            var count = Vector<double>.Count;
            int vectorizedEnd = (input.Length / count) * count;

            if (reductionType is "Sum" or "Mean")
            {
                var vSum = Vector<double>.Zero;
                for (; i < vectorizedEnd; i += count)
                {
                    vSum += new Vector<double>(input.Slice(i, count));
                }
                for (int j = 0; j < count; j++)
                {
                    result += vSum[j];
                }
            }
            else if (reductionType == "Max")
            {
                var vMax = new Vector<double>(double.MinValue);
                for (; i < vectorizedEnd; i += count)
                {
                    vMax = Vector.Max(vMax, new Vector<double>(input.Slice(i, count)));
                }
                for (int j = 0; j < count; j++)
                {
                    result = Math.Max(result, vMax[j]);
                }
            }
            else if (reductionType == "Min")
            {
                var vMin = new Vector<double>(double.MaxValue);
                for (; i < vectorizedEnd; i += count)
                {
                    vMin = Vector.Min(vMin, new Vector<double>(input.Slice(i, count)));
                }
                for (int j = 0; j < count; j++)
                {
                    result = Math.Min(result, vMin[j]);
                }
            }
        }

        for (; i < input.Length; i++)
        {
            result = ApplyReduction(result, input[i], reductionType);
        }

        if (reductionType == "Mean") result /= input.Length;

        return result;
    }

    /// <summary>
    /// Executes a vectorized matrix multiplication with tiling.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
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

        if (typeof(T) == typeof(float))
        {
            ExecuteVectorizedMatMulFloat(
                MemoryMarshal.Cast<T, float>(left.Data),
                MemoryMarshal.Cast<T, float>(right.Data),
                MemoryMarshal.Cast<T, float>(result),
                M, K, N, tileSize);
        }
        else if (typeof(T) == typeof(double))
        {
            ExecuteVectorizedMatMulDouble(
                MemoryMarshal.Cast<T, double>(left.Data),
                MemoryMarshal.Cast<T, double>(right.Data),
                MemoryMarshal.Cast<T, double>(result),
                M, K, N, tileSize);
        }
        else
        {
            // Fallback: naive matmul
            ExecuteNaiveMatMul(left.Data, right.Data, result, M, K, N);
        }

        return new Tensor<T>(result, new[] { M, N });
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
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
        int vectorCount = Vector<float>.Count;

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

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void ExecuteVectorizedMatMulDouble(
        ReadOnlySpan<double> A,
        ReadOnlySpan<double> B,
        Span<double> C,
        int M, int K, int N,
        int tileSize)
    {
        C.Clear();
        int vectorCount = Vector<double>.Count;

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

    private static void ExecuteNaiveMatMul<T>(
        T[] A, T[] B, T[] C,
        int M, int K, int N)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                double sum = 0;
                for (int k = 0; k < K; k++)
                {
                    sum += numOps.ToDouble(A[i * K + k]) * numOps.ToDouble(B[k * N + j]);
                }
                C[i * N + j] = numOps.FromDouble(sum);
            }
        }
    }

    private static Tensor<T> ReduceAlongAxes<T>(
        Tensor<T> input, int[] axes, string reductionType, bool keepDims)
    {
        // Simplified implementation - reduce along specific axes
        // Full implementation would handle arbitrary axes combinations
        var data = input.Data;
        var shape = input.Shape;

        // Calculate output shape
        var outputShape = new List<int>();
        for (int i = 0; i < shape.Length; i++)
        {
            if (!axes.Contains(i))
                outputShape.Add(shape[i]);
            else if (keepDims)
                outputShape.Add(1);
        }
        if (outputShape.Count == 0) outputShape.Add(1);

        var outputSize = outputShape.Aggregate(1, (a, b) => a * b);
        var result = new T[outputSize];

        // Initialize based on reduction type
        double initVal = reductionType switch
        {
            "Sum" or "Mean" => 0.0,
            "Max" => double.MinValue,
            "Min" => double.MaxValue,
            _ => 0.0
        };

        var resultDouble = new double[outputSize];
        Array.Fill(resultDouble, initVal);
        var counts = new int[outputSize];

        // Compute reduction
        var numOps = MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < data.Length; i++)
        {
            // Map flat index to multi-dimensional index
            int outputIndex = ComputeOutputIndex(i, shape, axes, outputShape.ToArray());
            resultDouble[outputIndex] = ApplyReduction(
                resultDouble[outputIndex],
                numOps.ToDouble(data[i]),
                reductionType);
            counts[outputIndex]++;
        }

        // Finalize (for mean)
        for (int i = 0; i < outputSize; i++)
        {
            if (reductionType == "Mean" && counts[i] > 0)
                resultDouble[i] /= counts[i];
            result[i] = numOps.FromDouble(resultDouble[i]);
        }

        return new Tensor<T>(result, outputShape.ToArray());
    }

    private static int ComputeOutputIndex(int flatIndex, int[] inputShape, int[] axes, int[] outputShape)
    {
        // Convert flat index to coordinates
        var coords = new int[inputShape.Length];
        int remaining = flatIndex;
        for (int i = inputShape.Length - 1; i >= 0; i--)
        {
            coords[i] = remaining % inputShape[i];
            remaining /= inputShape[i];
        }

        // Compute output index (skipping reduced axes)
        int outputIndex = 0;
        int outputStride = 1;
        int outputDim = outputShape.Length - 1;

        for (int i = inputShape.Length - 1; i >= 0; i--)
        {
            if (!axes.Contains(i))
            {
                outputIndex += coords[i] * outputStride;
                outputStride *= outputShape[outputDim];
                outputDim--;
            }
        }

        return outputIndex;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float ApplyUnaryOperation(float value, string operation)
    {
        return operation switch
        {
            "Negate" => -value,
            "ReLU" => MathF.Max(0, value),
            "Sigmoid" => 1f / (1f + MathF.Exp(-value)),
            "Tanh" => MathF.Tanh(value),
            "Exp" => MathF.Exp(value),
            "Log" => MathF.Log(value),
            "Sqrt" => MathF.Sqrt(value),
            _ => value
        };
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double ApplyUnaryOperation(double value, string operation)
    {
        return operation switch
        {
            "Negate" => -value,
            "ReLU" => Math.Max(0, value),
            "Sigmoid" => 1.0 / (1.0 + Math.Exp(-value)),
            "Tanh" => Math.Tanh(value),
            "Exp" => Math.Exp(value),
            "Log" => Math.Log(value),
            "Sqrt" => Math.Sqrt(value),
            _ => value
        };
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static T ApplyBinaryScalar<T>(T left, T right, string operation)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        double l = numOps.ToDouble(left);
        double r = numOps.ToDouble(right);
        double result = operation switch
        {
            "Add" => l + r,
            "Subtract" => l - r,
            "Multiply" => l * r,
            "Divide" => l / r,
            _ => l + r
        };
        return numOps.FromDouble(result);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static T ApplyUnaryScalar<T>(T value, string operation)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        double v = numOps.ToDouble(value);
        double result = ApplyUnaryOperation(v, operation);
        return numOps.FromDouble(result);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double ApplyReduction(double accumulator, double value, string reductionType)
    {
        return reductionType switch
        {
            "Sum" or "Mean" => accumulator + value,
            "Max" => Math.Max(accumulator, value),
            "Min" => Math.Min(accumulator, value),
            _ => accumulator + value
        };
    }
}
