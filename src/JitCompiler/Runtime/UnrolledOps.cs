using System.Numerics;
using System.Runtime.CompilerServices;
using AiDotNet.Autodiff;

namespace AiDotNet.JitCompiler.Runtime;

/// <summary>
/// Runtime support for unrolled loop operations.
/// </summary>
/// <remarks>
/// <para>
/// This class provides runtime implementations for operations that have been
/// unrolled by the LoopUnrollingPass. Unrolling replaces loops with repeated
/// inline code, reducing loop overhead and enabling better instruction pipelining.
/// </para>
/// <para><b>For Beginners:</b> These are the actual implementations of unrolled operations.
///
/// When the JIT compiler unrolls a loop:
/// - Instead of: for (i=0; i&lt;4; i++) a[i] = b[i] + c[i];
/// - It becomes: a[0]=b[0]+c[0]; a[1]=b[1]+c[1]; a[2]=b[2]+c[2]; a[3]=b[3]+c[3];
///
/// Benefits:
/// - No loop counter increments
/// - No loop condition checks
/// - Better instruction pipelining
/// - CPU can execute multiple operations in parallel
/// </para>
/// </remarks>
public static class UnrolledOps
{
    /// <summary>
    /// Executes an unrolled sequence of element-wise operations.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="input">Input tensor.</param>
    /// <param name="operations">List of operations to apply (Add, Multiply, ReLU, etc.).</param>
    /// <param name="unrollFactor">The unroll factor used.</param>
    /// <returns>Result tensor after all operations are applied.</returns>
    /// <remarks>
    /// <para>
    /// Executes a fused sequence of operations on the input tensor. Operations are
    /// applied in sequence, with each element processed through all operations before
    /// moving to the next element (loop fusion).
    /// </para>
    /// </remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Tensor<T> ExecuteUnrolledSequence<T>(
        Tensor<T> input,
        string[] operations,
        int unrollFactor)
    {
        var result = new T[input.Data.Length];
        var data = input.Data;
        var length = data.Length;

        // Process in blocks of unrollFactor
        int i = 0;
        int unrolledEnd = length - (length % unrollFactor);

        // Unrolled loop - process unrollFactor elements at a time
        for (; i < unrolledEnd; i += unrollFactor)
        {
            for (int u = 0; u < unrollFactor; u++)
            {
                var value = ConvertToDouble(data[i + u]);
                foreach (var op in operations)
                {
                    value = ApplyOperation(value, op);
                }
                result[i + u] = ConvertFromDouble<T>(value);
            }
        }

        // Handle remainder
        for (; i < length; i++)
        {
            var value = ConvertToDouble(data[i]);
            foreach (var op in operations)
            {
                value = ApplyOperation(value, op);
            }
            result[i] = ConvertFromDouble<T>(value);
        }

        return new Tensor<T>(result, input.Shape);
    }

    /// <summary>
    /// Executes an unrolled element-wise operation on small tensors.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="input">Input tensor.</param>
    /// <param name="operation">The operation to apply.</param>
    /// <param name="unrollFactor">The unroll factor.</param>
    /// <param name="totalElements">Total number of elements.</param>
    /// <returns>Result tensor.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Tensor<T> ExecuteUnrolledElementwise<T>(
        Tensor<T> input,
        string operation,
        int unrollFactor,
        int totalElements)
    {
        var result = new T[totalElements];
        var data = input.Data;

        int i = 0;
        int unrolledEnd = totalElements - (totalElements % unrollFactor);

        // Unrolled main loop
        for (; i < unrolledEnd; i += unrollFactor)
        {
            // Manually unroll based on common unroll factors
            if (unrollFactor >= 8)
            {
                result[i] = ApplyOp<T>(data[i], operation);
                result[i + 1] = ApplyOp<T>(data[i + 1], operation);
                result[i + 2] = ApplyOp<T>(data[i + 2], operation);
                result[i + 3] = ApplyOp<T>(data[i + 3], operation);
                result[i + 4] = ApplyOp<T>(data[i + 4], operation);
                result[i + 5] = ApplyOp<T>(data[i + 5], operation);
                result[i + 6] = ApplyOp<T>(data[i + 6], operation);
                result[i + 7] = ApplyOp<T>(data[i + 7], operation);
                for (int j = 8; j < unrollFactor; j++)
                {
                    result[i + j] = ApplyOp<T>(data[i + j], operation);
                }
            }
            else if (unrollFactor >= 4)
            {
                result[i] = ApplyOp<T>(data[i], operation);
                result[i + 1] = ApplyOp<T>(data[i + 1], operation);
                result[i + 2] = ApplyOp<T>(data[i + 2], operation);
                result[i + 3] = ApplyOp<T>(data[i + 3], operation);
                for (int j = 4; j < unrollFactor; j++)
                {
                    result[i + j] = ApplyOp<T>(data[i + j], operation);
                }
            }
            else
            {
                for (int j = 0; j < unrollFactor; j++)
                {
                    result[i + j] = ApplyOp<T>(data[i + j], operation);
                }
            }
        }

        // Handle remainder
        for (; i < totalElements; i++)
        {
            result[i] = ApplyOp<T>(data[i], operation);
        }

        return new Tensor<T>(result, input.Shape);
    }

    /// <summary>
    /// Executes an unrolled reduction operation using tree reduction.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="input">Input tensor.</param>
    /// <param name="reductionType">Type of reduction (Sum, Mean, Max).</param>
    /// <param name="unrollFactor">The unroll factor.</param>
    /// <returns>Reduced scalar as a 1-element tensor.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Tensor<T> ExecuteUnrolledReduction<T>(
        Tensor<T> input,
        string reductionType,
        int unrollFactor)
    {
        var data = input.Data;
        var length = data.Length;

        // Use accumulators for tree reduction
        var accumulators = new double[unrollFactor];

        // Initialize accumulators based on reduction type
        double initValue = reductionType switch
        {
            "Sum" or "Mean" => 0.0,
            "Max" => double.MinValue,
            "Min" => double.MaxValue,
            _ => 0.0
        };
        Array.Fill(accumulators, initValue);

        // Parallel accumulation
        int i = 0;
        int unrolledEnd = length - (length % unrollFactor);

        for (; i < unrolledEnd; i += unrollFactor)
        {
            for (int j = 0; j < unrollFactor; j++)
            {
                accumulators[j] = ApplyReduction(accumulators[j], ConvertToDouble(data[i + j]), reductionType);
            }
        }

        // Handle remainder
        for (; i < length; i++)
        {
            accumulators[i % unrollFactor] = ApplyReduction(
                accumulators[i % unrollFactor],
                ConvertToDouble(data[i]),
                reductionType);
        }

        // Final tree reduction of accumulators
        double result = accumulators[0];
        for (int j = 1; j < unrollFactor; j++)
        {
            result = ApplyReduction(result, accumulators[j], reductionType);
        }

        // For mean, divide by count
        if (reductionType == "Mean")
        {
            result /= length;
        }

        return new Tensor<T>(new[] { ConvertFromDouble<T>(result) }, new[] { 1 });
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double ApplyOperation(double value, string operation)
    {
        return operation switch
        {
            "Add" => value, // Pass through for unary
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
    private static T ApplyOp<T>(T value, string operation)
    {
        var dValue = ConvertToDouble(value);
        var result = ApplyOperation(dValue, operation);
        return ConvertFromDouble<T>(result);
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

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double ConvertToDouble<T>(T value)
    {
        return value switch
        {
            float f => f,
            double d => d,
            int i => i,
            long l => l,
            _ => Convert.ToDouble(value)
        };
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static T ConvertFromDouble<T>(double value)
    {
        if (typeof(T) == typeof(float))
            return (T)(object)(float)value;
        if (typeof(T) == typeof(double))
            return (T)(object)value;
        if (typeof(T) == typeof(int))
            return (T)(object)(int)value;
        if (typeof(T) == typeof(long))
            return (T)(object)(long)value;
        return (T)Convert.ChangeType(value, typeof(T));
    }
}
