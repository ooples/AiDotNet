using System;
using System.Numerics.Tensors;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Internal dispatcher for TensorPrimitives operations that handles type-specific dispatch.
/// Uses TensorPrimitives for float and double types when available, providing SIMD acceleration.
/// For other types, throws NotSupportedException as TensorPrimitives doesn't support them.
/// </summary>
/// <remarks>
/// This dispatcher centralizes all the type-specific TensorPrimitives calls in one place.
/// Callers should check UseGenericTensorPrimitives before calling these methods
/// and fall back to INumericOperations-based implementations for unsupported types.
/// Note: double overloads are only available in .NET 8.0+. For .NET Framework, only float is supported.
/// </remarks>
internal static class TensorPrimitivesDispatcher
{
    /// <summary>
    /// Performs element-wise addition using TensorPrimitives SIMD operations.
    /// </summary>
    public static void Add<T>(T[] x, T[] y, T[] destination)
    {
        if (typeof(T) == typeof(float))
        {
            TensorPrimitives.Add((float[])(object)x, (float[])(object)y, (float[])(object)destination);
        }
#if NET8_0_OR_GREATER
        else if (typeof(T) == typeof(double))
        {
            TensorPrimitives.Add((double[])(object)x, (double[])(object)y, (double[])(object)destination);
        }
#endif
        else
        {
            throw new NotSupportedException($"TensorPrimitives.Add is not supported for type {typeof(T)}. Use INumericOperations fallback.");
        }
    }

    /// <summary>
    /// Performs element-wise subtraction using TensorPrimitives SIMD operations.
    /// </summary>
    public static void Subtract<T>(T[] x, T[] y, T[] destination)
    {
        if (typeof(T) == typeof(float))
        {
            TensorPrimitives.Subtract((float[])(object)x, (float[])(object)y, (float[])(object)destination);
        }
#if NET8_0_OR_GREATER
        else if (typeof(T) == typeof(double))
        {
            TensorPrimitives.Subtract((double[])(object)x, (double[])(object)y, (double[])(object)destination);
        }
#endif
        else
        {
            throw new NotSupportedException($"TensorPrimitives.Subtract is not supported for type {typeof(T)}. Use INumericOperations fallback.");
        }
    }

    /// <summary>
    /// Performs element-wise multiplication using TensorPrimitives SIMD operations.
    /// </summary>
    public static void Multiply<T>(T[] x, T[] y, T[] destination)
    {
        if (typeof(T) == typeof(float))
        {
            TensorPrimitives.Multiply((float[])(object)x, (float[])(object)y, (float[])(object)destination);
        }
#if NET8_0_OR_GREATER
        else if (typeof(T) == typeof(double))
        {
            TensorPrimitives.Multiply((double[])(object)x, (double[])(object)y, (double[])(object)destination);
        }
#endif
        else
        {
            throw new NotSupportedException($"TensorPrimitives.Multiply is not supported for type {typeof(T)}. Use INumericOperations fallback.");
        }
    }

    /// <summary>
    /// Performs element-wise division using TensorPrimitives SIMD operations.
    /// </summary>
    public static void Divide<T>(T[] x, T[] y, T[] destination)
    {
        if (typeof(T) == typeof(float))
        {
            TensorPrimitives.Divide((float[])(object)x, (float[])(object)y, (float[])(object)destination);
        }
#if NET8_0_OR_GREATER
        else if (typeof(T) == typeof(double))
        {
            TensorPrimitives.Divide((double[])(object)x, (double[])(object)y, (double[])(object)destination);
        }
#endif
        else
        {
            throw new NotSupportedException($"TensorPrimitives.Divide is not supported for type {typeof(T)}. Use INumericOperations fallback.");
        }
    }

    /// <summary>
    /// Computes dot product using TensorPrimitives SIMD operations.
    /// </summary>
    public static T Dot<T>(T[] x, T[] y)
    {
        if (typeof(T) == typeof(float))
        {
            return (T)(object)TensorPrimitives.Dot((float[])(object)x, (float[])(object)y);
        }
#if NET8_0_OR_GREATER
        else if (typeof(T) == typeof(double))
        {
            return (T)(object)TensorPrimitives.Dot((double[])(object)x, (double[])(object)y);
        }
#endif
        throw new NotSupportedException($"TensorPrimitives.Dot is not supported for type {typeof(T)}. Use INumericOperations fallback.");
    }

    /// <summary>
    /// Computes sum using TensorPrimitives SIMD operations.
    /// </summary>
    public static T Sum<T>(T[] x)
    {
        if (typeof(T) == typeof(float))
        {
            return (T)(object)TensorPrimitives.Sum((float[])(object)x);
        }
#if NET8_0_OR_GREATER
        else if (typeof(T) == typeof(double))
        {
            return (T)(object)TensorPrimitives.Sum((double[])(object)x);
        }
#endif
        throw new NotSupportedException($"TensorPrimitives.Sum is not supported for type {typeof(T)}. Use INumericOperations fallback.");
    }

    /// <summary>
    /// Finds maximum value using TensorPrimitives SIMD operations.
    /// </summary>
    public static T Max<T>(T[] x)
    {
        if (typeof(T) == typeof(float))
        {
            return (T)(object)TensorPrimitives.Max((float[])(object)x);
        }
#if NET8_0_OR_GREATER
        else if (typeof(T) == typeof(double))
        {
            return (T)(object)TensorPrimitives.Max((double[])(object)x);
        }
#endif
        throw new NotSupportedException($"TensorPrimitives.Max is not supported for type {typeof(T)}. Use INumericOperations fallback.");
    }

    /// <summary>
    /// Finds minimum value using TensorPrimitives SIMD operations.
    /// </summary>
    public static T Min<T>(T[] x)
    {
        if (typeof(T) == typeof(float))
        {
            return (T)(object)TensorPrimitives.Min((float[])(object)x);
        }
#if NET8_0_OR_GREATER
        else if (typeof(T) == typeof(double))
        {
            return (T)(object)TensorPrimitives.Min((double[])(object)x);
        }
#endif
        throw new NotSupportedException($"TensorPrimitives.Min is not supported for type {typeof(T)}. Use INumericOperations fallback.");
    }

    /// <summary>
    /// Computes exponential using TensorPrimitives SIMD operations.
    /// </summary>
    public static void Exp<T>(T[] x, T[] destination)
    {
        if (typeof(T) == typeof(float))
        {
            TensorPrimitives.Exp((float[])(object)x, (float[])(object)destination);
        }
#if NET8_0_OR_GREATER
        else if (typeof(T) == typeof(double))
        {
            TensorPrimitives.Exp((double[])(object)x, (double[])(object)destination);
        }
#endif
        else
        {
            throw new NotSupportedException($"TensorPrimitives.Exp is not supported for type {typeof(T)}. Use INumericOperations fallback.");
        }
    }

    /// <summary>
    /// Computes natural logarithm using TensorPrimitives SIMD operations.
    /// </summary>
    public static void Log<T>(T[] x, T[] destination)
    {
        if (typeof(T) == typeof(float))
        {
            TensorPrimitives.Log((float[])(object)x, (float[])(object)destination);
        }
#if NET8_0_OR_GREATER
        else if (typeof(T) == typeof(double))
        {
            TensorPrimitives.Log((double[])(object)x, (double[])(object)destination);
        }
#endif
        else
        {
            throw new NotSupportedException($"TensorPrimitives.Log is not supported for type {typeof(T)}. Use INumericOperations fallback.");
        }
    }

    /// <summary>
    /// Computes hyperbolic tangent using TensorPrimitives SIMD operations.
    /// </summary>
    public static void Tanh<T>(T[] x, T[] destination)
    {
        if (typeof(T) == typeof(float))
        {
            TensorPrimitives.Tanh((float[])(object)x, (float[])(object)destination);
        }
#if NET8_0_OR_GREATER
        else if (typeof(T) == typeof(double))
        {
            TensorPrimitives.Tanh((double[])(object)x, (double[])(object)destination);
        }
#endif
        else
        {
            throw new NotSupportedException($"TensorPrimitives.Tanh is not supported for type {typeof(T)}. Use INumericOperations fallback.");
        }
    }

    /// <summary>
    /// Computes sigmoid using TensorPrimitives SIMD operations.
    /// </summary>
    public static void Sigmoid<T>(T[] x, T[] destination)
    {
        if (typeof(T) == typeof(float))
        {
            TensorPrimitives.Sigmoid((float[])(object)x, (float[])(object)destination);
        }
#if NET8_0_OR_GREATER
        else if (typeof(T) == typeof(double))
        {
            TensorPrimitives.Sigmoid((double[])(object)x, (double[])(object)destination);
        }
#endif
        else
        {
            throw new NotSupportedException($"TensorPrimitives.Sigmoid is not supported for type {typeof(T)}. Use INumericOperations fallback.");
        }
    }

    /// <summary>
    /// Computes base-2 logarithm using TensorPrimitives SIMD operations.
    /// </summary>
    public static void Log2<T>(T[] x, T[] destination)
    {
        if (typeof(T) == typeof(float))
        {
            TensorPrimitives.Log2((float[])(object)x, (float[])(object)destination);
        }
#if NET8_0_OR_GREATER
        else if (typeof(T) == typeof(double))
        {
            TensorPrimitives.Log2((double[])(object)x, (double[])(object)destination);
        }
#endif
        else
        {
            throw new NotSupportedException($"TensorPrimitives.Log2 is not supported for type {typeof(T)}. Use INumericOperations fallback.");
        }
    }

    /// <summary>
    /// Computes softmax using TensorPrimitives SIMD operations.
    /// </summary>
    public static void SoftMax<T>(T[] x, T[] destination)
    {
        if (typeof(T) == typeof(float))
        {
            TensorPrimitives.SoftMax((float[])(object)x, (float[])(object)destination);
        }
#if NET8_0_OR_GREATER
        else if (typeof(T) == typeof(double))
        {
            TensorPrimitives.SoftMax((double[])(object)x, (double[])(object)destination);
        }
#endif
        else
        {
            throw new NotSupportedException($"TensorPrimitives.SoftMax is not supported for type {typeof(T)}. Use INumericOperations fallback.");
        }
    }

    /// <summary>
    /// Computes cosine similarity using TensorPrimitives SIMD operations.
    /// </summary>
    public static T CosineSimilarity<T>(T[] x, T[] y)
    {
        if (typeof(T) == typeof(float))
        {
            return (T)(object)TensorPrimitives.CosineSimilarity((float[])(object)x, (float[])(object)y);
        }
#if NET8_0_OR_GREATER
        else if (typeof(T) == typeof(double))
        {
            return (T)(object)TensorPrimitives.CosineSimilarity((double[])(object)x, (double[])(object)y);
        }
#endif
        throw new NotSupportedException($"TensorPrimitives.CosineSimilarity is not supported for type {typeof(T)}. Use INumericOperations fallback.");
    }
}
