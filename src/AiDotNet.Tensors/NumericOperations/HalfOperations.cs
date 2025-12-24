using System;
#if NET8_0_OR_GREATER
using System.Numerics.Tensors;
#endif
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Tensors.NumericOperations;

/// <summary>
/// Provides numeric operations for the Half (FP16) data type.
/// </summary>
/// <remarks>
/// Half (FP16) is a 16-bit floating-point format with:
/// - 1 sign bit
/// - 5 exponent bits
/// - 10 mantissa bits
/// - Range: approximately Ãƒâ€šÃ‚Â±6.55ÃƒÆ’Ã¢â‚¬â€10ÃƒÂ¢Ã‚ÂÃ‚Â´
/// - Precision: approximately 3-4 decimal digits
///
/// Benefits for mixed-precision training:
/// - 2x memory reduction compared to float
/// - 2-3x faster on GPUs with Tensor Cores (V100, A100, RTX 3000+)
/// - Same numeric behavior as float, just reduced range and precision
///
/// Limitations:
/// - Limited range [6e-8, 65504] can cause underflow/overflow
/// - Reduced precision can accumulate errors
/// - Requires loss scaling to prevent gradient underflow
/// - Most operations internally convert to float for computation
///
/// Usage in mixed-precision training:
/// - Store weights and activations in FP16
/// - Accumulate gradients in FP32
/// - Keep master copy of weights in FP32
/// </remarks>
public class HalfOperations : INumericOperations<Half>
{
    /// <summary>
    /// Gets the zero value (0.0) for Half.
    /// </summary>
    public Half Zero => (Half)0.0f;

    /// <summary>
    /// Gets the value one (1.0) for Half.
    /// </summary>
    public Half One => (Half)1.0f;

    /// <summary>
    /// Gets the minimum value that can be represented by Half.
    /// </summary>
    /// <remarks>
    /// Half.MinValue = -65504
    /// Much more limited than float's -3.4ÃƒÆ’Ã¢â‚¬â€10Ãƒâ€šÃ‚Â³ÃƒÂ¢Ã‚ÂÃ‚Â¸
    /// </remarks>
    public Half MinValue => Half.MinValue;

    /// <summary>
    /// Gets the maximum value that can be represented by Half.
    /// </summary>
    /// <remarks>
    /// Half.MaxValue = 65504
    /// Much more limited than float's 3.4ÃƒÆ’Ã¢â‚¬â€10Ãƒâ€šÃ‚Â³ÃƒÂ¢Ã‚ÂÃ‚Â¸
    /// </remarks>
    public Half MaxValue => Half.MaxValue;

    /// <summary>
    /// Gets the number of bits used for precision (16 for Half).
    /// </summary>
    public int PrecisionBits => 16;

    /// <summary>
    /// Adds two Half values.
    /// </summary>
    /// <remarks>
    /// Note: Internally converts to float for computation to avoid precision issues.
    /// </remarks>
    public Half Add(Half a, Half b) => (Half)((float)a + (float)b);

    /// <summary>
    /// Subtracts two Half values.
    /// </summary>
    public Half Subtract(Half a, Half b) => (Half)((float)a - (float)b);

    /// <summary>
    /// Multiplies two Half values.
    /// </summary>
    public Half Multiply(Half a, Half b) => (Half)((float)a * (float)b);

    /// <summary>
    /// Divides two Half values.
    /// </summary>
    public Half Divide(Half a, Half b) => (Half)((float)a / (float)b);

    /// <summary>
    /// Negates a Half value.
    /// </summary>
    public Half Negate(Half a) => -a;

    /// <summary>
    /// Calculates the square root of a Half value.
    /// </summary>
    public Half Sqrt(Half value) => (Half)Math.Sqrt((float)value);

    /// <summary>
    /// Converts a double value to Half.
    /// </summary>
    /// <remarks>
    /// Warning: May lose precision and cause overflow if value is outside Half's range.
    /// </remarks>
    public Half FromDouble(double value) => (Half)value;

    /// <summary>
    /// Converts a Half value to a 32-bit integer.
    /// </summary>
    public int ToInt32(Half value) => (int)value;

    /// <summary>
    /// Compares two Half values for greater than.
    /// </summary>
    public bool GreaterThan(Half a, Half b) => a > b;

    /// <summary>
    /// Compares two Half values for less than.
    /// </summary>
    public bool LessThan(Half a, Half b) => a < b;

    /// <summary>
    /// Calculates the absolute value of a Half.
    /// </summary>
    public Half Abs(Half value) => (Half)Math.Abs((float)value);

    /// <summary>
    /// Calculates the square of a Half value.
    /// </summary>
    public Half Square(Half value) => (Half)((float)value * (float)value);

    /// <summary>
    /// Calculates the exponential function (e^value).
    /// </summary>
    /// <remarks>
    /// Warning: exp() can easily overflow Half's range. Use with loss scaling.
    /// </remarks>
    public Half Exp(Half value) => (Half)Math.Exp((float)value);

    /// <summary>
    /// Compares two Half values for equality.
    /// </summary>
    public bool Equals(Half a, Half b) => a == b;

    public int Compare(Half a, Half b) => a.CompareTo(b);

    /// <summary>
    /// Raises a Half value to a power.
    /// </summary>
    public Half Power(Half baseValue, Half exponent) =>
        (Half)Math.Pow((float)baseValue, (float)exponent);

    /// <summary>
    /// Calculates the natural logarithm of a Half value.
    /// </summary>
    public Half Log(Half value) => (Half)Math.Log((float)value);

    /// <summary>
    /// Compares two Half values for greater than or equal.
    /// </summary>
    public bool GreaterThanOrEquals(Half a, Half b) => a >= b;

    /// <summary>
    /// Compares two Half values for less than or equal.
    /// </summary>
    public bool LessThanOrEquals(Half a, Half b) => a <= b;

    /// <summary>
    /// Rounds a Half value to the nearest integer.
    /// </summary>
    public Half Round(Half value) => (Half)Math.Round((float)value);

    public Half Floor(Half value) => (Half)Math.Floor((float)value);
    public Half Ceiling(Half value) => (Half)Math.Ceiling((float)value);
    public Half Frac(Half value) => (Half)((float)value - Math.Floor((float)value));

    public Half Sin(Half value) => (Half)Math.Sin((float)value);
    public Half Cos(Half value) => (Half)Math.Cos((float)value);


    /// <summary>
    /// Determines whether a Half value is NaN (Not a Number).
    /// </summary>
    public bool IsNaN(Half value) => Half.IsNaN(value);

    /// <summary>
    /// Determines whether a Half value is infinity.
    /// </summary>
    public bool IsInfinity(Half value) => Half.IsInfinity(value);

    /// <summary>
    /// Returns the sign of a Half value (1, -1, or 0).
    /// </summary>
    public Half SignOrZero(Half value)
    {
        if (Half.IsNaN(value))
            return value;
        if (value > Zero)
            return One;
        if (value < Zero)
            return (Half)(-1.0f);
        return Zero;
    }

    /// <summary>
    /// Converts a Half value to float (FP32).
    /// </summary>
    /// <remarks>
    /// This is lossless - all Half values can be exactly represented in float.
    /// </remarks>
    public float ToFloat(Half value) => (float)value;

    /// <summary>
    /// Converts a float (FP32) value to Half.
    /// </summary>
    /// <remarks>
    /// Warning: May lose precision and cause overflow if value is outside Half's range.
    /// Values outside [-65504, 65504] will become infinity.
    /// </remarks>
    public Half FromFloat(float value) => (Half)value;

    /// <summary>
    /// Converts a Half value to Half (identity operation).
    /// </summary>
    public Half ToHalf(Half value) => value;

    /// <summary>
    /// Converts a Half value to Half (identity operation).
    /// </summary>
    public Half FromHalf(Half value) => value;

    /// <summary>
    /// Converts a Half value to double (FP64).
    /// </summary>
    /// <remarks>
    /// This is lossless - all Half values can be exactly represented in double.
    /// </remarks>
    public double ToDouble(Half value) => (double)value;

    /// <inheritdoc/>
    public bool SupportsCpuAcceleration => true;

    /// <inheritdoc/>
    public bool SupportsGpuAcceleration => false;

    #region IVectorizedOperations<Half> Implementation

    /// <summary>
    /// Performs element-wise addition. Uses SIMD on .NET 8+, falls back to loops on older frameworks.
    /// </summary>
    public void Add(ReadOnlySpan<Half> x, ReadOnlySpan<Half> y, Span<Half> destination)
    {
#if NET8_0_OR_GREATER
        TensorPrimitives.Add(x, y, destination);
#else
        VectorizedOperationsFallback.Add(this, x, y, destination);
#endif
    }

    /// <summary>
    /// Performs element-wise subtraction. Uses SIMD on .NET 8+, falls back to loops on older frameworks.
    /// </summary>
    public void Subtract(ReadOnlySpan<Half> x, ReadOnlySpan<Half> y, Span<Half> destination)
    {
#if NET8_0_OR_GREATER
        TensorPrimitives.Subtract(x, y, destination);
#else
        VectorizedOperationsFallback.Subtract(this, x, y, destination);
#endif
    }

    /// <summary>
    /// Performs element-wise multiplication. Uses SIMD on .NET 8+, falls back to loops on older frameworks.
    /// </summary>
    public void Multiply(ReadOnlySpan<Half> x, ReadOnlySpan<Half> y, Span<Half> destination)
    {
#if NET8_0_OR_GREATER
        TensorPrimitives.Multiply(x, y, destination);
#else
        VectorizedOperationsFallback.Multiply(this, x, y, destination);
#endif
    }

    /// <summary>
    /// Performs element-wise division. Uses SIMD on .NET 8+, falls back to loops on older frameworks.
    /// </summary>
    public void Divide(ReadOnlySpan<Half> x, ReadOnlySpan<Half> y, Span<Half> destination)
    {
#if NET8_0_OR_GREATER
        TensorPrimitives.Divide(x, y, destination);
#else
        VectorizedOperationsFallback.Divide(this, x, y, destination);
#endif
    }

    /// <summary>
    /// Computes dot product. Uses SIMD on .NET 8+, falls back to loops on older frameworks.
    /// </summary>
    public Half Dot(ReadOnlySpan<Half> x, ReadOnlySpan<Half> y)
    {
#if NET8_0_OR_GREATER
        return TensorPrimitives.Dot(x, y);
#else
        return VectorizedOperationsFallback.Dot(this, x, y);
#endif
    }

    /// <summary>
    /// Computes sum. Uses SIMD on .NET 8+, falls back to loops on older frameworks.
    /// </summary>
    public Half Sum(ReadOnlySpan<Half> x)
    {
#if NET8_0_OR_GREATER
        return TensorPrimitives.Sum(x);
#else
        return VectorizedOperationsFallback.Sum(this, x);
#endif
    }

    /// <summary>
    /// Finds maximum. Uses SIMD on .NET 8+, falls back to loops on older frameworks.
    /// </summary>
    public Half Max(ReadOnlySpan<Half> x)
    {
#if NET8_0_OR_GREATER
        return TensorPrimitives.Max(x);
#else
        return VectorizedOperationsFallback.Max(this, x);
#endif
    }

    /// <summary>
    /// Finds minimum. Uses SIMD on .NET 8+, falls back to loops on older frameworks.
    /// </summary>
    public Half Min(ReadOnlySpan<Half> x)
    {
#if NET8_0_OR_GREATER
        return TensorPrimitives.Min(x);
#else
        return VectorizedOperationsFallback.Min(this, x);
#endif
    }

    /// <summary>
    /// Computes exponential. Uses SIMD on .NET 8+, falls back to loops on older frameworks.
    /// </summary>
    public void Exp(ReadOnlySpan<Half> x, Span<Half> destination)
    {
#if NET8_0_OR_GREATER
        TensorPrimitives.Exp(x, destination);
#else
        VectorizedOperationsFallback.Exp(this, x, destination);
#endif
    }

    /// <summary>
    /// Computes natural logarithm. Uses SIMD on .NET 8+, falls back to loops on older frameworks.
    /// </summary>
    public void Log(ReadOnlySpan<Half> x, Span<Half> destination)
    {
#if NET8_0_OR_GREATER
        TensorPrimitives.Log(x, destination);
#else
        VectorizedOperationsFallback.Log(this, x, destination);
#endif
    }

    /// <summary>
    /// Computes hyperbolic tangent. Uses SIMD on .NET 8+, falls back to loops on older frameworks.
    /// </summary>
    public void Tanh(ReadOnlySpan<Half> x, Span<Half> destination)
    {
#if NET8_0_OR_GREATER
        TensorPrimitives.Tanh(x, destination);
#else
        VectorizedOperationsFallback.Tanh(this, x, destination);
#endif
    }

    /// <summary>
    /// Computes sigmoid. Uses SIMD on .NET 8+, falls back to loops on older frameworks.
    /// </summary>
    public void Sigmoid(ReadOnlySpan<Half> x, Span<Half> destination)
    {
#if NET8_0_OR_GREATER
        TensorPrimitives.Sigmoid(x, destination);
#else
        VectorizedOperationsFallback.Sigmoid(this, x, destination);
#endif
    }

    /// <summary>
    /// Computes base-2 logarithm. Uses SIMD on .NET 8+, falls back to loops on older frameworks.
    /// </summary>
    public void Log2(ReadOnlySpan<Half> x, Span<Half> destination)
    {
#if NET8_0_OR_GREATER
        TensorPrimitives.Log2(x, destination);
#else
        VectorizedOperationsFallback.Log2(this, x, destination);
#endif
    }

    /// <summary>
    /// Computes softmax. Uses SIMD on .NET 8+, falls back to loops on older frameworks.
    /// </summary>
    public void SoftMax(ReadOnlySpan<Half> x, Span<Half> destination)
    {
#if NET8_0_OR_GREATER
        TensorPrimitives.SoftMax(x, destination);
#else
        VectorizedOperationsFallback.SoftMax(this, x, destination);
#endif
    }

    /// <summary>
    /// Computes cosine similarity. Uses SIMD on .NET 8+, falls back to loops on older frameworks.
    /// </summary>
    public Half CosineSimilarity(ReadOnlySpan<Half> x, ReadOnlySpan<Half> y)
    {
#if NET8_0_OR_GREATER
        return TensorPrimitives.CosineSimilarity(x, y);
#else
        return VectorizedOperationsFallback.CosineSimilarity(this, x, y);
#endif
    }

    private static readonly HalfOperations _instance = new();

    /// <summary>
    /// Fills a span with a specified value.
    /// </summary>
    public void Fill(Span<Half> destination, Half value) => destination.Fill(value);

    /// <summary>
    /// Multiplies each element in a span by a scalar value.
    /// </summary>
    public void MultiplyScalar(ReadOnlySpan<Half> x, Half scalar, Span<Half> destination)
        => VectorizedOperationsFallback.MultiplyScalar(_instance, x, scalar, destination);

    /// <summary>
    /// Divides each element in a span by a scalar value.
    /// </summary>
    public void DivideScalar(ReadOnlySpan<Half> x, Half scalar, Span<Half> destination)
        => VectorizedOperationsFallback.DivideScalar(_instance, x, scalar, destination);

    /// <summary>
    /// Adds a scalar value to each element in a span.
    /// </summary>
    public void AddScalar(ReadOnlySpan<Half> x, Half scalar, Span<Half> destination)
        => VectorizedOperationsFallback.AddScalar(_instance, x, scalar, destination);

    /// <summary>
    /// Subtracts a scalar value from each element in a span.
    /// </summary>
    public void SubtractScalar(ReadOnlySpan<Half> x, Half scalar, Span<Half> destination)
        => VectorizedOperationsFallback.SubtractScalar(_instance, x, scalar, destination);

    /// <summary>
    /// Computes square root of each element. Uses SIMD on .NET 8+, falls back to loops on older frameworks.
    /// </summary>
#if NET8_0_OR_GREATER
    public void Sqrt(ReadOnlySpan<Half> x, Span<Half> destination)
        => System.Numerics.Tensors.TensorPrimitives.Sqrt<Half>(x, destination);
#else
    public void Sqrt(ReadOnlySpan<Half> x, Span<Half> destination)
        => VectorizedOperationsFallback.Sqrt(_instance, x, destination);
#endif

    /// <summary>
    /// Computes absolute value of each element. Uses SIMD on .NET 8+, falls back to loops on older frameworks.
    /// </summary>
#if NET8_0_OR_GREATER
    public void Abs(ReadOnlySpan<Half> x, Span<Half> destination)
        => System.Numerics.Tensors.TensorPrimitives.Abs<Half>(x, destination);
#else
    public void Abs(ReadOnlySpan<Half> x, Span<Half> destination)
        => VectorizedOperationsFallback.Abs(_instance, x, destination);
#endif

    /// <summary>
    /// Negates each element. Uses SIMD on .NET 8+, falls back to loops on older frameworks.
    /// </summary>
#if NET8_0_OR_GREATER
    public void Negate(ReadOnlySpan<Half> x, Span<Half> destination)
        => System.Numerics.Tensors.TensorPrimitives.Negate<Half>(x, destination);
#else
    public void Negate(ReadOnlySpan<Half> x, Span<Half> destination)
        => VectorizedOperationsFallback.Negate(_instance, x, destination);
#endif

    /// <summary>
    /// Clips each element to the specified range. Falls back to loops.
    /// </summary>
    public void Clip(ReadOnlySpan<Half> x, Half min, Half max, Span<Half> destination)
        => VectorizedOperationsFallback.Clip(_instance, x, min, max, destination);

    /// <summary>
    /// Raises each element to a specified power. Uses SIMD on .NET 8+, falls back to loops on older frameworks.
    /// </summary>
#if NET8_0_OR_GREATER
    public void Pow(ReadOnlySpan<Half> x, Half power, Span<Half> destination)
        => System.Numerics.Tensors.TensorPrimitives.Pow<Half>(x, power, destination);
#else
    public void Pow(ReadOnlySpan<Half> x, Half power, Span<Half> destination)
        => VectorizedOperationsFallback.Pow(_instance, x, power, destination);
#endif

    /// <summary>
    /// Copies elements from source to destination.
    /// </summary>
    public void Copy(ReadOnlySpan<Half> source, Span<Half> destination)
        => source.CopyTo(destination);

    #endregion

    public void Floor(ReadOnlySpan<Half> x, Span<Half> destination)
    {
        for (int i = 0; i < x.Length; i++)
            destination[i] = (Half)Math.Floor((float)x[i]);
    }

    public void Ceiling(ReadOnlySpan<Half> x, Span<Half> destination)
    {
        for (int i = 0; i < x.Length; i++)
            destination[i] = (Half)Math.Ceiling((float)x[i]);
    }

    public void Frac(ReadOnlySpan<Half> x, Span<Half> destination)
    {
        for (int i = 0; i < x.Length; i++)
            destination[i] = (Half)((float)x[i] - Math.Floor((float)x[i]));
    }

    public void Sin(ReadOnlySpan<Half> x, Span<Half> destination)
    {
        for (int i = 0; i < x.Length; i++)
            destination[i] = (Half)Math.Sin((float)x[i]);
    }

    public void Cos(ReadOnlySpan<Half> x, Span<Half> destination)
    {
        for (int i = 0; i < x.Length; i++)
            destination[i] = (Half)Math.Cos((float)x[i]);
    }

}
