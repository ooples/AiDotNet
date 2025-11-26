using System;
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
}
