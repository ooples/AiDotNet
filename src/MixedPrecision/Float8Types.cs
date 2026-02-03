using System.Runtime.InteropServices;

namespace AiDotNet.MixedPrecision;

/// <summary>
/// Helper methods for bit manipulation of floating point values.
/// </summary>
internal static class BitConverterHelper
{
    /// <summary>
    /// Converts a float to its bit representation as an int.
    /// </summary>
    public static int SingleToInt32Bits(float value)
    {
#if NETCOREAPP2_1_OR_GREATER || NETSTANDARD2_1_OR_GREATER || NET5_0_OR_GREATER
        return BitConverter.SingleToInt32Bits(value);
#else
        // For older frameworks, use unsafe code or union struct
        var union = new SingleInt32Union { Single = value };
        return union.Int32;
#endif
    }

    /// <summary>
    /// Converts an int bit representation to a float.
    /// </summary>
    public static float Int32BitsToSingle(int value)
    {
#if NETCOREAPP2_1_OR_GREATER || NETSTANDARD2_1_OR_GREATER || NET5_0_OR_GREATER
        return BitConverter.Int32BitsToSingle(value);
#else
        var union = new SingleInt32Union { Int32 = value };
        return union.Single;
#endif
    }

    [StructLayout(LayoutKind.Explicit)]
    private struct SingleInt32Union
    {
        [FieldOffset(0)]
        public float Single;
        [FieldOffset(0)]
        public int Int32;
    }
}

/// <summary>
/// Represents an 8-bit floating point number in E4M3 format (4 exponent bits, 3 mantissa bits).
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> E4M3 is like a compressed version of regular floating point numbers.
/// It uses only 8 bits (1 byte) instead of 32 bits, making it 4x smaller. The trade-off is
/// reduced precision and range.
/// </para>
/// <para>
/// <b>Format Details:</b>
/// <list type="bullet">
/// <item><description>1 sign bit</description></item>
/// <item><description>4 exponent bits (bias = 7)</description></item>
/// <item><description>3 mantissa bits</description></item>
/// <item><description>Range: ±448</description></item>
/// <item><description>Smallest positive: ~0.001953125</description></item>
/// </list>
/// </para>
/// <para>
/// <b>Use Cases:</b> Best for weights and activations in neural network forward passes
/// where values are typically well-bounded.
/// </para>
/// <para>
/// <b>Hardware:</b> NVIDIA H100/H200 GPUs have native FP8 Tensor Cores that can process
/// E4M3 values at 2x the throughput of FP16.
/// </para>
/// </remarks>
[StructLayout(LayoutKind.Sequential, Size = 1)]
public readonly struct Float8E4M3 : IEquatable<Float8E4M3>, IComparable<Float8E4M3>
{
    private readonly byte _value;

    // E4M3 constants
    private const int SignBit = 7;
    private const int ExponentBits = 4;
    private const int MantissaBits = 3;
    private const int ExponentBias = 7;
    private const int ExponentMask = 0x78; // bits 6-3
    private const int MantissaMask = 0x07; // bits 2-0
    private const int SignMask = 0x80;     // bit 7

    /// <summary>
    /// Maximum representable value in E4M3 format (448).
    /// </summary>
    public static readonly Float8E4M3 MaxValue = FromFloat(448f);

    /// <summary>
    /// Minimum representable positive value in E4M3 format.
    /// </summary>
    public static readonly Float8E4M3 MinPositive = FromFloat(0.001953125f);

    /// <summary>
    /// Represents positive zero.
    /// </summary>
    public static readonly Float8E4M3 Zero = new(0);

    /// <summary>
    /// Represents positive one.
    /// </summary>
    public static readonly Float8E4M3 One = FromFloat(1.0f);

    /// <summary>
    /// Represents NaN (Not a Number).
    /// </summary>
    public static readonly Float8E4M3 NaN = new(0x7F); // All exponent and mantissa bits set

    private Float8E4M3(byte value) => _value = value;

    /// <summary>
    /// Gets the raw byte value.
    /// </summary>
    public byte RawValue => _value;

    /// <summary>
    /// Returns true if this value is NaN.
    /// </summary>
    public bool IsNaN => (_value & 0x7F) == 0x7F;

    /// <summary>
    /// Returns true if this value is zero.
    /// </summary>
    public bool IsZero => (_value & 0x7F) == 0;

    /// <summary>
    /// Returns true if this value is negative.
    /// </summary>
    public bool IsNegative => (_value & SignMask) != 0;

    /// <summary>
    /// Creates a Float8E4M3 from a single-precision float.
    /// </summary>
    /// <param name="value">The float value to convert.</param>
    /// <returns>The E4M3 representation.</returns>
    public static Float8E4M3 FromFloat(float value)
    {
        if (float.IsNaN(value))
            return NaN;

        // Handle zero (including negative zero)
        // Check bits to distinguish +0 from -0 since (0f == -0f) is true
        int valueBits = BitConverterHelper.SingleToInt32Bits(value);
        if ((valueBits & 0x7FFFFFFF) == 0) // Exponent and mantissa are zero
        {
            return valueBits < 0 ? new Float8E4M3(0x80) : Zero;
        }

        int sign = value < 0 ? 1 : 0;
        value = Math.Abs(value);

        // Clamp to E4M3 range
        const float maxVal = 448f;
        const float minVal = 0.001953125f; // 2^-9

        if (value > maxVal)
            value = maxVal;
        else if (value < minVal)
            return sign == 1 ? new Float8E4M3(0x80) : Zero; // Flush to zero

        // Convert to E4M3
        int floatBits = BitConverterHelper.SingleToInt32Bits(value);
        int floatExponent = ((floatBits >> 23) & 0xFF) - 127; // FP32 bias
        int floatMantissa = floatBits & 0x7FFFFF;

        // Convert exponent (clamp to E4M3 range)
        int e4m3Exponent = floatExponent + ExponentBias;
        if (e4m3Exponent < 0)
            e4m3Exponent = 0;
        else if (e4m3Exponent > 15)
            e4m3Exponent = 15;

        // Convert mantissa (keep top 3 bits)
        int e4m3Mantissa = (floatMantissa >> 20) & MantissaMask;

        // Round to nearest
        if ((floatMantissa & 0x80000) != 0) // Check 4th bit
        {
            e4m3Mantissa++;
            if (e4m3Mantissa > 7)
            {
                e4m3Mantissa = 0;
                e4m3Exponent++;
            }
        }

        byte result = (byte)((sign << SignBit) | (e4m3Exponent << MantissaBits) | e4m3Mantissa);
        return new Float8E4M3(result);
    }

    /// <summary>
    /// Converts this Float8E4M3 to a single-precision float.
    /// </summary>
    /// <returns>The float representation.</returns>
    public float ToFloat()
    {
        if (IsNaN)
            return float.NaN;

        if (IsZero)
            return IsNegative ? -0f : 0f;

        int sign = (_value >> SignBit) & 1;
        int exponent = (_value >> MantissaBits) & 0xF;
        int mantissa = _value & MantissaMask;

        // Handle subnormal numbers (exponent == 0 but mantissa != 0)
        if (exponent == 0)
        {
            // Subnormal: value = 0.mantissa * 2^(1-bias) = mantissa/8 * 2^(-6)
            // Compute directly to avoid denormalized FP32 issues
            float value = mantissa * (1.0f / 8.0f) * (1.0f / 64.0f); // mantissa/8 * 2^-6
            return sign == 1 ? -value : value;
        }

        // Normal number: value = 1.mantissa * 2^(exponent-bias)
        // Convert exponent
        int fp32Exponent = exponent - ExponentBias + 127;

        // Convert mantissa (scale from 3 bits to 23 bits)
        int fp32Mantissa = mantissa << 20;

        int fp32Bits = (sign << 31) | (fp32Exponent << 23) | fp32Mantissa;
        return BitConverterHelper.Int32BitsToSingle(fp32Bits);
    }

    /// <inheritdoc />
    public bool Equals(Float8E4M3 other) => _value == other._value;

    /// <inheritdoc />
    public override bool Equals(object? obj) => obj is Float8E4M3 other && Equals(other);

    /// <inheritdoc />
    public override int GetHashCode() => _value;

    /// <inheritdoc />
    public int CompareTo(Float8E4M3 other) => ToFloat().CompareTo(other.ToFloat());

    /// <inheritdoc />
    public override string ToString() => ToFloat().ToString();

    public static bool operator ==(Float8E4M3 left, Float8E4M3 right) => left.Equals(right);
    public static bool operator !=(Float8E4M3 left, Float8E4M3 right) => !left.Equals(right);
    public static bool operator <(Float8E4M3 left, Float8E4M3 right) => left.CompareTo(right) < 0;
    public static bool operator >(Float8E4M3 left, Float8E4M3 right) => left.CompareTo(right) > 0;
    public static bool operator <=(Float8E4M3 left, Float8E4M3 right) => left.CompareTo(right) <= 0;
    public static bool operator >=(Float8E4M3 left, Float8E4M3 right) => left.CompareTo(right) >= 0;

    public static explicit operator float(Float8E4M3 value) => value.ToFloat();
    public static explicit operator Float8E4M3(float value) => FromFloat(value);
}

/// <summary>
/// Represents an 8-bit floating point number in E5M2 format (5 exponent bits, 2 mantissa bits).
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> E5M2 is an 8-bit format with a larger range but less precision than E4M3.
/// It's designed for gradients during backpropagation, which can have a wider range of values.
/// </para>
/// <para>
/// <b>Format Details:</b>
/// <list type="bullet">
/// <item><description>1 sign bit</description></item>
/// <item><description>5 exponent bits (bias = 15)</description></item>
/// <item><description>2 mantissa bits</description></item>
/// <item><description>Range: ±57344</description></item>
/// <item><description>Smallest positive: ~0.0000610352</description></item>
/// </list>
/// </para>
/// <para>
/// <b>Use Cases:</b> Best for gradients in neural network backward passes where the larger
/// dynamic range helps prevent gradient underflow/overflow.
/// </para>
/// <para>
/// <b>Best Practice:</b> Use E5M2 for gradients with dynamic loss scaling to handle the
/// reduced precision while leveraging the larger dynamic range.
/// </para>
/// </remarks>
[StructLayout(LayoutKind.Sequential, Size = 1)]
public readonly struct Float8E5M2 : IEquatable<Float8E5M2>, IComparable<Float8E5M2>
{
    private readonly byte _value;

    // E5M2 constants
    private const int SignBit = 7;
    private const int ExponentBits = 5;
    private const int MantissaBits = 2;
    private const int ExponentBias = 15;
    private const int ExponentMask = 0x7C; // bits 6-2
    private const int MantissaMask = 0x03; // bits 1-0
    private const int SignMask = 0x80;     // bit 7

    /// <summary>
    /// Maximum representable value in E5M2 format (57344).
    /// </summary>
    public static readonly Float8E5M2 MaxValue = FromFloat(57344f);

    /// <summary>
    /// Minimum representable positive value in E5M2 format.
    /// </summary>
    public static readonly Float8E5M2 MinPositive = FromFloat(0.0000610352f);

    /// <summary>
    /// Represents positive zero.
    /// </summary>
    public static readonly Float8E5M2 Zero = new(0);

    /// <summary>
    /// Represents positive one.
    /// </summary>
    public static readonly Float8E5M2 One = FromFloat(1.0f);

    /// <summary>
    /// Represents positive infinity.
    /// </summary>
    public static readonly Float8E5M2 PositiveInfinity = new(0x7C);

    /// <summary>
    /// Represents negative infinity.
    /// </summary>
    public static readonly Float8E5M2 NegativeInfinity = new(0xFC);

    /// <summary>
    /// Represents NaN (Not a Number).
    /// </summary>
    public static readonly Float8E5M2 NaN = new(0x7F);

    private Float8E5M2(byte value) => _value = value;

    /// <summary>
    /// Gets the raw byte value.
    /// </summary>
    public byte RawValue => _value;

    /// <summary>
    /// Returns true if this value is NaN.
    /// </summary>
    public bool IsNaN
    {
        get
        {
            int exp = (_value >> MantissaBits) & 0x1F;
            int mantissa = _value & MantissaMask;
            return exp == 31 && mantissa != 0;
        }
    }

    /// <summary>
    /// Returns true if this value is infinity (positive or negative).
    /// </summary>
    public bool IsInfinity
    {
        get
        {
            int exp = (_value >> MantissaBits) & 0x1F;
            int mantissa = _value & MantissaMask;
            return exp == 31 && mantissa == 0;
        }
    }

    /// <summary>
    /// Returns true if this value is zero.
    /// </summary>
    public bool IsZero => (_value & 0x7F) == 0;

    /// <summary>
    /// Returns true if this value is negative.
    /// </summary>
    public bool IsNegative => (_value & SignMask) != 0;

    /// <summary>
    /// Creates a Float8E5M2 from a single-precision float.
    /// </summary>
    /// <param name="value">The float value to convert.</param>
    /// <returns>The E5M2 representation.</returns>
    public static Float8E5M2 FromFloat(float value)
    {
        if (float.IsNaN(value))
            return NaN;

        if (float.IsPositiveInfinity(value))
            return PositiveInfinity;

        if (float.IsNegativeInfinity(value))
            return NegativeInfinity;

        // Handle zero (including negative zero)
        // Check bits to distinguish +0 from -0 since (0f == -0f) is true
        int valueBits = BitConverterHelper.SingleToInt32Bits(value);
        if ((valueBits & 0x7FFFFFFF) == 0) // Exponent and mantissa are zero
        {
            return valueBits < 0 ? new Float8E5M2(0x80) : Zero;
        }

        int sign = value < 0 ? 1 : 0;
        value = Math.Abs(value);

        // Clamp to E5M2 range
        const float maxVal = 57344f;
        const float minVal = 0.0000610352f; // 2^-14

        if (value > maxVal)
            return sign == 1 ? NegativeInfinity : PositiveInfinity;
        else if (value < minVal)
            return sign == 1 ? new Float8E5M2(0x80) : Zero; // Flush to zero

        // Convert to E5M2
        int floatBits = BitConverterHelper.SingleToInt32Bits(value);
        int floatExponent = ((floatBits >> 23) & 0xFF) - 127; // FP32 bias
        int floatMantissa = floatBits & 0x7FFFFF;

        // Convert exponent (clamp to E5M2 range)
        int e5m2Exponent = floatExponent + ExponentBias;
        if (e5m2Exponent < 0)
            e5m2Exponent = 0;
        else if (e5m2Exponent > 30)
            e5m2Exponent = 30;

        // Convert mantissa (keep top 2 bits)
        int e5m2Mantissa = (floatMantissa >> 21) & MantissaMask;

        // Round to nearest
        if ((floatMantissa & 0x100000) != 0) // Check 3rd bit
        {
            e5m2Mantissa++;
            if (e5m2Mantissa > 3)
            {
                e5m2Mantissa = 0;
                e5m2Exponent++;
            }
        }

        byte result = (byte)((sign << SignBit) | (e5m2Exponent << MantissaBits) | e5m2Mantissa);
        return new Float8E5M2(result);
    }

    /// <summary>
    /// Converts this Float8E5M2 to a single-precision float.
    /// </summary>
    /// <returns>The float representation.</returns>
    public float ToFloat()
    {
        if (IsNaN)
            return float.NaN;

        if (IsInfinity)
            return IsNegative ? float.NegativeInfinity : float.PositiveInfinity;

        if (IsZero)
            return IsNegative ? -0f : 0f;

        int sign = (_value >> SignBit) & 1;
        int exponent = (_value >> MantissaBits) & 0x1F;
        int mantissa = _value & MantissaMask;

        // Handle subnormal numbers (exponent == 0 but mantissa != 0)
        if (exponent == 0)
        {
            // Subnormal: value = 0.mantissa * 2^(1-bias) = mantissa/4 * 2^(-14)
            // Compute directly to avoid denormalized FP32 issues
            float value = mantissa * (1.0f / 4.0f) * (1.0f / 16384.0f); // mantissa/4 * 2^-14
            return sign == 1 ? -value : value;
        }

        // Normal number: value = 1.mantissa * 2^(exponent-bias)
        // Convert exponent
        int fp32Exponent = exponent - ExponentBias + 127;

        // Convert mantissa (scale from 2 bits to 23 bits)
        int fp32Mantissa = mantissa << 21;

        int fp32Bits = (sign << 31) | (fp32Exponent << 23) | fp32Mantissa;
        return BitConverterHelper.Int32BitsToSingle(fp32Bits);
    }

    /// <inheritdoc />
    public bool Equals(Float8E5M2 other) => _value == other._value;

    /// <inheritdoc />
    public override bool Equals(object? obj) => obj is Float8E5M2 other && Equals(other);

    /// <inheritdoc />
    public override int GetHashCode() => _value;

    /// <inheritdoc />
    public int CompareTo(Float8E5M2 other) => ToFloat().CompareTo(other.ToFloat());

    /// <inheritdoc />
    public override string ToString() => ToFloat().ToString();

    public static bool operator ==(Float8E5M2 left, Float8E5M2 right) => left.Equals(right);
    public static bool operator !=(Float8E5M2 left, Float8E5M2 right) => !left.Equals(right);
    public static bool operator <(Float8E5M2 left, Float8E5M2 right) => left.CompareTo(right) < 0;
    public static bool operator >(Float8E5M2 left, Float8E5M2 right) => left.CompareTo(right) > 0;
    public static bool operator <=(Float8E5M2 left, Float8E5M2 right) => left.CompareTo(right) <= 0;
    public static bool operator >=(Float8E5M2 left, Float8E5M2 right) => left.CompareTo(right) >= 0;

    public static explicit operator float(Float8E5M2 value) => value.ToFloat();
    public static explicit operator Float8E5M2(float value) => FromFloat(value);
}

/// <summary>
/// Provides utility methods for working with FP8 types.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This class provides helper methods to convert between FP8 formats
/// and standard floating-point types, as well as bulk conversion methods for arrays.
/// </para>
/// </remarks>
public static class Float8Extensions
{
    /// <summary>
    /// Converts an array of floats to E4M3 format.
    /// </summary>
    /// <param name="values">Float values to convert.</param>
    /// <returns>Array of E4M3 values.</returns>
    public static Float8E4M3[] ToE4M3(this float[] values)
    {
        var result = new Float8E4M3[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            result[i] = Float8E4M3.FromFloat(values[i]);
        }
        return result;
    }

    /// <summary>
    /// Converts an array of E4M3 values to floats.
    /// </summary>
    /// <param name="values">E4M3 values to convert.</param>
    /// <returns>Array of float values.</returns>
    public static float[] ToFloatArray(this Float8E4M3[] values)
    {
        var result = new float[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            result[i] = values[i].ToFloat();
        }
        return result;
    }

    /// <summary>
    /// Converts an array of floats to E5M2 format.
    /// </summary>
    /// <param name="values">Float values to convert.</param>
    /// <returns>Array of E5M2 values.</returns>
    public static Float8E5M2[] ToE5M2(this float[] values)
    {
        var result = new Float8E5M2[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            result[i] = Float8E5M2.FromFloat(values[i]);
        }
        return result;
    }

    /// <summary>
    /// Converts an array of E5M2 values to floats.
    /// </summary>
    /// <param name="values">E5M2 values to convert.</param>
    /// <returns>Array of float values.</returns>
    public static float[] ToFloatArray(this Float8E5M2[] values)
    {
        var result = new float[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            result[i] = values[i].ToFloat();
        }
        return result;
    }

    /// <summary>
    /// Converts E4M3 to E5M2 (for gradients).
    /// </summary>
    /// <param name="value">E4M3 value.</param>
    /// <returns>E5M2 representation.</returns>
    public static Float8E5M2 ToE5M2(this Float8E4M3 value) => Float8E5M2.FromFloat(value.ToFloat());

    /// <summary>
    /// Converts E5M2 to E4M3 (for weights/activations).
    /// </summary>
    /// <param name="value">E5M2 value.</param>
    /// <returns>E4M3 representation.</returns>
    public static Float8E4M3 ToE4M3(this Float8E5M2 value) => Float8E4M3.FromFloat(value.ToFloat());
}
