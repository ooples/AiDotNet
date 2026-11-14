using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.FederatedLearning.Privacy;

/// <summary>
/// Fixed-point encoding of real-valued model updates into the ring <c>Z_R</c>, <c>R = 2^b</c>, in which Bonawitz et al.
/// (2017) secure aggregation adds inputs and masks.
/// </summary>
/// <remarks>
/// <para>
/// The paper's protocol sums vectors in <c>Z_R^m</c> (section 3, "each user holds a vector x_u of length m with
/// elements in Z_R"). Masks only cancel exactly in modular integer arithmetic; added in floating point they leave
/// rounding residue. A value <c>x</c> is clipped to <c>[-ClipBound, ClipBound]</c>, scaled by
/// <c>2^FractionalBits</c>, rounded to the nearest integer (ties to even) and reduced modulo <c>R</c> in two's
/// complement. Decoding interprets a ring element as a signed integer in <c>[-R/2, R/2)</c> and divides by the scale.
/// </para>
/// <para>
/// The modular sum of <c>n</c> encodings decodes to the exact sum of the fixed-point values as long as that sum stays
/// in <c>[-R/2, R/2)</c>; <see cref="EnsureSumFits(int)"/> checks the sufficient condition
/// <c>n * ClipBound * 2^FractionalBits &lt; 2^(b-1)</c>.
/// </para>
/// <para><b>For Beginners:</b> Cryptographic masks are random whole numbers that wrap around like a clock. To add a
/// model parameter such as 0.375 to them, it is first turned into a whole number (0.375 * 2^24), and turned back after
/// all masks have cancelled. The defaults keep 24 fractional bits (finer than single-precision rounding near 1) inside
/// 64-bit ring elements.</para>
/// </remarks>
[ComponentType(ComponentType.CryptoPrimitive)]
[PipelineStage(PipelineStage.Training)]
public sealed class FixedPointRingEncoding
{
    /// <summary>The default ring bit width <c>b</c>.</summary>
    public const int DefaultBitWidth = 64;

    /// <summary>The default number of fractional bits.</summary>
    public const int DefaultFractionalBits = 24;

    /// <summary>The default clipping bound (the largest finite half-precision magnitude, rounded up to 2^16).</summary>
    public const double DefaultClipBound = 65536.0;

    private readonly ulong _mask;
    private readonly ulong _signBit;

    /// <summary>
    /// Creates an encoding for the ring <c>Z_{2^bitWidth}</c>.
    /// </summary>
    /// <param name="bitWidth">The ring bit width <c>b</c>, between 2 and 64.</param>
    /// <param name="fractionalBits">The number of fractional bits <c>f</c>; the scale is <c>2^f</c>. Between 0 and
    /// <c>b - 2</c>.</param>
    /// <param name="clipBound">The clipping bound <c>C &gt; 0</c>; inputs are clipped to <c>[-C, C]</c>. A single
    /// clipped input must fit in the signed range: <c>C * 2^f &lt; 2^(b-1)</c>.</param>
    public FixedPointRingEncoding(
        int bitWidth = DefaultBitWidth,
        int fractionalBits = DefaultFractionalBits,
        double clipBound = DefaultClipBound)
    {
        if (bitWidth < 2 || bitWidth > 64)
        {
            throw new ArgumentOutOfRangeException(nameof(bitWidth), "Bit width must be between 2 and 64.");
        }

        if (fractionalBits < 0 || fractionalBits > bitWidth - 2)
        {
            throw new ArgumentOutOfRangeException(nameof(fractionalBits), "Fractional bits must be between 0 and bitWidth - 2.");
        }

        if (double.IsNaN(clipBound) || double.IsInfinity(clipBound) || clipBound <= 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(clipBound), "Clip bound must be a positive finite number.");
        }

        BitWidth = bitWidth;
        FractionalBits = fractionalBits;
        ClipBound = clipBound;
        Scale = Math.Pow(2.0, fractionalBits);
        _mask = bitWidth == 64 ? ulong.MaxValue : (1UL << bitWidth) - 1UL;
        _signBit = 1UL << (bitWidth - 1);

        if (clipBound * Scale >= Math.Pow(2.0, bitWidth - 1))
        {
            throw new ArgumentOutOfRangeException(
                nameof(clipBound),
                "clipBound * 2^fractionalBits must be below 2^(bitWidth - 1) so one clipped input fits the signed ring range.");
        }
    }

    /// <summary>Gets the ring bit width <c>b</c>; the ring is <c>Z_R</c> with <c>R = 2^b</c>.</summary>
    public int BitWidth { get; }

    /// <summary>Gets the number of fractional bits <c>f</c>.</summary>
    public int FractionalBits { get; }

    /// <summary>Gets the clipping bound <c>C</c>.</summary>
    public double ClipBound { get; }

    /// <summary>Gets the fixed-point scale <c>2^f</c>.</summary>
    public double Scale { get; }

    /// <summary>Gets <c>R - 1</c>, the mask that reduces a 64-bit integer modulo <c>R</c>.</summary>
    public ulong ModulusMask => _mask;

    /// <summary>
    /// Encodes <paramref name="value"/>: clip to <c>[-C, C]</c>, multiply by <c>2^f</c>, round to nearest (ties to even),
    /// reduce modulo <c>R</c>.
    /// </summary>
    /// <exception cref="ArgumentException"><paramref name="value"/> is NaN.</exception>
    public ulong Encode(double value)
    {
        if (double.IsNaN(value))
        {
            throw new ArgumentException("NaN cannot be encoded for secure aggregation.", nameof(value));
        }

        double clipped = value > ClipBound ? ClipBound : (value < -ClipBound ? -ClipBound : value);
        long integer = (long)Math.Round(clipped * Scale, MidpointRounding.ToEven);
        return unchecked((ulong)integer) & _mask;
    }

    /// <summary>
    /// Decodes a ring element: interpret it as a signed integer in <c>[-R/2, R/2)</c> and divide by <c>2^f</c>.
    /// </summary>
    public double Decode(ulong ringElement) => ToSignedInteger(ringElement) / Scale;

    /// <summary>Interprets a ring element as the signed integer in <c>[-R/2, R/2)</c> congruent to it.</summary>
    public long ToSignedInteger(ulong ringElement)
    {
        ulong reduced = ringElement & _mask;
        if (BitWidth == 64)
        {
            return unchecked((long)reduced);
        }

        return (reduced & _signBit) != 0
            ? unchecked((long)reduced - (long)(_mask + 1UL))
            : (long)reduced;
    }

    /// <summary>Returns <c>(a + b) mod R</c>.</summary>
    public ulong Add(ulong a, ulong b) => unchecked(a + b) & _mask;

    /// <summary>Returns <c>(a - b) mod R</c>.</summary>
    public ulong Subtract(ulong a, ulong b) => unchecked(a - b) & _mask;

    /// <summary>Adds <paramref name="addend"/> into <paramref name="accumulator"/> element-wise modulo <c>R</c>.</summary>
    public void AddInPlace(ulong[] accumulator, ulong[] addend)
    {
        ValidatePair(accumulator, addend);
        for (int i = 0; i < accumulator.Length; i++)
        {
            accumulator[i] = unchecked(accumulator[i] + addend[i]) & _mask;
        }
    }

    /// <summary>Subtracts <paramref name="subtrahend"/> from <paramref name="accumulator"/> element-wise modulo <c>R</c>.</summary>
    public void SubtractInPlace(ulong[] accumulator, ulong[] subtrahend)
    {
        ValidatePair(accumulator, subtrahend);
        for (int i = 0; i < accumulator.Length; i++)
        {
            accumulator[i] = unchecked(accumulator[i] - subtrahend[i]) & _mask;
        }
    }

    /// <summary>
    /// Throws when the sum of <paramref name="clientCount"/> clipped inputs could leave the signed ring range, which
    /// would make the decoded aggregate wrap around: requires <c>n * C * 2^f &lt; 2^(b-1)</c>.
    /// </summary>
    public void EnsureSumFits(int clientCount)
    {
        if (clientCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(clientCount), "Client count must be positive.");
        }

        if (clientCount * ClipBound * Scale >= Math.Pow(2.0, BitWidth - 1))
        {
            throw new ArgumentException(
                $"The sum of {clientCount} inputs clipped to {ClipBound} with {FractionalBits} fractional bits can overflow a " +
                $"{BitWidth}-bit ring. Increase the bit width, or lower the clip bound or fractional bits.",
                nameof(clientCount));
        }
    }

    private static void ValidatePair(ulong[] accumulator, ulong[] other)
    {
        if (accumulator is null)
        {
            throw new ArgumentNullException(nameof(accumulator));
        }

        if (other is null)
        {
            throw new ArgumentNullException(nameof(other));
        }

        if (accumulator.Length != other.Length)
        {
            throw new ArgumentException("Ring vectors must have the same length.", nameof(other));
        }
    }
}
