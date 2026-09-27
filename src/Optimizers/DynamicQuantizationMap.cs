namespace AiDotNet.Optimizers;

/// <summary>
/// The 8-bit dynamic quantization data type of Dettmers et al., used for block-wise quantized optimizer state.
/// </summary>
/// <remarks>
/// <para>
/// A block is normalized by its absolute maximum, and each value is stored as the index of the nearest entry in a
/// fixed 256-entry codebook. The codebook is dense near zero (a dynamic exponent with a shrinking fraction), so small
/// values keep their RELATIVE precision instead of rounding to zero. Linear absmax quantization
/// (<c>round(x / (absmax / 255))</c>) rounds every second moment below about 1/510 of its block maximum to zero; the
/// Adam denominator then collapses to epsilon and the update becomes <c>lr * m / epsilon</c>. That is the failure
/// this data type exists to prevent.
/// </para>
/// <para>
/// The first moment uses the signed map and the second moment the unsigned map, as in the reference 8-bit
/// optimizers. The construction below reproduces <c>create_dynamic_map(signed, max_exponent_bits=7, total_bits=8)</c>
/// from bitsandbytes (entries computed through float32 like the reference, so they agree to rounding).
/// </para>
/// <para><b>Reference:</b> T. Dettmers, M. Lewis, S. Shleifer, L. Zettlemoyer, "8-bit Optimizers via Block-wise
/// Quantization", ICLR 2022; T. Dettmers, "8-Bit Approximations for Parallelism in Deep Learning", ICLR 2016.</para>
/// </remarks>
internal static class DynamicQuantizationMap
{
    /// <summary>Codebook for signed values (the first moment), sorted ascending, spanning [-1, 1].</summary>
    internal static readonly double[] Signed = Create(signed: true);

    /// <summary>Codebook for non-negative values (the second moment), sorted ascending, spanning [0, 1].</summary>
    internal static readonly double[] Unsigned = Create(signed: false);

    /// <summary>The index of 0.0 in <see cref="Signed"/>, the encoding of a zero first moment.</summary>
    internal static readonly byte SignedZeroIndex = (byte)Array.IndexOf(Signed, 0.0);

    /// <summary>
    /// The smallest block scale ever stored. A block of exact zeros would otherwise divide by zero; any positive floor
    /// works because every value in such a block encodes to the zero entry.
    /// </summary>
    internal const double MinScale = 1e-30;

    private static double[] Create(bool signed, int maxExponentBits = 7, int totalBits = 8)
    {
        var data = new System.Collections.Generic.List<double>(1 << totalBits);
        int nonSignBits = totalBits - 1;
        int additionalItems = (1 << (nonSignBits - maxExponentBits)) - 1;
        int i = 0;
        for (i = 0; i < maxExponentBits; i++)
        {
            int fractionItems = signed
                ? (1 << (i + nonSignBits - maxExponentBits)) + 1
                : (1 << (i + nonSignBits - maxExponentBits + 1)) + 1;
            AddMeans(data, fractionItems, Math.Pow(10, -(maxExponentBits - 1) + i), signed);
        }

        if (additionalItems > 0)
            AddMeans(data, additionalItems + 1, Math.Pow(10, -(maxExponentBits - 1) + i - 1), signed);

        data.Add(0.0);
        data.Add(1.0);
        if (data.Count != 1 << totalBits)
            throw new InvalidOperationException(
                $"Dynamic quantization map has {data.Count} entries, expected {1 << totalBits}.");
        data.Sort();
        return data.ToArray();
    }

    // torch.linspace(0.1, 1, n) in float32, then midpoints, scaled; float32 first so the entries match the reference.
    private static void AddMeans(System.Collections.Generic.List<double> data, int count, double magnitude, bool signed)
    {
        var boundaries = new float[count];
        for (int k = 0; k < count; k++)
            boundaries[k] = count == 1 ? 0.1f : (float)(0.1 + (1.0 - 0.1) * k / (count - 1));
        for (int k = 0; k < count - 1; k++)
        {
            double mean = (boundaries[k] + boundaries[k + 1]) / 2.0f;
            data.Add(magnitude * mean);
            if (signed) data.Add(-magnitude * mean);
        }
    }

    /// <summary>The block scale for values whose largest magnitude is <paramref name="absMax"/>.</summary>
    internal static double Scale(double absMax) => absMax > MinScale ? absMax : MinScale;

    /// <summary>Encodes <paramref name="value"/> in a block of the given scale as the nearest codebook index.</summary>
    internal static byte Encode(double value, double scale, double[] code)
    {
        double normalized = value / scale;
        double lo = code[0], hi = code[code.Length - 1];
        if (normalized <= lo) return 0;
        if (normalized >= hi) return (byte)(code.Length - 1);
        int upper = UpperIndex(normalized, code);
        int lower = upper - 1;
        return (byte)(normalized - code[lower] <= code[upper] - normalized ? lower : upper);
    }

    /// <summary>
    /// Encodes with stochastic rounding between the two neighbouring codebook entries, choosing the upper one with
    /// probability proportional to how close <paramref name="value"/> is to it, so the encoding is unbiased.
    /// </summary>
    internal static byte EncodeStochastic(double value, double scale, double[] code, double uniform01)
    {
        double normalized = value / scale;
        if (normalized <= code[0]) return 0;
        if (normalized >= code[code.Length - 1]) return (byte)(code.Length - 1);
        int upper = UpperIndex(normalized, code);
        int lower = upper - 1;
        double t = (normalized - code[lower]) / (code[upper] - code[lower]);
        return (byte)(uniform01 < t ? upper : lower);
    }

    /// <summary>Decodes a codebook index in a block of the given scale.</summary>
    internal static double Decode(byte index, double scale, double[] code) => code[index] * scale;

    // Smallest index whose entry is strictly greater than value (value is strictly inside the codebook's range).
    private static int UpperIndex(double value, double[] code)
    {
        int lo = 0, hi = code.Length - 1;
        while (lo < hi)
        {
            int mid = (lo + hi) >> 1;
            if (code[mid] <= value) lo = mid + 1; else hi = mid;
        }
        return lo;
    }
}
