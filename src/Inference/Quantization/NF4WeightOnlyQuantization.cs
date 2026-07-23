using AiDotNet.Deployment.Optimization.Quantization.Formats;

namespace AiDotNet.Inference.Quantization;

/// <summary>
/// Per-group NF4 (4-bit NormalFloat) weight-only quantization for inference.
/// </summary>
/// <remarks>
/// <para>
/// Stores weights as packed 4-bit NF4 indices with per-group FP32 scale factors.
/// Two NF4 values share one byte (low nibble = even index, high nibble = odd index).
/// During inference, weights are dequantized: w_fp32 = NF4Codebook[index] * groupScale.
/// </para>
/// <para><b>For Beginners:</b> NF4 compresses weights to just 4 bits (16 possible values)
/// chosen to be optimal for the bell-curve distribution that neural network weights
/// naturally follow. This gives 8x compression vs FP32. It is the same format used by
/// QLoRA for efficient fine-tuning of large language models.
/// </para>
/// </remarks>
internal static class NF4WeightOnlyQuantization
{
    // NF4 codebook from QLoRA (Dettmers et al., 2023)
    private static readonly double[] NF4Codebook =
    {
        -1.0, -0.6961928, -0.5250730, -0.3949460,
        -0.2844714, -0.1828020, -0.0911346,  0.0,
         0.0796089,  0.1609563,  0.2461107,  0.3379640,
         0.4407326,  0.5626170,  0.7229568,  1.0
    };

    internal readonly struct QuantizedWeights
    {
        public QuantizedWeights(byte[] packedWeights, float[] groupScales, int rows, int cols, int groupSize)
        {
            PackedWeights = packedWeights;
            GroupScales = groupScales;
            Rows = rows;
            Cols = cols;
            GroupSize = groupSize;
        }

        /// <summary>
        /// Packed NF4 indices. Two 4-bit indices per byte (low nibble = even, high nibble = odd).
        /// Length = ceil(rows * cols / 2).
        /// </summary>
        public byte[] PackedWeights { get; }

        /// <summary>Per-group absmax scale factors.</summary>
        public float[] GroupScales { get; }

        public int Rows { get; }
        public int Cols { get; }
        public int GroupSize { get; }
    }

    /// <summary>
    /// Quantizes a weight matrix to per-group NF4.
    /// </summary>
    /// <param name="weights">Row-major weight values.</param>
    /// <param name="rows">Number of rows.</param>
    /// <param name="cols">Number of columns.</param>
    /// <param name="groupSize">Number of elements per quantization group (default: 64).</param>
    public static QuantizedWeights QuantizePerGroup(ReadOnlySpan<float> weights, int rows, int cols, int groupSize = 64)
    {
        if (rows <= 0) throw new ArgumentOutOfRangeException(nameof(rows));
        if (cols <= 0) throw new ArgumentOutOfRangeException(nameof(cols));
        if (groupSize <= 0) throw new ArgumentOutOfRangeException(nameof(groupSize));
        int total = rows * cols;
        if (weights.Length < total)
            throw new ArgumentException("Weight span too small for given dimensions.", nameof(weights));

        int numGroups = (total + groupSize - 1) / groupSize;
        var groupScales = new float[numGroups];
        int packedLen = (total + 1) / 2;
        var packed = new byte[packedLen];

        // Temporary buffer for 4-bit indices
        var indices = new byte[total];

        for (int g = 0; g < numGroups; g++)
        {
            int start = g * groupSize;
            int end = Math.Min(start + groupSize, total);

            // Find group absmax
            float maxAbs = 0f;
            for (int i = start; i < end; i++)
            {
                float av = MathF.Abs(weights[i]);
                if (av > maxAbs) maxAbs = av;
            }

            float scale = maxAbs > 0f ? maxAbs : 1f;
            groupScales[g] = scale;

            // Quantize: normalize to [-1, 1], find nearest NF4 codebook entry
            float inv = 1f / scale;
            for (int i = start; i < end; i++)
            {
                double normalized = weights[i] * inv;
                indices[i] = (byte)FindNearestNF4Index(normalized);
            }
        }

        // Pack two indices per byte
        for (int i = 0; i < total; i += 2)
        {
            byte low = indices[i];
            byte high = (i + 1 < total) ? indices[i + 1] : (byte)0;
            packed[i / 2] = (byte)(low | (high << 4));
        }

        return new QuantizedWeights(packed, groupScales, rows, cols, groupSize);
    }

    /// <summary>
    /// Dequantizes a single NF4 value: float_value = NF4Codebook[index] * groupScale.
    /// </summary>
    public static float Dequantize(int nf4Index, float groupScale)
    {
        return (float)NF4Codebook[nf4Index & 0x0F] * groupScale;
    }

    /// <summary>
    /// Extracts a 4-bit NF4 index from a packed byte array.
    /// </summary>
    public static int ExtractIndex(byte[] packed, int elementIndex)
    {
        int byteIdx = elementIndex / 2;
        return (elementIndex & 1) == 0
            ? packed[byteIdx] & 0x0F
            : (packed[byteIdx] >> 4) & 0x0F;
    }

    private static int FindNearestNF4Index(double value)
    {
        // Clamp to codebook range
        if (value <= NF4Codebook[0]) return 0;
        if (value >= NF4Codebook[15]) return 15;

        int bestIdx = 0;
        double bestDist = Math.Abs(value - NF4Codebook[0]);

        for (int i = 1; i < 16; i++)
        {
            double dist = Math.Abs(value - NF4Codebook[i]);
            if (dist < bestDist)
            {
                bestDist = dist;
                bestIdx = i;
            }
        }

        return bestIdx;
    }
}
