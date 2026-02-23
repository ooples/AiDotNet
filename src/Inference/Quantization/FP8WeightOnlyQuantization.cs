using AiDotNet.Deployment.Optimization.Quantization.Formats;

namespace AiDotNet.Inference.Quantization;

/// <summary>
/// Per-row FP8 (E4M3) weight-only quantization for inference.
/// </summary>
/// <remarks>
/// <para>
/// Stores weights as bytes in FP8 E4M3 format with a per-row FP32 scale factor.
/// During inference, weights are dequantized on the fly: w_fp32 = FP8ToFloat(w_fp8) * scale.
/// </para>
/// <para><b>For Beginners:</b> FP8 uses 8 bits in a floating-point layout (4 exponent bits,
/// 3 mantissa bits) instead of a simple integer. This preserves outlier values better than
/// INT8 at the same compression ratio (4x vs FP32). It is the default quantization format
/// on NVIDIA H100 and newer GPUs.
/// </para>
/// </remarks>
internal static class FP8WeightOnlyQuantization
{
    internal readonly struct QuantizedWeights
    {
        public QuantizedWeights(byte[] weights, float[] scales, int rows, int cols)
        {
            Weights = weights;
            Scales = scales;
            Rows = rows;
            Cols = cols;
        }

        /// <summary>FP8 E4M3 encoded weight bytes, row-major [rows * cols].</summary>
        public byte[] Weights { get; }

        /// <summary>Per-row scale factors used to map original range to FP8 range.</summary>
        public float[] Scales { get; }

        public int Rows { get; }
        public int Cols { get; }
    }

    /// <summary>
    /// Quantizes a 2D weight matrix to per-row FP8 E4M3.
    /// </summary>
    public static QuantizedWeights QuantizePerRow(ReadOnlySpan<float> weights, int rows, int cols)
    {
        if (rows <= 0) throw new ArgumentOutOfRangeException(nameof(rows));
        if (cols <= 0) throw new ArgumentOutOfRangeException(nameof(cols));
        if (weights.Length < rows * cols)
            throw new ArgumentException("Weight span too small for given dimensions.", nameof(weights));

        const double fp8Max = 448.0; // E4M3 max value
        var q = new byte[rows * cols];
        var scales = new float[rows];

        for (int r = 0; r < rows; r++)
        {
            int baseIdx = r * cols;

            // Find per-row max absolute value
            float maxAbs = 0f;
            for (int c = 0; c < cols; c++)
            {
                float av = MathF.Abs(weights[baseIdx + c]);
                if (av > maxAbs) maxAbs = av;
            }

            // Scale: maps [−maxAbs, maxAbs] → [−fp8Max, fp8Max]
            float scale = maxAbs > 0f ? maxAbs / (float)fp8Max : 1f;
            scales[r] = scale;

            float inv = 1f / scale;
            for (int c = 0; c < cols; c++)
            {
                double scaled = weights[baseIdx + c] * inv;
                q[baseIdx + c] = FP8Quantizer<float, float[], float[]>.E4M3ToByte(scaled);
            }
        }

        return new QuantizedWeights(q, scales, rows, cols);
    }

    /// <summary>
    /// Dequantizes a single FP8 weight: float_value = FP8ToFloat(byte) * scale.
    /// </summary>
    public static float Dequantize(byte fp8Byte, float scale)
    {
        return (float)FP8Quantizer<float, float[], float[]>.ByteToE4M3(fp8Byte) * scale;
    }
}
