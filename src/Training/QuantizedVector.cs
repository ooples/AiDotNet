namespace AiDotNet.Training;

/// <summary>
/// An 8-bit block-quantized snapshot of an O(n) vector — the shared memory-bounding primitive for
/// the full-gradient streaming optimizers (streaming L-BFGS curvature history, streaming Conjugate
/// Gradient previous gradient/direction).
/// </summary>
/// <remarks>
/// Stores one signed int8 per element plus one fp64 scale per block (default 2048 elements), i.e.
/// ~4× smaller than fp32 and ~8× smaller than fp64. <see cref="Dequantize"/> reconstructs into a
/// caller-provided reused buffer so a method holding several of these still only spends O(n)
/// transient memory while it walks them one at a time.
/// </remarks>
internal sealed class QuantizedVector
{
    private readonly byte[] _quantized;
    private readonly double[] _scales;
    private readonly int _blockSize;

    private QuantizedVector(byte[] quantized, double[] scales, int blockSize)
    {
        _quantized = quantized;
        _scales = scales;
        _blockSize = blockSize;
    }

    /// <summary>Quantizes the first <paramref name="n"/> elements of <paramref name="v"/>.</summary>
    public static QuantizedVector Quantize(double[] v, int n, int blockSize)
    {
        int numBlocks = (n + blockSize - 1) / blockSize;
        var quantized = new byte[n];
        var scales = new double[numBlocks];
        for (int b = 0; b < numBlocks; b++)
        {
            int start = b * blockSize, end = Math.Min(start + blockSize, n);
            double max = 0.0;
            for (int i = start; i < end; i++)
            {
                double a = Math.Abs(v[i]);
                if (!double.IsNaN(a) && !double.IsInfinity(a) && a > max) max = a;
            }
            double scale = max / 127.0;
            if (scale < 1e-12 || double.IsNaN(scale) || double.IsInfinity(scale)) scale = 1e-12;
            scales[b] = scale;
            double inv = 1.0 / scale;
            for (int i = start; i < end; i++)
            {
                int qi = (int)Math.Round(v[i] * inv);
                if (qi < -127) qi = -127; else if (qi > 127) qi = 127;
                quantized[i] = (byte)(qi + 128);
            }
        }
        return new QuantizedVector(quantized, scales, blockSize);
    }

    /// <summary>Reconstructs the first <paramref name="n"/> elements into <paramref name="dst"/>.</summary>
    public void Dequantize(double[] dst, int n)
    {
        for (int b = 0; b * _blockSize < n; b++)
        {
            int start = b * _blockSize, end = Math.Min(start + _blockSize, n);
            double scale = _scales[b];
            for (int i = start; i < end; i++) dst[i] = (_quantized[i] - 128) * scale;
        }
    }
}
