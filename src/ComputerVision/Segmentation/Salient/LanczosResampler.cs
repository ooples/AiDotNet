namespace AiDotNet.ComputerVision.Segmentation.Salient;

/// <summary>
/// Pillow's <c>Image.resize(..., Image.Resampling.LANCZOS)</c>: separable Lanczos-3 resampling whose kernel widens
/// by the scale when shrinking (antialiasing), horizontal pass first, weights normalised per output sample
/// (<c>libImaging/Resample.c</c>, <c>precompute_coeffs</c>).
/// </summary>
internal static class LanczosResampler
{
    private const double Support = 3.0;

    /// <summary>Resizes a planar image [channels, height, width] (row-major).</summary>
    /// <param name="source">The planar source.</param>
    /// <param name="channels">Channel count.</param>
    /// <param name="height">Source height.</param>
    /// <param name="width">Source width.</param>
    /// <param name="outHeight">Output height.</param>
    /// <param name="outWidth">Output width.</param>
    /// <param name="clampMin">Lower clip of the result (Pillow clips 8-bit images to [0, 255]).</param>
    /// <param name="clampMax">Upper clip of the result.</param>
    /// <returns>The planar result [channels, outHeight, outWidth].</returns>
    public static double[] Resize(
        double[] source, int channels, int height, int width, int outHeight, int outWidth, double clampMin, double clampMax)
    {
        if (source is null) throw new ArgumentNullException(nameof(source));
        if (source.Length != channels * height * width)
            throw new ArgumentException("The source does not match its dimensions.", nameof(source));
        var horizontal = new double[channels * height * outWidth];
        var (xStart, xWeights) = Coefficients(width, outWidth);
        for (int c = 0; c < channels; c++)
        {
            for (int y = 0; y < height; y++)
            {
                int row = ((c * height) + y) * width;
                int outRow = ((c * height) + y) * outWidth;
                for (int x = 0; x < outWidth; x++)
                {
                    double sum = 0.0;
                    var w = xWeights[x];
                    for (int k = 0; k < w.Length; k++)
                        sum += w[k] * source[row + xStart[x] + k];
                    horizontal[outRow + x] = sum;
                }
            }
        }

        var output = new double[channels * outHeight * outWidth];
        var (yStart, yWeights) = Coefficients(height, outHeight);
        for (int c = 0; c < channels; c++)
        {
            for (int y = 0; y < outHeight; y++)
            {
                var w = yWeights[y];
                for (int x = 0; x < outWidth; x++)
                {
                    double sum = 0.0;
                    for (int k = 0; k < w.Length; k++)
                        sum += w[k] * horizontal[(((c * height) + yStart[y] + k) * outWidth) + x];
                    output[(((c * outHeight) + y) * outWidth) + x] = Math.Min(clampMax, Math.Max(clampMin, sum));
                }
            }
        }

        return output;
    }

    /// <summary>Pillow's <c>precompute_coeffs</c> for one axis.</summary>
    internal static (int[] Start, double[][] Weights) Coefficients(int inSize, int outSize)
    {
        double scale = (double)inSize / outSize;
        double filterScale = Math.Max(scale, 1.0);
        double support = Support * filterScale;
        var start = new int[outSize];
        var weights = new double[outSize][];
        for (int x = 0; x < outSize; x++)
        {
            double center = (x + 0.5) * scale;
            int min = Math.Max((int)(center - support + 0.5), 0);
            int max = Math.Min((int)(center + support + 0.5), inSize) - min;
            var w = new double[Math.Max(0, max)];
            double total = 0.0;
            for (int k = 0; k < w.Length; k++)
            {
                w[k] = Lanczos((k + min - center + 0.5) / filterScale);
                total += w[k];
            }

            if (total != 0.0)
            {
                for (int k = 0; k < w.Length; k++)
                    w[k] /= total;
            }

            start[x] = min;
            weights[x] = w;
        }

        return (start, weights);
    }

    private static double Lanczos(double x)
    {
        if (x > -Support && x < Support)
            return Sinc(x) * Sinc(x / Support);
        return 0.0;
    }

    private static double Sinc(double x)
    {
        if (x == 0.0) return 1.0;
        x *= Math.PI;
        return Math.Sin(x) / x;
    }
}
