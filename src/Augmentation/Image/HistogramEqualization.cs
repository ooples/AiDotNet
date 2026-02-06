namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Applies standard histogram equalization to improve image contrast.
/// </summary>
/// <remarks>
/// <para>
/// Histogram equalization redistributes pixel intensity values to produce a more uniform
/// histogram, effectively spreading out the most frequent intensity values. This improves
/// contrast, especially for images that are too dark or too bright.
/// </para>
/// <para><b>For Beginners:</b> If your image looks washed out or too dark, histogram
/// equalization stretches the range of colors to use the full spectrum, making details
/// more visible.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class HistogramEqualization<T> : ImageAugmenterBase<T>
{
    /// <summary>
    /// Gets whether to apply per-channel equalization.
    /// When false, converts to luminance, equalizes, then converts back.
    /// </summary>
    public bool PerChannel { get; }

    /// <summary>
    /// Creates a new histogram equalization augmentation.
    /// </summary>
    /// <param name="perChannel">Whether to equalize each channel independently. Default is false.</param>
    /// <param name="probability">Probability of applying. Default is 0.5.</param>
    public HistogramEqualization(bool perChannel = false, double probability = 0.5) : base(probability)
    {
        PerChannel = perChannel;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        int h = data.Height;
        int w = data.Width;
        int numBins = 256;
        double maxVal = data.IsNormalized ? 1.0 : 255.0;

        if (PerChannel || data.Channels == 1)
        {
            for (int c = 0; c < data.Channels; c++)
            {
                EqualizeChannel(data, result, h, w, c, numBins, maxVal);
            }
        }
        else
        {
            // Equalize based on luminance: build histogram from average luminance,
            // then apply the same mapping to all channels proportionally
            EqualizeLuminance(data, result, h, w, numBins, maxVal);
        }

        return result;
    }

    private static void EqualizeLuminance(ImageTensor<T> src, ImageTensor<T> dst,
        int h, int w, int numBins, double maxVal)
    {
        // Build luminance histogram (average of all channels)
        var histogram = new int[numBins];
        long totalPixels = (long)h * w;

        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                double lum = 0;
                for (int c = 0; c < src.Channels; c++)
                    lum += NumOps.ToDouble(src.GetPixel(y, x, c));
                lum /= src.Channels;
                int bin = Math.Max(0, Math.Min(numBins - 1, (int)(lum / maxVal * (numBins - 1))));
                histogram[bin]++;
            }

        // Build CDF
        var cdf = new double[numBins];
        cdf[0] = histogram[0];
        for (int i = 1; i < numBins; i++) cdf[i] = cdf[i - 1] + histogram[i];

        double cdfMin = 0;
        for (int i = 0; i < numBins; i++)
            if (cdf[i] > 0) { cdfMin = cdf[i]; break; }

        double denom = totalPixels - cdfMin;
        if (denom < 1) denom = 1;

        // Apply the same luminance-based mapping to all channels
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                double lum = 0;
                for (int c = 0; c < src.Channels; c++)
                    lum += NumOps.ToDouble(src.GetPixel(y, x, c));
                lum /= src.Channels;
                int bin = Math.Max(0, Math.Min(numBins - 1, (int)(lum / maxVal * (numBins - 1))));
                double scale = lum > 1e-10 ? ((cdf[bin] - cdfMin) / denom * maxVal) / lum : 1.0;

                for (int c = 0; c < src.Channels; c++)
                {
                    double val = NumOps.ToDouble(src.GetPixel(y, x, c)) * scale;
                    dst.SetPixel(y, x, c, NumOps.FromDouble(Math.Max(0, Math.Min(maxVal, val))));
                }
            }
    }

    private static void EqualizeChannel(ImageTensor<T> src, ImageTensor<T> dst,
        int h, int w, int channel, int numBins, double maxVal)
    {
        var histogram = new int[numBins];
        long totalPixels = (long)h * w;

        // Build histogram
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                double val = NumOps.ToDouble(src.GetPixel(y, x, channel));
                int bin = Math.Max(0, Math.Min(numBins - 1, (int)(val / maxVal * (numBins - 1))));
                histogram[bin]++;
            }
        }

        // Build CDF
        var cdf = new double[numBins];
        cdf[0] = histogram[0];
        for (int i = 1; i < numBins; i++)
            cdf[i] = cdf[i - 1] + histogram[i];

        // Find first non-zero CDF value
        double cdfMin = 0;
        for (int i = 0; i < numBins; i++)
        {
            if (cdf[i] > 0) { cdfMin = cdf[i]; break; }
        }

        // Apply equalization
        double denom = totalPixels - cdfMin;
        if (denom < 1) denom = 1;

        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                double val = NumOps.ToDouble(src.GetPixel(y, x, channel));
                int bin = Math.Max(0, Math.Min(numBins - 1, (int)(val / maxVal * (numBins - 1))));
                double equalized = (cdf[bin] - cdfMin) / denom * maxVal;
                dst.SetPixel(y, x, channel, NumOps.FromDouble(equalized));
            }
        }
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["per_channel"] = PerChannel;
        return p;
    }
}
