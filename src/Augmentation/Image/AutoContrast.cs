namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Maximizes image contrast by stretching the intensity range to fill [0, max].
/// </summary>
/// <remarks>
/// <para>AutoContrast finds the minimum and maximum pixel values per channel and linearly
/// scales them to span the full range. This is equivalent to a per-channel min-max normalization.</para>
/// <para><b>For Beginners:</b> If your darkest pixel is 50 and brightest is 200, this stretches
/// them to 0 and 255, making the image use the full brightness range.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class AutoContrast<T> : ImageAugmenterBase<T>
{
    /// <summary>
    /// Gets the percentage of lightest/darkest pixels to ignore when finding min/max.
    /// </summary>
    public double Cutoff { get; }

    public AutoContrast(double cutoff = 0, double probability = 0.5) : base(probability)
    {
        if (cutoff < 0 || cutoff >= 50)
            throw new ArgumentOutOfRangeException(nameof(cutoff), "Must be [0, 50).");
        Cutoff = cutoff;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        double maxVal = data.IsNormalized ? 1.0 : 255.0;
        int h = data.Height, w = data.Width;
        int totalPixels = h * w;
        int cutoffPixels = (int)(totalPixels * Cutoff / 100.0);

        for (int c = 0; c < data.Channels; c++)
        {
            // Collect all values for this channel
            var values = new double[totalPixels];
            int idx = 0;
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    values[idx++] = NumOps.ToDouble(data.GetPixel(y, x, c));

            Array.Sort(values);

            double lo = values[cutoffPixels];
            double hi = values[totalPixels - 1 - cutoffPixels];

            if (Math.Abs(hi - lo) < 1e-10) continue;

            double scale = maxVal / (hi - lo);

            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    double val = NumOps.ToDouble(data.GetPixel(y, x, c));
                    double mapped = (val - lo) * scale;
                    mapped = Math.Max(0, Math.Min(maxVal, mapped));
                    result.SetPixel(y, x, c, NumOps.FromDouble(mapped));
                }
        }

        return result;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters(); p["cutoff"] = Cutoff; return p;
    }
}
