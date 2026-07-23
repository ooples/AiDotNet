namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Applies color constancy correction using the Gray World assumption or Max-RGB method.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class ColorConstancy<T> : ImageAugmenterBase<T>
{
    public ColorConstancyMethod Method { get; }
    public double BlendFactor { get; }

    public ColorConstancy(ColorConstancyMethod method = ColorConstancyMethod.GrayWorld,
        double blendFactor = 1.0, double probability = 0.5) : base(probability)
    {
        Method = method; BlendFactor = blendFactor;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        if (data.Channels < 3) return data.Clone();

        var result = data.Clone();
        double maxVal = data.IsNormalized ? 1.0 : 255.0;
        int numPixels = data.Height * data.Width;

        var gains = new double[3];

        if (Method == ColorConstancyMethod.GrayWorld)
        {
            var channelMean = new double[3];
            for (int y = 0; y < data.Height; y++)
                for (int x = 0; x < data.Width; x++)
                    for (int c = 0; c < 3; c++)
                        channelMean[c] += NumOps.ToDouble(data.GetPixel(y, x, c));

            double overallMean = 0;
            for (int c = 0; c < 3; c++)
            {
                channelMean[c] /= numPixels;
                overallMean += channelMean[c];
            }
            overallMean /= 3.0;

            for (int c = 0; c < 3; c++)
                gains[c] = channelMean[c] > 1e-10 ? overallMean / channelMean[c] : 1.0;
        }
        else // MaxRGB
        {
            var channelMax = new double[3];
            for (int y = 0; y < data.Height; y++)
                for (int x = 0; x < data.Width; x++)
                    for (int c = 0; c < 3; c++)
                    {
                        double val = NumOps.ToDouble(data.GetPixel(y, x, c));
                        if (val > channelMax[c]) channelMax[c] = val;
                    }

            double overallMax = Math.Max(channelMax[0], Math.Max(channelMax[1], channelMax[2]));
            for (int c = 0; c < 3; c++)
                gains[c] = channelMax[c] > 1e-10 ? overallMax / channelMax[c] : 1.0;
        }

        // Blend gains toward 1.0
        for (int c = 0; c < 3; c++)
            gains[c] = 1.0 + (gains[c] - 1.0) * BlendFactor;

        for (int y = 0; y < data.Height; y++)
            for (int x = 0; x < data.Width; x++)
                for (int c = 0; c < 3; c++)
                {
                    double val = NumOps.ToDouble(data.GetPixel(y, x, c)) * gains[c];
                    result.SetPixel(y, x, c, NumOps.FromDouble(Math.Max(0, Math.Min(maxVal, val))));
                }

        return result;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["method"] = Method.ToString(); p["blend_factor"] = BlendFactor;
        return p;
    }
}

/// <summary>
/// Color constancy correction method.
/// </summary>
public enum ColorConstancyMethod
{
    /// <summary>Gray World assumption - adjusts so channel means are equal.</summary>
    GrayWorld,
    /// <summary>Max-RGB - adjusts so channel maximums are equal.</summary>
    MaxRgb
}
