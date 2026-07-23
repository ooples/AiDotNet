namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Adaptive Instance Normalization (AdaIN) for style transfer augmentation.
/// Normalizes content features and applies style statistics.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class AdaIN<T> : ImageAugmenterBase<T>
{
    public double Alpha { get; }

    public AdaIN(double alpha = 1.0, double probability = 0.5) : base(probability)
    {
        Alpha = alpha;
    }

    /// <summary>
    /// Applies AdaIN: content structure with style statistics.
    /// </summary>
    public ImageTensor<T> ApplyAdaIN(ImageTensor<T> content, ImageTensor<T> style,
        AugmentationContext<T> context)
    {
        var result = content.Clone();
        double maxVal = content.IsNormalized ? 1.0 : 255.0;
        int numPixels = content.Height * content.Width;

        for (int c = 0; c < Math.Min(content.Channels, style.Channels); c++)
        {
            // Content statistics
            double cMean = 0;
            for (int y = 0; y < content.Height; y++)
                for (int x = 0; x < content.Width; x++)
                    cMean += NumOps.ToDouble(content.GetPixel(y, x, c));
            cMean /= numPixels;

            double cVar = 0;
            for (int y = 0; y < content.Height; y++)
                for (int x = 0; x < content.Width; x++)
                {
                    double d = NumOps.ToDouble(content.GetPixel(y, x, c)) - cMean;
                    cVar += d * d;
                }
            double cStd = Math.Sqrt(cVar / numPixels + 1e-8);

            // Style statistics
            int sPixels = style.Height * style.Width;
            double sMean = 0;
            for (int y = 0; y < style.Height; y++)
                for (int x = 0; x < style.Width; x++)
                    sMean += NumOps.ToDouble(style.GetPixel(y, x, c));
            sMean /= sPixels;

            double sVar = 0;
            for (int y = 0; y < style.Height; y++)
                for (int x = 0; x < style.Width; x++)
                {
                    double d = NumOps.ToDouble(style.GetPixel(y, x, c)) - sMean;
                    sVar += d * d;
                }
            double sStd = Math.Sqrt(sVar / sPixels + 1e-8);

            // AdaIN: normalize content, apply style stats
            for (int y = 0; y < content.Height; y++)
                for (int x = 0; x < content.Width; x++)
                {
                    double val = NumOps.ToDouble(content.GetPixel(y, x, c));
                    double normalized = (val - cMean) / cStd;
                    double styled = normalized * sStd + sMean;
                    double blended = val * (1 - Alpha) + styled * Alpha;
                    result.SetPixel(y, x, c,
                        NumOps.FromDouble(Math.Max(0, Math.Min(maxVal, blended))));
                }
        }

        return result;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        return data.Clone();
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["alpha"] = Alpha;
        return p;
    }
}
