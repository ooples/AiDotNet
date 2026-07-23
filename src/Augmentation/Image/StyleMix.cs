namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Style mixing augmentation - transfers statistical style (mean/variance) from a reference image.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class StyleMix<T> : ImageAugmenterBase<T>
{
    public double Alpha { get; }

    public StyleMix(double alpha = 0.5, double probability = 0.5) : base(probability)
    {
        Alpha = alpha;
    }

    /// <summary>
    /// Transfers style statistics from reference to content image.
    /// </summary>
    public ImageTensor<T> ApplyStyleMix(ImageTensor<T> content, ImageTensor<T> reference,
        AugmentationContext<T> context)
    {
        var result = content.Clone();
        double maxVal = content.IsNormalized ? 1.0 : 255.0;
        int numPixels = content.Height * content.Width;

        for (int c = 0; c < Math.Min(content.Channels, reference.Channels); c++)
        {
            // Compute content stats
            double contentMean = 0, contentVar = 0;
            for (int y = 0; y < content.Height; y++)
                for (int x = 0; x < content.Width; x++)
                    contentMean += NumOps.ToDouble(content.GetPixel(y, x, c));
            contentMean /= numPixels;

            for (int y = 0; y < content.Height; y++)
                for (int x = 0; x < content.Width; x++)
                {
                    double diff = NumOps.ToDouble(content.GetPixel(y, x, c)) - contentMean;
                    contentVar += diff * diff;
                }
            contentVar /= numPixels;
            double contentStd = Math.Sqrt(contentVar + 1e-8);

            // Compute reference stats
            int refPixels = reference.Height * reference.Width;
            double refMean = 0, refVar = 0;
            for (int y = 0; y < reference.Height; y++)
                for (int x = 0; x < reference.Width; x++)
                    refMean += NumOps.ToDouble(reference.GetPixel(y, x, c));
            refMean /= refPixels;

            for (int y = 0; y < reference.Height; y++)
                for (int x = 0; x < reference.Width; x++)
                {
                    double diff = NumOps.ToDouble(reference.GetPixel(y, x, c)) - refMean;
                    refVar += diff * diff;
                }
            refVar /= refPixels;
            double refStd = Math.Sqrt(refVar + 1e-8);

            // Transfer: normalize content, apply reference stats, blend
            double targetMean = contentMean + Alpha * (refMean - contentMean);
            double targetStd = contentStd + Alpha * (refStd - contentStd);

            for (int y = 0; y < content.Height; y++)
                for (int x = 0; x < content.Width; x++)
                {
                    double val = NumOps.ToDouble(content.GetPixel(y, x, c));
                    double normalized = (val - contentMean) / contentStd;
                    double transferred = normalized * targetStd + targetMean;
                    result.SetPixel(y, x, c,
                        NumOps.FromDouble(Math.Max(0, Math.Min(maxVal, transferred))));
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
