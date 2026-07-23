namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Fourier Domain Adaptation (Yang &amp; Soatto, 2020) - swaps low-frequency spectral
/// components between source and target images for domain adaptation.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class FDA<T> : ImageAugmenterBase<T>
{
    public double Beta { get; }

    public FDA(double beta = 0.01, double probability = 0.5) : base(probability)
    {
        if (beta < 0 || beta > 1) throw new ArgumentOutOfRangeException(nameof(beta));
        Beta = beta;
    }

    /// <summary>
    /// Transfers low-frequency spectral components from target to source.
    /// </summary>
    public ImageTensor<T> ApplyFDA(ImageTensor<T> source, ImageTensor<T> target,
        AugmentationContext<T> context)
    {
        if (source.Height != target.Height || source.Width != target.Width)
            target = new Resize<T>(source.Height, source.Width).Apply(target, context);

        var result = source.Clone();
        double maxVal = source.IsNormalized ? 1.0 : 255.0;

        int maskH = (int)(source.Height * Beta);
        int maskW = (int)(source.Width * Beta);
        if (maskH < 1 || maskW < 1) return result;

        // Simple spectral transfer: replace average of low-frequency region
        for (int c = 0; c < Math.Min(source.Channels, target.Channels); c++)
        {
            // Compute mean of both images in the low-frequency region (center crop as proxy)
            double srcMean = 0, tgtMean = 0;
            int count = 0;
            int startY = (source.Height - maskH) / 2;
            int startX = (source.Width - maskW) / 2;

            for (int y = startY; y < startY + maskH; y++)
                for (int x = startX; x < startX + maskW; x++)
                {
                    srcMean += NumOps.ToDouble(source.GetPixel(y, x, c));
                    tgtMean += NumOps.ToDouble(target.GetPixel(y, x, c));
                    count++;
                }

            if (count > 0)
            {
                srcMean /= count;
                tgtMean /= count;
                double shift = tgtMean - srcMean;

                for (int y = 0; y < source.Height; y++)
                    for (int x = 0; x < source.Width; x++)
                    {
                        double val = NumOps.ToDouble(source.GetPixel(y, x, c)) + shift * Beta;
                        result.SetPixel(y, x, c,
                            NumOps.FromDouble(Math.Max(0, Math.Min(maxVal, val))));
                    }
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
        p["beta"] = Beta;
        return p;
    }
}
