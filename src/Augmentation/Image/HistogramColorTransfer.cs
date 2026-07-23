namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Transfers color distribution from a reference image using histogram matching.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class HistogramColorTransfer<T> : ImageAugmenterBase<T>
{
    public double BlendFactor { get; }

    public HistogramColorTransfer(double blendFactor = 1.0,
        double probability = 0.5) : base(probability)
    {
        BlendFactor = blendFactor;
    }

    /// <summary>
    /// Transfers color histogram from reference to source.
    /// </summary>
    public ImageTensor<T> ApplyTransfer(ImageTensor<T> source, ImageTensor<T> reference,
        AugmentationContext<T> context)
    {
        var result = source.Clone();
        double maxVal = source.IsNormalized ? 1.0 : 255.0;
        int numBins = 256;

        for (int c = 0; c < Math.Min(source.Channels, reference.Channels); c++)
        {
            // Build source CDF
            var srcHist = new double[numBins];
            for (int y = 0; y < source.Height; y++)
                for (int x = 0; x < source.Width; x++)
                {
                    int bin = (int)(NumOps.ToDouble(source.GetPixel(y, x, c)) / maxVal * (numBins - 1));
                    bin = Math.Max(0, Math.Min(numBins - 1, bin));
                    srcHist[bin]++;
                }

            var srcCdf = new double[numBins];
            srcCdf[0] = srcHist[0];
            for (int i = 1; i < numBins; i++) srcCdf[i] = srcCdf[i - 1] + srcHist[i];
            double srcTotal = srcCdf[numBins - 1];
            if (srcTotal > 0)
                for (int i = 0; i < numBins; i++) srcCdf[i] /= srcTotal;

            // Build reference CDF
            var refHist = new double[numBins];
            for (int y = 0; y < reference.Height; y++)
                for (int x = 0; x < reference.Width; x++)
                {
                    int bin = (int)(NumOps.ToDouble(reference.GetPixel(y, x, c)) / maxVal * (numBins - 1));
                    bin = Math.Max(0, Math.Min(numBins - 1, bin));
                    refHist[bin]++;
                }

            var refCdf = new double[numBins];
            refCdf[0] = refHist[0];
            for (int i = 1; i < numBins; i++) refCdf[i] = refCdf[i - 1] + refHist[i];
            double refTotal = refCdf[numBins - 1];
            if (refTotal > 0)
                for (int i = 0; i < numBins; i++) refCdf[i] /= refTotal;

            // Build mapping LUT
            var lut = new int[numBins];
            for (int i = 0; i < numBins; i++)
            {
                int j = 0;
                while (j < numBins - 1 && refCdf[j] < srcCdf[i]) j++;
                lut[i] = j;
            }

            // Apply mapping with blend
            for (int y = 0; y < source.Height; y++)
                for (int x = 0; x < source.Width; x++)
                {
                    double original = NumOps.ToDouble(source.GetPixel(y, x, c));
                    int bin = (int)(original / maxVal * (numBins - 1));
                    bin = Math.Max(0, Math.Min(numBins - 1, bin));
                    double mapped = (double)lut[bin] / (numBins - 1) * maxVal;
                    double blended = original * (1 - BlendFactor) + mapped * BlendFactor;
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
        p["blend_factor"] = BlendFactor;
        return p;
    }
}
