namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Matches the histogram of an image to a reference histogram or target distribution.
/// </summary>
/// <remarks>
/// <para>
/// Histogram matching transforms pixel intensities so the output histogram matches a specified
/// reference. This is useful for normalizing images from different sources to have similar
/// appearance, or for style transfer by matching color distributions.
/// </para>
/// <para><b>For Beginners:</b> If you have images taken under different lighting conditions,
/// histogram matching can make them look more similar by adjusting one image's color
/// distribution to match another's.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class HistogramMatching<T> : ImageAugmenterBase<T>
{
    /// <summary>
    /// Gets the reference histogram to match (per channel, normalized CDF).
    /// If null, matches to a uniform distribution.
    /// </summary>
    public double[][]? ReferenceHistograms { get; }

    /// <summary>
    /// Gets the blend factor [0, 1]. 1.0 = full match, 0.0 = no change.
    /// </summary>
    public double BlendFactor { get; }

    /// <summary>
    /// Creates a new histogram matching augmentation.
    /// </summary>
    /// <param name="referenceHistograms">Reference histograms per channel (each 256 bins, normalized). Null = uniform.</param>
    /// <param name="blendFactor">How much to blend toward the reference. Default is 1.0.</param>
    /// <param name="probability">Probability of applying. Default is 0.5.</param>
    public HistogramMatching(double[][]? referenceHistograms = null, double blendFactor = 1.0,
        double probability = 0.5) : base(probability)
    {
        if (blendFactor < 0 || blendFactor > 1)
            throw new ArgumentOutOfRangeException(nameof(blendFactor));
        ReferenceHistograms = referenceHistograms;
        BlendFactor = blendFactor;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        int h = data.Height;
        int w = data.Width;
        int numBins = 256;
        double maxVal = data.IsNormalized ? 1.0 : 255.0;

        for (int c = 0; c < data.Channels; c++)
        {
            MatchChannel(data, result, h, w, c, numBins, maxVal);
        }

        return result;
    }

    private void MatchChannel(ImageTensor<T> src, ImageTensor<T> dst,
        int h, int w, int channel, int numBins, double maxVal)
    {
        // Build source CDF
        var srcHist = new int[numBins];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                double val = NumOps.ToDouble(src.GetPixel(y, x, channel));
                int bin = Math.Max(0, Math.Min(numBins - 1, (int)(val / maxVal * (numBins - 1))));
                srcHist[bin]++;
            }

        var srcCdf = new double[numBins];
        srcCdf[0] = srcHist[0];
        for (int i = 1; i < numBins; i++) srcCdf[i] = srcCdf[i - 1] + srcHist[i];
        double srcTotal = srcCdf[numBins - 1];
        if (srcTotal > 0)
            for (int i = 0; i < numBins; i++) srcCdf[i] /= srcTotal;

        // Build reference CDF
        var refCdf = new double[numBins];
        if (ReferenceHistograms != null && channel < ReferenceHistograms.Length)
        {
            var refHist = ReferenceHistograms[channel];
            refCdf[0] = refHist[0];
            for (int i = 1; i < Math.Min(numBins, refHist.Length); i++)
                refCdf[i] = refCdf[i - 1] + refHist[i];
            double refTotal = refCdf[numBins - 1];
            if (refTotal > 0)
                for (int i = 0; i < numBins; i++) refCdf[i] /= refTotal;
        }
        else
        {
            // Uniform distribution
            for (int i = 0; i < numBins; i++) refCdf[i] = (i + 1.0) / numBins;
        }

        // Build mapping: for each source bin, find closest reference bin
        var mapping = new int[numBins];
        for (int i = 0; i < numBins; i++)
        {
            double bestDist = double.MaxValue;
            int bestJ = 0;
            for (int j = 0; j < numBins; j++)
            {
                double dist = Math.Abs(srcCdf[i] - refCdf[j]);
                if (dist < bestDist)
                {
                    bestDist = dist;
                    bestJ = j;
                }
            }
            mapping[i] = bestJ;
        }

        // Apply mapping
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                double val = NumOps.ToDouble(src.GetPixel(y, x, channel));
                int bin = Math.Max(0, Math.Min(numBins - 1, (int)(val / maxVal * (numBins - 1))));
                double mapped = (double)mapping[bin] / (numBins - 1) * maxVal;
                double blended = val * (1 - BlendFactor) + mapped * BlendFactor;
                dst.SetPixel(y, x, channel, NumOps.FromDouble(blended));
            }
        }
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["blend_factor"] = BlendFactor;
        p["has_reference"] = ReferenceHistograms != null;
        return p;
    }
}
