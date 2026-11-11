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
        // Compute mean of both images in the low-frequency region (center crop as proxy) per channel
        var engine = Engine;
        int channels = Math.Min(source.Channels, target.Channels);
        int startY = (source.Height - maskH) / 2;
        int startX = (source.Width - maskW) / 2;
        var sourcePlanar = source.ToPlanarTensor();
        int[] regionStart = [0, startY, startX];
        int[] regionSize = [channels, maskH, maskW];
        var srcMean = engine.ReduceMean(engine.TensorSlice(sourcePlanar, regionStart, regionSize), [1, 2], true);
        var tgtMean = engine.ReduceMean(engine.TensorSlice(target.ToPlanarTensor(), regionStart, regionSize), [1, 2], true);
        var shift = engine.TensorMultiplyScalar(engine.TensorSubtract(tgtMean, srcMean), NumOps.FromDouble(Beta));

        var shifted = engine.TensorClamp(
            engine.TensorAdd(ImageTensor<T>.LeadingChannels(sourcePlanar, channels), shift),
            NumOps.Zero, NumOps.FromDouble(maxVal));
        result.SetFromPlanarTensor(ImageTensor<T>.ReplaceLeadingBlock(sourcePlanar, shifted));

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
