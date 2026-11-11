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

        var engine = Engine;
        int channels = Math.Min(source.Channels, reference.Channels);
        var sourcePlanar = source.ToPlanarTensor();
        var referencePlanar = reference.ToPlanarTensor();
        T keep = NumOps.FromDouble(1 - BlendFactor);
        T blend = NumOps.FromDouble(BlendFactor);
        T zero = NumOps.Zero;
        T max = NumOps.FromDouble(maxVal);
        T binSpan = NumOps.FromDouble(numBins - 1.0);
        var planes = new Tensor<T>[channels];

        for (int c = 0; c < channels; c++)
        {
            // Normalized source and reference CDFs
            var srcPlane = engine.TensorSliceAxis(sourcePlanar, 0, c);
            var srcBins = ImageTensor<T>.HistogramBins(srcPlane, maxVal, numBins);
            var srcCdf = NormalizedCdf(engine, srcBins, numBins);
            var refCdf = NormalizedCdf(engine,
                ImageTensor<T>.HistogramBins(engine.TensorSliceAxis(referencePlanar, 0, c), maxVal, numBins), numBins);

            // Mapping LUT: first reference bin whose CDF reaches the source CDF (last bin if none)
            var lut = engine.TensorClampMax(engine.TensorSearchSorted(refCdf, srcCdf, false), numBins - 1);

            // Apply mapping with blend
            var levels = engine.TensorMultiplyScalar(engine.TensorDivideScalar(lut.Cast<T>(), binSpan), max);
            var mapped = engine.TensorTake(levels, srcBins);
            planes[c] = engine.TensorClamp(
                engine.TensorAdd(engine.TensorMultiplyScalar(srcPlane, keep), engine.TensorMultiplyScalar(mapped, blend)),
                zero, max);
        }

        result.SetFromPlanarTensor(ImageTensor<T>.ReplaceLeadingChannels(sourcePlanar, planes));

        return result;
    }

    /// <summary>
    /// CDF of the bin histogram, divided by the total count when it is positive.
    /// </summary>
    private static Tensor<T> NormalizedCdf(IEngine engine, Tensor<int> bins, int numBins)
    {
        var cdf = engine.TensorCumSum(engine.TensorBinCount(engine.Reshape(bins, [bins.Length]), numBins), 0).Cast<T>();
        int total = bins.Length;
        return total > 0 ? engine.TensorDivideScalar(cdf, NumOps.FromDouble(total)) : cdf;
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
