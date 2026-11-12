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

        // Content and reference statistics per channel (channels present in both images)
        var engine = Engine;
        int channels = Math.Min(content.Channels, reference.Channels);
        var contentPlanar = content.ToPlanarTensor();
        var contentRgb = ImageTensor<T>.LeadingChannels(contentPlanar, channels);
        var (contentMean, contentStd) = ImageTensor<T>.ChannelMeanStd(contentRgb);
        var (refMean, refStd) = ImageTensor<T>.ChannelMeanStd(ImageTensor<T>.LeadingChannels(reference.ToPlanarTensor(), channels));

        // Transfer: normalize content, apply reference stats, blend
        T alpha = NumOps.FromDouble(Alpha);
        var targetMean = engine.TensorAdd(contentMean, engine.TensorMultiplyScalar(engine.TensorSubtract(refMean, contentMean), alpha));
        var targetStd = engine.TensorAdd(contentStd, engine.TensorMultiplyScalar(engine.TensorSubtract(refStd, contentStd), alpha));

        var normalized = engine.TensorDivide(engine.TensorSubtract(contentRgb, contentMean), contentStd);
        var transferred = engine.TensorAdd(engine.TensorMultiply(normalized, targetStd), targetMean);
        var clamped = engine.TensorClamp(transferred, NumOps.Zero, NumOps.FromDouble(maxVal));
        result.SetFromPlanarTensor(ImageTensor<T>.ReplaceLeadingBlock(contentPlanar, clamped));

        return result;
    }

    /// <inheritdoc />
    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        return data.Clone();
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["alpha"] = Alpha;
        return p;
    }
}
