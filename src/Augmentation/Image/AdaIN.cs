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

        // Content and style statistics per channel (channels present in both images)
        var engine = Engine;
        int channels = Math.Min(content.Channels, style.Channels);
        var contentPlanar = content.ToPlanarTensor();
        var contentRgb = ImageTensor<T>.LeadingChannels(contentPlanar, channels);
        var (cMean, cStd) = ImageTensor<T>.ChannelMeanStd(contentRgb);
        var (sMean, sStd) = ImageTensor<T>.ChannelMeanStd(ImageTensor<T>.LeadingChannels(style.ToPlanarTensor(), channels));

        // AdaIN: normalize content, apply style stats
        var normalized = engine.TensorDivide(engine.TensorSubtract(contentRgb, cMean), cStd);
        var styled = engine.TensorAdd(engine.TensorMultiply(normalized, sStd), sMean);
        var blended = engine.TensorAdd(
            engine.TensorMultiplyScalar(contentRgb, NumOps.FromDouble(1 - Alpha)),
            engine.TensorMultiplyScalar(styled, NumOps.FromDouble(Alpha)));
        var clamped = engine.TensorClamp(blended, NumOps.Zero, NumOps.FromDouble(maxVal));
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
