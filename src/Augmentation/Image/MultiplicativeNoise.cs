namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Applies multiplicative Gaussian noise: output = input * (1 + noise).
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class MultiplicativeNoise<T> : ImageAugmenterBase<T>
{
    public double MinMultiplier { get; }
    public double MaxMultiplier { get; }
    public bool PerChannel { get; }

    public MultiplicativeNoise(double minMultiplier = 0.9, double maxMultiplier = 1.1,
        bool perChannel = true, double probability = 0.5) : base(probability)
    {
        MinMultiplier = minMultiplier; MaxMultiplier = maxMultiplier; PerChannel = perChannel;
    }

    /// <inheritdoc />
    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        double maxVal = data.IsNormalized ? 1.0 : 255.0;
        var result = data.Clone();

        int h = data.Height, w = data.Width, channels = data.Channels;
        Tensor<T> multipliers;
        if (!PerChannel)
        {
            // Same multiplier for all channels at each pixel
            multipliers = AugmentationRandom<T>.Uniform(context, [1, h, w], MinMultiplier, MaxMultiplier);
        }
        else
        {
            var mult = AugmentationRandom<T>.Uniform(context, [h, w, channels], MinMultiplier, MaxMultiplier);
            multipliers = ImageTensor<T>.PlanarFromInterleaved(mult, h, w, channels);
        }

        var engine = Engine;
        var scaled = engine.TensorMultiply(data.ToPlanarTensor(), multipliers);
        result.SetFromPlanarTensor(engine.TensorClamp(scaled, NumOps.Zero, NumOps.FromDouble(maxVal)));

        return result;
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["min_multiplier"] = MinMultiplier; p["max_multiplier"] = MaxMultiplier;
        p["per_channel"] = PerChannel;
        return p;
    }
}
