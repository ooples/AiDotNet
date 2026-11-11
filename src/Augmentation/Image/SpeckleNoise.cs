namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Adds multiplicative speckle noise: output = input + input * noise.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class SpeckleNoise<T> : ImageAugmenterBase<T>
{
    public double MinStd { get; }
    public double MaxStd { get; }

    public SpeckleNoise(double minStd = 0.05, double maxStd = 0.2, double probability = 0.5)
        : base(probability)
    {
        MinStd = minStd; MaxStd = maxStd;
    }

    /// <inheritdoc />
    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        double std = context.GetRandomDouble(MinStd, MaxStd);
        double maxVal = data.IsNormalized ? 1.0 : 255.0;
        var result = data.Clone();

        int h = data.Height, w = data.Width, channels = data.Channels;
        var noise = AugmentationRandom<T>.Gaussian(context, [h, w, channels], 0, std);

        // noisy = val + val * noise
        var engine = Engine;
        var image = data.ToPlanarTensor();
        var noisy = engine.TensorAdd(image, engine.TensorMultiply(image, ImageTensor<T>.PlanarFromInterleaved(noise, h, w, channels)));
        result.SetFromPlanarTensor(engine.TensorClamp(noisy, NumOps.Zero, NumOps.FromDouble(maxVal)));

        return result;
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters(); p["min_std"] = MinStd; p["max_std"] = MaxStd; return p;
    }
}
