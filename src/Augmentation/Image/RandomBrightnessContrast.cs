namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Randomly adjusts brightness and contrast simultaneously.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class RandomBrightnessContrast<T> : ImageAugmenterBase<T>
{
    public double BrightnessLimit { get; }
    public double ContrastLimit { get; }

    public RandomBrightnessContrast(double brightnessLimit = 0.2, double contrastLimit = 0.2,
        double probability = 0.5) : base(probability)
    {
        if (brightnessLimit < 0) throw new ArgumentOutOfRangeException(nameof(brightnessLimit), "BrightnessLimit must be non-negative.");
        if (contrastLimit < 0) throw new ArgumentOutOfRangeException(nameof(contrastLimit), "ContrastLimit must be non-negative.");
        BrightnessLimit = brightnessLimit; ContrastLimit = contrastLimit;
    }

    /// <inheritdoc />
    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        double maxVal = data.IsNormalized ? 1.0 : 255.0;
        double brightnessFactor = context.GetRandomDouble(-BrightnessLimit, BrightnessLimit);
        double contrastFactor = context.GetRandomDouble(1 - ContrastLimit, 1 + ContrastLimit);

        // Compute per-channel mean for contrast adjustment (prevents color shifts)
        var engine = Engine;
        var image = data.ToPlanarTensor();
        var channelMean = engine.ReduceMean(image, [1, 2], true);

        var contrasted = engine.TensorAdd(
            engine.TensorMultiplyScalar(engine.TensorSubtract(image, channelMean), NumOps.FromDouble(contrastFactor)),
            channelMean);
        var adjusted = engine.TensorAddScalar(contrasted, NumOps.FromDouble(brightnessFactor * maxVal));
        result.SetFromPlanarTensor(engine.TensorClamp(adjusted, NumOps.Zero, NumOps.FromDouble(maxVal)));

        return result;
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["brightness_limit"] = BrightnessLimit; p["contrast_limit"] = ContrastLimit;
        return p;
    }
}
