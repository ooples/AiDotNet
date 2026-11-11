namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Simulates camera sensor noise (ISO noise) with separate color and intensity components.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class ISONoise<T> : ImageAugmenterBase<T>
{
    public double MinColorShift { get; }
    public double MaxColorShift { get; }
    public double MinIntensity { get; }
    public double MaxIntensity { get; }

    public ISONoise(double minColorShift = 0.01, double maxColorShift = 0.05,
        double minIntensity = 0.1, double maxIntensity = 0.5, double probability = 0.5)
        : base(probability)
    {
        MinColorShift = minColorShift; MaxColorShift = maxColorShift;
        MinIntensity = minIntensity; MaxIntensity = maxIntensity;
    }

    /// <inheritdoc />
    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        double colorShift = context.GetRandomDouble(MinColorShift, MaxColorShift);
        double intensity = context.GetRandomDouble(MinIntensity, MaxIntensity);
        double maxVal = data.IsNormalized ? 1.0 : 255.0;
        var result = data.Clone();

        // Draw luminance noise (one per pixel) and colour noise (one per channel) in loop order
        int h = data.Height, w = data.Width, channels = data.Channels;
        var engine = Engine;
        var draws = AugmentationRandom<T>.StandardNormal(context, [h * w, 1 + channels]);
        var lumNoise = engine.TensorMultiplyScalar(
            engine.Reshape(engine.TensorSlice(draws, [0, 0], [h * w, 1]), [1, h, w]),
            NumOps.FromDouble(intensity * 0.01 * maxVal));
        var colorNoise = engine.TensorMultiplyScalar(
            engine.TensorSlice(draws, [0, 1], [h * w, channels]), NumOps.FromDouble(colorShift * maxVal));

        // noisy = val + lumNoise + colorNoise
        var noisy = engine.TensorAdd(
            engine.TensorAdd(data.ToPlanarTensor(), lumNoise),
            ImageTensor<T>.PlanarFromInterleaved(colorNoise.Contiguous(), h, w, channels));
        result.SetFromPlanarTensor(engine.TensorClamp(noisy, NumOps.Zero, NumOps.FromDouble(maxVal)));

        return result;
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["min_color_shift"] = MinColorShift; p["max_color_shift"] = MaxColorShift;
        p["min_intensity"] = MinIntensity; p["max_intensity"] = MaxIntensity;
        return p;
    }
}
