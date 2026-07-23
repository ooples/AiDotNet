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

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        double colorShift = context.GetRandomDouble(MinColorShift, MaxColorShift);
        double intensity = context.GetRandomDouble(MinIntensity, MaxIntensity);
        double maxVal = data.IsNormalized ? 1.0 : 255.0;
        var result = data.Clone();

        for (int y = 0; y < data.Height; y++)
            for (int x = 0; x < data.Width; x++)
            {
                // Luminance noise (same for all channels)
                double lumNoise = context.SampleGaussian(0, intensity * 0.01 * maxVal);
                for (int c = 0; c < data.Channels; c++)
                {
                    double val = NumOps.ToDouble(data.GetPixel(y, x, c));
                    // Per-channel color noise
                    double colorNoise = context.SampleGaussian(0, colorShift * maxVal);
                    double noisy = val + lumNoise + colorNoise;
                    result.SetPixel(y, x, c, NumOps.FromDouble(Math.Max(0, Math.Min(maxVal, noisy))));
                }
            }

        return result;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["min_color_shift"] = MinColorShift; p["max_color_shift"] = MaxColorShift;
        p["min_intensity"] = MinIntensity; p["max_intensity"] = MaxIntensity;
        return p;
    }
}
