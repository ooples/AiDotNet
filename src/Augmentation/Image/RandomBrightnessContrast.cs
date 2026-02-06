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

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        double maxVal = data.IsNormalized ? 1.0 : 255.0;
        double brightnessFactor = context.GetRandomDouble(-BrightnessLimit, BrightnessLimit);
        double contrastFactor = context.GetRandomDouble(1 - ContrastLimit, 1 + ContrastLimit);

        // Compute per-channel mean for contrast adjustment (prevents color shifts)
        int pixelCount = data.Height * data.Width;
        var channelMean = new double[data.Channels];
        for (int y = 0; y < data.Height; y++)
            for (int x = 0; x < data.Width; x++)
                for (int c = 0; c < data.Channels; c++)
                    channelMean[c] += NumOps.ToDouble(data.GetPixel(y, x, c));
        for (int c = 0; c < data.Channels; c++)
            channelMean[c] /= pixelCount;

        for (int y = 0; y < data.Height; y++)
            for (int x = 0; x < data.Width; x++)
                for (int c = 0; c < data.Channels; c++)
                {
                    double val = NumOps.ToDouble(data.GetPixel(y, x, c));
                    // Apply contrast around per-channel mean, then brightness
                    val = (val - channelMean[c]) * contrastFactor + channelMean[c] + brightnessFactor * maxVal;
                    result.SetPixel(y, x, c, NumOps.FromDouble(Math.Max(0, Math.Min(maxVal, val))));
                }

        return result;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["brightness_limit"] = BrightnessLimit; p["contrast_limit"] = ContrastLimit;
        return p;
    }
}
