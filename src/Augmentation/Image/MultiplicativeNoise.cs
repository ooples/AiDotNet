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

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        double maxVal = data.IsNormalized ? 1.0 : 255.0;
        var result = data.Clone();

        if (!PerChannel)
        {
            // Same multiplier for all channels at each pixel
            for (int y = 0; y < data.Height; y++)
                for (int x = 0; x < data.Width; x++)
                {
                    double mult = context.GetRandomDouble(MinMultiplier, MaxMultiplier);
                    for (int c = 0; c < data.Channels; c++)
                    {
                        double val = NumOps.ToDouble(data.GetPixel(y, x, c)) * mult;
                        result.SetPixel(y, x, c, NumOps.FromDouble(Math.Max(0, Math.Min(maxVal, val))));
                    }
                }
        }
        else
        {
            for (int y = 0; y < data.Height; y++)
                for (int x = 0; x < data.Width; x++)
                    for (int c = 0; c < data.Channels; c++)
                    {
                        double mult = context.GetRandomDouble(MinMultiplier, MaxMultiplier);
                        double val = NumOps.ToDouble(data.GetPixel(y, x, c)) * mult;
                        result.SetPixel(y, x, c, NumOps.FromDouble(Math.Max(0, Math.Min(maxVal, val))));
                    }
        }

        return result;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["min_multiplier"] = MinMultiplier; p["max_multiplier"] = MaxMultiplier;
        p["per_channel"] = PerChannel;
        return p;
    }
}
