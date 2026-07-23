namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Random Erasing augmentation (Zhong et al. 2020) that randomly erases rectangular regions.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class RandomErasing<T> : ImageAugmenterBase<T>
{
    public double MinArea { get; }
    public double MaxArea { get; }
    public double MinAspectRatio { get; }
    public double MaxAspectRatio { get; }
    public double FillValue { get; }
    public bool FillWithNoise { get; }
    public int MaxAttempts { get; }

    public RandomErasing(double minArea = 0.02, double maxArea = 0.33,
        double minAspectRatio = 0.3, double maxAspectRatio = 3.3,
        double fillValue = 0.0, bool fillWithNoise = false,
        int maxAttempts = 10, double probability = 0.5) : base(probability)
    {
        MinArea = minArea; MaxArea = maxArea;
        MinAspectRatio = minAspectRatio; MaxAspectRatio = maxAspectRatio;
        FillValue = fillValue; FillWithNoise = fillWithNoise;
        MaxAttempts = maxAttempts;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        double maxVal = data.IsNormalized ? 1.0 : 255.0;
        double imageArea = (double)data.Height * data.Width;

        for (int attempt = 0; attempt < MaxAttempts; attempt++)
        {
            double targetArea = imageArea * context.GetRandomDouble(MinArea, MaxArea);
            double logRatioMin = Math.Log(MinAspectRatio);
            double logRatioMax = Math.Log(MaxAspectRatio);
            double aspectRatio = Math.Exp(context.GetRandomDouble(logRatioMin, logRatioMax));

            int eraseH = (int)Math.Round(Math.Sqrt(targetArea * aspectRatio));
            int eraseW = (int)Math.Round(Math.Sqrt(targetArea / aspectRatio));

            if (eraseH >= data.Height || eraseW >= data.Width) continue;

            int y1 = context.GetRandomInt(0, data.Height - eraseH);
            int x1 = context.GetRandomInt(0, data.Width - eraseW);

            for (int y = y1; y < y1 + eraseH; y++)
                for (int x = x1; x < x1 + eraseW; x++)
                    for (int c = 0; c < data.Channels; c++)
                    {
                        double val = FillWithNoise
                            ? context.GetRandomDouble(0, maxVal)
                            : FillValue;
                        result.SetPixel(y, x, c, NumOps.FromDouble(val));
                    }

            break; // Successfully erased
        }

        return result;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["min_area"] = MinArea; p["max_area"] = MaxArea;
        p["min_aspect_ratio"] = MinAspectRatio; p["max_aspect_ratio"] = MaxAspectRatio;
        p["fill_value"] = FillValue; p["fill_with_noise"] = FillWithNoise;
        return p;
    }
}
