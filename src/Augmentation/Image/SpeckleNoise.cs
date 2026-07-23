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

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        double std = context.GetRandomDouble(MinStd, MaxStd);
        double maxVal = data.IsNormalized ? 1.0 : 255.0;
        var result = data.Clone();

        for (int y = 0; y < data.Height; y++)
            for (int x = 0; x < data.Width; x++)
                for (int c = 0; c < data.Channels; c++)
                {
                    double val = NumOps.ToDouble(data.GetPixel(y, x, c));
                    double noise = context.SampleGaussian(0, std);
                    double noisy = val + val * noise;
                    result.SetPixel(y, x, c, NumOps.FromDouble(Math.Max(0, Math.Min(maxVal, noisy))));
                }

        return result;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters(); p["min_std"] = MinStd; p["max_std"] = MaxStd; return p;
    }
}
