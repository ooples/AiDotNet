namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Adds Poisson (shot) noise that scales with pixel intensity, simulating photon counting noise.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class PoissonNoise<T> : ImageAugmenterBase<T>
{
    public double Scale { get; }

    public PoissonNoise(double scale = 1.0, double probability = 0.5) : base(probability)
    {
        Scale = scale;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        double maxVal = data.IsNormalized ? 1.0 : 255.0;
        var result = data.Clone();

        for (int y = 0; y < data.Height; y++)
            for (int x = 0; x < data.Width; x++)
                for (int c = 0; c < data.Channels; c++)
                {
                    double val = NumOps.ToDouble(data.GetPixel(y, x, c));
                    // Poisson noise approximated by Gaussian with variance = value
                    double normalizedVal = val / maxVal;
                    double stdDev = Math.Sqrt(Math.Max(normalizedVal, 1e-10)) * Scale * 0.1;
                    double noisy = val + context.SampleGaussian(0, stdDev * maxVal);
                    result.SetPixel(y, x, c, NumOps.FromDouble(Math.Max(0, Math.Min(maxVal, noisy))));
                }

        return result;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters(); p["scale"] = Scale; return p;
    }
}
