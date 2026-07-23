namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Simulates snow effect on the image.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class Snow<T> : ImageAugmenterBase<T>
{
    public double MinSnowPoint { get; }
    public double MaxSnowPoint { get; }
    public double MinBrightness { get; }
    public double MaxBrightness { get; }

    public Snow(double minSnowPoint = 0.1, double maxSnowPoint = 0.3,
        double minBrightness = 1.5, double maxBrightness = 2.5,
        double probability = 0.5) : base(probability)
    {
        MinSnowPoint = minSnowPoint; MaxSnowPoint = maxSnowPoint;
        MinBrightness = minBrightness; MaxBrightness = maxBrightness;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        double maxVal = data.IsNormalized ? 1.0 : 255.0;
        double snowPoint = context.GetRandomDouble(MinSnowPoint, MaxSnowPoint);
        double brightness = context.GetRandomDouble(MinBrightness, MaxBrightness);

        // Add snow particles
        int numParticles = (int)(snowPoint * data.Height * data.Width * 0.1);
        for (int i = 0; i < numParticles; i++)
        {
            int y = context.GetRandomInt(0, data.Height);
            int x = context.GetRandomInt(0, data.Width);
            int radius = context.GetRandomInt(1, 3);

            for (int dy = -radius; dy <= radius; dy++)
                for (int dx = -radius; dx <= radius; dx++)
                {
                    int ny = y + dy; int nx = x + dx;
                    if (ny < 0 || ny >= data.Height || nx < 0 || nx >= data.Width) continue;
                    if (dy * dy + dx * dx > radius * radius) continue;

                    for (int c = 0; c < data.Channels; c++)
                        result.SetPixel(ny, nx, c, NumOps.FromDouble(maxVal));
                }
        }

        // Brighten the overall image
        for (int y = 0; y < data.Height; y++)
            for (int x = 0; x < data.Width; x++)
            {
                // Compute luminance
                double lum = 0;
                for (int c = 0; c < data.Channels; c++)
                    lum += NumOps.ToDouble(result.GetPixel(y, x, c));
                lum /= data.Channels;

                double normLum = lum / maxVal;
                if (normLum > (1 - snowPoint))
                {
                    for (int c = 0; c < data.Channels; c++)
                    {
                        double val = NumOps.ToDouble(result.GetPixel(y, x, c)) * brightness;
                        result.SetPixel(y, x, c, NumOps.FromDouble(Math.Max(0, Math.Min(maxVal, val))));
                    }
                }
            }

        return result;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["min_snow_point"] = MinSnowPoint; p["max_snow_point"] = MaxSnowPoint;
        p["min_brightness"] = MinBrightness; p["max_brightness"] = MaxBrightness;
        return p;
    }
}
