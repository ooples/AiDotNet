namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Adds random shadow regions to the image.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class Shadow<T> : ImageAugmenterBase<T>
{
    public int MinShadows { get; }
    public int MaxShadows { get; }
    public double MinDarkness { get; }
    public double MaxDarkness { get; }

    public Shadow(int minShadows = 1, int maxShadows = 3,
        double minDarkness = 0.3, double maxDarkness = 0.7,
        double probability = 0.5) : base(probability)
    {
        if (minShadows < 0) throw new ArgumentOutOfRangeException(nameof(minShadows));
        if (maxShadows < minShadows) throw new ArgumentOutOfRangeException(nameof(maxShadows), "maxShadows must be >= minShadows.");
        if (minDarkness < 0 || minDarkness > 1) throw new ArgumentOutOfRangeException(nameof(minDarkness));
        if (maxDarkness < minDarkness || maxDarkness > 1) throw new ArgumentOutOfRangeException(nameof(maxDarkness), "maxDarkness must be >= minDarkness and <= 1.");
        MinShadows = minShadows; MaxShadows = maxShadows;
        MinDarkness = minDarkness; MaxDarkness = maxDarkness;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        double maxVal = data.IsNormalized ? 1.0 : 255.0;
        int numShadows = context.GetRandomInt(MinShadows, MaxShadows + 1);

        for (int s = 0; s < numShadows; s++)
        {
            double darkness = context.GetRandomDouble(MinDarkness, MaxDarkness);

            // Use half-plane shadow with gradient
            double x1 = context.GetRandomDouble(0, data.Width);
            double y1 = context.GetRandomDouble(0, data.Height);
            double angle = context.GetRandomDouble(0, Math.PI * 2);
            double nx = Math.Cos(angle);
            double ny = Math.Sin(angle);

            for (int y = 0; y < data.Height; y++)
                for (int x = 0; x < data.Width; x++)
                {
                    double dist = (x - x1) * nx + (y - y1) * ny;
                    double shadowFactor;

                    if (dist < -20)
                        shadowFactor = 0;
                    else if (dist > 20)
                        shadowFactor = 1;
                    else
                        shadowFactor = (dist + 20) / 40.0;

                    double multiplier = 1.0 - shadowFactor * darkness;

                    for (int c = 0; c < data.Channels; c++)
                    {
                        double val = NumOps.ToDouble(result.GetPixel(y, x, c));
                        result.SetPixel(y, x, c,
                            NumOps.FromDouble(Math.Max(0, Math.Min(maxVal, val * multiplier))));
                    }
                }
        }

        return result;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["min_shadows"] = MinShadows; p["max_shadows"] = MaxShadows;
        p["min_darkness"] = MinDarkness; p["max_darkness"] = MaxDarkness;
        return p;
    }
}
