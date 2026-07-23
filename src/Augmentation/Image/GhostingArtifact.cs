namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Simulates MRI ghosting artifacts caused by motion during acquisition.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class GhostingArtifact<T> : ImageAugmenterBase<T>
{
    public int NumGhosts { get; }
    public double MinIntensity { get; }
    public double MaxIntensity { get; }

    public GhostingArtifact(int numGhosts = 3, double minIntensity = 0.05,
        double maxIntensity = 0.3, double probability = 0.5) : base(probability)
    {
        NumGhosts = numGhosts; MinIntensity = minIntensity; MaxIntensity = maxIntensity;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        double maxVal = data.IsNormalized ? 1.0 : 255.0;
        double ghostIntensity = context.GetRandomDouble(MinIntensity, MaxIntensity);

        // Ghosts appear as shifted copies along the phase-encode direction
        bool horizontal = context.GetRandomBool();

        for (int g = 1; g <= NumGhosts; g++)
        {
            int shift = (int)(g * (horizontal ? data.Width : data.Height) / (NumGhosts + 1.0) * 0.3);
            double weight = ghostIntensity / g;

            for (int y = 0; y < data.Height; y++)
                for (int x = 0; x < data.Width; x++)
                {
                    int srcY = horizontal ? y : (y + shift) % data.Height;
                    int srcX = horizontal ? (x + shift) % data.Width : x;

                    for (int c = 0; c < data.Channels; c++)
                    {
                        double current = NumOps.ToDouble(result.GetPixel(y, x, c));
                        double ghost = NumOps.ToDouble(data.GetPixel(srcY, srcX, c));
                        double val = current + ghost * weight;
                        result.SetPixel(y, x, c,
                            NumOps.FromDouble(Math.Max(0, Math.Min(maxVal, val))));
                    }
                }
        }

        return result;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["num_ghosts"] = NumGhosts;
        p["min_intensity"] = MinIntensity; p["max_intensity"] = MaxIntensity;
        return p;
    }
}
