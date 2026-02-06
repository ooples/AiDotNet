namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Simulates spatter effects (mud, rain drops) on the image.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class Spatter<T> : ImageAugmenterBase<T>
{
    public double Intensity { get; }
    public double MinSpotSize { get; }
    public double MaxSpotSize { get; }
    public SpatterMode Mode { get; }

    public Spatter(double intensity = 0.3, double minSpotSize = 0.01, double maxSpotSize = 0.05,
        SpatterMode mode = SpatterMode.Mud, double probability = 0.5) : base(probability)
    {
        Intensity = intensity; MinSpotSize = minSpotSize; MaxSpotSize = maxSpotSize; Mode = mode;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        double maxVal = data.IsNormalized ? 1.0 : 255.0;
        int numSpots = (int)(Intensity * 50);

        for (int s = 0; s < numSpots; s++)
        {
            double spotRadius = context.GetRandomDouble(MinSpotSize, MaxSpotSize) *
                                Math.Min(data.Height, data.Width);
            int centerY = context.GetRandomInt(0, data.Height);
            int centerX = context.GetRandomInt(0, data.Width);

            int r = (int)Math.Ceiling(spotRadius);
            int yMin = Math.Max(0, centerY - r);
            int yMax = Math.Min(data.Height - 1, centerY + r);
            int xMin = Math.Max(0, centerX - r);
            int xMax = Math.Min(data.Width - 1, centerX + r);

            for (int y = yMin; y <= yMax; y++)
            {
                for (int x = xMin; x <= xMax; x++)
                {
                    double dist = Math.Sqrt((y - centerY) * (y - centerY) + (x - centerX) * (x - centerX));
                    if (dist > spotRadius) continue;

                    double alpha = 1.0 - (dist / spotRadius);
                    alpha *= context.GetRandomDouble(0.5, 1.0);

                    for (int c = 0; c < data.Channels; c++)
                    {
                        double original = NumOps.ToDouble(data.GetPixel(y, x, c));
                        double spotColor = GetSpotColor(c, maxVal);
                        double blended = original * (1 - alpha) + spotColor * alpha;
                        result.SetPixel(y, x, c,
                            NumOps.FromDouble(Math.Max(0, Math.Min(maxVal, blended))));
                    }
                }
            }
        }

        return result;
    }

    private double GetSpotColor(int channel, double maxVal)
    {
        return Mode switch
        {
            SpatterMode.Mud => channel switch
            {
                0 => 0.4 * maxVal,  // brownish R
                1 => 0.3 * maxVal,  // brownish G
                _ => 0.15 * maxVal  // brownish B
            },
            SpatterMode.Rain => 0.85 * maxVal, // bright, semi-transparent
            _ => 0.4 * maxVal
        };
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["intensity"] = Intensity; p["mode"] = Mode.ToString();
        p["min_spot_size"] = MinSpotSize; p["max_spot_size"] = MaxSpotSize;
        return p;
    }
}

/// <summary>
/// Spatter effect type.
/// </summary>
public enum SpatterMode
{
    /// <summary>Mud-like brown spatter.</summary>
    Mud,
    /// <summary>Rain drop-like transparent spatter.</summary>
    Rain
}
