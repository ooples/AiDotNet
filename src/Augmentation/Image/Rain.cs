namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Simulates rain streaks on the image.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class Rain<T> : ImageAugmenterBase<T>
{
    public double MinSlant { get; }
    public double MaxSlant { get; }
    public double DropLength { get; }
    public double DropWidth { get; }
    public double Intensity { get; }
    public double Brightness { get; }

    public Rain(double minSlant = -10, double maxSlant = 10,
        double dropLength = 0.05, double dropWidth = 1,
        double intensity = 0.3, double brightness = 0.7,
        double probability = 0.5) : base(probability)
    {
        MinSlant = minSlant; MaxSlant = maxSlant;
        DropLength = dropLength; DropWidth = dropWidth;
        Intensity = intensity; Brightness = brightness;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        double maxVal = data.IsNormalized ? 1.0 : 255.0;
        double slant = context.GetRandomDouble(MinSlant, MaxSlant);
        double slantRad = slant * Math.PI / 180.0;
        int numDrops = (int)(Intensity * data.Height * data.Width / 100.0);
        int dropLen = Math.Max(3, (int)(DropLength * data.Height));

        for (int d = 0; d < numDrops; d++)
        {
            int startY = context.GetRandomInt(0, data.Height);
            int startX = context.GetRandomInt(0, data.Width);

            int halfWidth = Math.Max(0, (int)((DropWidth - 1) / 2));

            for (int i = 0; i < dropLen; i++)
            {
                int y = startY + i;
                int xCenter = startX + (int)(i * Math.Tan(slantRad));

                for (int dx = -halfWidth; dx <= halfWidth; dx++)
                {
                    int x = xCenter + dx;
                    if (y < 0 || y >= data.Height || x < 0 || x >= data.Width) continue;

                    for (int c = 0; c < data.Channels; c++)
                    {
                        double val = NumOps.ToDouble(data.GetPixel(y, x, c));
                        double drop = val + Brightness * maxVal * 0.3;
                        result.SetPixel(y, x, c, NumOps.FromDouble(Math.Max(0, Math.Min(maxVal, drop))));
                    }
                }
            }
        }

        // Slight overall brightness reduction for overcast
        for (int y = 0; y < data.Height; y++)
            for (int x = 0; x < data.Width; x++)
                for (int c = 0; c < data.Channels; c++)
                {
                    double val = NumOps.ToDouble(result.GetPixel(y, x, c));
                    result.SetPixel(y, x, c, NumOps.FromDouble(Math.Max(0, val * 0.95)));
                }

        return result;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["min_slant"] = MinSlant; p["max_slant"] = MaxSlant;
        p["drop_length"] = DropLength; p["drop_width"] = DropWidth;
        p["intensity"] = Intensity; p["brightness"] = Brightness;
        return p;
    }
}
