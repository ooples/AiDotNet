namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Simulates sun flare / lens flare effect on the image.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class SunFlare<T> : ImageAugmenterBase<T>
{
    public double MinRadius { get; }
    public double MaxRadius { get; }
    public int NumFlareCircles { get; }

    public SunFlare(double minRadius = 0.1, double maxRadius = 0.4,
        int numFlareCircles = 5, double probability = 0.5) : base(probability)
    {
        MinRadius = minRadius; MaxRadius = maxRadius;
        NumFlareCircles = numFlareCircles;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        double maxVal = data.IsNormalized ? 1.0 : 255.0;

        // Sun position (typically upper half)
        int sunY = context.GetRandomInt(0, data.Height / 2);
        int sunX = context.GetRandomInt(0, data.Width);
        double mainRadius = context.GetRandomDouble(MinRadius, MaxRadius) * Math.Min(data.Height, data.Width);

        // Draw main sun glow
        DrawGlow(result, sunY, sunX, mainRadius, maxVal, 0.8);

        // Draw secondary flare circles along a line from center
        int centerY = data.Height / 2;
        int centerX = data.Width / 2;

        for (int i = 0; i < NumFlareCircles; i++)
        {
            double t = context.GetRandomDouble(0.3, 1.5);
            int flareY = (int)(sunY + (centerY - sunY) * t);
            int flareX = (int)(sunX + (centerX - sunX) * t);
            double flareRadius = mainRadius * context.GetRandomDouble(0.1, 0.5);

            if (flareY >= 0 && flareY < data.Height && flareX >= 0 && flareX < data.Width)
                DrawGlow(result, flareY, flareX, flareRadius, maxVal, 0.3);
        }

        return result;
    }

    private static void DrawGlow(ImageTensor<T> image, int centerY, int centerX,
        double radius, double maxVal, double intensity)
    {
        int r = (int)Math.Ceiling(radius);
        int yMin = Math.Max(0, centerY - r);
        int yMax = Math.Min(image.Height - 1, centerY + r);
        int xMin = Math.Max(0, centerX - r);
        int xMax = Math.Min(image.Width - 1, centerX + r);

        for (int y = yMin; y <= yMax; y++)
            for (int x = xMin; x <= xMax; x++)
            {
                double dist = Math.Sqrt((y - centerY) * (y - centerY) + (x - centerX) * (x - centerX));
                if (dist > radius) continue;

                double alpha = (1 - dist / radius) * intensity;
                for (int c = 0; c < image.Channels; c++)
                {
                    double val = NumOps.ToDouble(image.GetPixel(y, x, c));
                    double glowed = val + alpha * maxVal;
                    image.SetPixel(y, x, c, NumOps.FromDouble(Math.Max(0, Math.Min(maxVal, glowed))));
                }
            }
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["min_radius"] = MinRadius; p["max_radius"] = MaxRadius;
        p["num_flare_circles"] = NumFlareCircles;
        return p;
    }
}
