namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Simulates frost/ice crystal patterns on the image.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class Frost<T> : ImageAugmenterBase<T>
{
    public double MinSeverity { get; }
    public double MaxSeverity { get; }

    public Frost(double minSeverity = 0.2, double maxSeverity = 0.7,
        double probability = 0.5) : base(probability)
    {
        MinSeverity = minSeverity; MaxSeverity = maxSeverity;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        double maxVal = data.IsNormalized ? 1.0 : 255.0;
        double severity = context.GetRandomDouble(MinSeverity, MaxSeverity);

        // Generate frost pattern using diamond-square-like noise
        var frostPattern = new double[data.Height, data.Width];
        int gridSize = Math.Max(4, Math.Min(data.Height, data.Width) / 16);

        // Multiple octaves for realistic frost texture
        for (int octave = 0; octave < 3; octave++)
        {
            int currentGrid = gridSize * (1 << octave);
            if (currentGrid < 2) currentGrid = 2;
            double weight = 1.0 / (1 << octave);

            int gH = data.Height / currentGrid + 2;
            int gW = data.Width / currentGrid + 2;
            var grid = new double[gH, gW];
            for (int gy = 0; gy < gH; gy++)
                for (int gx = 0; gx < gW; gx++)
                    grid[gy, gx] = context.GetRandomDouble(0, 1);

            for (int y = 0; y < data.Height; y++)
                for (int x = 0; x < data.Width; x++)
                {
                    double fy = (double)y / currentGrid;
                    double fx = (double)x / currentGrid;
                    int y0 = (int)fy; int x0 = (int)fx;
                    int y1 = Math.Min(y0 + 1, gH - 1);
                    int x1 = Math.Min(x0 + 1, gW - 1);
                    double ty = fy - y0; double tx = fx - x0;

                    double val = grid[y0, x0] * (1 - ty) * (1 - tx) +
                                 grid[y1, x0] * ty * (1 - tx) +
                                 grid[y0, x1] * (1 - ty) * tx +
                                 grid[y1, x1] * ty * tx;

                    frostPattern[y, x] += val * weight;
                }
        }

        // Normalize frost pattern to [0, 1]
        double minP = double.MaxValue, maxP = double.MinValue;
        for (int y = 0; y < data.Height; y++)
            for (int x = 0; x < data.Width; x++)
            {
                if (frostPattern[y, x] < minP) minP = frostPattern[y, x];
                if (frostPattern[y, x] > maxP) maxP = frostPattern[y, x];
            }

        double range = maxP - minP;
        if (range < 1e-10) range = 1;

        for (int y = 0; y < data.Height; y++)
            for (int x = 0; x < data.Width; x++)
            {
                double frost = (frostPattern[y, x] - minP) / range;
                frost = frost * frost; // Sharpen the pattern
                double alpha = frost * severity;

                for (int c = 0; c < data.Channels; c++)
                {
                    double val = NumOps.ToDouble(data.GetPixel(y, x, c));
                    double frosted = val * (1 - alpha) + maxVal * alpha;
                    result.SetPixel(y, x, c, NumOps.FromDouble(Math.Max(0, Math.Min(maxVal, frosted))));
                }
            }

        return result;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["min_severity"] = MinSeverity; p["max_severity"] = MaxSeverity;
        return p;
    }
}
