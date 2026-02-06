namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Plasma fractal transformation - applies a procedural plasma-like color perturbation.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class PlasmaTransform<T> : ImageAugmenterBase<T>
{
    public double MinIntensity { get; }
    public double MaxIntensity { get; }

    public PlasmaTransform(double minIntensity = 0.05, double maxIntensity = 0.2,
        double probability = 0.5) : base(probability)
    {
        MinIntensity = minIntensity; MaxIntensity = maxIntensity;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        double maxVal = data.IsNormalized ? 1.0 : 255.0;
        double intensity = context.GetRandomDouble(MinIntensity, MaxIntensity);

        // Generate plasma pattern using diamond-square algorithm approximation
        int size = Math.Max(data.Height, data.Width);

        // Multi-octave noise for each channel
        for (int c = 0; c < data.Channels; c++)
        {
            var plasma = new double[data.Height, data.Width];

            for (int octave = 0; octave < 4; octave++)
            {
                int freq = Math.Max(2, 4 << octave);
                double weight = 1.0 / (1 << octave);
                int gH = data.Height / freq + 2;
                int gW = data.Width / freq + 2;

                var grid = new double[gH, gW];
                for (int gy = 0; gy < gH; gy++)
                    for (int gx = 0; gx < gW; gx++)
                        grid[gy, gx] = context.GetRandomDouble(-1, 1);

                for (int y = 0; y < data.Height; y++)
                    for (int x = 0; x < data.Width; x++)
                    {
                        double fy = (double)y / freq;
                        double fx = (double)x / freq;
                        int y0 = (int)fy; int x0 = (int)fx;
                        int y1 = Math.Min(y0 + 1, gH - 1);
                        int x1 = Math.Min(x0 + 1, gW - 1);
                        double ty = fy - y0; double tx = fx - x0;
                        // Smoothstep
                        ty = ty * ty * (3 - 2 * ty);
                        tx = tx * tx * (3 - 2 * tx);

                        double val = grid[y0, x0] * (1 - ty) * (1 - tx) +
                                     grid[y1, x0] * ty * (1 - tx) +
                                     grid[y0, x1] * (1 - ty) * tx +
                                     grid[y1, x1] * ty * tx;
                        plasma[y, x] += val * weight;
                    }
            }

            // Apply plasma perturbation
            for (int y = 0; y < data.Height; y++)
                for (int x = 0; x < data.Width; x++)
                {
                    double val = NumOps.ToDouble(data.GetPixel(y, x, c));
                    val += plasma[y, x] * intensity * maxVal;
                    result.SetPixel(y, x, c,
                        NumOps.FromDouble(Math.Max(0, Math.Min(maxVal, val))));
                }
        }

        return result;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["min_intensity"] = MinIntensity; p["max_intensity"] = MaxIntensity;
        return p;
    }
}
